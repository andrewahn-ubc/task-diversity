from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

from .config import load_config
from .continual_adam import ContinualAdam
from .continual_backprop import ContinualBackprop
from .env import VectorBanyan
from .layouts import build_layout_catalog
from .model import RecurrentActorCritic
from .taskgen import build_catalog


NARVAL_VERSIONS = {"matplotlib": "3.9.2", "numpy": "1.26.4", "torch": "2.6.0"}


def run_preflight(
    config_path: str,
    require_cuda: bool,
    require_python_311: bool = False,
    require_narval_runtime: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    if require_python_311 and sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Python 3.11 is required, found {platform.python_version()}")
    installed_versions = {
        package: importlib.metadata.version(package) for package in NARVAL_VERSIONS
    }
    if require_narval_runtime:
        base_versions = {
            package: version.split("+", 1)[0]
            for package, version in installed_versions.items()
        }
        if base_versions != NARVAL_VERSIONS:
            raise RuntimeError(
                f"Narval package mismatch: expected {NARVAL_VERSIONS}, found {installed_versions}"
            )
        non_alliance = {
            package: version
            for package, version in installed_versions.items()
            if "+computecanada" not in version
        }
        if non_alliance:
            raise RuntimeError(
                f"Expected Alliance +computecanada direct wheels, found {non_alliance}"
            )
        if torch.version.cuda is None:
            raise RuntimeError(
                "The installed Torch wheel is CPU-only; Narval requires an Alliance CUDA build"
            )
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight failed: torch.cuda.is_available() is false")
    device = torch.device("cuda" if require_cuda else "cpu")
    if require_cuda:
        capability = torch.cuda.get_device_capability(device)
        if capability < (8, 0):
            raise RuntimeError(f"An Ampere-or-newer GPU is required, found capability {capability}")
        if "sm_80" not in torch.cuda.get_arch_list():
            raise RuntimeError(
                f"Torch lacks A100 sm_80 kernels: compiled arches={torch.cuda.get_arch_list()}"
            )
        if not torch.backends.cudnn.is_available():
            raise RuntimeError("Torch cannot access its cuDNN runtime")
    catalog = build_catalog(
        config.experiment.num_distributions,
        max(config.experiment.diversities),
        config.environment.max_depth,
        config.environment.max_leaves,
        config.experiment.task_seed,
    )
    layout_catalog = build_layout_catalog(
        max(config.experiment.diversities),
        config.environment.grid_size,
        config.environment.max_leaves,
        config.experiment.task_seed + 7_919,
    )
    layout_signatures = {
        layout.cpu().numpy().tobytes() for layout in layout_catalog.walls
    }
    if len(layout_signatures) != max(config.experiment.diversities):
        raise RuntimeError("Layout generator did not produce the requested distinct layouts")
    phase_sets = [
        {
            catalog.tasks[int(task_id)].topology_signature
            for task_id in catalog.task_ids(phase, max(config.experiment.diversities)).tolist()
            if catalog.tasks[int(task_id)].depth == config.environment.max_depth
        }
        for phase in range(config.experiment.num_distributions)
    ]
    for left in range(len(phase_sets)):
        for right in range(left + 1, len(phase_sets)):
            if phase_sets[left] & phase_sets[right]:
                raise RuntimeError(f"Topology overlap between phases {left + 1} and {right + 1}")
    for phase in range(config.experiment.num_distributions):
        anchor_id = catalog.phase_topology_tasks[phase][0][0]
        anchor = catalog.tasks[anchor_id]
        if anchor.depth != config.environment.min_depth or len(anchor.binary_rules) != 1:
            raise RuntimeError(
                f"Phase {phase + 1} does not begin with a one-merge curriculum task"
            )
    slot_distances = (
        layout_catalog.object_slots - layout_catalog.agent_starts[:, None, :]
    ).abs().sum(dim=2)
    if int(slot_distances.max()) > 3:
        raise RuntimeError("Object placement is not compact enough for the merge curriculum")
    probe_env = VectorBanyan(
        catalog,
        catalog.task_ids(0, 1),
        num_envs=8,
        grid_size=config.environment.grid_size,
        max_episode_steps=config.environment.max_episode_steps,
        device=device,
        seed=123,
    )
    obs, reward, done, info = probe_env.step(torch.zeros(8, dtype=torch.int64, device=device))
    if obs.objects.shape != (8, config.environment.grid_size, config.environment.grid_size):
        raise RuntimeError("Environment observation shape check failed")
    if not torch.isfinite(reward).all() or done.shape != (8,):
        raise RuntimeError("Environment step check failed")
    environment_state = probe_env.state_dict()
    probe_env.reset()
    expected_task_index = probe_env.task_index.clone()
    expected_objects = probe_env.objects.clone()
    probe_env.load_state_dict(environment_state)
    probe_env.reset()
    if not torch.equal(probe_env.task_index, expected_task_index) or not torch.equal(
        probe_env.objects, expected_objects
    ):
        raise RuntimeError("Environment checkpoint RNG restoration is not deterministic")
    update_generator = torch.Generator(device=device)
    update_generator.manual_seed(991)
    update_generator_state = update_generator.get_state().cpu()
    expected_random = torch.rand(8, generator=update_generator, device=device)
    update_generator.set_state(update_generator_state)
    actual_random = torch.rand(8, generator=update_generator, device=device)
    if not torch.equal(actual_random, expected_random):
        raise RuntimeError("PPO generator checkpoint restoration is not deterministic")
    cpu_rng_state = torch.get_rng_state().cpu()
    torch.set_rng_state(cpu_rng_state)
    if require_cuda:
        cuda_rng_states = [state.cpu() for state in torch.cuda.get_rng_state_all()]
        torch.cuda.set_rng_state_all(cuda_rng_states)
    probe_model = RecurrentActorCritic(
        grid_size=config.environment.grid_size,
        object_feature_dim=config.environment.object_feature_dim,
        hidden_size=32,
        walls=probe_env.walls[0],
    ).to(device)
    logits, values, hidden = probe_model.forward_step(
        obs,
        probe_model.initial_hidden(8, device),
        torch.ones(8, dtype=torch.bool, device=device),
    )
    probe_loss = logits.square().mean() + values.square().mean() + hidden.square().mean()
    probe_optimizer = ContinualAdam(
        probe_model.parameters(), lr=config.ppo.learning_rate, eps=1e-5
    )
    probe_loss.backward()
    probe_optimizer.step()
    cbp_probe_ok = False
    if config.cbp.enabled:
        probe_cbp = ContinualBackprop(
            probe_model,
            probe_optimizer,
            dataclasses.replace(
                config.cbp, replacement_rate=1.0, maturity_threshold=0
            ),
        )
        probe_stats = {
            "conv1": (torch.ones(32, device=device), torch.ones(32, device=device)),
            "conv2": (
                torch.ones(32, config.environment.grid_size**2, device=device),
                torch.ones(32, config.environment.grid_size**2, device=device),
            ),
            "pre_gru": (torch.ones(32, device=device), torch.ones(32, device=device)),
            "gru": (torch.ones(32, device=device), torch.ones(32, device=device)),
        }
        expected_replacements = {"conv1": 32, "conv2": 32, "pre_gru": 32, "gru": 32}
        if probe_cbp.step(probe_stats) != expected_replacements:
            raise RuntimeError("PPO + CBP optimizer/reset probe failed")
        cbp_probe_ok = True
    if require_cuda:
        torch.cuda.synchronize(device)
    if not torch.isfinite(probe_loss):
        raise RuntimeError("Convolution/GRU/PPO optimizer probe returned a non-finite loss")
    payload: dict[str, Any] = {
        "status": "ok",
        "python": platform.python_version(),
        "python_311": sys.version_info[:2] == (3, 11),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "installed_versions": installed_versions,
        "platform": platform.platform(),
        "config": str(config_path),
        "config_fingerprint": config.fingerprint(),
        "catalog_tasks": len(catalog.tasks),
        "topologies_per_phase": [len(items) for items in catalog.phase_topology_tasks],
        "available_layouts": layout_catalog.count,
        "rollout_batch_env_steps": (
            config.environment.num_envs * config.ppo.rollout_steps
        ),
        "ppo_updates_per_distribution": (
            config.experiment.steps_per_distribution
            // (config.environment.num_envs * config.ppo.rollout_steps)
        ),
        "curriculum_enabled": config.curriculum.enabled,
        "curriculum_stage_steps": config.curriculum.stage_steps,
        "curriculum_total_steps": config.curriculum.stage_steps
        * (config.environment.max_depth - config.environment.min_depth + 1),
        "curriculum_neutral_dead_end": config.curriculum.neutral_dead_end,
        "anchor_depth1_binary": True,
        "maximum_object_slot_manhattan_distance": int(slot_distances.max()),
        "cuda_required": require_cuda,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if require_cuda else None,
        "gpu_capability": torch.cuda.get_device_capability(device) if require_cuda else None,
        "torch_arch_list": torch.cuda.get_arch_list() if require_cuda else None,
        "cudnn_version": torch.backends.cudnn.version() if require_cuda else None,
        "cbp_enabled": config.cbp.enabled,
        "cbp_optimizer_reset_probe": cbp_probe_ok,
        "checkpoint_rng_restore_probe": True,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-python-311", action="store_true")
    parser.add_argument("--require-narval-runtime", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    payload = run_preflight(
        args.config,
        args.require_cuda,
        args.require_python_311,
        args.require_narval_runtime,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.replace(temporary, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
