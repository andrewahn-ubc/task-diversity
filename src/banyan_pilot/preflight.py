from __future__ import annotations

import argparse
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
from .env import VectorBanyan
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
    if require_cuda:
        probe_model = RecurrentActorCritic(
            grid_size=config.environment.grid_size,
            object_feature_dim=config.environment.object_feature_dim,
            hidden_size=32,
            walls=probe_env.walls,
        ).to(device)
        logits, values, hidden = probe_model.forward_step(
            obs,
            probe_model.initial_hidden(8, device),
            torch.ones(8, dtype=torch.bool, device=device),
        )
        probe_loss = logits.square().mean() + values.square().mean() + hidden.square().mean()
        probe_loss.backward()
        torch.cuda.synchronize(device)
        if not torch.isfinite(probe_loss):
            raise RuntimeError("CUDA convolution/GRU/autograd probe returned a non-finite loss")
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
        "cuda_required": require_cuda,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if require_cuda else None,
        "gpu_capability": torch.cuda.get_device_capability(device) if require_cuda else None,
        "torch_arch_list": torch.cuda.get_arch_list() if require_cuda else None,
        "cudnn_version": torch.backends.cudnn.version() if require_cuda else None,
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
