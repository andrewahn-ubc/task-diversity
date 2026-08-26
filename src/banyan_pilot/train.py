from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import signal
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Config, load_config
from .continual_adam import ContinualAdam
from .continual_backprop import ContinualBackprop
from .env import CompactObs, VectorBanyan
from .model import RecurrentActorCritic
from .ppo import collect_rollout, evaluate_policy, gradient_cosines, update_ppo
from .taskgen import TaskCatalog, build_catalog


REQUEUE_EXIT_CODE = 75


class RequeueRequested:
    value = False


def _handle_signal(signum: int, frame: Any) -> None:
    del signum, frame
    RequeueRequested.value = True


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _jsonl_append(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _device_from_argument(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class Trainer:
    def __init__(
        self,
        config: Config,
        *,
        diversity: int,
        seed: int,
        output_root: Path,
        device: torch.device,
    ) -> None:
        self.config = config
        self.diversity = diversity
        self.seed = seed
        self.device = device
        if diversity not in config.experiment.diversities:
            raise ValueError(f"Diversity {diversity} is not configured")
        if seed not in config.experiment.seeds:
            raise ValueError(f"Seed {seed} is not configured")
        self.run_dir = output_root / f"n{diversity}" / f"seed_{seed}"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.diagnostics_path = self.run_dir / "diagnostics.jsonl"
        self.checkpoint_path = self.run_dir / "checkpoints" / "latest.pt"
        self.completed_path = self.run_dir / "completed.json"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.catalog: TaskCatalog = build_catalog(
            config.experiment.num_distributions,
            max(config.experiment.diversities),
            config.environment.max_depth,
            config.environment.max_leaves,
            config.experiment.task_seed,
        )
        env_seed = 10_000_019 * seed + 1009 * diversity + config.experiment.task_seed
        initial_ids = self.catalog.task_ids(0, diversity)
        self.env = self._make_env(initial_ids, env_seed, config.environment.num_envs)
        self.model = RecurrentActorCritic(
            grid_size=config.environment.grid_size,
            object_feature_dim=config.environment.object_feature_dim,
            hidden_size=config.ppo.hidden_size,
            walls=self.env.walls,
        ).to(device)
        if config.cbp.enabled:
            self.optimizer: torch.optim.Optimizer = ContinualAdam(
                self.model.parameters(), lr=config.ppo.learning_rate, eps=1e-5
            )
            self.continual_backprop: ContinualBackprop | None = ContinualBackprop(
                self.model, self.optimizer, config.cbp
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=config.ppo.learning_rate, eps=1e-5
            )
            self.continual_backprop = None
        self.update_generator = torch.Generator(device=device)
        self.update_generator.manual_seed(env_seed + 31)
        self.phase = 0
        self.phase_steps = 0
        self.total_steps = 0
        self.update_index = 0
        self.obs = self.env.observe()
        self.hidden = self.model.initial_hidden(self.env.num_envs, device)
        self.episode_start = torch.ones(self.env.num_envs, dtype=torch.bool, device=device)
        self.logged_diagnostics: set[tuple[int, int]] = set()
        self.logged_evaluations: set[tuple[int, int, str]] = set()
        self.started_at = time.time()

    def _make_env(
        self, task_ids: torch.Tensor, seed: int, num_envs: int | None = None
    ) -> VectorBanyan:
        return VectorBanyan(
            self.catalog,
            task_ids,
            num_envs=num_envs or self.config.environment.num_envs,
            grid_size=self.config.environment.grid_size,
            max_episode_steps=self.config.environment.max_episode_steps,
            device=self.device,
            seed=seed,
        )

    def _eval_factory(self, task_ids: torch.Tensor, seed: int) -> VectorBanyan:
        num_envs = min(self.config.diagnostics.num_envs, self.config.experiment.eval_episodes)
        return self._make_env(task_ids, seed, num_envs)

    def _evaluation(self, phase: int, phase_steps: int, kind: str) -> dict[str, Any]:
        key = (phase, phase_steps, kind)
        if key in self.logged_evaluations:
            return {}
        eval_seed = self.seed * 1_000_003 + phase * 10_007 + phase_steps + 17
        task_ids = self.catalog.task_ids(phase, self.diversity)
        self.model.eval()
        result = evaluate_policy(
            self.model,
            task_ids,
            self._eval_factory,
            episodes=self.config.experiment.eval_episodes,
            seed=eval_seed,
        )
        payload: dict[str, Any] = {
            "event": "evaluation",
            "kind": kind,
            "seed": self.seed,
            "diversity": self.diversity,
            "phase": phase + 1,
            "phase_env_steps": phase_steps,
            "total_env_steps": phase * self.config.experiment.steps_per_distribution
            + phase_steps,
            "wall_time_seconds": time.time() - self.started_at,
            **result,
        }
        _jsonl_append(self.metrics_path, payload)
        self.logged_evaluations.add(key)
        self.model.train()
        return payload

    def _backward_d1_evaluation(self, after_phase: int) -> None:
        eval_seed = self.seed * 1_000_003 + after_phase * 20_011 + 29
        self.model.eval()
        result = evaluate_policy(
            self.model,
            self.catalog.task_ids(0, self.diversity),
            self._eval_factory,
            episodes=self.config.experiment.eval_episodes,
            seed=eval_seed,
        )
        payload: dict[str, Any] = {
            "event": "backward_evaluation",
            "kind": "phase_end_d1",
            "seed": self.seed,
            "diversity": self.diversity,
            "phase": 1,
            "after_phase": after_phase + 1,
            "total_env_steps": (after_phase + 1)
            * self.config.experiment.steps_per_distribution,
            "wall_time_seconds": time.time() - self.started_at,
            **result,
        }
        _jsonl_append(self.metrics_path, payload)
        self.model.train()

    def _diagnostic(self, phase: int, checkpoint_steps: int) -> None:
        if phase == 0 or (phase, checkpoint_steps) in self.logged_diagnostics:
            return
        diag = self.config.diagnostics
        current_ids = self.catalog.task_ids(phase, self.diversity)
        previous_ids = self.catalog.mixed_task_ids(range(phase), self.diversity)
        base_seed = self.seed * 10_000_019 + phase * 100_003 + checkpoint_steps
        current_env = self._make_env(current_ids, base_seed + 1, diag.num_envs)
        previous_env = self._make_env(previous_ids, base_seed + 2, diag.num_envs)
        self.model.train()
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(base_seed + 3)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(base_seed + 3)
            current_rollout = collect_rollout(
                self.model,
                current_env,
                current_env.observe(),
                self.model.initial_hidden(diag.num_envs, self.device),
                torch.ones(diag.num_envs, dtype=torch.bool, device=self.device),
                rollout_steps=diag.rollout_steps,
                gamma=self.config.ppo.gamma,
                gae_lambda=self.config.ppo.gae_lambda,
            )
            previous_rollout = collect_rollout(
                self.model,
                previous_env,
                previous_env.observe(),
                self.model.initial_hidden(diag.num_envs, self.device),
                torch.ones(diag.num_envs, dtype=torch.bool, device=self.device),
                rollout_steps=diag.rollout_steps,
                gamma=self.config.ppo.gamma,
                gae_lambda=self.config.ppo.gae_lambda,
            )
            cosines = gradient_cosines(
                self.model, current_rollout, previous_rollout, self.config.ppo
            )
        checkpoint_eval = evaluate_policy(
            self.model,
            current_ids,
            self._eval_factory,
            episodes=diag.eval_episodes,
            seed=base_seed + 4,
        )
        payload: dict[str, Any] = {
            "event": "gradient_diagnostic",
            "seed": self.seed,
            "diversity": self.diversity,
            "phase": phase + 1,
            "phase_env_steps": checkpoint_steps,
            "phase_fraction": checkpoint_steps
            / self.config.experiment.steps_per_distribution,
            "total_env_steps": phase * self.config.experiment.steps_per_distribution
            + checkpoint_steps,
            "previous_distribution_count": phase,
            "checkpoint_success_rate": checkpoint_eval["success_rate"],
            "current_batch_mean_reward": float(current_rollout.rewards.mean().cpu()),
            "previous_batch_mean_reward": float(previous_rollout.rewards.mean().cpu()),
            "wall_time_seconds": time.time() - self.started_at,
            **cosines,
        }
        _jsonl_append(self.diagnostics_path, payload)
        self.logged_diagnostics.add((phase, checkpoint_steps))

    def save_checkpoint(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "config_fingerprint": self.config.fingerprint(),
            "diversity": self.diversity,
            "seed": self.seed,
            "phase": self.phase,
            "phase_steps": self.phase_steps,
            "total_steps": self.total_steps,
            "update_index": self.update_index,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "continual_backprop": (
                self.continual_backprop.state_dict()
                if self.continual_backprop is not None
                else None
            ),
            "env": self.env.state_dict(),
            "obs": tuple(value.cpu() for value in self.obs.as_tuple()),
            "hidden": self.hidden.cpu(),
            "episode_start": self.episode_start.cpu(),
            "update_generator": self.update_generator.get_state().cpu(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "logged_diagnostics": sorted(self.logged_diagnostics),
            "logged_evaluations": sorted(self.logged_evaluations),
        }
        with tempfile.NamedTemporaryFile(
            dir=self.checkpoint_path.parent, prefix="checkpoint-", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, self.checkpoint_path)
        finally:
            temporary.unlink(missing_ok=True)

    def load_checkpoint(self) -> bool:
        if not self.checkpoint_path.exists():
            return False
        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        expected = (self.config.fingerprint(), self.diversity, self.seed)
        actual = (
            checkpoint["config_fingerprint"],
            checkpoint["diversity"],
            checkpoint["seed"],
        )
        if actual != expected:
            raise RuntimeError(f"Checkpoint identity mismatch: expected {expected}, got {actual}")
        self.phase = checkpoint["phase"]
        self.phase_steps = checkpoint["phase_steps"]
        self.total_steps = checkpoint["total_steps"]
        self.update_index = checkpoint["update_index"]
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.continual_backprop is not None:
            if checkpoint.get("continual_backprop") is None:
                raise RuntimeError("CBP checkpoint state is missing")
            self.continual_backprop.load_state_dict(checkpoint["continual_backprop"])
        elif checkpoint.get("continual_backprop") is not None:
            raise RuntimeError("Unexpected CBP state in a plain-PPO checkpoint")
        self.env.load_state_dict(checkpoint["env"])
        self.obs = CompactObs(*(value.to(self.device) for value in checkpoint["obs"]))
        self.hidden = checkpoint["hidden"].to(self.device)
        self.episode_start = checkpoint["episode_start"].to(self.device)
        self.update_generator.set_state(checkpoint["update_generator"].to(self.device))
        torch.set_rng_state(checkpoint["torch_rng"])
        if torch.cuda.is_available() and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        self.logged_diagnostics = {tuple(item) for item in checkpoint["logged_diagnostics"]}
        self.logged_evaluations = {tuple(item) for item in checkpoint["logged_evaluations"]}
        return True

    def _write_run_metadata(self) -> None:
        payload = {
            "config": dataclasses.asdict(self.config),
            "config_fingerprint": self.config.fingerprint(),
            "diversity": self.diversity,
            "seed": self.seed,
            "device": str(self.device),
            "algorithm": "ppo_cbp" if self.continual_backprop is not None else "ppo",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "started_unix": self.started_at,
        }
        _json_dump(self.run_dir / "run.json", payload)

    def run(self, *, stop_after_distributions: int | None = None) -> int:
        if self.completed_path.exists():
            completed = json.loads(self.completed_path.read_text(encoding="utf-8"))
            if completed.get("config_fingerprint") != self.config.fingerprint():
                raise RuntimeError(
                    f"Completed run at {self.run_dir} belongs to a different configuration; "
                    "use a new output root and preserve the existing result"
                )
            print(f"Run already complete: {self.completed_path}", flush=True)
            return 0
        resumed = self.load_checkpoint()
        if not resumed:
            self._write_run_metadata()
        batch_steps = self.config.environment.num_envs * self.config.ppo.rollout_steps
        target = self.config.experiment.steps_per_distribution
        if target % batch_steps:
            raise ValueError(
                f"steps_per_distribution={target} must be divisible by rollout batch={batch_steps}"
            )
        phase_limit = self.config.experiment.num_distributions
        if stop_after_distributions is not None:
            if not 1 <= stop_after_distributions <= phase_limit:
                raise ValueError(
                    "stop_after_distributions must be between 1 and "
                    f"{phase_limit}, got {stop_after_distributions}"
                )
            phase_limit = stop_after_distributions
        while self.phase < phase_limit:
            phase = self.phase
            if self.phase_steps == 0:
                self.env.set_task_pool(self.catalog.task_ids(phase, self.diversity))
                self.obs = self.env.observe()
                self.hidden.zero_()
                self.episode_start.fill_(True)
                self._evaluation(phase, 0, "phase_start")
                self._diagnostic(phase, 0)
                self.save_checkpoint()
            diagnostic_steps = {
                int(round(fraction * target)) for fraction in self.config.experiment.diagnostic_fractions
            }
            while self.phase_steps < target:
                rollout = collect_rollout(
                    self.model,
                    self.env,
                    self.obs,
                    self.hidden,
                    self.episode_start,
                    rollout_steps=self.config.ppo.rollout_steps,
                    gamma=self.config.ppo.gamma,
                    gae_lambda=self.config.ppo.gae_lambda,
                )
                losses = update_ppo(
                    self.model,
                    self.optimizer,
                    rollout,
                    self.config.ppo,
                    self.update_generator,
                    self.continual_backprop,
                )
                self.obs = rollout.next_obs
                self.hidden = rollout.next_hidden.detach()
                self.episode_start = rollout.next_episode_start
                self.phase_steps += batch_steps
                self.total_steps += batch_steps
                self.update_index += 1
                if self.phase_steps % self.config.experiment.eval_interval_steps == 0:
                    evaluation = self._evaluation(phase, self.phase_steps, "periodic")
                    _jsonl_append(
                        self.metrics_path,
                        {
                            "event": "optimization",
                            "seed": self.seed,
                            "diversity": self.diversity,
                            "phase": phase + 1,
                            "phase_env_steps": self.phase_steps,
                            "total_env_steps": self.total_steps,
                            **losses,
                        },
                    )
                if self.phase_steps in diagnostic_steps:
                    self._diagnostic(phase, self.phase_steps)
                if (
                    self.update_index % self.config.ppo.checkpoint_interval_updates == 0
                    or RequeueRequested.value
                ):
                    self.save_checkpoint()
                if RequeueRequested.value:
                    print("Saved checkpoint after USR1; requesting SLURM requeue", flush=True)
                    return REQUEUE_EXIT_CODE
            self._evaluation(phase, target, "phase_end")
            self._diagnostic(phase, target)
            self._backward_d1_evaluation(phase)
            self.phase += 1
            self.phase_steps = 0
            self.save_checkpoint()
        if self.phase < self.config.experiment.num_distributions:
            pilot = {
                "status": "pilot_complete",
                "completed_distributions": self.phase,
                "seed": self.seed,
                "diversity": self.diversity,
                "total_env_steps": self.total_steps,
                "wall_time_seconds": time.time() - self.started_at,
                "config_fingerprint": self.config.fingerprint(),
                "algorithm": "ppo_cbp" if self.continual_backprop is not None else "ppo",
                "cbp_total_replacements": (
                    self.continual_backprop.totals()
                    if self.continual_backprop is not None
                    else None
                ),
            }
            _json_dump(self.run_dir / "pilot_complete.json", pilot)
            print(
                f"Pilot checkpoint complete after {self.phase} distribution(s): "
                f"{self.run_dir}",
                flush=True,
            )
            return 0
        completed = {
            "status": "complete",
            "seed": self.seed,
            "diversity": self.diversity,
            "total_env_steps": self.total_steps,
            "wall_time_seconds": time.time() - self.started_at,
            "config_fingerprint": self.config.fingerprint(),
            "algorithm": "ppo_cbp" if self.continual_backprop is not None else "ppo",
            "cbp_total_replacements": (
                self.continual_backprop.totals()
                if self.continual_backprop is not None
                else None
            ),
        }
        _json_dump(self.completed_path, completed)
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--diversity", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--override-steps", type=int)
    parser.add_argument("--stop-after-distributions", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.override_steps is not None:
        batch_steps = config.environment.num_envs * config.ppo.rollout_steps
        rounded = max(batch_steps, (args.override_steps // batch_steps) * batch_steps)
        experiment = dataclasses.replace(
            config.experiment,
            steps_per_distribution=rounded,
            eval_interval_steps=rounded,
            eval_episodes=min(64, config.experiment.eval_episodes),
            diagnostic_fractions=(0.0, 1.0),
        )
        diagnostics = dataclasses.replace(
            config.diagnostics,
            num_envs=min(32, config.diagnostics.num_envs),
            rollout_steps=min(16, config.diagnostics.rollout_steps),
            eval_episodes=min(64, config.diagnostics.eval_episodes),
        )
        config = dataclasses.replace(config, experiment=experiment, diagnostics=diagnostics)
    output_root = Path(args.output_root or config.experiment.output_root)
    device = _device_from_argument(args.device)
    _seed_everything(args.seed)
    signal.signal(signal.SIGUSR1, _handle_signal)
    trainer = Trainer(
        config,
        diversity=args.diversity,
        seed=args.seed,
        output_root=output_root,
        device=device,
    )
    return trainer.run(stop_after_distributions=args.stop_after_distributions)


if __name__ == "__main__":
    raise SystemExit(main())
