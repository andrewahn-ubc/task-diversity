from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    diversities: tuple[int, ...]
    seeds: tuple[int, ...]
    num_distributions: int
    task_seed: int
    steps_per_distribution: int
    eval_interval_steps: int
    eval_episodes: int
    diagnostic_fractions: tuple[float, ...]
    output_root: str


@dataclasses.dataclass(frozen=True)
class EnvironmentConfig:
    grid_size: int
    max_depth: int
    min_depth: int
    max_leaves: int
    object_feature_dim: int
    max_episode_steps: int
    num_envs: int


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int
    update_epochs: int
    minibatch_envs: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_coef: float
    value_coef: float
    entropy_coef: float
    max_grad_norm: float
    hidden_size: int
    checkpoint_interval_updates: int


@dataclasses.dataclass(frozen=True)
class DiagnosticConfig:
    num_envs: int
    rollout_steps: int
    eval_episodes: int


@dataclasses.dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig
    environment: EnvironmentConfig
    ppo: PPOConfig
    diagnostics: DiagnosticConfig

    def fingerprint(self) -> str:
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def _convert(cls: type[Any], values: dict[str, Any]) -> Any:
    fields = {field.name: field for field in dataclasses.fields(cls)}
    unknown = set(values) - set(fields)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    converted = dict(values)
    for name, field in fields.items():
        if name not in converted:
            raise ValueError(f"Missing {cls.__name__}.{name}")
        if str(field.type).startswith("tuple") or "tuple[" in str(field.type):
            converted[name] = tuple(converted[name])
    return cls(**converted)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    expected = {"experiment", "environment", "ppo", "diagnostics"}
    if set(raw) != expected:
        raise ValueError(f"Config sections must be {sorted(expected)}, got {sorted(raw)}")
    config = Config(
        experiment=_convert(ExperimentConfig, raw["experiment"]),
        environment=_convert(EnvironmentConfig, raw["environment"]),
        ppo=_convert(PPOConfig, raw["ppo"]),
        diagnostics=_convert(DiagnosticConfig, raw["diagnostics"]),
    )
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    exp, env, ppo = config.experiment, config.environment, config.ppo
    if tuple(exp.diversities) != (1, 16, 256):
        raise ValueError("Primary diversity levels must remain exactly (1, 16, 256)")
    if exp.num_distributions != 4:
        raise ValueError("The pilot requires exactly four distributions")
    if not 1 <= env.min_depth <= env.max_depth:
        raise ValueError("Invalid depth range")
    if exp.steps_per_distribution < ppo.rollout_steps * env.num_envs:
        raise ValueError("Each phase must contain at least one PPO rollout")
    if ppo.minibatch_envs > env.num_envs or env.num_envs % ppo.minibatch_envs:
        raise ValueError("minibatch_envs must evenly divide num_envs")
    if any(not 0.0 <= fraction <= 1.0 for fraction in exp.diagnostic_fractions):
        raise ValueError("Diagnostic fractions must be in [0, 1]")
