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
class CBPConfig:
    enabled: bool
    replacement_rate: float
    decay_rate: float
    maturity_threshold: int
    utility: str


@dataclasses.dataclass(frozen=True)
class CurriculumConfig:
    enabled: bool
    stage_steps: int
    neutral_dead_end: bool


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
    cbp: CBPConfig
    curriculum: CurriculumConfig
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
    expected = {
        "experiment",
        "environment",
        "ppo",
        "cbp",
        "curriculum",
        "diagnostics",
    }
    if set(raw) != expected:
        raise ValueError(f"Config sections must be {sorted(expected)}, got {sorted(raw)}")
    config = Config(
        experiment=_convert(ExperimentConfig, raw["experiment"]),
        environment=_convert(EnvironmentConfig, raw["environment"]),
        ppo=_convert(PPOConfig, raw["ppo"]),
        cbp=_convert(CBPConfig, raw["cbp"]),
        curriculum=_convert(CurriculumConfig, raw["curriculum"]),
        diagnostics=_convert(DiagnosticConfig, raw["diagnostics"]),
    )
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    exp, env, ppo, cbp, curriculum = (
        config.experiment,
        config.environment,
        config.ppo,
        config.cbp,
        config.curriculum,
    )
    if not exp.diversities or any(value < 1 for value in exp.diversities):
        raise ValueError("Diversity levels must be positive")
    if tuple(sorted(set(exp.diversities))) != tuple(exp.diversities):
        raise ValueError("Diversity levels must be unique and increasing")
    if not exp.seeds or len(set(exp.seeds)) != len(exp.seeds):
        raise ValueError("Training seeds must be nonempty and unique")
    if exp.num_distributions < 1:
        raise ValueError("num_distributions must be positive")
    if not 1 <= env.min_depth <= env.max_depth:
        raise ValueError("Invalid depth range")
    if exp.steps_per_distribution < ppo.rollout_steps * env.num_envs:
        raise ValueError("Each phase must contain at least one PPO rollout")
    if ppo.minibatch_envs > env.num_envs or env.num_envs % ppo.minibatch_envs:
        raise ValueError("minibatch_envs must evenly divide num_envs")
    batch_steps = ppo.rollout_steps * env.num_envs
    if exp.steps_per_distribution % batch_steps:
        raise ValueError("steps_per_distribution must contain complete PPO rollout batches")
    if exp.eval_interval_steps % batch_steps:
        raise ValueError("eval_interval_steps must align to PPO rollout batches")
    if not 0.0 <= cbp.replacement_rate <= 1.0:
        raise ValueError("CBP replacement_rate must be in [0, 1]")
    if not 0.0 <= cbp.decay_rate < 1.0:
        raise ValueError("CBP decay_rate must be in [0, 1)")
    if cbp.maturity_threshold < 0:
        raise ValueError("CBP maturity_threshold must be nonnegative")
    if cbp.utility != "contribution":
        raise ValueError("CBP utility must be 'contribution'")
    if curriculum.stage_steps < 0:
        raise ValueError("Curriculum stage_steps must be nonnegative")
    if curriculum.enabled:
        if curriculum.stage_steps < batch_steps:
            raise ValueError("An enabled curriculum stage must contain at least one PPO rollout")
        if curriculum.stage_steps % batch_steps:
            raise ValueError("Curriculum stage_steps must align to PPO rollout batches")
        curriculum_steps = curriculum.stage_steps * (
            env.max_depth - env.min_depth + 1
        )
        if curriculum_steps >= exp.steps_per_distribution:
            raise ValueError(
                "The curriculum must leave a nonempty all-depth, original-reward training period"
            )
    if any(not 0.0 <= fraction <= 1.0 for fraction in exp.diagnostic_fractions):
        raise ValueError("Diagnostic fractions must be in [0, 1]")
