import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import wandb
import yaml

from banyan_grid.utils.dataset_loader import validate_packed_ruleset_dataset
from banyan_grid.utils.logging import shutdown_wandb_async
from banyan_grid.rounds.metrics import (
    tensor_to_wandb_table,
    update_transfer_tensor,
)
from banyan_grid.rounds.round import run_eval_round, run_round_groups_parallel
from banyan_grid.tasks.ruleset_dataset_compact import (
    load_packed_u32_bz2,
)

_tournament_executor: ThreadPoolExecutor | None = None
_tournament_executor_lock = Lock()


def _get_tournament_executor(max_workers: int = 8) -> ThreadPoolExecutor:
    global _tournament_executor
    with _tournament_executor_lock:
        if _tournament_executor is None:
            _tournament_executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="tournament-eval",
            )
        return _tournament_executor


def _shutdown_tournament_executor() -> None:
    global _tournament_executor
    with _tournament_executor_lock:
        if _tournament_executor is not None:
            _tournament_executor.shutdown(wait=False)
            _tournament_executor = None


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Loads a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_dataset_metadata(dataset_path: str | Path) -> dict[str, Any] | None:
    """Load metadata JSON for a dataset file.

    Expects metadata file alongside dataset: foo.npy.bz2 -> foo_meta.json
    """
    dataset_path = Path(dataset_path)
    # Strip .npy.bz2 or .npy suffix to get base name
    name = dataset_path.name
    if name.endswith(".npy.bz2"):
        base = name[:-8]
    elif name.endswith(".npy"):
        base = name[:-4]
    else:
        base = name

    meta_path = dataset_path.parent / f"{base}_meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    return None


def _resolve_config_path(
    path_value: str | Path,
    *,
    dataset_dir: str | Path | None = None,
) -> Path:
    """Resolve a config path against CWD, repo root, and optional dataset dir."""
    raw = Path(path_value).expanduser()
    if raw.is_absolute():
        return raw

    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        (Path.cwd() / raw).resolve(),
        (repo_root / raw).resolve(),
    ]
    if dataset_dir is not None:
        candidates.append((Path(dataset_dir).resolve() / raw).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _extract_serializable_model_state(state: Any) -> Any:
    """Convert trainer state objects into a serialization-friendly pytree."""
    if state is None:
        return None
    if hasattr(state, "params"):
        payload: dict[str, Any] = {"params": state.params}
        batch_stats = getattr(state, "batch_stats", None)
        if batch_stats is not None:
            payload["batch_stats"] = batch_stats
        step_val = getattr(state, "step", None)
        if step_val is not None:
            try:
                payload["step"] = int(np.asarray(jax.device_get(step_val)).item())
            except Exception:
                pass
        return payload
    if isinstance(state, dict):
        return {str(k): _extract_serializable_model_state(v) for k, v in state.items()}
    if isinstance(state, tuple):
        return tuple(_extract_serializable_model_state(v) for v in state)
    if isinstance(state, list):
        return [_extract_serializable_model_state(v) for v in state]
    return state


def save_final_model_checkpoint(
    config: dict[str, Any],
    baseline_config: dict[str, Any],
    agent_params: dict[int, Any | None],
) -> Path | None:
    """Save final agent parameters at end-of-run and optionally log to W&B."""
    save_enabled = bool(
        config.get(
            "SAVE_FINAL_MODEL",
            baseline_config.get("SAVE_FINAL_MODEL", False),
        )
    )
    if not save_enabled:
        return None

    try:
        from flax import serialization as flax_serialization
    except Exception as exc:
        print(f"Skipping final model save (flax serialization unavailable): {exc}")
        return None

    output_root = Path(config.get("output_dir", "outputs"))
    ckpt_dir = output_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.run
    run_id = run.id if run is not None else "offline"
    model_path = ckpt_dir / f"final_agent_params_{run_id}.msgpack"
    meta_path = ckpt_dir / f"final_agent_params_{run_id}.meta.json"

    serializable_params = {
        str(agent_id): _extract_serializable_model_state(state)
        for agent_id, state in sorted(agent_params.items())
    }
    payload = {
        "format": "banyan_final_agent_params_v1",
        "baseline": str(config.get("baseline", "")),
        "num_agents": int(config.get("num_agents", len(agent_params))),
        "single_agent": bool(baseline_config.get("SINGLE_AGENT", True)),
        "single_agent_id": int(baseline_config.get("SINGLE_AGENT_ID", 0)),
        "agent_params": serializable_params,
    }

    with model_path.open("wb") as f:
        f.write(flax_serialization.to_bytes(payload))

    metadata = {
        "model_file": str(model_path),
        "baseline": str(config.get("baseline", "")),
        "wandb_run_id": run_id,
        "wandb_project": str(config.get("wandb_project", "")),
        "wandb_group": str(config.get("wandb_group", "")),
        "wandb_name": str(config.get("wandb_name", "")),
        "ruleset_datasets": dict(config.get("ruleset_datasets", {})),
        "save_final_model": True,
    }
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved final model checkpoint: {model_path}")

    if run is not None:
        try:
            artifact = wandb.Artifact(name=f"final-model-{run.id}", type="model")
            artifact.add_file(str(model_path), name=model_path.name)
            artifact.add_file(str(meta_path), name=meta_path.name)
            wandb.log_artifact(artifact)
            print(f"Logged final model artifact to W&B: final-model-{run.id}")
        except Exception as exc:
            print(f"Failed to log final model artifact: {exc}")

    return model_path


def save_round_checkpoint(
    config: dict[str, Any],
    baseline_config: dict[str, Any],
    agent_params: dict[int, Any | None],
    round_idx: int,
) -> Path | None:
    """Save agent parameters after a training round for offline eval."""
    save_enabled = bool(
        config.get(
            "SAVE_ROUND_CHECKPOINTS",
            baseline_config.get("SAVE_ROUND_CHECKPOINTS", False),
        )
    )
    if not save_enabled:
        return None

    try:
        from flax import serialization as flax_serialization
    except Exception as exc:
        print(f"Skipping round checkpoint (flax serialization unavailable): {exc}")
        return None

    output_root = Path(config.get("output_dir", "outputs"))
    ckpt_dir = output_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.run
    run_id = run.id if run is not None else "offline"
    model_path = ckpt_dir / f"agent_params_round{round_idx}_{run_id}.msgpack"
    meta_path = ckpt_dir / f"agent_params_round{round_idx}_{run_id}.meta.json"

    serializable_params = {
        str(agent_id): _extract_serializable_model_state(state)
        for agent_id, state in sorted(agent_params.items())
    }
    payload = {
        "format": "banyan_round_agent_params_v1",
        "baseline": str(config.get("baseline", "")),
        "round_idx": round_idx,
        "num_agents": int(config.get("num_agents", len(agent_params))),
        "single_agent": bool(baseline_config.get("SINGLE_AGENT", True)),
        "single_agent_id": int(baseline_config.get("SINGLE_AGENT_ID", 0)),
        "agent_params": serializable_params,
    }

    with model_path.open("wb") as f:
        f.write(flax_serialization.to_bytes(payload))

    metadata = {
        "model_file": str(model_path),
        "baseline": str(config.get("baseline", "")),
        "round_idx": round_idx,
        "wandb_run_id": run_id,
        "wandb_project": str(config.get("wandb_project", "")),
        "wandb_group": str(config.get("wandb_group", "")),
        "wandb_name": str(config.get("wandb_name", "")),
        "ruleset_datasets": dict(config.get("ruleset_datasets", {})),
    }
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved round {round_idx} checkpoint: {model_path}")
    return model_path


def run_tournament(
    config: dict[str, Any] | str | Path,
    make_train,
    make_eval=None,
) -> dict[str, Any]:
    """Run a multi-round continual-learning tournament.

    Args:
        config: Tournament config (dict or path to a YAML file).
        make_train: Callable(config_dict) -> train_fn. The returned train_fn
            is called as train_fn(rng, init_params, pairing_meta) and must
            return a dict with at least "final_params".
        make_eval: Optional Callable(config_dict) -> eval_fn. The returned
            eval_fn is called as eval_fn(rng, params, eval_meta) and must
            return a mapping of metric name to value. When omitted,
            cross-round evaluation is skipped.
    """
    if isinstance(config, (str, Path)):
        config = load_yaml(config)

    baseline_config = load_yaml(config["baseline_config_path"])

    baseline = config["baseline"]
    num_agents = config["num_agents"]
    rounds = config["rounds"]
    ruleset_datasets = config["ruleset_datasets"]
    round_replay_enabled = bool(
        config.get(
            "ROUND_REPLAY_ENABLED",
            baseline_config.get("ROUND_REPLAY_ENABLED", False),
        )
    )
    round_replay_ratio = float(
        config.get("ROUND_REPLAY_RATIO", baseline_config.get("ROUND_REPLAY_RATIO", 0.0))
    )
    round_replay_max_past_rounds = int(
        config.get(
            "ROUND_REPLAY_MAX_PAST_ROUNDS",
            baseline_config.get("ROUND_REPLAY_MAX_PAST_ROUNDS", 0),
        )
    )
    round_replay_sampling = str(
        config.get(
            "ROUND_REPLAY_SAMPLING",
            baseline_config.get("ROUND_REPLAY_SAMPLING", "uniform"),
        )
    ).lower()
    if round_replay_sampling not in ("uniform", "recency"):
        raise ValueError(
            "ROUND_REPLAY_SAMPLING must be one of: uniform, recency "
            f"(got {round_replay_sampling!r})"
        )
    if round_replay_ratio < 0.0:
        raise ValueError(
            f"ROUND_REPLAY_RATIO must be non-negative (got {round_replay_ratio})"
        )
    if round_replay_max_past_rounds < 0:
        raise ValueError(
            "ROUND_REPLAY_MAX_PAST_ROUNDS must be >= 0 "
            f"(got {round_replay_max_past_rounds})"
        )
    reset_optimizer_each_round = bool(
        config.get(
            "reset_optimizer_each_round",
            config.get("RESET_OPTIMIZER_EACH_ROUND", False),
        )
    )
    log_effective_rank = bool(
        config.get(
            "LOG_EFFECTIVE_RANK_BOUNDARY",
            baseline_config.get("LOG_EFFECTIVE_RANK_BOUNDARY", True),
        )
    )

    def _extract_actor_param_tree(state: Any) -> Any | None:
        actor_state = state
        if isinstance(actor_state, dict) and "actor" in actor_state:
            actor_state = actor_state["actor"]
        if hasattr(actor_state, "params"):
            actor_state = actor_state.params
        if actor_state is None:
            return None
        host_tree = jax.device_get(actor_state)
        return host_tree

    def _get_kernel(
        params_tree: Any,
        path: tuple[str, ...],
    ) -> np.ndarray | None:
        node: Any = params_tree
        try:
            for key in path:
                node = node[key]
        except Exception:
            return None
        arr = np.asarray(node)
        if arr.ndim != 2:
            return None
        return arr

    def _effective_rank(kernel: np.ndarray) -> float:
        singular_vals = np.linalg.svd(kernel, compute_uv=False)
        singular_vals = singular_vals.astype(np.float64, copy=False)
        denom = float(singular_vals.sum())
        if denom <= 0.0:
            return 0.0
        probs = singular_vals / denom
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        return float(np.exp(entropy))

    def _task_key(
        task_set: str,
        task_label: str | None,
        layout_bank_file: str | None = None,
        layout_bank_num_active: int | None = None,
    ) -> str:
        if task_label:
            return task_label
        layout_suffix = ""
        if layout_bank_file not in (None, "") or layout_bank_num_active not in (
            None,
            0,
        ):
            bank_stem = (
                Path(str(layout_bank_file)).stem
                if layout_bank_file not in (None, "")
                else "bank"
            )
            active = (
                "all"
                if layout_bank_num_active in (None, 0)
                else str(int(layout_bank_num_active))
            )
            layout_suffix = f"|layout_{bank_stem}_n{active}"
        return f"{task_set}{layout_suffix}"

    def _freeze_mask_from_config(
        agents: tuple[int, int],
        freeze_agents: list[int] | None,
        freeze_left: bool,
        freeze_right: bool,
    ) -> tuple[bool, bool]:
        if freeze_agents is None:
            return (freeze_left, freeze_right)
        if not isinstance(freeze_agents, (list, tuple, set)):
            freeze_agents = [freeze_agents]
        freeze_set = {int(a) for a in freeze_agents}
        return (agents[0] in freeze_set, agents[1] in freeze_set)

    def _reset_optimizer_state(state: Any | None) -> Any | None:
        if state is None:
            return None
        if not hasattr(state, "tx") or not hasattr(state, "params"):
            return state
        try:
            opt_state = state.tx.init(state.params)
            return state.replace(step=0, opt_state=opt_state)
        except Exception:
            return state

    def _normalize_round(round_cfg: dict[str, Any]) -> list[dict[str, Any]]:
        default_task_set = round_cfg.get("task_set")
        default_train_steps = round_cfg.get("train_steps")
        default_freeze_agents = round_cfg.get("freeze_agents", None)
        default_freeze_left = bool(round_cfg.get("freeze_left", False))
        default_freeze_right = bool(round_cfg.get("freeze_right", False))
        default_layout_bank_file = round_cfg.get("layout_bank_file", None)
        default_layout_bank_num_active = round_cfg.get("layout_bank_num_active", None)
        normalized = []
        for entry in round_cfg.get("pairings", []):
            if isinstance(entry, dict):
                agents = (
                    entry.get("agents") or entry.get("pairing") or entry.get("pair")
                )
                if agents is None:
                    raise ValueError(
                        "Pairing entry must include 'agents' (or 'pairing'/'pair')."
                    )
                task_set = entry.get("task_set", default_task_set)
                train_steps = entry.get("train_steps", default_train_steps)
                task_label = entry.get("task_label", None)
                freeze_agents = entry.get("freeze_agents", default_freeze_agents)
                freeze_left = bool(entry.get("freeze_left", default_freeze_left))
                freeze_right = bool(entry.get("freeze_right", default_freeze_right))
                layout_bank_file = entry.get(
                    "layout_bank_file", default_layout_bank_file
                )
                layout_bank_num_active = entry.get(
                    "layout_bank_num_active", default_layout_bank_num_active
                )
            else:
                agents = entry
                task_set = default_task_set
                train_steps = default_train_steps
                task_label = None
                freeze_agents = default_freeze_agents
                freeze_left = default_freeze_left
                freeze_right = default_freeze_right
                layout_bank_file = default_layout_bank_file
                layout_bank_num_active = default_layout_bank_num_active

            if task_set is None:
                raise ValueError("Missing task_set for pairing.")
            if train_steps is None:
                raise ValueError("Missing train_steps for pairing.")
            if task_set not in ruleset_datasets:
                raise ValueError(
                    f"Task set {task_set} not found in ruleset_datasets config"
                )

            task_key = _task_key(
                task_set,
                task_label,
                layout_bank_file=layout_bank_file,
                layout_bank_num_active=(
                    int(layout_bank_num_active)
                    if layout_bank_num_active is not None
                    else None
                ),
            )
            freeze_mask = _freeze_mask_from_config(
                tuple(agents), freeze_agents, freeze_left, freeze_right
            )
            normalized.append(
                {
                    "agents": tuple(agents),
                    "task_set": task_set,
                    "task_key": task_key,
                    "train_steps": int(train_steps),
                    "layout_bank_file": layout_bank_file,
                    "layout_bank_num_active": layout_bank_num_active,
                    "freeze_mask": freeze_mask,
                }
            )
        return normalized

    normalized_rounds: list[list[dict[str, Any]]] = []
    task_set_names: list[str] = []
    task_key_to_dataset: dict[str, str] = {}
    task_key_to_overrides: dict[str, dict[str, Any]] = {}

    for round_cfg in rounds:
        normalized = _normalize_round(round_cfg)
        normalized_rounds.append(normalized)
        for entry in normalized:
            key = entry["task_key"]
            if key not in task_key_to_dataset:
                task_set_names.append(key)
                task_key_to_dataset[key] = entry["task_set"]
                task_key_to_overrides[key] = {
                    "layout_bank_file": entry["layout_bank_file"],
                    "layout_bank_num_active": entry["layout_bank_num_active"],
                }
            else:
                if task_key_to_dataset[key] != entry["task_set"]:
                    raise ValueError(
                        f"Task key {key} maps to multiple datasets: "
                        f"{task_key_to_dataset[key]} vs {entry['task_set']}"
                    )
                overrides = task_key_to_overrides[key]
                if (
                    overrides["layout_bank_file"] != entry["layout_bank_file"]
                    or overrides["layout_bank_num_active"]
                    != entry["layout_bank_num_active"]
                ):
                    raise ValueError(f"Task key {key} maps to multiple env settings.")

    task_set_to_idx = {name: idx for idx, name in enumerate(task_set_names)}

    # Preflight path validation so missing files fail before any training starts.
    resolved_ruleset_datasets: dict[str, str] = {}
    for task_set, dataset_ref in ruleset_datasets.items():
        resolved_dataset = _resolve_config_path(str(dataset_ref))
        if not resolved_dataset.exists():
            raise FileNotFoundError(
                "Ruleset dataset not found for task set "
                f"{task_set}: {resolved_dataset} (config value: {dataset_ref})"
            )
        resolved_ruleset_datasets[task_set] = str(resolved_dataset)
    ruleset_datasets = resolved_ruleset_datasets
    config["ruleset_datasets"] = dict(ruleset_datasets)

    print("[dataset-paths] resolved ruleset datasets:")
    for task_set in sorted(ruleset_datasets):
        print(f"  {task_set}: {ruleset_datasets[task_set]}")

    validation_config = {**baseline_config, **config}
    for task_set, dataset_path_str in sorted(ruleset_datasets.items()):
        dataset_path = Path(dataset_path_str)
        packed_host = load_packed_u32_bz2(str(dataset_path.parent), dataset_path.name)
        validate_packed_ruleset_dataset(
            packed_host,
            validation_config,
            dataset_dir=str(dataset_path.parent),
            dataset_file=dataset_path.name,
        )
    if bool(validation_config.get("VALIDATE_RULESET_DATASET", True)):
        print("[dataset-validation] collect tile/item invariants passed")

    task_key_to_resolved_layout_bank: dict[str, str] = {}
    for task_key, task_set in task_key_to_dataset.items():
        overrides = task_key_to_overrides[task_key]
        layout_bank_ref = overrides.get("layout_bank_file", None)
        if layout_bank_ref in (None, ""):
            continue
        dataset_dir = Path(ruleset_datasets[task_set]).parent
        resolved_layout_bank = _resolve_config_path(
            str(layout_bank_ref), dataset_dir=dataset_dir
        )
        if not resolved_layout_bank.exists():
            raise FileNotFoundError(
                "Layout bank not found for "
                f"task_key={task_key}, task_set={task_set}: {resolved_layout_bank} "
                f"(config value: {layout_bank_ref})"
            )
        task_key_to_resolved_layout_bank[task_key] = str(resolved_layout_bank)

    if task_key_to_resolved_layout_bank:
        print("[dataset-paths] resolved layout banks:")
        for task_key in sorted(task_key_to_resolved_layout_bank):
            task_set = task_key_to_dataset.get(task_key, "<unknown>")
            print(
                f"  {task_key} (task_set={task_set}): "
                f"{task_key_to_resolved_layout_bank[task_key]}"
            )

    # Load ruleset metadata for wandb config
    ruleset_info: dict[str, Any] = {}
    for task_name, dataset_path in ruleset_datasets.items():
        meta = load_dataset_metadata(dataset_path)
        if meta:
            ruleset_info[task_name] = {
                "max_depth": meta.get("max_depth"),
                "pool_size": meta.get("pool_size"),
                "n": meta.get("n"),
                "tree_topology": meta.get("tree_topology", "balanced"),
                "rules_per_depth": meta.get("rules_per_depth"),
                "rule_shape": meta.get("rule_shape"),
                "topology_unique_per_depth": meta.get("topology_unique_per_depth"),
                "topology_unique_fraction_per_depth": meta.get(
                    "topology_unique_fraction_per_depth"
                ),
                "topology_unary_fraction_per_depth": meta.get(
                    "topology_unary_fraction_per_depth"
                ),
                "topology_binary_fraction_per_depth": meta.get(
                    "topology_binary_fraction_per_depth"
                ),
                "topology_leaf_count_distribution_per_depth": meta.get(
                    "topology_leaf_count_distribution_per_depth"
                ),
                "topology_rule_count_distribution_per_depth": meta.get(
                    "topology_rule_count_distribution_per_depth"
                ),
            }
            if meta.get("topology_unique_per_depth") is not None:
                print(
                    f"[dataset-topology] {task_name}: "
                    f"unique_per_depth={meta.get('topology_unique_per_depth')}"
                )
            if (
                meta.get("topology_depth5_leaf_count_distribution") is not None
                or meta.get("topology_depth5_rule_count_distribution") is not None
            ):
                print(
                    f"[dataset-topology] {task_name}: "
                    f"d5_leaf_dist={meta.get('topology_depth5_leaf_count_distribution')}, "
                    f"d5_rule_dist={meta.get('topology_depth5_rule_count_distribution')}, "
                    f"d5_unary_frac={meta.get('topology_depth5_unary_fraction')}, "
                    f"d5_binary_frac={meta.get('topology_depth5_binary_fraction')}"
                )

    # Keep observation dimensionality stable across rounds by using
    # the maximum *actual active* rule count among all task-set datasets.
    # This avoids padding to the formula worst-case (e.g. 31 for depth 6)
    # when mixed topologies rarely use more than half that.
    _TRIM_HEADROOM = 4
    global_max_rules = 0
    for _ds_name, _ds_path in ruleset_datasets.items():
        try:
            from banyan_grid.tasks.ruleset_dataset_compact import (
                load_packed_u32_bz2 as _load_packed,
            )

            _packed = _load_packed(
                str(Path(_ds_path).parent), Path(_ds_path).name
            )
            _active = int(np.max(np.sum(_packed != np.uint32(5), axis=1)))
            _trimmed = _active + _TRIM_HEADROOM
            global_max_rules = max(global_max_rules, _trimmed)
        except Exception:
            # Fallback to metadata rule_shape if loading fails.
            _info = ruleset_info.get(_ds_name, {})
            _rs = _info.get("rule_shape")
            if isinstance(_rs, (list, tuple)) and len(_rs) > 0:
                try:
                    global_max_rules = max(global_max_rules, int(_rs[0]))
                except (TypeError, ValueError):
                    pass
    if global_max_rules > 0:
        print(
            f"[tournament] global_max_rules={global_max_rules} "
            f"(from actual active rule counts + {_TRIM_HEADROOM} headroom)"
        )

    wandb.init(
        project=config.get("wandb_project", "banyan"),
        name=config.get("wandb_name", f"{baseline}_tournament"),
        group=config.get("wandb_group"),
        config={
            "baseline": baseline,
            "num_agents": num_agents,
            "num_rounds": len(rounds),
            "ruleset_info": ruleset_info,
            "ROUND_REPLAY_ENABLED": round_replay_enabled,
            "ROUND_REPLAY_RATIO": round_replay_ratio,
            "ROUND_REPLAY_MAX_PAST_ROUNDS": round_replay_max_past_rounds,
            "ROUND_REPLAY_SAMPLING": round_replay_sampling,
            **baseline_config,
        },
    )

    # Define custom x-axis for all metrics (use env_step instead of wandb's default step)
    # Trajectory metrics use per-task-set cumulative_env_step to avoid collisions
    # NOTE: WandB only supports glob suffixes (pattern/*), so we structure as:
    #   trajectories_s1/agent_0, trajectories_s1/agent_0_d1, etc.
    for ts_idx in range(len(task_set_names)):
        ts_num = ts_idx + 1
        wandb.define_metric(
            f"trajectories_s{ts_num}/*",
            step_metric=f"cumulative_env_step_s{ts_num}",
        )
    # Keep round-boundary diagnostics on their own axis.
    wandb.define_metric("round_boundary/*", step_metric="round_boundary/round_idx")
    wandb.define_metric("*", step_metric="env_step")

    agent_params: dict[int, Any | None] = {i: None for i in range(num_agents)}

    seed = config.get("seed", 0)
    rng = jax.random.PRNGKey(seed)

    round_metrics_by_idx: dict[int, dict[tuple[int, int, str], dict[str, float]]] = {}
    training_history: set[tuple[int, int, str]] = set()

    transfer_tensor: dict[tuple[int, int], dict[str, dict[int, dict[str, float]]]] = {}

    run_evals = config.get("run_eval", True) and make_eval is not None
    async_round_eval = bool(config.get("ASYNC_ROUND_EVAL", False))
    async_round_eval_workers = int(config.get("ASYNC_ROUND_EVAL_WORKERS", 1))
    if async_round_eval and not run_evals:
        async_round_eval = False

    train_fn_cache: dict[tuple, Any] = {}
    eval_fn_cache: dict[tuple, Any] = {}
    pending_round_eval_futures: list[tuple[int, Any]] = []
    round_eval_executor: ThreadPoolExecutor | None = None

    def _build_task_config(
        task_key: str,
        *,
        total_timesteps: int | None = None,
        round_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        dataset_name = task_key_to_dataset[task_key]
        dataset_path = Path(ruleset_datasets[dataset_name])
        cfg = baseline_config.copy()
        cfg["RULESET_DATASET_DIR"] = str(dataset_path.parent)
        cfg["RULESET_DATASET_FILE"] = dataset_path.name
        cfg["task_set_names"] = task_set_names
        if global_max_rules > 0:
            cfg["GLOBAL_MAX_RULES"] = int(global_max_rules)
        if total_timesteps is not None:
            cfg["TOTAL_TIMESTEPS"] = total_timesteps
        if round_overrides is not None:
            layout_bank_file = round_overrides.get("layout_bank_file", None)
            if layout_bank_file not in (None, ""):
                cfg["LAYOUT_BANK_FILE"] = task_key_to_resolved_layout_bank.get(
                    task_key, str(layout_bank_file)
                )
            if round_overrides.get("layout_bank_num_active", None) is not None:
                cfg["LAYOUT_BANK_NUM_ACTIVE"] = int(
                    round_overrides["layout_bank_num_active"]
                )
        return cfg

    def _train_cache_key(task_key: str, cfg: dict[str, Any]) -> tuple:
        # Key by effective training config, not task label, so rounds that alias
        # the same dataset/settings can reuse compiled train fns.
        return (
            int(cfg["TOTAL_TIMESTEPS"]),
            str(cfg.get("RULESET_DATASET_DIR", "")),
            str(cfg.get("RULESET_DATASET_FILE", "")),
            _layout_cache_key(cfg),
        )

    def _eval_cache_key(task_key: str, cfg: dict[str, Any]) -> tuple:
        # Key by effective eval config, not task label, for maximal reuse.
        return (
            str(cfg.get("RULESET_DATASET_DIR", "")),
            str(cfg.get("RULESET_DATASET_FILE", "")),
            int(cfg.get("EVAL_EPISODES", 100)),
            _layout_cache_key(cfg),
        )

    def _layout_cache_key(cfg: dict[str, Any]) -> tuple:
        return (
            str(cfg.get("LAYOUT_BANK_FILE", "")),
            int(cfg.get("LAYOUT_BANK_NUM_ACTIVE", 0)),
        )

    def _collect_replay_candidates(current_round_idx: int) -> list[dict[str, Any]]:
        if current_round_idx <= 0:
            return []
        if round_replay_max_past_rounds <= 0:
            return []

        start_idx = max(0, current_round_idx - round_replay_max_past_rounds)
        candidates: list[dict[str, Any]] = []
        for past_round_idx in range(start_idx, current_round_idx):
            # Avoid overweighting rounds that happen to have many pairings with
            # identical settings by keeping unique settings per source round.
            seen_keys: set[
                tuple[
                    str,
                    str | None,
                    int | None,
                ]
            ] = set()
            for entry in normalized_rounds[past_round_idx]:
                dedupe_key = (
                    entry["task_key"],
                    entry.get("layout_bank_file"),
                    entry.get("layout_bank_num_active"),
                )
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                candidates.append(
                    {
                        "source_round_idx": past_round_idx,
                        "task_key": entry["task_key"],
                        "layout_bank_file": entry.get("layout_bank_file"),
                        "layout_bank_num_active": entry.get("layout_bank_num_active"),
                    }
                )
        return candidates

    def _sample_replay_candidate(
        sample_rng: jax.Array,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if len(candidates) == 1:
            return candidates[0]

        if round_replay_sampling == "uniform":
            sample_idx = int(
                jax.random.randint(
                    sample_rng, shape=(), minval=0, maxval=len(candidates)
                )
            )
            return candidates[sample_idx]

        # Recency-biased: linearly upweight more recent source rounds.
        min_source_round_idx = min(
            int(candidate["source_round_idx"]) for candidate in candidates
        )
        weights = []
        for candidate in candidates:
            source_round_idx = int(candidate["source_round_idx"])
            recency_rank = source_round_idx - min_source_round_idx + 1
            weights.append(float(recency_rank))
        probs = jnp.asarray(weights, dtype=jnp.float32)
        probs = probs / probs.sum()
        sample_idx = int(
            jax.random.choice(sample_rng, jnp.arange(len(candidates)), p=probs)
        )
        return candidates[sample_idx]

    def _get_train_fn(
        task_key: str,
        *,
        total_timesteps: int,
        round_overrides: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if round_overrides is None:
            round_overrides = task_key_to_overrides.get(task_key)
        cfg = _build_task_config(
            task_key,
            total_timesteps=total_timesteps,
            round_overrides=round_overrides,
        )
        key = _train_cache_key(task_key, cfg)
        if key not in train_fn_cache:
            train_fn_cache[key] = make_train(cfg)
        return train_fn_cache[key], cfg

    # If TRANSFER_EVAL_EPISODES is set in the baseline config, use it for
    # cross-round transfer tensor evals (which need higher sample counts for
    # stable per-depth success rates).  Falls back to EVAL_EPISODES otherwise.
    transfer_eval_episodes = baseline_config.get("TRANSFER_EVAL_EPISODES", None)
    if transfer_eval_episodes is not None:
        transfer_eval_episodes = int(transfer_eval_episodes)

    def _get_eval_fn(
        task_key: str,
        *,
        round_overrides: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if round_overrides is None:
            round_overrides = task_key_to_overrides.get(task_key)
        cfg = _build_task_config(task_key, round_overrides=round_overrides)
        if transfer_eval_episodes is not None:
            cfg["EVAL_EPISODES"] = transfer_eval_episodes
        key = _eval_cache_key(task_key, cfg)
        if key not in eval_fn_cache:
            assert make_eval is not None, "make_eval is required when evals run"
            eval_fn_cache[key] = make_eval(cfg)
        return eval_fn_cache[key], cfg

    all_pairs_for_eval = [
        (i, j) for i in range(num_agents) for j in range(i + 1, num_agents)
    ]
    total_eval_combos_per_round = len(all_pairs_for_eval) * len(task_set_names)

    def _run_round_eval_for_state(
        round_idx_eval: int,
        agent_params_eval: dict[int, Any | None],
        eval_rng_eval: jax.Array,
        *,
        parallel_tasks: bool,
    ) -> dict[tuple[int, int, str], dict[str, float]]:
        # Evaluate ALL agent pairs on ALL task sets (not just historical combos)
        # This gives us zero-shot performance data for forward transfer calculations.
        all_eval_combos = [
            (i, j, ts) for (i, j) in all_pairs_for_eval for ts in task_set_names
        ]

        combos_by_task: dict[str, list[tuple[int, int, str]]] = {}
        for combo in all_eval_combos:
            task_key = combo[2]
            if task_key not in combos_by_task:
                combos_by_task[task_key] = []
            combos_by_task[task_key].append(combo)

        task_keys_list = list(combos_by_task.keys())
        eval_rngs_local = jax.random.split(eval_rng_eval, len(task_keys_list))
        task_rngs = {tk: eval_rngs_local[i] for i, tk in enumerate(task_keys_list)}

        def _run_task_eval(
            task_key: str,
        ) -> dict[tuple[int, int, str], dict[str, float]]:
            eval_fn, _ = _get_eval_fn(task_key)
            return run_eval_round(
                eval_combos=combos_by_task[task_key],
                agent_params=agent_params_eval,
                eval_rng=task_rngs[task_key],
                round_idx=round_idx_eval,
                eval_fn=eval_fn,
                task_set_to_idx=task_set_to_idx,
            )

        round_eval_results: dict[tuple[int, int, str], dict[str, float]] = {}
        if (not parallel_tasks) or len(task_keys_list) == 1:
            for task_key in task_keys_list:
                round_eval_results.update(_run_task_eval(task_key))
        else:
            executor = _get_tournament_executor(max_workers=len(task_keys_list))
            futures = {executor.submit(_run_task_eval, tk): tk for tk in task_keys_list}
            for future in as_completed(futures):
                task_results = future.result()
                round_eval_results.update(task_results)
        return round_eval_results

    def _record_round_eval(
        round_idx_eval: int,
        round_eval_results: dict[tuple[int, int, str], dict[str, float]],
    ) -> None:
        update_transfer_tensor(transfer_tensor, round_eval_results, round_idx_eval)
        round_metrics_by_idx[round_idx_eval] = round_eval_results
        print(
            f"  Eval complete (round {round_idx_eval + 1}): "
            f"{len(round_eval_results)} results"
        )

    def _log_transfer_summary(metric_key: str = "success_rate") -> None:
        if not transfer_tensor:
            return

        bwt_values: list[float] = []
        fwt_values: list[float] = []
        forgetting_values: list[float] = []
        per_task: dict[str, dict[str, list[float]]] = {}

        for pair_metrics in transfer_tensor.values():
            for task_key, round_metrics in pair_metrics.items():
                ordered = sorted(round_metrics.items(), key=lambda x: int(x[0]))
                values: list[float] = []
                for _, metrics in ordered:
                    raw = metrics.get(metric_key, None)
                    if raw is None:
                        continue
                    try:
                        val = float(raw)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(val):
                        values.append(val)
                if not values:
                    continue

                initial = values[0]
                final = values[-1]
                peak = max(values)
                bwt = final - initial
                # By convention we use 0.0 as the zero-shot baseline.
                fwt = initial
                forgetting = peak - final

                bwt_values.append(bwt)
                fwt_values.append(fwt)
                forgetting_values.append(forgetting)

                bucket = per_task.setdefault(
                    task_key,
                    {"bwt": [], "fwt": [], "forgetting": []},
                )
                bucket["bwt"].append(bwt)
                bucket["fwt"].append(fwt)
                bucket["forgetting"].append(forgetting)

        if not bwt_values:
            return

        payload: dict[str, float] = {
            "transfer/num_series": float(len(bwt_values)),
            "transfer/bwt_mean": float(np.mean(bwt_values)),
            "transfer/fwt_mean": float(np.mean(fwt_values)),
            "transfer/forgetting_mean": float(np.mean(forgetting_values)),
        }

        for task_key, task_vals in per_task.items():
            task_slug = (
                str(task_key).replace("|", "_").replace("/", "_").replace(" ", "_")
            )
            payload[f"transfer/by_task/{task_slug}/bwt_mean"] = float(
                np.mean(task_vals["bwt"])
            )
            payload[f"transfer/by_task/{task_slug}/fwt_mean"] = float(
                np.mean(task_vals["fwt"])
            )
            payload[f"transfer/by_task/{task_slug}/forgetting_mean"] = float(
                np.mean(task_vals["forgetting"])
            )

        wandb.log(payload)

    def _consume_ready_round_evals(*, block: bool) -> None:
        nonlocal pending_round_eval_futures
        if not pending_round_eval_futures:
            return
        if block:
            futures_by_round = {
                future: ridx for ridx, future in pending_round_eval_futures
            }
            pending_round_eval_futures = []
            for future in as_completed(futures_by_round):
                ridx = futures_by_round[future]
                results = future.result()
                _record_round_eval(ridx, results)
            return

        still_pending: list[tuple[int, Any]] = []
        for ridx, future in pending_round_eval_futures:
            if future.done():
                results = future.result()
                _record_round_eval(ridx, results)
            else:
                still_pending.append((ridx, future))
        pending_round_eval_futures = still_pending

    if async_round_eval:
        round_eval_executor = ThreadPoolExecutor(
            max_workers=max(1, async_round_eval_workers),
            thread_name_prefix="round-eval",
        )
        print(f"Async round eval enabled (workers={max(1, async_round_eval_workers)}).")

    def _log_effective_rank_boundary(boundary_round_idx: int) -> None:
        if not log_effective_rank:
            return

        rank_payload: dict[str, float] = {
            "round_boundary/round_idx": float(boundary_round_idx)
        }
        per_agent_means: list[float] = []
        for agent_id, state in agent_params.items():
            params_tree = _extract_actor_param_tree(state)
            if params_tree is None:
                continue
            embed_kernel = _get_kernel(
                params_tree,
                ("params", "embed_dense", "kernel"),
            )
            actor_kernel = _get_kernel(
                params_tree,
                ("params", "actor_fc", "kernel"),
            )
            if embed_kernel is None and actor_kernel is None:
                continue
            agent_vals: list[float] = []
            if embed_kernel is not None:
                embed_rank = _effective_rank(embed_kernel)
                rank_payload[
                    f"round_boundary/effective_rank/agent_{agent_id}/embed_dense"
                ] = embed_rank
                agent_vals.append(embed_rank)
            if actor_kernel is not None:
                actor_rank = _effective_rank(actor_kernel)
                rank_payload[
                    f"round_boundary/effective_rank/agent_{agent_id}/actor_fc"
                ] = actor_rank
                agent_vals.append(actor_rank)
            if agent_vals:
                mean_rank = float(sum(agent_vals) / len(agent_vals))
                rank_payload[f"round_boundary/effective_rank/agent_{agent_id}/mean"] = (
                    mean_rank
                )
                per_agent_means.append(mean_rank)
        if per_agent_means:
            rank_payload["round_boundary/effective_rank/mean"] = float(
                sum(per_agent_means) / len(per_agent_means)
            )
        if len(rank_payload) > 1:
            wandb.log(rank_payload)

    def _log_td_error_boundary(
        boundary_round_idx: int,
        td_error_summary: dict[int, float],
    ) -> None:
        if not td_error_summary:
            return
        td_payload: dict[str, float] = {
            "round_boundary/round_idx": float(boundary_round_idx)
        }
        per_agent_vals: list[float] = []
        for agent_id in sorted(td_error_summary.keys()):
            td_val = float(td_error_summary[agent_id])
            td_payload[f"round_boundary/td_error/agent_{agent_id}"] = td_val
            per_agent_vals.append(td_val)
        if per_agent_vals:
            td_payload["round_boundary/td_error/mean"] = float(
                sum(per_agent_vals) / len(per_agent_vals)
            )
        if len(td_payload) > 1:
            wandb.log(td_payload)

    latest_td_error_summary: dict[int, float] = {}

    for round_idx, _round_config in enumerate(rounds):
        if reset_optimizer_each_round and round_idx > 0:
            agent_params = {
                agent_id: _reset_optimizer_state(state)
                for agent_id, state in agent_params.items()
            }
            print(f"Reset optimizer state for round {round_idx + 1}")
        rng, round_rng = jax.random.split(rng)
        print(f"\n{'=' * 60}")
        print(f"Round {round_idx + 1}/{len(rounds)}")
        print(f"{'=' * 60}")
        pair_entries = normalized_rounds[round_idx]
        pairings = [entry["agents"] for entry in pair_entries]
        task_keys = sorted({entry["task_key"] for entry in pair_entries})
        train_steps_set = sorted({entry["train_steps"] for entry in pair_entries})
        if len(task_keys) == 1:
            print(f"Task: {task_keys[0]}")
        else:
            print(f"Tasks: {task_keys}")
        print(f"Pairings: {pairings}")
        print(f"Train steps: {train_steps_set}")

        pairings_by_config: dict[
            tuple[
                str,
                int,
                str | None,
                int | None,
            ],
            list[dict[str, Any]],
        ] = {}
        for entry in pair_entries:
            key = (
                entry["task_key"],
                entry["train_steps"],
                entry.get("layout_bank_file"),
                entry.get("layout_bank_num_active"),
            )
            pairings_by_config.setdefault(key, []).append(entry)
        num_groups = len(pairings_by_config)
        if num_groups == 1:
            print(f"  Running {len(pairings)} pairings (vmapped, single dataset)...")
        else:
            print(
                f"  Running {len(pairings)} pairings across {num_groups} datasets "
                f"(parallel threads)..."
            )
        for idx, (i, j) in enumerate(pairings):
            print(f"    Pairing {idx + 1}: agents {i} and {j}")

        # Build groups for parallel execution
        groups = []
        for group_key, group_entries in pairings_by_config.items():
            (
                task_key,
                train_steps,
                layout_bank_file,
                layout_bank_num_active,
            ) = group_key
            groups.append(
                {
                    "task_key": task_key,
                    "train_steps": train_steps,
                    "layout_bank_file": layout_bank_file,
                    "layout_bank_num_active": layout_bank_num_active,
                    "pairings": [entry["agents"] for entry in group_entries],
                    "freeze_masks": [entry["freeze_mask"] for entry in group_entries],
                }
            )

        group_rngs = list(jax.random.split(round_rng, len(groups))) if groups else []
        round_td_error_summary: dict[int, float] = {}
        agent_params, train_td_summary = run_round_groups_parallel(
            groups=groups,
            agent_params=agent_params,
            round_rngs=group_rngs,
            round_idx=round_idx,
            task_set_to_idx=task_set_to_idx,
            get_train_fn=_get_train_fn,
        )
        round_td_error_summary.update(train_td_summary)

        for entry in pair_entries:
            i, j = entry["agents"]
            task_key = entry["task_key"]
            combo = (i, j, task_key)
            training_history.add(combo)

        if round_replay_enabled and round_replay_ratio > 0.0 and round_idx > 0:
            replay_candidates = _collect_replay_candidates(round_idx)
            if replay_candidates:
                replay_entries = []
                rng, replay_rng = jax.random.split(rng)
                replay_pair_rngs = list(jax.random.split(replay_rng, len(pair_entries)))

                for entry, sample_rng in zip(pair_entries, replay_pair_rngs):
                    replay_steps = int(entry["train_steps"] * round_replay_ratio)
                    if replay_steps <= 0:
                        continue
                    sampled = _sample_replay_candidate(
                        sample_rng,
                        replay_candidates,
                    )
                    replay_entries.append(
                        {
                            "agents": entry["agents"],
                            "freeze_mask": entry["freeze_mask"],
                            "task_key": sampled["task_key"],
                            "train_steps": replay_steps,
                            "layout_bank_file": sampled.get("layout_bank_file"),
                            "layout_bank_num_active": sampled.get(
                                "layout_bank_num_active"
                            ),
                            "source_round_idx": sampled["source_round_idx"],
                        }
                    )

                if replay_entries:
                    replay_by_config: dict[
                        tuple[
                            str,
                            int,
                            str | None,
                            int | None,
                        ],
                        list[dict[str, Any]],
                    ] = {}
                    for replay_entry in replay_entries:
                        key = (
                            replay_entry["task_key"],
                            replay_entry["train_steps"],
                            replay_entry.get("layout_bank_file"),
                            replay_entry.get("layout_bank_num_active"),
                        )
                        replay_by_config.setdefault(key, []).append(replay_entry)

                    replay_groups = []
                    for group_key, group_entries in replay_by_config.items():
                        (
                            replay_task_key,
                            replay_train_steps,
                            replay_layout_bank_file,
                            replay_layout_bank_num_active,
                        ) = group_key
                        replay_groups.append(
                            {
                                "task_key": replay_task_key,
                                "train_steps": replay_train_steps,
                                "layout_bank_file": replay_layout_bank_file,
                                "layout_bank_num_active": replay_layout_bank_num_active,
                                "pairings": [
                                    replay_entry["agents"]
                                    for replay_entry in group_entries
                                ],
                                "freeze_masks": [
                                    replay_entry["freeze_mask"]
                                    for replay_entry in group_entries
                                ],
                            }
                        )

                    replay_rngs = []
                    if replay_groups:
                        rng, replay_groups_rng = jax.random.split(rng)
                        replay_rngs = list(
                            jax.random.split(replay_groups_rng, len(replay_groups))
                        )
                    if replay_groups:
                        print(
                            "  Round replay: "
                            f"{len(replay_entries)} pairings, "
                            f"{len(replay_groups)} groups, "
                            f"ratio={round_replay_ratio}, "
                            f"window={round_replay_max_past_rounds}, "
                            f"sampling={round_replay_sampling}"
                        )
                        agent_params, replay_td_summary = run_round_groups_parallel(
                            groups=replay_groups,
                            agent_params=agent_params,
                            round_rngs=replay_rngs,
                            round_idx=round_idx,
                            task_set_to_idx=task_set_to_idx,
                            get_train_fn=_get_train_fn,
                        )
                        round_td_error_summary.update(replay_td_summary)
                        replay_task_counts: dict[str, int] = {}
                        for replay_entry in replay_entries:
                            i, j = replay_entry["agents"]
                            replay_task_key = replay_entry["task_key"]
                            replay_task_counts[replay_task_key] = (
                                replay_task_counts.get(replay_task_key, 0) + 1
                            )
                            training_history.add((i, j, replay_task_key))
                        print(
                            "  Replay task usage: "
                            + ", ".join(
                                f"{task_key} x{count}"
                                for task_key, count in sorted(
                                    replay_task_counts.items()
                                )
                            )
                        )

        _log_effective_rank_boundary(round_idx)
        _log_td_error_boundary(round_idx, round_td_error_summary)
        latest_td_error_summary = round_td_error_summary.copy()

        if run_evals:
            rng, eval_rng = jax.random.split(rng)
            print(
                f"  Running evaluation on {total_eval_combos_per_round} combos "
                f"({len(all_pairs_for_eval)} pairs x {len(task_set_names)} tasks)..."
            )
            if async_round_eval and round_eval_executor is not None:
                # Snapshot params for this round's eval so training can continue.
                eval_params_snapshot = agent_params.copy()
                future = round_eval_executor.submit(
                    _run_round_eval_for_state,
                    round_idx,
                    eval_params_snapshot,
                    eval_rng,
                    parallel_tasks=False,
                )
                pending_round_eval_futures.append((round_idx, future))
                print("  Eval dispatched asynchronously.")
                _consume_ready_round_evals(block=False)
            else:
                round_eval_results = _run_round_eval_for_state(
                    round_idx,
                    agent_params,
                    eval_rng,
                    parallel_tasks=True,
                )
                _record_round_eval(round_idx, round_eval_results)

        save_round_checkpoint(config, baseline_config, agent_params, round_idx)
        print(f"Round {round_idx + 1} complete")

    if async_round_eval:
        print("Waiting for pending async round eval jobs...")
        _consume_ready_round_evals(block=True)

    # Explicit end-of-run boundary snapshot (e.g., round 2 in a 2-round run).
    _log_effective_rank_boundary(len(rounds))
    _log_td_error_boundary(len(rounds), latest_td_error_summary)

    print(f"\n{'=' * 60}")
    print("Tournament complete!")
    print(f"{'=' * 60}")

    # Log transfer tensor and round metrics to wandb
    if transfer_tensor:
        try:
            tensor_table = tensor_to_wandb_table(transfer_tensor)
            wandb.log({"transfer_tensor": tensor_table})
            _log_transfer_summary(metric_key="success_rate")
            print("\nTransfer tensor logged to WandB")
        except Exception as e:
            print(f"\nFailed to log to WandB: {e}")

    save_final_model_checkpoint(config, baseline_config, agent_params)

    shutdown_wandb_async(timeout_s=5.0)
    if round_eval_executor is not None:
        round_eval_executor.shutdown(wait=False)
    _shutdown_tournament_executor()
    wandb.finish()

    all_round_metrics = [
        round_metrics_by_idx[idx] for idx in sorted(round_metrics_by_idx.keys())
    ]

    return {
        "agent_params": agent_params,
        "round_metrics": all_round_metrics,
        "transfer_tensor": transfer_tensor,
        "training_history": training_history,
    }
