import atexit
import logging
import queue
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import wandb

_logger = logging.getLogger(__name__)

_viz_executor: ThreadPoolExecutor | None = None
_viz_executor_lock = Lock()
_wandb_lock = Lock()
_wandb_async_enabled = True
_wandb_queue_maxsize = 2048
_wandb_queue: queue.Queue[tuple[dict[str, Any], Any]] = queue.Queue(maxsize=_wandb_queue_maxsize)
_wandb_worker_thread: Thread | None = None
_wandb_worker_stop = Event()
_wandb_worker_lock = Lock()
_wandb_drop_count = 0
_dataset_context_lock = Lock()
_dataset_context_logged: set[tuple[Any, Any, Any, str]] = set()


def _get_viz_executor(max_workers: int = 1) -> ThreadPoolExecutor:
    global _viz_executor
    with _viz_executor_lock:
        if _viz_executor is None:
            _viz_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="viz")
        return _viz_executor


def _submit_viz_task(fn, *args) -> None:
    try:
        _get_viz_executor().submit(fn, *args)
    except Exception as e:
        _logger.debug("Viz executor submit failed, running synchronously: %s", e)
        fn(*args)


def _to_host(x):
    try:
        return np.asarray(jax.device_get(x))
    except Exception:
        return np.asarray(x)


def _wandb_log_sync(payload, step=None) -> None:
    """Synchronous WandB logging path."""
    run = wandb.run
    if run is None or getattr(run, "disabled", False):
        return
    try:
        with _wandb_lock:
            wandb.log(payload, step=step)
    except Exception as e:
        _logger.debug("WandB log failed: %s", e)


def _wandb_worker_loop() -> None:
    while True:
        if _wandb_worker_stop.is_set() and _wandb_queue.empty():
            break
        try:
            payload, step = _wandb_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            _wandb_log_sync(payload, step=step)
        finally:
            _wandb_queue.task_done()


def _ensure_wandb_worker() -> None:
    global _wandb_worker_thread
    if _wandb_worker_thread is not None and _wandb_worker_thread.is_alive():
        return
    with _wandb_worker_lock:
        if _wandb_worker_thread is not None and _wandb_worker_thread.is_alive():
            return
        _wandb_worker_stop.clear()
        _wandb_worker_thread = Thread(
            target=_wandb_worker_loop,
            name="wandb-log-worker",
            daemon=True,
        )
        _wandb_worker_thread.start()


def configure_wandb_async(enabled: bool) -> None:
    """Enable or disable async WandB logging."""
    global _wandb_async_enabled
    _wandb_async_enabled = bool(enabled)
    if not _wandb_async_enabled:
        shutdown_wandb_async(timeout_s=0.5)


def shutdown_wandb_async(timeout_s: float = 2.0) -> None:
    """Flush and stop async WandB worker."""
    global _wandb_worker_thread
    with _wandb_worker_lock:
        worker = _wandb_worker_thread
    if worker is None:
        return
    _wandb_worker_stop.set()
    try:
        worker.join(timeout=timeout_s)
    except Exception:
        pass
    with _wandb_worker_lock:
        if _wandb_worker_thread is worker:
            _wandb_worker_thread = None


atexit.register(shutdown_wandb_async)


def _wandb_log(payload, step=None):
    """Log to WandB with optional async queueing."""
    run = wandb.run
    if run is None or getattr(run, "disabled", False):
        return
    if not _wandb_async_enabled:
        _wandb_log_sync(payload, step=step)
        return

    global _wandb_drop_count
    _ensure_wandb_worker()
    try:
        _wandb_queue.put_nowait((payload, step))
    except queue.Full:
        # Backpressure policy: drop oldest log item to keep training non-blocking.
        try:
            _wandb_queue.get_nowait()
            _wandb_queue.task_done()
        except queue.Empty:
            pass
        try:
            _wandb_queue.put_nowait((payload, step))
        except queue.Full:
            _wandb_drop_count += 1
            if _wandb_drop_count in (1, 10, 100) or (_wandb_drop_count % 500 == 0):
                _logger.warning(
                    "WandB async queue saturated; dropped %d log payload(s)",
                    _wandb_drop_count,
                )


def _safe_action_names(num_actions, given=None):
    """Return action names with a fallback list."""
    if given is not None and len(given) == num_actions:
        return list(given)
    return [f"A{i}" for i in range(num_actions)]


def _get_matplotlib():
    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        try:
            matplotlib.use("Agg")
        except Exception:
            pass

    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    return gridspec, plt


def _first_scalar(x, default=None):
    """Extract first scalar value from array-like, or return default."""
    if x is None:
        return default
    arr = np.asarray(x)
    if arr.size == 0:
        return default
    val = arr.flat[0]
    return val.item() if hasattr(val, "item") else val


def _default_log_every_updates(config: dict[str, Any]) -> int:
    """Default metric logging cadence when LOG_EVERY_UPDATES is unset.

    Targets ~60 logs/round (3x denser than the previous ~20 logs/round default).
    """
    return max(1, int(config.get("NUM_UPDATES", 1000)) // 60)


def _make_prefix(meta: dict[str, Any] | None, task_set_names: list[str] | None = None) -> str:
    """Build WandB prefix: r{round}_a{agent0}_a{agent1}_s{task_set}/"""
    if meta is None:
        return ""

    agent_ids = meta.get("agent_ids")
    if agent_ids is None:
        return ""
    if hasattr(agent_ids, "tolist"):
        agent_ids = agent_ids.tolist()
    if len(agent_ids) < 2:
        return ""

    parts = []
    round_idx = _first_scalar(meta.get("round_idx"))
    if round_idx is not None:
        parts.append(f"r{round_idx}")

    a0 = _first_scalar(agent_ids[0], agent_ids[0])
    a1 = _first_scalar(agent_ids[1], agent_ids[1])
    parts.append(f"a{a0}_a{a1}")

    task_set_idx = _first_scalar(meta.get("task_set_idx"))
    if task_set_idx is not None:
        parts.append(f"s{task_set_idx + 1}")

    return "_".join(parts) + "/"


def _build_prefixed_metric_payload(
    metric: dict[str, Any],
    meta: dict[str, Any] | None,
    task_set_names: list[str] | None,
    steps_per_round: int,
    *,
    log_action_diagnostics: bool = False,
) -> dict[str, Any]:
    """Build prefixed payload with trajectory stitching keys for a single log row."""
    prefix = _make_prefix(meta, task_set_names)
    # Add prefix to all metric keys (exclude env_step since it's used as x-axis, not a chart)
    prefixed_metric = {
        f"{prefix}{k}": v
        for k, v in metric.items()
        if k != "env_step" and (log_action_diagnostics or not k.startswith(("action/", "toggle/")))
    }

    env_step = metric.get("env_step", None)
    plotted_env_step = env_step

    if meta is not None and env_step is not None and steps_per_round > 0:
        round_idx = meta.get("round_idx", None)
        if round_idx is not None:
            plotted_env_step = int(_first_scalar(round_idx, 0)) * int(steps_per_round) + int(
                _first_scalar(env_step, 0)
            )

    # env_step is the global x-axis used by wandb.define_metric("*").
    # Per-round trainers reset local env_step, so offset by round here to keep
    # W&B's step metric monotonic across tournament rounds.
    if env_step is not None:
        prefixed_metric["env_step"] = plotted_env_step

    # Also log trajectory metrics (without round in prefix, for stitching across rounds)
    # These will be plotted as continuous lines per agent
    if meta is not None:
        agent_ids = meta.get("agent_ids", None)
        round_idx = meta.get("round_idx", None)
        task_set_idx = meta.get("task_set_idx", None)
        if agent_ids is not None:
            if hasattr(agent_ids, "tolist"):
                agent_ids = agent_ids.tolist()
            a0 = int(agent_ids[0])
            a1 = int(agent_ids[1])

            # Build trajectory key prefix for this pairing
            # Structure: trajectories_s{N}/agent_{id} for WandB glob compatibility
            ts_idx = 0
            if task_set_idx is not None:
                ts_idx = int(_first_scalar(task_set_idx, 0))
            ts_num = ts_idx + 1
            traj_prefix = f"trajectories_s{ts_num}"

            # Compute cumulative env_step: round_idx * steps_per_round + env_step
            # Log per-task-set cumulative step to avoid collisions from parallel pairings
            cumulative = None
            if plotted_env_step is not None and round_idx is not None and steps_per_round > 0:
                cumulative = int(_first_scalar(plotted_env_step, 0))
                # Each task set gets its own cumulative step metric
                prefixed_metric[f"cumulative_env_step_s{ts_num}"] = cumulative

            # Log overall success rate for each agent's trajectory
            sr = metric.get("success/rate", None)
            if sr is not None:
                prefixed_metric[f"{traj_prefix}/agent_{a0}"] = sr
                prefixed_metric[f"{traj_prefix}/agent_{a1}"] = sr

            # Log per-depth success rates for trajectories (up to d6)
            for d in range(1, 7):
                sr_d = metric.get(f"success/rate_d{d}", None)
                if sr_d is not None:
                    prefixed_metric[f"{traj_prefix}/agent_{a0}_d{d}"] = sr_d
                    prefixed_metric[f"{traj_prefix}/agent_{a1}_d{d}"] = sr_d

            # Log per-depth episode lengths for trajectories
            for d in range(1, 7):
                el_d = metric.get(f"episode/length_d{d}", None)
                if el_d is not None:
                    prefixed_metric[f"{traj_prefix}/ep_length_d{d}"] = el_d

            # Log per-depth timestep averages for trajectories.
            for d in range(1, 7):
                ts_d = metric.get(f"depth/timesteps_avg_per_episode_d{d}", None)
                if ts_d is not None:
                    prefixed_metric[f"{traj_prefix}/timesteps_avg_d{d}"] = ts_d

    return prefixed_metric


def log_round_dataset_host(
    pairing_meta: dict[str, Any] | None,
    config: dict[str, Any],
) -> None:
    """Log dataset context for a round/task-set once per run."""
    run = wandb.run
    if run is None or getattr(run, "disabled", False):
        return

    dataset_dir_raw = config.get("RULESET_DATASET_DIR", None)
    dataset_file_raw = config.get("RULESET_DATASET_FILE", None)
    if not dataset_dir_raw or not dataset_file_raw:
        return

    dataset_dir = str(dataset_dir_raw)
    dataset_file = str(dataset_file_raw)
    dataset_dir_path = Path(dataset_dir).expanduser()
    dataset_path = str((dataset_dir_path / dataset_file).resolve())

    round_idx: int | None = None
    task_set_idx: int | None = None
    if pairing_meta is not None:
        round_idx = int(_first_scalar(pairing_meta.get("round_idx"), 0))
        task_set_idx = int(_first_scalar(pairing_meta.get("task_set_idx"), 0))
    elif bool(config.get("STANDALONE_ROUND_LOGGING", False)):
        round_idx = int(config.get("STANDALONE_ROUND_IDX", 0))
        task_set_idx = int(config.get("STANDALONE_TASK_SET_IDX", 0))

    task_set_name: str | None = None
    task_set_names = config.get("task_set_names", None)
    if (
        task_set_idx is not None
        and isinstance(task_set_names, (list, tuple))
        and 0 <= task_set_idx < len(task_set_names)
    ):
        task_set_name = str(task_set_names[task_set_idx])
    elif config.get("STANDALONE_TASK_SET_NAME", None) is not None:
        task_set_name = str(config.get("STANDALONE_TASK_SET_NAME"))

    context_key = (getattr(run, "id", None), round_idx, task_set_idx, dataset_path)
    with _dataset_context_lock:
        if context_key in _dataset_context_logged:
            return
        _dataset_context_logged.add(context_key)

    round_token = f"r{round_idx}" if round_idx is not None else "r_unknown"
    task_token = f"s{task_set_idx + 1}" if task_set_idx is not None else "s_unknown"
    scoped = f"round_dataset/{round_token}_{task_token}"

    payload: dict[str, Any] = {
        f"{scoped}/path": dataset_path,
        f"{scoped}/dir": str(dataset_dir_path),
        f"{scoped}/file": dataset_file,
        "round_dataset/current/path": dataset_path,
        "round_dataset/current/file": dataset_file,
    }
    if round_idx is not None:
        payload[f"{scoped}/round_idx"] = round_idx
        payload["round_dataset/current/round_idx"] = round_idx
    if task_set_idx is not None:
        payload[f"{scoped}/task_set_idx"] = task_set_idx
        payload["round_dataset/current/task_set_idx"] = task_set_idx
    if task_set_name is not None:
        payload[f"{scoped}/task_set_name"] = task_set_name
        payload["round_dataset/current/task_set_name"] = task_set_name

    _wandb_log(payload)


def build_common_metrics(
    traj_batch: Any,
    info: dict[str, jax.Array],
    env_step: jax.Array,
    num_agents: int,
    config: dict[str, Any],
    custom_metrics: Optional[dict[str, jax.Array]] = None,
) -> dict[str, jax.Array]:
    """Build common scalar metrics in JAX (no side effects)."""

    def _safe_mean(values: jax.Array) -> jax.Array:
        vals = jnp.asarray(values, dtype=jnp.float32)
        return jnp.mean(vals)

    def _safe_weighted_mean(values: jax.Array, weights: jax.Array) -> jax.Array:
        vals = jnp.asarray(values, dtype=jnp.float32)
        w = jnp.asarray(weights, dtype=jnp.float32)
        denom = jnp.sum(w)
        return jnp.where(denom > 0, jnp.sum(vals * w) / denom, 0.0)

    succ = info["episode_success"]
    end = info["episode_end"]
    eps_successes = jnp.sum(succ) / num_agents
    eps_ended = jnp.sum(end) / num_agents
    success_rate = jnp.where(eps_ended > 0, eps_successes / eps_ended, 0.0)

    # Episode length from info dict (time_step is set every step; mask by episode_end)
    ep_lengths = info.get("episode_length", None)
    if ep_lengths is not None:
        # Only count episode lengths at episode boundaries
        total_length = (
            jnp.sum(ep_lengths * end) / num_agents
        )  # divide by num_agents since it's tiled
        mean_ep_length = jnp.where(eps_ended > 0, total_length / eps_ended, 0.0)
    else:
        mean_ep_length = None

    metric: dict[str, jax.Array] = {
        "success/rate": success_rate,
        "env_step": env_step,
    }

    dead_end = info.get("dead_end", None)
    timeout = info.get("timeout", None)
    if dead_end is not None:
        dead_end_eps = jnp.sum(dead_end * end) / num_agents
        metric["dead_end/rate"] = jnp.where(eps_ended > 0, dead_end_eps / eps_ended, 0.0)
    if timeout is not None:
        timeout_eps = jnp.sum(timeout * end) / num_agents
        metric["timeout/rate"] = jnp.where(eps_ended > 0, timeout_eps / eps_ended, 0.0)

    if mean_ep_length is not None:
        metric["episode/length_mean"] = mean_ep_length

    if "original_reward" in info:
        # Mean reward per episode (sum rewards at episode ends / num episodes)
        total_reward = jnp.sum(info["original_reward"])
        metric["reward/mean"] = total_reward / (traj_batch.reward.shape[0] * num_agents)

    # Step-level diagnostics (global across all depths/episodes)
    reward_shape = info.get("reward_shape", None)
    reward_distractor_penalty = info.get("reward_distractor_penalty", None)
    reward_timeout_penalty = info.get("reward_timeout_penalty", None)
    action_pickup = info.get("action_pickup", None)
    action_drop = info.get("action_drop", None)
    action_drop_success = info.get("action_drop_success", None)
    action_drop_failed = info.get("action_drop_failed", None)
    action_toggle = info.get("action_toggle", None)
    action_move = info.get("action_move", None)
    action_push = info.get("action_push", None)
    toggle_attempt = info.get("toggle_attempt", None)
    toggle_transform_success = info.get("toggle_transform_success", None)

    if reward_shape is not None:
        metric["reward_shape/mean"] = _safe_mean(reward_shape)
    if reward_distractor_penalty is not None:
        metric["reward_distractor_penalty/mean"] = _safe_mean(reward_distractor_penalty)
    if reward_timeout_penalty is not None:
        metric["reward_timeout_penalty/mean"] = _safe_mean(reward_timeout_penalty)

    action_fields = (
        ("action/pickup_rate", action_pickup),
        ("action/drop_rate", action_drop),
        ("action/drop_success_rate", action_drop_success),
        ("action/drop_failed_rate", action_drop_failed),
        ("action/toggle_rate", action_toggle),
        ("action/move_rate", action_move),
        ("action/push_rate", action_push),
    )
    for metric_key, action_vals in action_fields:
        if action_vals is not None:
            metric[metric_key] = _safe_mean(action_vals)

    if toggle_attempt is not None and toggle_transform_success is not None:
        total_attempts = jnp.sum(toggle_attempt)
        total_success = jnp.sum(toggle_transform_success)
        metric["toggle/attempt_rate"] = _safe_mean(toggle_attempt)
        metric["toggle/success_rate"] = jnp.where(
            total_attempts > 0, total_success / total_attempts, 0.0
        )

    # Always log depth metrics if masks are available (supports d1..d6)
    depth_masks = {d: info.get(f"d{d}_mask", None) for d in range(1, 7)}
    available_depths = [d for d, mask in depth_masks.items() if mask is not None]
    if available_depths:

        def _rate(mask):
            eps_succ = jnp.sum(succ * mask) / num_agents
            eps_end = jnp.sum(end * mask) / num_agents
            return jnp.where(eps_end > 0, eps_succ / eps_end, 0.0)

        def _event_rate(event, mask):
            eps_event = jnp.sum(event * end * mask) / num_agents
            eps_end = jnp.sum(end * mask) / num_agents
            return jnp.where(eps_end > 0, eps_event / eps_end, 0.0)

        for d in available_depths:
            metric[f"success/rate_d{d}"] = _rate(depth_masks[d])
            if dead_end is not None:
                metric[f"dead_end/rate_d{d}"] = _event_rate(dead_end, depth_masks[d])
            if timeout is not None:
                metric[f"timeout/rate_d{d}"] = _event_rate(timeout, depth_masks[d])

            mask_d = depth_masks[d]
            if reward_shape is not None:
                metric[f"reward_shape/mean_d{d}"] = _safe_weighted_mean(reward_shape, mask_d)
            if reward_distractor_penalty is not None:
                metric[f"reward_distractor_penalty/mean_d{d}"] = _safe_weighted_mean(
                    reward_distractor_penalty, mask_d
                )
            if reward_timeout_penalty is not None:
                metric[f"reward_timeout_penalty/mean_d{d}"] = _safe_weighted_mean(
                    reward_timeout_penalty, mask_d
                )
            for base_key, action_vals in action_fields:
                if action_vals is not None:
                    metric[f"{base_key}_d{d}"] = _safe_weighted_mean(action_vals, mask_d)
            if toggle_attempt is not None and toggle_transform_success is not None:
                attempts_d = jnp.sum(toggle_attempt * mask_d)
                success_d = jnp.sum(toggle_transform_success * mask_d)
                metric[f"toggle/attempt_rate_d{d}"] = _safe_weighted_mean(toggle_attempt, mask_d)
                metric[f"toggle/success_rate_d{d}"] = jnp.where(
                    attempts_d > 0, success_d / attempts_d, 0.0
                )

        def _timesteps(mask):
            return jnp.sum(mask) / num_agents

        def _timesteps_avg_per_episode(mask):
            ts = _timesteps(mask)
            ep_end = jnp.sum(end * mask) / num_agents
            return jnp.where(ep_end > 0, ts / ep_end, 0.0)

        ts_by_depth: dict[int, jax.Array] = {}
        ts_total = jnp.zeros((), dtype=jnp.float32)
        for d in available_depths:
            ts_d = _timesteps(depth_masks[d])
            ts_by_depth[d] = ts_d
            ts_total = ts_total + ts_d
            metric[f"depth/timesteps_d{d}"] = ts_d
        metric["depth/timesteps_total"] = ts_total
        for d in available_depths:
            ts_d = ts_by_depth[d]
            metric[f"depth/timesteps_frac_d{d}"] = jnp.where(ts_total > 0, ts_d / ts_total, 0.0)
            metric[f"depth/timesteps_avg_per_episode_d{d}"] = _timesteps_avg_per_episode(
                depth_masks[d]
            )

        if ep_lengths is not None:

            def _mean_len(mask):
                len_sum = jnp.sum(ep_lengths * end * mask) / num_agents
                ep_end = jnp.sum(end * mask) / num_agents
                return jnp.where(ep_end > 0, len_sum / ep_end, 0.0)

            for d in available_depths:
                metric[f"episode/length_d{d}"] = _mean_len(depth_masks[d])

    # Add custom baseline-specific metrics (filter to only include what we want)
    if custom_metrics is not None:
        # Only include per-agent loss metrics we care about
        wanted_keys = {
            "ppo/agent0/loss_total",
            "ppo/agent0/loss_policy",
            "ppo/agent0/entropy",
            "ppo/agent0/td_error_mse",
            "ppo/agent1/loss_total",
            "ppo/agent1/loss_policy",
            "ppo/agent1/entropy",
            "ppo/agent1/td_error_mse",
            "ppo/critic/td_error_mse",
        }
        if config.get("E3T_ENABLED", False):
            wanted_keys.update(
                {
                    "ppo/agent0/moa_nll_loss",
                    "ppo/agent1/moa_nll_loss",
                }
            )
        if config.get("LOG_DYNAMICS", False):
            dynamic_suffixes = (
                "grad_norm",
                "grad_l0_frac",
                "grad_l1_norm",
                "param_norm",
                "param_l1_norm",
                "adv_mean",
                "adv_std",
                "ratio_mean",
                "ratio_std",
                "explained_variance",
                "act_norm_embed",
                "act_norm_gru",
                "act_norm_actor",
                "act_norm_critic",
                "dead_frac",
            )
            for agent_idx in (0, 1):
                wanted_keys.update(
                    {f"ppo/agent{agent_idx}/{suffix}" for suffix in dynamic_suffixes}
                )
        for k, v in custom_metrics.items():
            if k in wanted_keys or "/cbp/" in k:
                # Simplify key: ppo/agent0/loss_total -> agent0/loss_total
                simple_key = k.replace("ppo/", "")
                metric[simple_key] = v

    return metric


def log_step_metrics_host(
    metrics_host: dict[str, Any],
    pairing_meta: dict[str, Any] | None,
    config: dict[str, Any],
) -> None:
    """Log one update-step metric row from host Python."""
    if not metrics_host:
        return
    run = wandb.run
    if run is None or getattr(run, "disabled", False):
        return

    row: dict[str, Any] = {}
    for key, value in metrics_host.items():
        arr = np.asarray(value)
        if arr.ndim != 0:
            continue
        scalar = arr.item() if hasattr(arr, "item") else arr
        row[key] = scalar

    if not row:
        return

    task_set_names = config.get("task_set_names", None)
    steps_per_round = int(config.get("TOTAL_TIMESTEPS", 0))

    meta: dict[str, Any] | None = None
    if pairing_meta is not None:
        agent_ids = pairing_meta.get("agent_ids")
        if agent_ids is not None:
            agent_arr = np.asarray(agent_ids).reshape(-1)
            if agent_arr.size >= 2:
                agent_ids = [int(agent_arr[0]), int(agent_arr[1])]
            else:
                agent_ids = None
        meta = {
            "round_idx": int(_first_scalar(pairing_meta.get("round_idx"), 0)),
            "task_set_idx": int(_first_scalar(pairing_meta.get("task_set_idx"), 0)),
            "agent_ids": agent_ids,
        }

    payload = _build_prefixed_metric_payload(
        row,
        meta,
        task_set_names,
        steps_per_round,
        log_action_diagnostics=bool(config.get("LOG_ACTION_DIAGNOSTICS", False)),
    )
    _wandb_log(payload)


def log_host_round_metrics(
    all_metrics: dict[str, Any],
    pairings: list[tuple[int, int]],
    round_idx: int,
    task_set_idx: int,
    config: dict[str, Any],
    update_start: int = 0,
) -> None:
    """Host-side logging for vmapped training outputs (no jax.debug.callback)."""
    if not all_metrics or not pairings:
        return
    run = wandb.run
    if run is None or getattr(run, "disabled", False):
        return

    task_set_names = config.get("task_set_names", None)
    steps_per_round = int(config.get("TOTAL_TIMESTEPS", 0))
    log_every = int(
        max(
            1,
            config.get("LOG_EVERY_UPDATES", _default_log_every_updates(config)),
        )
    )

    metrics_host = jax.device_get(all_metrics)
    first_key = next(iter(metrics_host.keys()), None)
    if first_key is None:
        return
    first_val = np.asarray(metrics_host[first_key])
    if first_val.ndim < 2:
        # Expected shape [num_pairings, num_updates] for vmapped outputs.
        return
    num_pairings, num_updates = first_val.shape[0], first_val.shape[1]
    if num_pairings <= 0 or num_updates <= 0:
        return

    num_pairings_to_log = min(len(pairings), num_pairings)
    for update_idx in range(num_updates):
        if (update_idx % log_every) != 0:
            continue
        for pairing_idx in range(num_pairings_to_log):
            row: dict[str, Any] = {}
            for key, val in metrics_host.items():
                arr = np.asarray(val)
                if arr.ndim < 2:
                    continue
                if pairing_idx >= arr.shape[0] or update_idx >= arr.shape[1]:
                    continue
                scalar = arr[pairing_idx, update_idx]
                if np.isscalar(scalar):
                    row[key] = scalar.item() if hasattr(scalar, "item") else scalar
                else:
                    # Ignore non-scalar metrics for host-side scalar logging.
                    continue

            if not row:
                continue
            row.setdefault(
                "env_step",
                int(update_start + update_idx)
                * int(config.get("NUM_STEPS", 1))
                * int(config.get("NUM_ENVS", 1)),
            )
            a0, a1 = pairings[pairing_idx]
            meta = {
                "round_idx": round_idx,
                "task_set_idx": task_set_idx,
                "agent_ids": [a0, a1],
            }
            payload = _build_prefixed_metric_payload(
                row,
                meta,
                task_set_names,
                steps_per_round,
                log_action_diagnostics=bool(config.get("LOG_ACTION_DIAGNOSTICS", False)),
            )
            _wandb_log(payload)


def log_common_metrics(
    traj_batch: Any,
    info: dict[str, jax.Array],
    update_step: jax.Array,
    env_step: jax.Array,
    num_agents: int,
    config: dict[str, Any],
    custom_metrics: Optional[dict[str, jax.Array]] = None,
    viz_callback: Optional[Callable] = None,
    runner_state: Optional[Any] = None,
    eval_callback: Optional[Callable] = None,
    eval_callback_args: Optional[tuple] = None,
    eval_interval: int = 1,
    pairing_meta: Optional[dict[str, Any]] = None,
) -> None:
    """Log shared metrics and optional viz/eval callbacks."""
    metric = build_common_metrics(
        traj_batch=traj_batch,
        info=info,
        env_step=env_step,
        num_agents=num_agents,
        config=config,
        custom_metrics=custom_metrics,
    )

    # Log to WandB via callback with pairing prefix
    task_set_names = config.get("task_set_names", None)
    steps_per_round = int(config.get("TOTAL_TIMESTEPS", 0))

    def callback(m, meta, steps_per_round):
        prefixed_metric = _build_prefixed_metric_payload(
            m,
            meta,
            task_set_names,
            steps_per_round,
            log_action_diagnostics=bool(config.get("LOG_ACTION_DIAGNOSTICS", False)),
        )
        # Don't pass step= to avoid vmap ordering conflicts; WandB auto-increments
        _wandb_log(prefixed_metric)

    # Gate the metrics callback to fire every LOG_EVERY_UPDATES steps instead of
    # every update. Each jax.debug.callback is a device-host sync that stalls the
    # GPU, so we keep logs sparse by default (~60 logs/round) unless overridden.
    log_every = int(
        max(
            1,
            config.get("LOG_EVERY_UPDATES", _default_log_every_updates(config)),
        )
    )
    do_log = (update_step % log_every) == 0

    # Capture non-JAX values (strings, dicts) via closure; only pass JAX
    # arrays through jax.lax.cond to avoid "not a valid JAX type" errors.
    def _log_true(m):
        jax.debug.callback(callback, m, pairing_meta, steps_per_round)
        return jnp.array(0, dtype=jnp.int32)

    def _log_false(m):
        return jnp.array(0, dtype=jnp.int32)

    _ = jax.lax.cond(do_log, _log_true, _log_false, metric)

    # Visualization callback
    if viz_callback is not None and config.get("LOG_VIZ", True):
        viz_every = int(
            max(
                1,
                config.get("LOG_EVERY_UPDATES", _default_log_every_updates(config)),
            )
        )
        do_viz = (update_step % viz_every) == 0

        state = getattr(traj_batch, "state", None)
        if state is not None:
            # Capture task_set_names via closure (not passed through jax.lax.cond)
            def _viz_true(args):
                m, meta, tb, st, rs = args
                payload = _slice_viz_window(
                    tb,
                    st,
                    k=min(12, config.get("NUM_STEPS", 128)),
                )

                def _viz_host(mm, meta_host, pl, rs_host):
                    prefix = _make_prefix(meta_host, task_set_names)
                    prefixed_m = {f"{prefix}{k}": v for k, v in mm.items()}
                    prefixed_m["prefix"] = prefix
                    viz_callback(prefixed_m, config, rs_host, pl)

                jax.debug.callback(_viz_host, m, meta, payload, rs)
                return jnp.array(0, dtype=jnp.int32)

            def _viz_false(args):
                return jnp.array(0, dtype=jnp.int32)

            _ = jax.lax.cond(
                do_viz,
                _viz_true,
                _viz_false,
                (metric, pairing_meta, traj_batch, state, runner_state),
            )

    # Evaluation callback (if provided)
    if eval_callback is not None:
        do_eval = (update_step % eval_interval) == 0

        def _eval_true(args):
            cb, step, cb_args = args
            if cb_args is not None:
                jax.debug.callback(cb, *cb_args, step)
            else:
                jax.debug.callback(cb, step)
            return jnp.array(0, dtype=jnp.int32)

        def _eval_false(args):
            return jnp.array(0, dtype=jnp.int32)

        _ = jax.lax.cond(
            do_eval,
            _eval_true,
            _eval_false,
            (eval_callback, update_step, eval_callback_args),
        )


def _slice_viz_window(traj_batch, state, k=12) -> dict[str, jax.Array]:
    T = traj_batch.reward.shape[0]
    k = min(k, T)
    positions = state.positions
    N = positions.shape[1]
    M = 1  # single actor per env

    rew_all = traj_batch.reward.reshape(T, N, M)
    env_reward_mean = jnp.mean(rew_all, axis=(0, 2))
    env_id = jnp.argmax(env_reward_mean).astype(jnp.int32)

    time_steps_env = state.time_step[:, env_id]
    idx = jnp.arange(T, dtype=jnp.int32)
    reset_mask = time_steps_env == 0
    last_reset = jnp.max(jnp.where(reset_mask, idx, -1))
    s_if = jnp.where(last_reset <= T - k - 1, last_reset + 1, T - k)
    s_else = T - k
    s = jax.lax.select(jnp.any(reset_mask), s_if, s_else)
    sel = s + jnp.arange(k, dtype=jnp.int32)
    frame_idx = jnp.maximum(sel - 1, 0)

    actor_idxs = env_id * M + jnp.arange(M, dtype=jnp.int32)

    def take_time(x):
        return jnp.take(x, sel, axis=0)

    def take_frame(x):
        return jnp.take(x, frame_idx, axis=0)

    actions = jnp.take(take_time(traj_batch.action), actor_idxs, axis=1)
    probs = jnp.take(take_time(traj_batch.action_probs), actor_idxs, axis=1)
    rewards = jnp.take(take_time(traj_batch.reward), actor_idxs, axis=1)
    values = jnp.take(take_time(traj_batch.value), actor_idxs, axis=1)
    targets = jnp.take(
        take_time(getattr(traj_batch, "targets", traj_batch.value)),
        actor_idxs,
        axis=1,
    )
    advantages = jnp.take(
        take_time(getattr(traj_batch, "advantages", jnp.zeros_like(traj_batch.value))),
        actor_idxs,
        axis=1,
    )

    map_array = take_frame(state.map_array)[:, env_id]
    color_map = take_frame(state.color_map)[:, env_id]
    pos = take_frame(state.positions)[:, env_id]
    directions = take_frame(state.directions)[:, env_id]
    inventories = take_frame(state.inventories)[:, env_id]
    inventory_colors = take_frame(state.inventory_colors)[:, env_id]
    ep_timesteps = jnp.take(time_steps_env, frame_idx, axis=0)

    return {
        "map_array": map_array,
        "color_map": color_map,
        "positions": pos,
        "directions": directions,
        "inventories": inventories,
        "inventory_colors": inventory_colors,
        "ep_timesteps": ep_timesteps,
        "actions": actions,
        "rewards": rewards,
        "values": values,
        "targets": targets,
        "advantages": advantages,
        "probs": probs,
    }


def _collect_rollout_window_from_payload(payload, action_names=None):
    from banyan_grid.utils.render import render_grid_cpu

    data = {k: _to_host(v) for k, v in payload.items()}
    map_array = data["map_array"]
    color_map = data["color_map"]
    positions = data["positions"]
    directions = data["directions"]
    inventories = data["inventories"]
    inventory_colors = data["inventory_colors"]
    ep_timesteps = data["ep_timesteps"]

    k = map_array.shape[0]
    M = data["actions"].shape[1]
    A = data["probs"].shape[-1]

    frames = []
    for i in range(k):
        fr = render_grid_cpu(
            map_array[i],
            color_map[i],
            positions[i],
            directions[i],
            inventories[i],
            inventory_colors[i],
            int(ep_timesteps[i]),
        )
        frames.append(np.asarray(fr))

    action_names = _safe_action_names(A, given=action_names)
    return dict(
        frames=frames,
        sel=np.arange(k),
        ep_timesteps=[int(t) for t in ep_timesteps],
        M=M,
        A=A,
        actions=data["actions"],
        rewards=data["rewards"],
        values=data["values"],
        targets=data["targets"],
        advantages=data["advantages"],
        probs=data["probs"],
        action_names=action_names,
    )


def _build_rollout_panel_from_win(config, win):
    """Build the rollout panel figure from window data."""
    gridspec, plt = _get_matplotlib()
    sel, M, A = win["sel"], win["M"], win["A"]
    rewards, values, targets, advantages, probs = (
        win["rewards"],
        win["values"],
        win["targets"],
        win["advantages"],
        win["probs"],
    )
    names = win["action_names"]

    k = len(sel)
    show_value_plots = config.get("SHOW_VALUE_PLOTS", False)
    xs = np.arange(k)

    num_heatmaps = min(M, 2)
    metric_rows = 3 if show_value_plots else 1
    total_rows = metric_rows + num_heatmaps
    fig_h = 2.0 * metric_rows + 2.5 * num_heatmaps
    fig_w = 12

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs = gridspec.GridSpec(total_rows, 1, figure=fig)

    if show_value_plots:
        axR = fig.add_subplot(gs[0, 0])
        axV = fig.add_subplot(gs[1, 0])
        axA = fig.add_subplot(gs[2, 0])

        if M > 0:
            axR.plot(xs, rewards[sel, 0], label="A0")
        if M > 1:
            axR.plot(xs, rewards[sel, 1], linestyle="--", label="A1")
        axR.set_title("Reward")
        axR.set_xlabel("∆ steps from reset")
        axR.grid(True)
        axR.legend(fontsize=8)

        if M > 0:
            axV.plot(xs, values[sel, 0], label="A0 V")
            axV.plot(xs, targets[sel, 0], label="A0 T")
        if M > 1:
            axV.plot(xs, values[sel, 1], linestyle="--", label="A1 V")
            axV.plot(xs, targets[sel, 1], linestyle="--", label="A1 T")
        axV.set_title("Value & Target")
        axV.set_xlabel("∆ steps from reset")
        axV.grid(True)
        axV.legend(fontsize=8, ncol=2)

        if M > 0:
            axA.plot(xs, advantages[sel, 0], label="A0")
        if M > 1:
            axA.plot(xs, advantages[sel, 1], linestyle="--", label="A1")
        axA.set_title("Advantage")
        axA.set_xlabel("∆ steps from reset")
        axA.grid(True)
        axA.legend(fontsize=8)
    else:
        axR = fig.add_subplot(gs[0, 0])
        if M > 0:
            axR.plot(xs, rewards[sel, 0], label="A0")
        if M > 1:
            axR.plot(xs, rewards[sel, 1], linestyle="--", label="A1")
        axR.set_title("Reward")
        axR.set_xlabel("∆ steps from reset")
        axR.grid(True)
        axR.legend(fontsize=8)

    heatmap_start = metric_rows
    for idx, agent in enumerate(range(num_heatmaps)):
        axH = fig.add_subplot(gs[heatmap_start + idx, 0])
        p = probs[sel, agent, :].T  # (A, k)
        im = axH.imshow(p, aspect="auto", origin="lower", extent=(0, k, 0, A))
        axH.set_title(f"Action probabilities — agent {agent}")
        axH.set_xlabel("∆ steps from reset")
        axH.set_yticks(np.arange(A) + 0.5)
        axH.set_yticklabels(names)
        fig.colorbar(im, ax=axH, fraction=0.015)

    return fig, plt


def _build_frame_gallery_from_win(win):
    """Build frame gallery images with captions from window data."""
    frames, sel, M = win["frames"], win["sel"], win["M"]
    ep_timesteps = win["ep_timesteps"]
    actions, rewards, values = win["actions"], win["rewards"], win["values"]
    names = win["action_names"]

    imgs = []
    for i, t in enumerate(sel):
        ep_t = ep_timesteps[i]
        caps = [f"t={ep_t}"]
        if M > 0:
            a0 = int(actions[t, 0])
            r0 = float(rewards[t, 0])
            v0 = float(values[t, 0])
            caps.append(f"A0:{names[a0]} r0={r0:.2f} v0={v0:.2f}")
        if M > 1:
            a1 = int(actions[t, 1])
            r1 = float(rewards[t, 1])
            v1 = float(values[t, 1])
            caps.append(f"A1:{names[a1]} r1={r1:.2f} v1={v1:.2f}")
        imgs.append(wandb.Image(frames[i], caption=" | ".join(caps)))

    return imgs


def default_viz_callback(metric, config, runner_state, traj_batch):
    """Default visualization callback."""
    prefix = metric.get("prefix", "")
    log_panel = config.get("LOG_ROLLOUT_PANEL", True)
    log_gallery = config.get("LOG_FRAME_GALLERY", False)
    if not log_panel and not log_gallery:
        return

    action_names = config.get("action_names", None)

    def _log_payload(payload):
        try:
            names = action_names
            if names is None:
                num_actions = payload["probs"].shape[-1]
                names = [
                    "RIGHT",
                    "LEFT",
                    "UP",
                    "DOWN",
                    "STAY",
                    "DROP",
                    "PICKUP",
                    "TOGGLE",
                    "MERGE",
                ][:num_actions]
            win = _collect_rollout_window_from_payload(payload, action_names=names)
            out = {}
            if log_panel:
                fig, plt = _build_rollout_panel_from_win(config, win)
                out[f"{prefix}viz/rollout_panel"] = wandb.Image(fig)
                plt.close(fig)
            if log_gallery:
                imgs = _build_frame_gallery_from_win(win)
                out[f"{prefix}viz/frame_gallery"] = imgs
            if out:
                _wandb_log(out)
        except Exception as e:
            _logger.warning("Viz callback failed for %s: %s", prefix, e)

    _submit_viz_task(_log_payload, traj_batch)
