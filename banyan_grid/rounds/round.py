import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np

from banyan_grid.utils.logging import (
    configure_wandb_async,
    log_host_round_metrics,
)

_logger = logging.getLogger(__name__)

_TRAIN_VMAP_CACHE: dict[tuple[int, int, str], Callable] = {}
_EVAL_VMAP_CACHE: dict[tuple[int, int, str], Callable] = {}


def _extract_round_td_error_summary(
    all_metrics: Any | None,
    pairings: list[tuple[int, int]],
) -> dict[int, float]:
    """Extract per-agent round-boundary TD error from final training update."""
    if all_metrics is None or not pairings:
        return {}

    metrics_host = jax.device_get(all_metrics)
    summary: dict[int, float] = {}

    def _final_pair_value(metric_value: Any, pairing_idx: int) -> float | None:
        if metric_value is None:
            return None
        arr = np.asarray(metric_value)
        if arr.size == 0:
            return None
        if arr.ndim >= 2:
            if pairing_idx >= arr.shape[0]:
                return None
            scalar = arr[pairing_idx, -1]
        elif arr.ndim == 1:
            scalar = arr[-1]
        else:
            scalar = arr
        scalar_arr = np.asarray(scalar)
        if scalar_arr.size == 0:
            return None
        value = float(scalar_arr.reshape(-1)[0])
        if not np.isfinite(value):
            return None
        return value

    if isinstance(metrics_host, dict):
        left_metric = metrics_host.get("agent0/td_error_mse")
        right_metric = metrics_host.get("agent1/td_error_mse")
        critic_metric = metrics_host.get("critic/td_error_mse")
        for pairing_idx, (agent_left, agent_right) in enumerate(pairings):
            left_val = _final_pair_value(left_metric, pairing_idx)
            right_val = _final_pair_value(right_metric, pairing_idx)
            critic_val = _final_pair_value(critic_metric, pairing_idx)
            if left_val is None:
                left_val = critic_val
            if right_val is None:
                right_val = critic_val
            if left_val is not None:
                summary[agent_left] = left_val
            if right_val is not None:
                summary[agent_right] = right_val
        return summary

    if isinstance(metrics_host, list):
        for pairing_idx, pairing_metrics in enumerate(metrics_host):
            if pairing_idx >= len(pairings) or not isinstance(pairing_metrics, dict):
                continue
            agent_left, agent_right = pairings[pairing_idx]
            left_val = _final_pair_value(pairing_metrics.get("agent0/td_error_mse"), 0)
            right_val = _final_pair_value(pairing_metrics.get("agent1/td_error_mse"), 0)
            critic_val = _final_pair_value(
                pairing_metrics.get("critic/td_error_mse"), 0
            )
            if left_val is None:
                left_val = critic_val
            if right_val is None:
                right_val = critic_val
            if left_val is not None:
                summary[agent_left] = left_val
            if right_val is not None:
                summary[agent_right] = right_val

    return summary


def run_round_groups_parallel(
    groups: list[dict[str, Any]],
    agent_params: dict[int, Any | None],
    round_rngs: list[jax.Array],
    round_idx: int,
    task_set_to_idx: dict[str, int],
    get_train_fn: Callable,
) -> tuple[dict[int, Any | None], dict[int, float]]:
    """Run multiple pairing groups in parallel using threads.

    Each group contains pairings that share the same dataset/config and can be
    vmapped together. Different groups (different datasets) run in parallel threads.

    Args:
        groups: List of group dicts, each with keys:
            - task_key: str
            - train_steps: int
            - layout_bank_file: str | None
            - layout_bank_num_active: int | None
            - pairings: list of (agent_i, agent_j) tuples
            - freeze_masks: optional list of (freeze_left, freeze_right) tuples
        agent_params: Current agent parameters
        round_rngs: One RNG per group
        round_idx: Current round index
        task_set_to_idx: Mapping from task_key to task_set index
        get_train_fn: Callable(task_key, total_timesteps, round_overrides) -> (train_fn, config)

    Returns:
        Tuple of (updated agent params, per-agent TD error summary).
    """
    if not groups:
        return agent_params, {}

    # Validate that agent sets are disjoint across groups
    # This is required for safe parallel execution
    all_agents_seen: set[int] = set()
    for group in groups:
        group_agents = set()
        for i, j in group["pairings"]:
            group_agents.add(i)
            group_agents.add(j)
        overlap = all_agents_seen & group_agents
        if overlap:
            raise ValueError(
                f"Cannot parallelize: agent(s) {overlap} appear in multiple groups "
                f"within the same round. Each agent can only be in one group per round."
            )
        all_agents_seen.update(group_agents)

    # If only one group, run directly without threading overhead
    if len(groups) == 1:
        group = groups[0]
        train_fn, train_cfg = get_train_fn(
            group["task_key"],
            total_timesteps=group["train_steps"],
            round_overrides={
                "layout_bank_file": group.get("layout_bank_file"),
                "layout_bank_num_active": group.get("layout_bank_num_active"),
            },
        )
        task_set_idx = task_set_to_idx[group["task_key"]]
        return run_round(
            pairings=group["pairings"],
            agent_params=agent_params,
            round_rng=round_rngs[0],
            round_idx=round_idx,
            task_set_idx=task_set_idx,
            train_fn=train_fn,
            freeze_masks=group.get("freeze_masks"),
            train_config=train_cfg,
        )

    # Collect param snapshots for each group (they may need different subsets)
    # Run groups in parallel threads
    results: dict[int, tuple[dict[int, Any | None], dict[int, float]]] = {}

    def run_group(
        group_idx: int,
    ) -> tuple[int, dict[int, Any | None], dict[int, float]]:
        group = groups[group_idx]
        train_fn, train_cfg = get_train_fn(
            group["task_key"],
            total_timesteps=group["train_steps"],
            round_overrides={
                "layout_bank_file": group.get("layout_bank_file"),
                "layout_bank_num_active": group.get("layout_bank_num_active"),
            },
        )
        task_set_idx = task_set_to_idx[group["task_key"]]
        updated, td_summary = run_round(
            pairings=group["pairings"],
            agent_params=agent_params,
            round_rng=round_rngs[group_idx],
            round_idx=round_idx,
            task_set_idx=task_set_idx,
            train_fn=train_fn,
            freeze_masks=group.get("freeze_masks"),
            train_config=train_cfg,
        )
        # Return only the agents that were updated in this group
        group_agents = set()
        for i, j in group["pairings"]:
            group_agents.add(i)
            group_agents.add(j)
        return group_idx, {a: updated[a] for a in group_agents}, td_summary

    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        futures = {executor.submit(run_group, i): i for i in range(len(groups))}
        for future in as_completed(futures):
            group_idx, group_updated, td_summary = future.result()
            results[group_idx] = (group_updated, td_summary)

    # Merge results - each group updates disjoint sets of agents
    updated_params = agent_params.copy()
    merged_td_summary: dict[int, float] = {}
    for group_idx in sorted(results.keys()):
        group_updated, group_td_summary = results[group_idx]
        for agent_id, params in group_updated.items():
            updated_params[agent_id] = params
        merged_td_summary.update(group_td_summary)

    return updated_params, merged_td_summary


def run_round(
    pairings: list[tuple[int, int]],
    agent_params: dict[int, Any | None],
    round_rng: jax.Array,
    round_idx: int,
    task_set_idx: int,
    train_fn: Callable,
    *,
    freeze_masks: list[tuple[bool, bool]] | None = None,
    train_config: dict[str, Any] | None = None,
    jit: bool = True,
) -> tuple[dict[int, Any | None], dict[int, float]]:
    """Run training for all pairings in a round (vmapped)."""
    num_pairings = len(pairings)

    # Generate RNG keys for all pairings from the round-level RNG
    rngs = jax.random.split(round_rng, num_pairings)

    if freeze_masks is None:
        freeze_masks = [(False, False) for _ in range(num_pairings)]
    if len(freeze_masks) != num_pairings:
        raise ValueError(
            "freeze_masks must match number of pairings "
            f"({len(freeze_masks)} vs {num_pairings})."
        )

    # Build stacked pairing_meta with JAX arrays
    pairing_meta = {
        "round_idx": jnp.full((num_pairings,), round_idx, dtype=jnp.int32),
        "task_set_idx": jnp.full((num_pairings,), task_set_idx, dtype=jnp.int32),
        "agent_ids": jnp.array(pairings, dtype=jnp.int32),  # (num_pairings, 2)
        "freeze_mask": jnp.array(freeze_masks, dtype=jnp.bool_),
    }

    # Prepare init_params for all pairings
    # For random baseline: all None
    # For IPPO: need to stack params into batched pytree
    all_init_params = []
    for pairing in pairings:
        i, j = pairing
        params_i = agent_params.get(i)
        params_j = agent_params.get(j)

        if params_i is None and params_j is None:
            # Random baseline case
            all_init_params.append(None)
        else:
            # IPPO case: build tuple (left_params, right_params)
            # Position 0 = agent i (left), Position 1 = agent j (right)
            all_init_params.append((params_i, params_j))

    # Check if we can vmap (all params None or all have same structure)
    all_none = all(p is None for p in all_init_params)
    all_same_structure = False
    supports_vmap = bool(getattr(train_fn, "supports_vmap", True))

    if not all_none:
        # Check if all non-None params have the same structure
        non_none_params = [p for p in all_init_params if p is not None]
        if len(non_none_params) >= 1:
            first_structure = jax.tree_util.tree_structure(non_none_params[0])
            all_same_structure = all(
                jax.tree_util.tree_structure(p) == first_structure
                for p in non_none_params
            )

    def _run_sequential_results() -> dict[str, Any]:
        sequential_results = []
        for idx in range(num_pairings):
            single_meta = jax.tree_util.tree_map(lambda x: x[idx], pairing_meta)
            result = train_fn(rngs[idx], all_init_params[idx], single_meta)
            sequential_results.append(result)
        return {
            "final_params": [r.get("final_params") for r in sequential_results],
            "all_metrics": [r.get("all_metrics") for r in sequential_results],
        }

    if not supports_vmap:
        _logger.info(
            "Sequential execution: train_fn does not support vmap (%d pairings)",
            num_pairings,
        )
        results = _run_sequential_results()
    elif all_none:
        # Random baseline: vmap with None params
        # The train_fn expects pairing_meta as a dict, so we pass the batched version
        # Each call will receive a slice via vmap
        if jit:
            cache_key = (id(train_fn), num_pairings, "none")
            vmapped_train = _TRAIN_VMAP_CACHE.get(cache_key)
            if vmapped_train is None:
                vmapped_train = jax.jit(
                    jax.vmap(
                        lambda rng, meta: train_fn(rng, None, meta), in_axes=(0, 0)
                    )
                )
                _TRAIN_VMAP_CACHE[cache_key] = vmapped_train
        else:
            vmapped_train = jax.vmap(
                lambda rng, meta: train_fn(rng, None, meta), in_axes=(0, 0)
            )
        results = vmapped_train(rngs, pairing_meta)

    elif all_same_structure:
        # IPPO case: stack params and vmap
        # Stack init_params into batched pytree
        # Note: This assumes all pairings have the same agent param structure
        stacked_params = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs),
            *all_init_params,
        )
        if jit:
            cache_key = (id(train_fn), num_pairings, "params")
            vmapped_train = _TRAIN_VMAP_CACHE.get(cache_key)
            if vmapped_train is None:
                vmapped_train = jax.jit(
                    jax.vmap(
                        lambda rng, params, meta: train_fn(rng, params, meta),
                        in_axes=(0, 0, 0),
                    )
                )
                _TRAIN_VMAP_CACHE[cache_key] = vmapped_train
        else:
            vmapped_train = jax.vmap(
                lambda rng, params, meta: train_fn(rng, params, meta),
                in_axes=(0, 0, 0),
            )
        results = vmapped_train(rngs, stacked_params, pairing_meta)

    else:
        # Fallback: sequential execution if structures differ
        _logger.warning(
            "Sequential fallback: %d pairings have mixed param structures", num_pairings
        )
        results = _run_sequential_results()

    round_td_error_summary: dict[int, float] = {}
    if isinstance(results, dict):
        round_td_error_summary = _extract_round_td_error_summary(
            results.get("all_metrics"),
            pairings,
        )
    elif isinstance(results, list):
        for pairing_idx, pairing_result in enumerate(results):
            if pairing_idx >= len(pairings) or not isinstance(pairing_result, dict):
                continue
            boundary_metrics = pairing_result.get("boundary_metrics")
            if not isinstance(boundary_metrics, dict):
                continue
            agent_left, agent_right = pairings[pairing_idx]
            left_val = boundary_metrics.get("agent0/td_error_mse")
            right_val = boundary_metrics.get("agent1/td_error_mse")
            critic_val = boundary_metrics.get("critic/td_error_mse")

            def _to_scalar(value: Any) -> float | None:
                if value is None:
                    return None
                arr = np.asarray(value)
                if arr.size == 0:
                    return None
                scalar = float(arr.reshape(-1)[0])
                if not np.isfinite(scalar):
                    return None
                return scalar

            left_scalar = _to_scalar(left_val)
            right_scalar = _to_scalar(right_val)
            critic_scalar = _to_scalar(critic_val)
            if left_scalar is None:
                left_scalar = critic_scalar
            if right_scalar is None:
                right_scalar = critic_scalar
            if left_scalar is not None:
                round_td_error_summary[agent_left] = left_scalar
            if right_scalar is not None:
                round_td_error_summary[agent_right] = right_scalar

    if train_config is not None and isinstance(results, dict):
        configure_wandb_async(bool(train_config.get("LIVE_LOG_ASYNC", True)))
        chunk_updates = int(train_config.get("LIVE_LOG_CHUNK_UPDATES", 0))
        maybe_metrics = results.get("all_metrics")
        if maybe_metrics is None:
            pass
        elif isinstance(maybe_metrics, dict):
            first_key = next(iter(maybe_metrics.keys()), None)
            if first_key is not None and chunk_updates > 0:
                first_shape = getattr(maybe_metrics[first_key], "shape", ())
                num_updates = int(first_shape[1]) if len(first_shape) >= 2 else 0
                if num_updates > 0:
                    for start in range(0, num_updates, chunk_updates):
                        end = min(num_updates, start + chunk_updates)
                        chunk_metrics = jax.tree_util.tree_map(
                            lambda x, s=start, e=end: x[:, s:e], maybe_metrics
                        )
                        log_host_round_metrics(
                            all_metrics=chunk_metrics,
                            pairings=pairings,
                            round_idx=round_idx,
                            task_set_idx=task_set_idx,
                            config=train_config,
                            update_start=start,
                        )
                else:
                    log_host_round_metrics(
                        all_metrics=maybe_metrics,
                        pairings=pairings,
                        round_idx=round_idx,
                        task_set_idx=task_set_idx,
                        config=train_config,
                    )
            else:
                log_host_round_metrics(
                    all_metrics=maybe_metrics,
                    pairings=pairings,
                    round_idx=round_idx,
                    task_set_idx=task_set_idx,
                    config=train_config,
                )
        elif isinstance(maybe_metrics, list):
            # Sequential fallback returns per-pairing metric dicts.
            # Log each pairing independently by adding a leading batch axis.
            for idx, pairing_metrics in enumerate(maybe_metrics):
                if (
                    idx >= len(pairings)
                    or pairing_metrics is None
                    or not isinstance(pairing_metrics, dict)
                ):
                    continue
                first_key = next(iter(pairing_metrics.keys()), None)
                if first_key is None:
                    continue
                first_shape = getattr(pairing_metrics[first_key], "shape", ())
                num_updates = int(first_shape[0]) if len(first_shape) >= 1 else 0
                if chunk_updates > 0 and num_updates > 0:
                    for start in range(0, num_updates, chunk_updates):
                        end = min(num_updates, start + chunk_updates)
                        sliced_metrics = jax.tree_util.tree_map(
                            lambda x, s=start, e=end: jnp.asarray(x)[s:e],
                            pairing_metrics,
                        )
                        batched_metrics = jax.tree_util.tree_map(
                            lambda x: jnp.expand_dims(jnp.asarray(x), axis=0),
                            sliced_metrics,
                        )
                        log_host_round_metrics(
                            all_metrics=batched_metrics,
                            pairings=[pairings[idx]],
                            round_idx=round_idx,
                            task_set_idx=task_set_idx,
                            config=train_config,
                            update_start=start,
                        )
                else:
                    batched_metrics = jax.tree_util.tree_map(
                        lambda x: jnp.expand_dims(jnp.asarray(x), axis=0),
                        pairing_metrics,
                    )
                    log_host_round_metrics(
                        all_metrics=batched_metrics,
                        pairings=[pairings[idx]],
                        round_idx=round_idx,
                        task_set_idx=task_set_idx,
                        config=train_config,
                    )

    # Update agent_params from results
    updated_params = agent_params.copy()

    # Handle results - could be batched dict (from vmap) or list (from sequential)
    if isinstance(results, dict):
        final_params = results.get("final_params")
        if final_params is None:
            # Random baseline - no params to update
            pass
        elif isinstance(final_params, tuple) and len(final_params) == 2:
            # Tuple case: (left_params, right_params) - batched under vmap
            # Position 0 = left agent, Position 1 = right agent
            left_batched, right_batched = final_params
            for idx, (i, j) in enumerate(pairings):
                if left_batched is not None:
                    single_left = jax.tree_util.tree_map(lambda x: x[idx], left_batched)
                    updated_params[i] = single_left
                if right_batched is not None:
                    single_right = jax.tree_util.tree_map(
                        lambda x: x[idx], right_batched
                    )
                    updated_params[j] = single_right
        elif isinstance(final_params, dict):
            # Batched dict case: final_params is {agent_id: batched_params}
            # Unstack by indexing into each agent's batched params
            for idx, (i, j) in enumerate(pairings):
                if i in final_params:
                    # Extract params for agent i at this pairing index
                    batched_params_i = final_params[i]
                    if batched_params_i is not None:
                        # Index into batched pytree to get single pairing's params
                        single_params_i = jax.tree_util.tree_map(
                            lambda x: x[idx], batched_params_i
                        )
                        updated_params[i] = single_params_i

                if j in final_params:
                    batched_params_j = final_params[j]
                    if batched_params_j is not None:
                        single_params_j = jax.tree_util.tree_map(
                            lambda x: x[idx], batched_params_j
                        )
                        updated_params[j] = single_params_j
        elif isinstance(final_params, list):
            # List case (from sequential fallback)
            for idx, (i, j) in enumerate(pairings):
                if idx < len(final_params):
                    pairing_params = final_params[idx]
                    if pairing_params is not None:
                        if isinstance(pairing_params, tuple):
                            # Tuple from sequential: (left, right)
                            updated_params[i] = pairing_params[0]
                            updated_params[j] = pairing_params[1]
                        elif isinstance(pairing_params, dict):
                            updated_params[i] = pairing_params.get(
                                i, updated_params.get(i)
                            )
                            updated_params[j] = pairing_params.get(
                                j, updated_params.get(j)
                            )

    return updated_params, round_td_error_summary


def run_eval_round(
    eval_combos: list[tuple[int, int, str]],
    agent_params: dict[int, Any | None],
    eval_rng: jax.Array,
    round_idx: int,
    eval_fn: Callable,
    task_set_to_idx: dict[str, int],
    batch_size: int = 8,
    *,
    jit: bool = True,
) -> dict[tuple[int, int, str], dict[str, float]]:
    """Run eval for (pair, task_set) combos in batches."""
    if not eval_combos:
        return {}

    output = {}

    def _strip_params(params: Any | None) -> Any | None:
        if params is None:
            return None
        if hasattr(params, "batch_stats"):
            return {"params": params.params, "batch_stats": params.batch_stats}
        return params.params if hasattr(params, "params") else params

    param_combos: list[tuple[int, int, str]] = []
    param_pairs: list[tuple[Any, Any]] = []
    param_rngs: list[jax.Array] = []
    scratch_combos: list[tuple[int, int, str]] = []
    scratch_rngs: list[jax.Array] = []

    rngs = jax.random.split(eval_rng, len(eval_combos))
    for (i, j, task_set), combo_rng in zip(eval_combos, rngs):
        params_i = _strip_params(agent_params.get(i))
        params_j = _strip_params(agent_params.get(j))
        if params_i is None or params_j is None:
            scratch_combos.append((i, j, task_set))
            scratch_rngs.append(combo_rng)
        else:
            param_combos.append((i, j, task_set))
            param_pairs.append((params_i, params_j))
            param_rngs.append(combo_rng)

    def _eval_batched(
        batch_combos: list[tuple[int, int, str]],
        batch_rngs: jax.Array,
        *,
        stacked_params: Any | None,
        mode: str,
    ) -> None:
        batch_len = len(batch_combos)
        if batch_len == 0:
            return
        batch_meta = {
            "round_idx": jnp.full((batch_len,), round_idx, dtype=jnp.int32),
            "task_set_idx": jnp.array(
                [task_set_to_idx[ts] for _, _, ts in batch_combos],
                dtype=jnp.int32,
            ),
            "agent_ids": jnp.array(
                [(i, j) for i, j, _ in batch_combos],
                dtype=jnp.int32,
            ),
        }

        if jit:
            cache_key = (id(eval_fn), batch_len, mode)
            vmapped_eval = _EVAL_VMAP_CACHE.get(cache_key)
            if vmapped_eval is None:
                if mode == "none":
                    vmapped_eval = jax.jit(
                        jax.vmap(
                            lambda rng, meta: eval_fn(rng, None, meta),
                            in_axes=(0, 0),
                        )
                    )
                else:
                    vmapped_eval = jax.jit(
                        jax.vmap(
                            lambda rng, params, meta: eval_fn(rng, params, meta),
                            in_axes=(0, 0, 0),
                        )
                    )
                _EVAL_VMAP_CACHE[cache_key] = vmapped_eval
        else:
            if mode == "none":
                vmapped_eval = jax.vmap(
                    lambda rng, meta: eval_fn(rng, None, meta),
                    in_axes=(0, 0),
                )
            else:
                vmapped_eval = jax.vmap(
                    lambda rng, params, meta: eval_fn(rng, params, meta),
                    in_axes=(0, 0, 0),
                )

        if mode == "none":
            batch_results = vmapped_eval(batch_rngs, batch_meta)
        else:
            batch_results = vmapped_eval(batch_rngs, stacked_params, batch_meta)

        def _extract_result(x, idx):
            """Extract result at batch index, handling both scalars and arrays."""
            val = x[idx]
            # If result is a scalar (0-dim or shape ()), convert to float
            # If result is an array (e.g., probe_success_rates), keep as list
            if hasattr(val, "shape") and val.shape != ():
                return val.tolist()
            return float(val)

        for idx, (i, j, task_set) in enumerate(batch_combos):
            combo_result = jax.tree_util.tree_map(
                lambda x, idx=idx: _extract_result(x, idx), batch_results
            )
            output[(i, j, task_set)] = combo_result

    # Evaluate combos with params (if any).
    if param_combos:
        all_same_structure = True
        if param_pairs:
            first_structure = jax.tree_util.tree_structure(param_pairs[0])
            all_same_structure = all(
                jax.tree_util.tree_structure(p) == first_structure for p in param_pairs
            )
        if all_same_structure:
            for batch_start in range(0, len(param_combos), batch_size):
                batch_end = min(batch_start + batch_size, len(param_combos))
                batch_combos = param_combos[batch_start:batch_end]
                batch_params = param_pairs[batch_start:batch_end]
                batch_rngs = jnp.stack(param_rngs[batch_start:batch_end])
                stacked_params = jax.tree_util.tree_map(
                    lambda *xs: jnp.stack(xs),
                    *batch_params,
                )
                _eval_batched(
                    batch_combos,
                    batch_rngs,
                    stacked_params=stacked_params,
                    mode="params",
                )
        else:
            for combo, params, combo_rng in zip(param_combos, param_pairs, param_rngs):
                i, j, task_set = combo
                meta = {
                    "round_idx": jnp.array(round_idx, dtype=jnp.int32),
                    "task_set_idx": jnp.array(
                        task_set_to_idx[task_set], dtype=jnp.int32
                    ),
                    "agent_ids": jnp.array([i, j], dtype=jnp.int32),
                }
                result = eval_fn(combo_rng, params, meta)
                result = jax.device_get(result)
                output[(i, j, task_set)] = {
                    k: v.tolist() if hasattr(v, "shape") and v.shape != () else float(v)
                    for k, v in result.items()
                }

    # Evaluate scratch combos (if any).
    if scratch_combos:
        for batch_start in range(0, len(scratch_combos), batch_size):
            batch_end = min(batch_start + batch_size, len(scratch_combos))
            batch_combos = scratch_combos[batch_start:batch_end]
            batch_rngs = jnp.stack(scratch_rngs[batch_start:batch_end])
            _eval_batched(
                batch_combos,
                batch_rngs,
                stacked_params=None,
                mode="none",
            )

    return output
