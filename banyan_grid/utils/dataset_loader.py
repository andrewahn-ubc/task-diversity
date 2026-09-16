"""Shared dataset and layout-bank helpers for Banyan."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from banyan_grid.environment.constants import (
    ITEM_TO_TILE,
    NUM_COLORS,
    NUM_RULESET_ITEMS,
    NUM_TILE_TYPES,
    RULE_TYPE_COLLECT,
    RULE_TYPE_COMBINE,
    RULE_TYPE_DISTRACTOR_COMBINE,
    RULE_TYPE_TERNARY_COMBINE,
    RULE_TYPE_TRANSFORM,
    TileType,
    TILE_TO_ITEM,
)
from banyan_grid.environment.rules import (
    DEFAULT_MAX_COMBINE_CASCADES,
    DEFAULT_MAX_TRANSFORM_APPLIES,
)
from banyan_grid.tasks.ruleset_codec import (
    unpack_rules_uint32_jit,
)


def resolve_layout_bank_path(layout_bank_file: str, dataset_dir: str) -> Path:
    raw = Path(layout_bank_file).expanduser()
    if raw.is_absolute():
        return raw
    repo_root = Path(__file__).resolve().parents[2]
    candidates = (
        (Path.cwd() / raw).resolve(),
        (repo_root / raw).resolve(),
        (Path(dataset_dir).resolve() / raw).resolve(),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def build_reachable_masks_from_masks_np(obstacle_masks: np.ndarray) -> np.ndarray:
    """Return cells reachable from the canonical spawn seeds for each layout."""
    num_layouts, h, w = obstacle_masks.shape
    reachable_masks = np.zeros((num_layouts, h, w), dtype=np.bool_)
    spawn_cells = [(0, 0), (1, 0)]
    nbr4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for idx in range(num_layouts):
        obs = obstacle_masks[idx]
        walkable = ~obs
        reachable = np.zeros_like(walkable, dtype=np.bool_)
        stack: list[tuple[int, int]] = []
        for sy, sx in spawn_cells:
            if 0 <= sy < h and 0 <= sx < w and walkable[sy, sx]:
                reachable[sy, sx] = True
                stack.append((sy, sx))
        while stack:
            y, x = stack.pop()
            for dy, dx in nbr4:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if not walkable[ny, nx] or reachable[ny, nx]:
                    continue
                reachable[ny, nx] = True
                stack.append((ny, nx))
        reachable_masks[idx] = reachable

    return reachable_masks


def build_reserved_cells_from_masks_np(
    obstacle_masks: np.ndarray,
    reachable_masks: np.ndarray | None = None,
) -> np.ndarray:
    """Convert [L,H,W] masks to reserved cells (obstacles + unreachable open cells)."""
    num_layouts, h, w = obstacle_masks.shape
    max_cells = h * w
    reserved = np.full((num_layouts, max_cells, 2), -1, dtype=np.int32)
    if reachable_masks is None:
        reachable_masks = build_reachable_masks_from_masks_np(obstacle_masks)

    for idx in range(num_layouts):
        obs = obstacle_masks[idx]
        reachable = reachable_masks[idx]
        reserve_mask = obs | (~reachable)
        ys, xs = np.where(reserve_mask)
        n = int(min(max_cells, ys.shape[0]))
        if n > 0:
            reserved[idx, :n, 0] = ys[:n]
            reserved[idx, :n, 1] = xs[:n]
    return reserved


def load_layout_bank(
    config: dict[str, Any],
    *,
    dataset_dir: str,
    dataset_file: str,
    grid_size: int,
    num_rulesets: int,
    device: jax.Device | None = None,
) -> dict[str, Any] | None:
    layout_bank_file = config.get("LAYOUT_BANK_FILE", None)
    if layout_bank_file in (None, ""):
        return None

    layout_path = resolve_layout_bank_path(str(layout_bank_file), dataset_dir)
    if not layout_path.exists():
        raise FileNotFoundError(f"Layout bank not found: {layout_path}")

    bank = np.load(layout_path, allow_pickle=False)
    obstacle_tiles = np.array(
        [
            int(TileType.BLOCK),
        ],
        dtype=np.int32,
    )
    if "obstacle_mask" in bank:
        obstacle_masks_np = np.asarray(bank["obstacle_mask"], dtype=np.bool_)
    elif "obstacle_map" in bank:
        obstacle_map = np.asarray(bank["obstacle_map"], dtype=np.int32)
        obstacle_masks_np = np.isin(obstacle_map, obstacle_tiles)
    elif "map_array" in bank:
        maps_np = np.asarray(bank["map_array"], dtype=np.int32)
        obstacle_masks_np = np.isin(maps_np, obstacle_tiles)
    else:
        raise ValueError(
            "Layout bank must contain one of: obstacle_mask, obstacle_map, map_array."
        )

    if obstacle_masks_np.ndim != 3:
        raise ValueError(
            f"Layout bank obstacle masks must be rank-3 [L,H,W], got shape={obstacle_masks_np.shape}."
        )
    if obstacle_masks_np.shape[1] != grid_size or obstacle_masks_np.shape[2] != grid_size:
        raise ValueError(
            f"Layout bank grid size mismatch: expected {grid_size}x{grid_size}, got "
            f"{obstacle_masks_np.shape[1]}x{obstacle_masks_np.shape[2]}."
        )

    total_layouts = int(obstacle_masks_np.shape[0])
    requested_active = int(config.get("LAYOUT_BANK_NUM_ACTIVE", 0))
    if requested_active <= 0:
        active_layouts = total_layouts
    else:
        active_layouts = min(total_layouts, requested_active)
    if active_layouts <= 0:
        raise ValueError("Layout bank has no usable layouts.")

    obstacle_masks_np = obstacle_masks_np[:active_layouts]
    spawn_reachable_masks_np = build_reachable_masks_from_masks_np(obstacle_masks_np)
    reserved_masks_np = obstacle_masks_np | (~spawn_reachable_masks_np)
    reserved_cells_np = build_reserved_cells_from_masks_np(
        obstacle_masks_np,
        reachable_masks=spawn_reachable_masks_np,
    )

    precomputed_maps = None
    precomputed_colors = None
    precomputed_task_layout_indices = None
    precomputed_task_layout_counts = None
    precomputed_coverage = 0
    precomputed_max_choices = 0
    precomputed_source_match = False
    has_full_map_bank = False

    maps_np_raw = np.asarray(bank["map_array"], dtype=np.int32) if "map_array" in bank else None
    colors_np_raw = (
        np.asarray(bank["color_map"], dtype=np.int32) if "color_map" in bank else None
    )
    ruleset_indices_np_raw = (
        np.asarray(bank["ruleset_indices"], dtype=np.int32)
        if "ruleset_indices" in bank
        else None
    )
    source_dataset_raw = bank["source_dataset"].item() if "source_dataset" in bank else None
    dataset_path = (Path(dataset_dir) / dataset_file).resolve()
    if source_dataset_raw not in (None, ""):
        try:
            precomputed_source_match = Path(str(source_dataset_raw)).expanduser().resolve() == dataset_path
        except OSError:
            precomputed_source_match = False

    if (
        maps_np_raw is not None
        and colors_np_raw is not None
        and ruleset_indices_np_raw is not None
        and precomputed_source_match
    ):
        has_full_map_bank = True
        maps_np = maps_np_raw[:active_layouts]
        colors_np = colors_np_raw[:active_layouts]
        ruleset_indices_np = ruleset_indices_np_raw[:active_layouts]
        valid_task_mask = (ruleset_indices_np >= 0) & (ruleset_indices_np < num_rulesets)
        if np.any(valid_task_mask):
            counts_np = np.bincount(
                ruleset_indices_np[valid_task_mask], minlength=num_rulesets
            ).astype(np.int32, copy=False)
            max_choices = int(counts_np.max()) if counts_np.size > 0 else 0
            if max_choices > 0:
                index_table_np = np.full(
                    (num_rulesets, max_choices), -1, dtype=np.int32
                )
                cursor_np = np.zeros((num_rulesets,), dtype=np.int32)
                for layout_idx, task_idx in enumerate(ruleset_indices_np.tolist()):
                    if task_idx < 0 or task_idx >= num_rulesets:
                        continue
                    slot = int(cursor_np[task_idx])
                    index_table_np[task_idx, slot] = int(layout_idx)
                    cursor_np[task_idx] = slot + 1
                precomputed_coverage = int(np.sum(counts_np > 0))
                precomputed_max_choices = max_choices
                if device is not None:
                    precomputed_maps = jax.device_put(maps_np, device)
                    precomputed_colors = jax.device_put(colors_np, device)
                    precomputed_task_layout_indices = jax.device_put(index_table_np, device)
                    precomputed_task_layout_counts = jax.device_put(counts_np, device)
                else:
                    precomputed_maps = jnp.asarray(maps_np, dtype=jnp.int32)
                    precomputed_colors = jnp.asarray(colors_np, dtype=jnp.int32)
                    precomputed_task_layout_indices = jnp.asarray(
                        index_table_np, dtype=jnp.int32
                    )
                    precomputed_task_layout_counts = jnp.asarray(
                        counts_np, dtype=jnp.int32
                    )

    if device is not None:
        obstacle_masks = jax.device_put(obstacle_masks_np, device)
        reserved_masks = jax.device_put(reserved_masks_np, device)
        reserved_cells = jax.device_put(reserved_cells_np, device)
        spawn_reachable_masks = jax.device_put(spawn_reachable_masks_np, device)
    else:
        obstacle_masks = jnp.asarray(obstacle_masks_np, dtype=jnp.bool_)
        reserved_masks = jnp.asarray(reserved_masks_np, dtype=jnp.bool_)
        reserved_cells = jnp.asarray(reserved_cells_np, dtype=jnp.int32)
        spawn_reachable_masks = jnp.asarray(
            spawn_reachable_masks_np, dtype=jnp.bool_
        )

    return {
        "path": str(layout_path),
        "num_total": total_layouts,
        "num_active": active_layouts,
        "obstacle_masks": obstacle_masks,
        "reserved_masks": reserved_masks,
        "reserved_cells": reserved_cells,
        "spawn_reachable_masks": spawn_reachable_masks,
        "precomputed_maps": precomputed_maps,
        "precomputed_colors": precomputed_colors,
        "precomputed_task_layout_indices": precomputed_task_layout_indices,
        "precomputed_task_layout_counts": precomputed_task_layout_counts,
        "precomputed_coverage": precomputed_coverage,
        "precomputed_max_choices": precomputed_max_choices,
        "precomputed_source_match": precomputed_source_match,
        "has_full_map_bank": has_full_map_bank,
    }


def resolve_global_max_rules(config: dict[str, Any]) -> int:
    """Resolve cross-round max rule count, if provided."""
    raw = config.get("GLOBAL_MAX_RULES", 0)
    try:
        v = int(raw)
        if v > 0:
            return v
    except (TypeError, ValueError):
        pass

    info = config.get("ruleset_info", config.get("RULESET_INFO"))
    if isinstance(info, dict):
        best = 0
        for task_info in info.values():
            if not isinstance(task_info, dict):
                continue
            rule_shape = task_info.get("rule_shape")
            if isinstance(rule_shape, (list, tuple)) and len(rule_shape) > 0:
                try:
                    best = max(best, int(rule_shape[0]))
                except (TypeError, ValueError):
                    pass
        return best
    return 0


def active_rule_counts_from_packed_host(packed_host: np.ndarray) -> np.ndarray:
    counts = np.sum(packed_host != np.uint32(5), axis=1, dtype=np.int32)
    return np.maximum(counts.astype(np.int32, copy=False), 1)


def compute_rule_execution_signatures_host(
    rulesets: np.ndarray,
    active_rule_counts: np.ndarray | None = None,
) -> np.ndarray:
    rules = np.asarray(rulesets, dtype=np.int32)
    if rules.ndim != 3:
        raise ValueError(
            f"Expected decoded rulesets with shape (N, R, C), got {rules.shape}"
        )
    row_count = int(rules.shape[1])
    row_type = rules[:, :, 0]
    row_idx = np.arange(row_count, dtype=np.int32)[None, :]
    if active_rule_counts is None:
        active_mask = np.ones((rules.shape[0], row_count), dtype=bool)
    else:
        active_counts = np.asarray(active_rule_counts, dtype=np.int32)
        if active_counts.shape != (rules.shape[0],):
            raise ValueError(
                "active_rule_counts must have shape "
                f"({rules.shape[0]},), got {active_counts.shape}"
            )
        active_counts = np.clip(active_counts, 0, row_count)
        active_mask = row_idx < active_counts[:, None]

    is_move = active_mask & (row_type == 0)
    has_move = np.any(is_move, axis=1)
    first_move = np.where(has_move, np.argmax(is_move, axis=1), row_count).astype(
        np.int32, copy=False
    )
    pre_mask = active_mask & (row_idx < first_move[:, None])
    post_mask = active_mask & (row_idx > first_move[:, None])

    is_binary = active_mask & (
        (row_type == RULE_TYPE_COMBINE) | (row_type == RULE_TYPE_DISTRACTOR_COMBINE)
    )
    is_ternary = active_mask & (row_type == RULE_TYPE_TERNARY_COMBINE)
    is_transform = active_mask & (row_type == RULE_TYPE_TRANSFORM)

    return np.stack(
        [
            np.full((rules.shape[0],), row_count, dtype=np.int32),
            np.sum(is_binary & pre_mask, axis=1, dtype=np.int32),
            np.sum(is_ternary & pre_mask, axis=1, dtype=np.int32),
            np.sum(is_transform & pre_mask, axis=1, dtype=np.int32),
            np.sum(is_binary & post_mask, axis=1, dtype=np.int32),
            np.sum(is_ternary & post_mask, axis=1, dtype=np.int32),
            np.sum(is_transform & post_mask, axis=1, dtype=np.int32),
        ],
        axis=1,
    ).astype(np.int32, copy=False)


def global_rule_program_bounds_from_signatures(
    execution_signatures_host: np.ndarray,
) -> dict[str, int]:
    signatures = np.asarray(execution_signatures_host, dtype=np.int32)
    if signatures.ndim != 2 or signatures.shape[0] == 0:
        return {}
    pre_program_counts = np.sum(signatures[:, 1:4], axis=1, dtype=np.int32)
    post_program_counts = np.sum(signatures[:, 4:7], axis=1, dtype=np.int32)
    pre_combine_counts = signatures[:, 1] + signatures[:, 2]
    post_combine_counts = signatures[:, 4] + signatures[:, 5]
    pre_transform_counts = signatures[:, 3]
    post_transform_counts = signatures[:, 6]
    transform_counts = signatures[:, 3] + signatures[:, 6]
    bounds = {
        "pre_move_program_size": int(np.max(pre_program_counts)),
        "post_move_program_size": int(np.max(post_program_counts)),
        "transform_program_size": int(np.max(transform_counts)),
    }
    max_pre_combine = int(np.max(pre_combine_counts))
    max_post_combine = int(np.max(post_combine_counts))
    max_pre_transform = int(np.max(pre_transform_counts))
    max_post_transform = int(np.max(post_transform_counts))
    max_transform = int(np.max(transform_counts))
    if max_pre_combine < DEFAULT_MAX_COMBINE_CASCADES:
        bounds["pre_move_combine_steps"] = max_pre_combine
    if max_post_combine < DEFAULT_MAX_COMBINE_CASCADES:
        bounds["post_move_combine_steps"] = max_post_combine
    if max_pre_transform < DEFAULT_MAX_TRANSFORM_APPLIES:
        bounds["pre_move_transform_steps"] = max_pre_transform
    if max_post_transform < DEFAULT_MAX_TRANSFORM_APPLIES:
        bounds["post_move_transform_steps"] = max_post_transform
    if max_transform < DEFAULT_MAX_TRANSFORM_APPLIES:
        bounds["all_transform_steps"] = max_transform
    return bounds


def validate_collect_tile_invariant(
    packed_host: np.ndarray,
    *,
    dataset_dir: str | None = None,
    dataset_file: str | None = None,
    max_examples: int = 5,
) -> None:
    """Fail if collect rows encode a tile inconsistent with their item id.

    The codec should decode the dataset faithfully. Dataset generation is
    responsible for ensuring collect rows use a valid collectible tile for the
    semantic item/color target.
    """
    if packed_host.size == 0:
        return

    decoded = np.asarray(
        jax.device_get(unpack_rules_uint32_jit(jnp.asarray(packed_host))),
        dtype=np.int32,
    )
    if decoded.ndim != 3 or decoded.shape[-1] < 4:
        raise ValueError(
            "Packed ruleset dataset decoded to an unexpected shape: "
            f"{decoded.shape}."
        )

    collect_mask = decoded[:, :, 0] == int(RULE_TYPE_COLLECT)
    if not np.any(collect_mask):
        return

    tiles = decoded[:, :, 1]
    items = decoded[:, :, 2]
    colors = decoded[:, :, 3]

    tile_to_item = np.asarray(jax.device_get(TILE_TO_ITEM), dtype=np.int32)
    item_to_tile = np.asarray(jax.device_get(ITEM_TO_TILE), dtype=np.int32)

    item_valid = (items >= 0) & (items < int(NUM_RULESET_ITEMS))
    color_valid = (colors >= 0) & (colors < int(NUM_COLORS))
    tile_valid = (tiles >= 0) & (tiles < int(NUM_TILE_TYPES))
    safe_tiles = np.clip(tiles, 0, int(NUM_TILE_TYPES) - 1)
    tile_matches_item = tile_to_item[safe_tiles] == np.clip(
        items,
        0,
        int(NUM_RULESET_ITEMS) - 1,
    )
    invalid = collect_mask & (
        (~item_valid) | (~color_valid) | (~tile_valid) | (~tile_matches_item)
    )
    if not np.any(invalid):
        return

    dataset_label = (
        str((Path(dataset_dir) / str(dataset_file)).resolve())
        if dataset_dir is not None and dataset_file is not None
        else "<unknown dataset>"
    )
    rows = np.argwhere(invalid)
    examples: list[str] = []
    for task_idx, row_idx in rows[:max_examples]:
        tile = int(tiles[task_idx, row_idx])
        item = int(items[task_idx, row_idx])
        color = int(colors[task_idx, row_idx])
        expected = (
            int(item_to_tile[item])
            if 0 <= item < int(NUM_RULESET_ITEMS)
            else "invalid-item"
        )
        examples.append(
            f"task={int(task_idx)} row={int(row_idx)} "
            f"tile={tile} item={item} color={color} expected_tile={expected}"
        )

    raise ValueError(
        "Invalid ruleset dataset: collect-row tile/item invariant failed for "
        f"{dataset_label}. The dataset should be regenerated or excluded; "
        "the codec does not repair invalid collect tiles. Examples: "
        + "; ".join(examples)
    )


def validate_packed_ruleset_dataset(
    packed_host: np.ndarray,
    config: dict[str, Any],
    *,
    dataset_dir: str,
    dataset_file: str,
) -> None:
    if not bool(config.get("VALIDATE_RULESET_DATASET", True)):
        return
    validate_collect_tile_invariant(
        packed_host,
        dataset_dir=dataset_dir,
        dataset_file=dataset_file,
    )


def merge_reset_params(
    reset_params: dict[str, Any],
    task_reset_metadata: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(reset_params)
    merged.update(task_reset_metadata)
    return merged


def strip_single_reset_batch_dim(tree: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda x: x[0]
        if hasattr(x, "shape") and x.shape[:1] == (1,)
        else x,
        tree,
    )
