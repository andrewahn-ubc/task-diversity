"""Generate a diverse layout bank in JAX from compact Banyan ruleset datasets.

This script is intentionally standalone:
- It reads compact rulesets (.uint32.npy.bz2 + _meta.json)
- Generates item-valid base maps from rulesets
- Applies fast JAX block-layout mutations
- Filters for reachability/solvability constraints
- Saves a reusable layout-bank artifact (.npz + _meta.json)
- Logs preview layouts (default: 20) to Weights & Biases
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import wandb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from banyan_grid.utils.banyan import get_map
from banyan_grid.environment.constants import (
    COLLECTIBLE_TILES_ARRAY,
    NUM_COLORS,
    NUM_TILE_TYPES,
    WALKABLE_MASK,
    Colors,
    TileType,
)
from banyan_grid.tasks.ruleset_codec import unpack_rules_uint32
from banyan_grid.tasks.ruleset_dataset_compact import (
    device_put_packed,
    load_meta,
    load_packed_u32_bz2,
)


OPEN_TILES = jnp.array(
    [TileType.OPEN_FAST, TileType.OPEN_MEDIUM], dtype=jnp.int32
)
SPAWN_CELLS = jnp.array([[0, 0], [1, 0]], dtype=jnp.int32)
OBSTACLE_TILES = jnp.array([TileType.BLOCK], dtype=jnp.int32)
COLLECTIBLE_LUT = (
    jnp.zeros((NUM_TILE_TYPES,), dtype=jnp.bool_).at[COLLECTIBLE_TILES_ARRAY].set(True)
)
OBSTACLE_LUT = jnp.zeros((NUM_TILE_TYPES,), dtype=jnp.bool_).at[OBSTACLE_TILES].set(True)


def _resolve_dataset_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.suffixes[-3:] != [".uint32", ".npy", ".bz2"] and path.suffixes[-2:] == [
        ".npy",
        ".bz2",
    ]:
        if path.name.endswith(".npy.bz2"):
            path = path.with_name(path.name.replace(".npy.bz2", ".uint32.npy.bz2"))
    if path.suffixes[-3:] != [".uint32", ".npy", ".bz2"]:
        if path.name.endswith(".uint32"):
            path = path.with_name(path.name + ".npy.bz2")
        else:
            path = path.with_name(path.name + ".uint32.npy.bz2")
    return path


def _protected_spawn_mask(grid_size: int) -> jnp.ndarray:
    """Protect spawn neighborhood so mutation won't trap agents immediately."""
    mask = np.zeros((grid_size, grid_size), dtype=np.bool_)
    protected = [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (2, 0),
        (2, 1),
        (0, 2),
        (1, 2),
    ]
    for y, x in protected:
        if 0 <= y < grid_size and 0 <= x < grid_size:
            mask[y, x] = True
    return jnp.asarray(mask)


def _flood_fill_reachable(walkable: jax.Array, seeds: jax.Array) -> jax.Array:
    """4-neighbor flood fill using fixed iterations (JIT-friendly)."""
    reach0 = seeds & walkable

    def _step(_i: int, reach: jax.Array) -> jax.Array:
        up = jnp.pad(reach[1:, :], ((0, 1), (0, 0)))
        down = jnp.pad(reach[:-1, :], ((1, 0), (0, 0)))
        left = jnp.pad(reach[:, 1:], ((0, 0), (0, 1)))
        right = jnp.pad(reach[:, :-1], ((0, 0), (1, 0)))
        nbr = reach | up | down | left | right
        return nbr & walkable

    iters = int(walkable.shape[0] * walkable.shape[1])
    return jax.lax.fori_loop(0, iters, _step, reach0)


def _obstacle_mask_key_bytes_np(map_array: np.ndarray) -> bytes:
    """Exact packed key for the obstacle mask of one layout."""
    mask = np.isin(map_array, np.asarray(OBSTACLE_TILES, dtype=np.int32))
    packed = np.packbits(mask.reshape(-1).astype(np.uint8, copy=False), bitorder="little")
    return packed.tobytes()


def _obstacle_layout_id_np(map_array: np.ndarray) -> np.uint64:
    """Stable obstacle-layout id for metadata/debugging.

    For <=64 cells this is an exact bit-packed integer. For larger grids we fall back
    to a CRC32 of the packed obstacle mask; dedupe still uses the exact byte key.
    """
    packed_bytes = _obstacle_mask_key_bytes_np(map_array)
    if len(packed_bytes) <= 8:
        return np.uint64(int.from_bytes(packed_bytes.ljust(8, b"\x00"), "little"))
    return np.uint64(zlib.crc32(packed_bytes))


def _sample_ruleset_indices_uniform(
    key: jax.Array, n: int, num_rulesets: int
) -> jax.Array:
    return jax.random.randint(key, (n,), 0, num_rulesets)


def _build_depth_sampler(meta: dict[str, Any], num_rulesets: int):
    structure = meta.get("structure", None)
    if not isinstance(structure, dict):
        return None

    depths_sorted = sorted(int(d) for d in structure.keys())
    if not depths_sorted:
        return None

    starts = []
    counts = []
    cursor = 0
    for d in depths_sorted:
        c = int(structure[str(d)])
        starts.append(cursor)
        counts.append(c)
        cursor += c
    if cursor != num_rulesets:
        return None

    starts_arr = jnp.asarray(starts, dtype=jnp.int32)
    counts_arr = jnp.asarray(counts, dtype=jnp.int32)
    depths_arr = np.asarray(depths_sorted, dtype=np.int32)

    def _sample(key: jax.Array, n: int) -> jax.Array:
        k1, k2 = jax.random.split(key)
        num_depths = starts_arr.shape[0]
        depth_idx = jax.random.randint(k1, (n,), 0, num_depths)
        max_offsets = counts_arr[depth_idx]
        raw = jax.random.randint(k2, (n,), 0, jnp.max(counts_arr))
        offsets = raw % max_offsets
        return starts_arr[depth_idx] + offsets

    return _sample, starts_arr, counts_arr, depths_arr


def _depth_from_indices_np(
    indices: np.ndarray,
    starts_arr: np.ndarray | None,
    counts_arr: np.ndarray | None,
    depths_arr: np.ndarray | None,
) -> np.ndarray:
    if starts_arr is None or counts_arr is None or depths_arr is None:
        return np.full(indices.shape[0], -1, dtype=np.int32)
    ends = starts_arr + counts_arr
    depth_ids = np.searchsorted(ends, indices, side="right")
    depth_ids = np.clip(depth_ids, 0, len(depths_arr) - 1)
    return depths_arr[depth_ids]


def _build_renderer():
    tile_palette = np.zeros((NUM_TILE_TYPES, 3), dtype=np.uint8)
    tile_palette[int(TileType.OPEN_FAST)] = np.array([22, 26, 29], dtype=np.uint8)
    tile_palette[int(TileType.OPEN_MEDIUM)] = np.array([36, 41, 46], dtype=np.uint8)
    tile_palette[int(TileType.BLOCK)] = np.array([115, 115, 115], dtype=np.uint8)
    tile_palette[int(TileType.GOAL)] = np.array([40, 163, 84], dtype=np.uint8)

    color_palette = np.array(
        [
            [255, 0, 0],  # red
            [27, 174, 96],  # green
            [0, 0, 255],  # blue
            [112, 39, 195],  # purple
            [241, 196, 15],  # yellow
            [100, 100, 100],  # grey
            [255, 255, 255],  # white
            [165, 42, 42],  # brown
            [255, 20, 147],  # pink
            [255, 165, 0],  # orange
            [0, 206, 209],  # cyan
            [50, 205, 50],  # lime
            [0, 0, 0],  # black (sentinel)
        ],
        dtype=np.uint8,
    )

    collectible_mask = np.zeros((NUM_TILE_TYPES,), dtype=np.bool_)
    collectible_mask[np.asarray(COLLECTIBLE_TILES_ARRAY)] = True

    def _render(map_array: np.ndarray, color_map: np.ndarray, scale: int) -> np.ndarray:
        img = tile_palette[map_array]
        paint_by_color = collectible_mask[map_array]
        img = np.where(
            paint_by_color[..., None],
            color_palette[np.clip(color_map, 0, NUM_COLORS - 1)],
            img,
        )
        if scale > 1:
            img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
            img[::scale, :, :] = 18
            img[:, ::scale, :] = 18
        return img

    return _render


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="Path to .uint32.npy.bz2 dataset.")
    p.add_argument("--name", default="layout_bank", help="Output artifact prefix.")
    p.add_argument("--out-dir", default="CUR_LAYOUTS", help="Output directory.")
    p.add_argument("--grid-size", type=int, default=6)
    p.add_argument("--num-layouts", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--oversample-rounds", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--balanced-depth-sampling", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-open-ratio", type=float, default=0.30)
    p.add_argument("--min-reachable-ratio", type=float, default=0.55)
    p.add_argument(
        "--min-extra-block-frac",
        type=float,
        default=0.06,
        help="Min extra block fraction over total grid cells.",
    )
    p.add_argument(
        "--max-extra-block-frac",
        type=float,
        default=0.28,
        help="Max extra block fraction over total grid cells.",
    )
    p.add_argument("--preview-count", type=int, default=20)
    p.add_argument("--preview-scale", type=int, default=32)
    p.add_argument(
        "--bank-mode",
        type=str,
        default="obstacle_only",
        choices=["obstacle_only", "ruleset_mutation"],
        help=(
            "obstacle_only (recommended scientific mode): generate obstacle layouts "
            "independent of ruleset item placement. "
            "ruleset_mutation: mutate ruleset-derived maps (legacy behavior)."
        ),
    )
    p.add_argument(
        "--dedupe-mode",
        type=str,
        default="obstacle",
        choices=["none", "obstacle", "obstacle_ruleset", "full"],
        help=(
            "How to dedupe accepted layouts: "
            "none=accept all valid, "
            "obstacle=dedupe by obstacle mask only, "
            "obstacle_ruleset=dedupe by (obstacle mask, ruleset_idx), "
            "full=dedupe by full (map,color) bytes."
        ),
    )
    p.add_argument("--wandb-project", default="banyan-layout-bank")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = _resolve_dataset_path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    dataset_dir = str(dataset_path.parent)
    dataset_file = dataset_path.name

    meta = load_meta(dataset_dir, dataset_file)
    packed_host = load_packed_u32_bz2(dataset_dir, dataset_file)
    packed_buffer = device_put_packed(packed_host)
    num_rulesets, _ = packed_host.shape

    depth_sampler = _build_depth_sampler(meta, num_rulesets)
    sample_depth_fn = None
    starts_arr_np: np.ndarray | None = None
    counts_arr_np: np.ndarray | None = None
    depths_arr_np: np.ndarray | None = None
    if depth_sampler is not None:
        sample_depth_fn, starts_arr, counts_arr, depths_arr = depth_sampler
        starts_arr_np = np.asarray(starts_arr)
        counts_arr_np = np.asarray(counts_arr)
        depths_arr_np = np.asarray(depths_arr)

    cell_count = args.grid_size * args.grid_size
    min_extra_blocks = int(np.floor(args.min_extra_block_frac * cell_count))
    max_extra_blocks = int(np.floor(args.max_extra_block_frac * cell_count))
    max_extra_blocks = max(1, min(max_extra_blocks, cell_count - 1))
    min_extra_blocks = max(0, min(min_extra_blocks, max_extra_blocks))

    protected_mask = _protected_spawn_mask(args.grid_size)
    protected_flat = protected_mask.reshape(-1)

    @jax.jit
    def _mutate_and_validate(
        key: jax.Array, base_map: jax.Array, base_color: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        k_count, k_pick = jax.random.split(key)
        flat_map = base_map.reshape(-1)
        flat_color = base_color.reshape(-1)

        open_mask = jnp.isin(flat_map, OPEN_TILES)
        mutable_mask = open_mask & (~protected_flat)
        mutable_count = jnp.sum(mutable_mask.astype(jnp.int32))

        k_hi = jnp.minimum(jnp.int32(max_extra_blocks), mutable_count)
        k_lo = jnp.minimum(jnp.int32(min_extra_blocks), k_hi)

        k_draw = jax.lax.cond(
            k_hi > 0,
            lambda _: jax.random.randint(k_count, (), minval=k_lo, maxval=k_hi + 1),
            lambda _: jnp.int32(0),
            operand=None,
        )

        scores = jax.random.uniform(k_pick, shape=(cell_count,), minval=0.0, maxval=1.0)
        scores = jnp.where(mutable_mask, scores, jnp.full_like(scores, -1.0))
        top_idx = jax.lax.top_k(scores, max_extra_blocks)[1]
        select_mask = jnp.arange(max_extra_blocks, dtype=jnp.int32) < k_draw

        new_vals_map = jnp.where(
            select_mask, jnp.int32(TileType.BLOCK), flat_map[top_idx]
        )
        new_vals_color = jnp.where(
            select_mask, jnp.int32(Colors.BLACK), flat_color[top_idx]
        )
        flat_map = flat_map.at[top_idx].set(new_vals_map)
        flat_color = flat_color.at[top_idx].set(new_vals_color)

        map_out = flat_map.reshape(args.grid_size, args.grid_size)
        color_out = flat_color.reshape(args.grid_size, args.grid_size)

        walkable = WALKABLE_MASK[map_out]
        seeds = jnp.zeros_like(walkable, dtype=jnp.bool_)
        for rc in np.asarray(SPAWN_CELLS):
            y, x = int(rc[0]), int(rc[1])
            if y < args.grid_size and x < args.grid_size:
                seeds = seeds.at[y, x].set(True)

        reachable = _flood_fill_reachable(walkable, seeds)
        walkable_count = jnp.sum(walkable.astype(jnp.float32))
        reachable_count = jnp.sum(reachable.astype(jnp.float32))
        open_ratio = jnp.mean(walkable.astype(jnp.float32))
        reachable_ratio = jnp.where(
            walkable_count > 0.0, reachable_count / walkable_count, 0.0
        )

        collectible_mask = COLLECTIBLE_LUT[map_out]
        unreachable_collectibles = jnp.sum(
            (collectible_mask & (~reachable)).astype(jnp.int32)
        )

        spawn_ok = jnp.all(jnp.where(seeds, walkable, True))
        valid = (
            spawn_ok
            & (open_ratio >= jnp.float32(args.min_open_ratio))
            & (reachable_ratio >= jnp.float32(args.min_reachable_ratio))
            & (unreachable_collectibles == 0)
        )
        block_count = jnp.sum((map_out == TileType.BLOCK).astype(jnp.int32))
        return map_out, color_out, valid, open_ratio, reachable_ratio, block_count

    @jax.jit
    def _generate_batch_ruleset(
        key: jax.Array,
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        k_idx, k_map, k_mut, k_next = jax.random.split(key, 4)
        if args.balanced_depth_sampling and sample_depth_fn is not None:
            rs_idx = sample_depth_fn(k_idx, args.batch_size)
        else:
            rs_idx = _sample_ruleset_indices_uniform(k_idx, args.batch_size, num_rulesets)

        rulesets = unpack_rules_uint32(packed_buffer[rs_idx])
        map_keys = jax.random.split(k_map, args.batch_size)
        mut_keys = jax.random.split(k_mut, args.batch_size)

        base_maps, base_colors = jax.vmap(
            lambda kk, rr: get_map(
                kk,
                args.grid_size,
                rr,
            )
        )(map_keys, rulesets)

        maps, colors, valid, open_ratio, reach_ratio, block_count = jax.vmap(
            _mutate_and_validate
        )(mut_keys, base_maps, base_colors)
        return k_next, rs_idx, maps, colors, valid, open_ratio, reach_ratio, block_count

    @jax.jit
    def _generate_batch_obstacle_only(
        key: jax.Array,
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        k_mut, k_next = jax.random.split(key, 2)
        mut_keys = jax.random.split(k_mut, args.batch_size)
        base_map = jnp.full(
            (args.batch_size, args.grid_size, args.grid_size),
            jnp.int32(TileType.OPEN_FAST),
            dtype=jnp.int32,
        )
        base_color = jnp.full(
            (args.batch_size, args.grid_size, args.grid_size),
            jnp.int32(Colors.BLACK),
            dtype=jnp.int32,
        )
        rs_idx = jnp.full((args.batch_size,), -1, dtype=jnp.int32)
        maps, colors, valid, open_ratio, reach_ratio, block_count = jax.vmap(
            _mutate_and_validate
        )(mut_keys, base_map, base_color)
        return k_next, rs_idx, maps, colors, valid, open_ratio, reach_ratio, block_count

    print("=" * 68)
    print("Layout Bank Generation (JAX)")
    print("=" * 68)
    print(f"Dataset: {dataset_path}")
    print(f"Num rulesets: {num_rulesets}")
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"Target layouts: {args.num_layouts}")
    print(f"Batch size: {args.batch_size}")
    print(f"Extra blocks per layout: [{min_extra_blocks}, {max_extra_blocks}]")
    print(f"Bank mode: {args.bank_mode}")
    print(f"Dedupe mode: {args.dedupe_mode}")
    print(
        f"Validation: min_open_ratio={args.min_open_ratio}, "
        f"min_reachable_ratio={args.min_reachable_ratio}"
    )
    print("=" * 68)

    rng = jax.random.PRNGKey(args.seed)
    accepted: list[dict[str, Any]] = []
    seen_keys: set[Any] = set()

    for round_idx in range(args.oversample_rounds):
        rng, k_round = jax.random.split(rng)
        if args.bank_mode == "ruleset_mutation":
            (
                _,
                rs_idx,
                maps,
                colors,
                valid,
                open_ratio,
                reach_ratio,
                block_count,
            ) = _generate_batch_ruleset(k_round)
        else:
            (
                _,
                rs_idx,
                maps,
                colors,
                valid,
                open_ratio,
                reach_ratio,
                block_count,
            ) = _generate_batch_obstacle_only(k_round)

        rs_idx_np = np.asarray(rs_idx)
        maps_np = np.asarray(maps)
        colors_np = np.asarray(colors)
        valid_np = np.asarray(valid)
        open_ratio_np = np.asarray(open_ratio)
        reach_ratio_np = np.asarray(reach_ratio)
        block_count_np = np.asarray(block_count)
        added = 0
        for i in range(args.batch_size):
            if not bool(valid_np[i]):
                continue
            obstacle_key = _obstacle_mask_key_bytes_np(maps_np[i])
            layout_id = int(_obstacle_layout_id_np(maps_np[i]))
            if args.dedupe_mode == "none":
                key = None
            elif args.dedupe_mode == "obstacle":
                key = ("o", obstacle_key)
            elif args.dedupe_mode == "obstacle_ruleset":
                key = ("or", obstacle_key, int(rs_idx_np[i]))
            else:  # full
                key = ("f", maps_np[i].tobytes(), colors_np[i].tobytes())

            if key is not None:
                if key in seen_keys:
                    continue
                seen_keys.add(key)

            accepted.append(
                {
                    "ruleset_idx": int(rs_idx_np[i]),
                    "map": maps_np[i].astype(np.int32),
                    "color": colors_np[i].astype(np.int32),
                    "valid": True,
                    "open_ratio": float(open_ratio_np[i]),
                    "reachable_ratio": float(reach_ratio_np[i]),
                    "block_count": int(block_count_np[i]),
                    "layout_hash": layout_id,
                }
            )
            added += 1
            if len(accepted) >= args.num_layouts:
                break

        print(
            f"[round {round_idx + 1}/{args.oversample_rounds}] "
            f"added={added} total={len(accepted)}"
        )
        if len(accepted) >= args.num_layouts:
            break

    if len(accepted) < args.num_layouts:
        raise RuntimeError(
            f"Only generated {len(accepted)} unique valid layouts (target {args.num_layouts}). "
            "Increase --oversample-rounds or relax block/validation constraints."
        )

    accepted = accepted[: args.num_layouts]
    ruleset_indices = np.asarray([x["ruleset_idx"] for x in accepted], dtype=np.int32)
    maps_arr = np.stack([x["map"] for x in accepted], axis=0).astype(np.int32)
    colors_arr = np.stack([x["color"] for x in accepted], axis=0).astype(np.int32)
    obstacle_mask_arr = np.isin(maps_arr, np.asarray(OBSTACLE_TILES, dtype=np.int32))
    open_ratio_arr = np.asarray([x["open_ratio"] for x in accepted], dtype=np.float32)
    reach_ratio_arr = np.asarray([x["reachable_ratio"] for x in accepted], dtype=np.float32)
    block_count_arr = np.asarray([x["block_count"] for x in accepted], dtype=np.int32)
    hash_arr = np.asarray([x["layout_hash"] for x in accepted], dtype=np.uint64)

    if args.bank_mode == "ruleset_mutation":
        depth_arr = _depth_from_indices_np(
            ruleset_indices,
            starts_arr_np,
            counts_arr_np,
            depths_arr_np,
        )
    else:
        depth_arr = np.full((ruleset_indices.shape[0],), -1, dtype=np.int32)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bank_file = out_dir / f"{args.name}.layouts.npz"
    meta_file = out_dir / f"{args.name}.layouts_meta.json"

    np.savez_compressed(
        bank_file,
        obstacle_mask=obstacle_mask_arr,
        map_array=maps_arr,
        color_map=colors_arr,
        ruleset_indices=ruleset_indices,
        depth=depth_arr,
        block_count=block_count_arr,
        open_ratio=open_ratio_arr,
        reachable_ratio=reach_ratio_arr,
        layout_hash=hash_arr,
        source_dataset=np.asarray(str(dataset_path), dtype=np.str_),
    )

    output_meta = {
        "name": args.name,
        "source_dataset": str(dataset_path),
        "num_layouts": int(args.num_layouts),
        "grid_size": int(args.grid_size),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "oversample_rounds": int(args.oversample_rounds),
        "balanced_depth_sampling": bool(args.balanced_depth_sampling),
        "bank_mode": str(args.bank_mode),
        "dedupe_mode": str(args.dedupe_mode),
        "min_open_ratio": float(args.min_open_ratio),
        "min_reachable_ratio": float(args.min_reachable_ratio),
        "min_extra_block_frac": float(args.min_extra_block_frac),
        "max_extra_block_frac": float(args.max_extra_block_frac),
        "num_unique_layout_hashes": int(len(np.unique(hash_arr))),
        "block_count_stats": {
            "min": int(block_count_arr.min()),
            "mean": float(block_count_arr.mean()),
            "max": int(block_count_arr.max()),
        },
        "open_ratio_stats": {
            "min": float(open_ratio_arr.min()),
            "mean": float(open_ratio_arr.mean()),
            "max": float(open_ratio_arr.max()),
        },
        "reachable_ratio_stats": {
            "min": float(reach_ratio_arr.min()),
            "mean": float(reach_ratio_arr.mean()),
            "max": float(reach_ratio_arr.max()),
        },
        "depth_counts": {
            str(int(d)): int((depth_arr == d).sum())
            for d in sorted(np.unique(depth_arr))
            if int(d) >= 0
        },
    }
    with meta_file.open("w") as f:
        json.dump(output_meta, f, indent=2)

    print(f"Wrote layout bank: {bank_file}")
    print(f"Wrote layout metadata: {meta_file}")

    if args.wandb_mode != "disabled":
        run_name = args.wandb_run_name or args.name
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            mode=args.wandb_mode,
            config=output_meta,
            tags=["layout-bank", "jax", "banyan"],
        )
        renderer = _build_renderer()
        k = int(min(args.preview_count, args.num_layouts))
        preview_idx = np.linspace(0, args.num_layouts - 1, num=k, dtype=np.int32)
        images = []
        for i in preview_idx:
            img = renderer(maps_arr[i], colors_arr[i], scale=args.preview_scale)
            caption = (
                f"layout={i} depth={int(depth_arr[i])} ruleset_idx={int(ruleset_indices[i])} "
                f"blocks={int(block_count_arr[i])} open={open_ratio_arr[i]:.3f} "
                f"reach={reach_ratio_arr[i]:.3f} hash={int(hash_arr[i])}"
            )
            images.append(wandb.Image(img, caption=caption))

        wandb.log(
            {
                "layout_bank/num_layouts": args.num_layouts,
                "layout_bank/num_unique_hashes": len(np.unique(hash_arr)),
                "layout_bank/block_count_mean": float(block_count_arr.mean()),
                "layout_bank/open_ratio_mean": float(open_ratio_arr.mean()),
                "layout_bank/reachable_ratio_mean": float(reach_ratio_arr.mean()),
                "layout_bank/previews": images,
            }
        )
        run.finish()


if __name__ == "__main__":
    main()
