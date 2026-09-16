#!/usr/bin/env python3
"""Build strict disjoint layout pools for 2-round experiments.

Given one or more source layout-bank files, this script:
1) Extracts obstacle-mask layouts.
2) Deduplicates exact layouts globally.
3) Splits into disjoint pools:
   - one R2 pool of size --r2-size
   - multiple R1 pools from --r1-sizes
4) Writes each pool as a standalone .layouts.npz.

Disjointness is defined on exact obstacle-mask equality (bit-identical masks).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# Keep this script JAX-free (works cleanly on login nodes).
# Tile IDs from banyan_grid/environment/constants.py:
TILE_OPEN_FAST = 0
TILE_BLOCK = 3
TILE_MOVEABLE_BLOCK_1 = 14
TILE_MOVEABLE_BLOCK_2 = 15
TILE_MOVEABLE_BLOCK_3 = 16
TILE_DOOR = 18
OBSTACLE_TILES = np.array(
    [
        TILE_BLOCK,
        TILE_DOOR,
        TILE_MOVEABLE_BLOCK_1,
        TILE_MOVEABLE_BLOCK_2,
        TILE_MOVEABLE_BLOCK_3,
    ],
    dtype=np.int32,
)


@dataclass(frozen=True)
class LayoutRecord:
    source_bank_idx: int
    source_layout_idx: int
    mask_packed: bytes
    mask_shape: tuple[int, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--banks",
        nargs="+",
        required=True,
        help="Input layout-bank .npz files (one or more).",
    )
    p.add_argument(
        "--name",
        default="exp3_disjoint",
        help="Output prefix name.",
    )
    p.add_argument(
        "--out-dir",
        default="CUR_LAYOUTS",
        help="Output directory.",
    )
    p.add_argument(
        "--r2-size",
        type=int,
        default=10000,
        help="Number of layouts in the R2 pool.",
    )
    p.add_argument(
        "--r1-sizes",
        default="1,10,100,1000,10000",
        help="Comma-separated R1 pool sizes to generate.",
    )
    p.add_argument(
        "--r1-mode",
        choices=["independent", "nested"],
        default="independent",
        help=(
            "independent: every R1 pool is disjoint from all others and R2; "
            "nested: smaller R1 pools are prefixes of R1 max pool (not disjoint among R1)."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shuffle seed before splitting.",
    )
    return p.parse_args()


def _load_obstacle_masks(bank_path: Path) -> np.ndarray:
    with np.load(bank_path, allow_pickle=False) as bank:
        if "obstacle_mask" in bank:
            masks = np.asarray(bank["obstacle_mask"], dtype=np.bool_)
        elif "obstacle_map" in bank:
            obstacle_map = np.asarray(bank["obstacle_map"], dtype=np.int32)
            masks = np.isin(obstacle_map, OBSTACLE_TILES)
        elif "map_array" in bank:
            map_array = np.asarray(bank["map_array"], dtype=np.int32)
            masks = np.isin(map_array, OBSTACLE_TILES)
        else:
            keys = ", ".join(bank.files)
            raise ValueError(
                f"{bank_path} has no obstacle layout key. "
                f"Expected one of obstacle_mask/obstacle_map/map_array, got: {keys}"
            )
    if masks.ndim != 3:
        raise ValueError(f"{bank_path} masks must be [L,H,W], got shape={masks.shape}")
    return masks


def _pack_masks(masks: np.ndarray) -> np.ndarray:
    flat = masks.reshape(masks.shape[0], -1).astype(np.uint8, copy=False)
    return np.packbits(flat, axis=1, bitorder="little")


def _write_layout_bank(
    out_path: Path,
    records: list[LayoutRecord],
    source_paths: list[str],
    grid_h: int,
    grid_w: int,
) -> None:
    n = len(records)
    packed = np.frombuffer(b"".join(r.mask_packed for r in records), dtype=np.uint8)
    packed_width = len(records[0].mask_packed) if records else (grid_h * grid_w + 7) // 8
    packed = packed.reshape(n, packed_width)
    masks = np.unpackbits(packed, axis=1, bitorder="little")[:, : grid_h * grid_w]
    masks = masks.reshape(n, grid_h, grid_w).astype(np.bool_)

    obstacle_map = np.where(masks, TILE_BLOCK, TILE_OPEN_FAST).astype(np.int32)
    src_bank_idx = np.asarray([r.source_bank_idx for r in records], dtype=np.int32)
    src_layout_idx = np.asarray([r.source_layout_idx for r in records], dtype=np.int32)

    np.savez_compressed(
        out_path,
        obstacle_mask=masks,
        obstacle_map=obstacle_map,
        source_bank_index=src_bank_idx,
        source_layout_index=src_layout_idx,
        source_banks=np.asarray(source_paths, dtype=np.str_),
    )


def _ensure_nonempty_sizes(r1_sizes: Iterable[int], r2_size: int) -> list[int]:
    parsed = sorted({int(x) for x in r1_sizes if int(x) > 0})
    if not parsed:
        raise ValueError("At least one positive R1 size is required.")
    if r2_size <= 0:
        raise ValueError("--r2-size must be positive.")
    return parsed


def main() -> None:
    args = parse_args()
    bank_paths = [Path(p).expanduser().resolve() for p in args.banks]
    for p in bank_paths:
        if not p.exists():
            raise FileNotFoundError(f"Bank not found: {p}")

    r1_sizes = _ensure_nonempty_sizes(
        [x.strip() for x in args.r1_sizes.split(",") if x.strip()],
        args.r2_size,
    )

    print("=" * 72)
    print("Building Disjoint Layout Pools")
    print("=" * 72)
    print(f"Input banks: {len(bank_paths)}")
    for p in bank_paths:
        print(f"  - {p}")
    print(f"R2 size: {args.r2_size}")
    print(f"R1 sizes: {r1_sizes}")
    print(f"R1 mode: {args.r1_mode}")
    print(f"Seed: {args.seed}")
    print("=" * 72)

    dedup: dict[bytes, LayoutRecord] = {}
    total_seen = 0
    grid_shape: tuple[int, int] | None = None

    for bank_idx, bank_path in enumerate(bank_paths):
        masks = _load_obstacle_masks(bank_path)
        if grid_shape is None:
            grid_shape = (int(masks.shape[1]), int(masks.shape[2]))
        elif grid_shape != (int(masks.shape[1]), int(masks.shape[2])):
            raise ValueError(
                "All banks must have same grid shape. "
                f"Expected {grid_shape}, got {(masks.shape[1], masks.shape[2])} at {bank_path}"
            )

        packed = _pack_masks(masks)
        total_seen += int(masks.shape[0])
        added = 0
        for i in range(masks.shape[0]):
            key = packed[i].tobytes()
            if key in dedup:
                continue
            dedup[key] = LayoutRecord(
                source_bank_idx=bank_idx,
                source_layout_idx=i,
                mask_packed=key,
                mask_shape=grid_shape,
            )
            added += 1
        print(
            f"[scan] {bank_path.name}: total={masks.shape[0]} added_unique={added} "
            f"running_unique={len(dedup)}"
        )

    if grid_shape is None:
        raise RuntimeError("No layouts loaded.")

    unique_records = list(dedup.values())
    rng = np.random.default_rng(int(args.seed))
    perm = rng.permutation(len(unique_records))
    unique_records = [unique_records[i] for i in perm]

    if args.r1_mode == "independent":
        total_needed = int(args.r2_size + sum(r1_sizes))
    else:
        total_needed = int(args.r2_size + max(r1_sizes))

    if len(unique_records) < total_needed:
        raise RuntimeError(
            "Not enough unique layouts for requested disjoint split: "
            f"need {total_needed}, found {len(unique_records)} "
            f"(from {total_seen} total layouts across source banks)."
        )

    assignments: dict[str, list[LayoutRecord]] = {}
    cursor = 0

    r2_key = f"r2_n{args.r2_size}"
    assignments[r2_key] = unique_records[cursor : cursor + args.r2_size]
    cursor += args.r2_size

    if args.r1_mode == "independent":
        for n in sorted(r1_sizes, reverse=True):
            key = f"r1_n{n}"
            assignments[key] = unique_records[cursor : cursor + n]
            cursor += n
    else:
        max_n = max(r1_sizes)
        base = unique_records[cursor : cursor + max_n]
        for n in r1_sizes:
            assignments[f"r1_n{n}"] = base[:n]

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_paths = [str(p) for p in bank_paths]
    grid_h, grid_w = grid_shape

    manifest = {
        "name": args.name,
        "source_banks": source_paths,
        "grid_shape": [grid_h, grid_w],
        "seed": int(args.seed),
        "r2_size": int(args.r2_size),
        "r1_sizes": [int(x) for x in r1_sizes],
        "r1_mode": args.r1_mode,
        "total_source_layouts": int(total_seen),
        "total_unique_layouts": int(len(dedup)),
        "total_needed_layouts": int(total_needed),
        "outputs": {},
    }

    for key, recs in sorted(assignments.items()):
        out_path = out_dir / f"{args.name}_{key}.layouts.npz"
        _write_layout_bank(out_path, recs, source_paths, grid_h, grid_w)
        manifest["outputs"][key] = {
            "path": str(out_path),
            "count": int(len(recs)),
        }
        print(f"[write] {key}: {len(recs)} -> {out_path}")

    manifest_path = out_dir / f"{args.name}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[write] manifest: {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()
