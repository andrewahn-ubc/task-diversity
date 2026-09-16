#!/usr/bin/env python3
"""Verify that multiple layout banks are strictly disjoint.

Disjointness is checked on exact obstacle-mask equality.
No probabilistic hashing is used for final identity:
we compare packed obstacle-mask bytes directly.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np


# Keep script JAX-free.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--banks",
        nargs="+",
        required=True,
        help="Layout-bank .npz files to compare.",
    )
    p.add_argument(
        "--allow-within-bank-duplicates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If false, duplicate layouts inside a single bank are treated as failure.",
    )
    p.add_argument(
        "--report-json",
        default="",
        help="Optional path to save detailed JSON report.",
    )
    return p.parse_args()


def _load_masks(bank_path: Path) -> np.ndarray:
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
            raise ValueError(
                f"{bank_path} missing obstacle layout keys "
                "(need obstacle_mask/obstacle_map/map_array)."
            )
    if masks.ndim != 3:
        raise ValueError(f"{bank_path}: expected [L,H,W], got {masks.shape}")
    return masks


def _pack_masks(masks: np.ndarray) -> np.ndarray:
    flat = masks.reshape(masks.shape[0], -1).astype(np.uint8, copy=False)
    return np.packbits(flat, axis=1, bitorder="little")


def main() -> None:
    args = parse_args()
    bank_paths = [Path(p).expanduser().resolve() for p in args.banks]
    for p in bank_paths:
        if not p.exists():
            raise FileNotFoundError(f"Bank not found: {p}")

    masks_by_bank: list[np.ndarray] = []
    grid_shape: tuple[int, int] | None = None
    for p in bank_paths:
        masks = _load_masks(p)
        shape = (int(masks.shape[1]), int(masks.shape[2]))
        if grid_shape is None:
            grid_shape = shape
        elif grid_shape != shape:
            raise ValueError(
                "All banks must share the same grid shape for strict comparison. "
                f"Expected {grid_shape}, got {shape} at {p}"
            )
        masks_by_bank.append(masks)

    assert grid_shape is not None
    n_banks = len(bank_paths)

    # Exact signature map: packed-bytes -> list[(bank_idx, layout_idx)].
    signature_hits: dict[bytes, list[tuple[int, int]]] = {}

    total_layouts = 0
    for bank_idx, masks in enumerate(masks_by_bank):
        packed = _pack_masks(masks)
        total_layouts += int(masks.shape[0])
        for row_idx in range(packed.shape[0]):
            sig = packed[row_idx].tobytes()
            signature_hits.setdefault(sig, []).append((bank_idx, row_idx))

    # Count overlaps.
    pair_overlap_counts = np.zeros((n_banks, n_banks), dtype=np.int64)
    within_dup_counts = np.zeros((n_banks,), dtype=np.int64)
    overlap_examples: list[dict[str, object]] = []

    for sig, hits in signature_hits.items():
        by_bank: dict[int, list[int]] = {}
        for bank_idx, layout_idx in hits:
            by_bank.setdefault(bank_idx, []).append(layout_idx)

        for bank_idx, indices in by_bank.items():
            if len(indices) > 1:
                within_dup_counts[bank_idx] += len(indices) - 1

        present = sorted(by_bank.keys())
        if len(present) > 1:
            for i, j in itertools.combinations(present, 2):
                pair_overlap_counts[i, j] += 1
                pair_overlap_counts[j, i] += 1
            if len(overlap_examples) < 50:
                overlap_examples.append(
                    {
                        "banks": {
                            str(bank_paths[b]): by_bank[b]
                            for b in present
                        }
                    }
                )

    cross_overlap_total = int(np.triu(pair_overlap_counts, k=1).sum())
    within_dup_total = int(within_dup_counts.sum())
    unique_layouts = int(len(signature_hits))

    print("=" * 72)
    print("Layout Bank Disjointness Report")
    print("=" * 72)
    print(f"Grid shape: {grid_shape[0]}x{grid_shape[1]}")
    print(f"Banks: {n_banks}")
    for i, p in enumerate(bank_paths):
        print(f"  [{i}] {p}  (layouts={masks_by_bank[i].shape[0]})")
    print(f"Total layouts scanned: {total_layouts}")
    print(f"Unique obstacle layouts: {unique_layouts}")
    print(f"Cross-bank overlap count: {cross_overlap_total}")
    print(f"Within-bank duplicate count: {within_dup_total}")
    print("-" * 72)
    print("Pairwise overlap matrix (unique overlapping layouts):")
    header = "      " + " ".join([f"{i:>8d}" for i in range(n_banks)])
    print(header)
    for i in range(n_banks):
        row = " ".join([f"{int(pair_overlap_counts[i, j]):>8d}" for j in range(n_banks)])
        print(f"{i:>4d}  {row}")

    report = {
        "banks": [str(p) for p in bank_paths],
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "layouts_per_bank": [int(m.shape[0]) for m in masks_by_bank],
        "total_layouts": total_layouts,
        "unique_layouts": unique_layouts,
        "cross_overlap_total": cross_overlap_total,
        "within_dup_total": within_dup_total,
        "within_dup_per_bank": within_dup_counts.tolist(),
        "pair_overlap_counts": pair_overlap_counts.tolist(),
        "overlap_examples": overlap_examples,
    }

    if args.report_json:
        out = Path(args.report_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[write] {out}")

    failed = False
    if cross_overlap_total > 0:
        print("FAIL: cross-bank overlaps found.")
        failed = True
    if (not args.allow_within_bank_duplicates) and within_dup_total > 0:
        print("FAIL: within-bank duplicates found.")
        failed = True

    if not failed:
        print("PASS: strict disjointness verified.")
        return
    sys.exit(1)


if __name__ == "__main__":
    main()
