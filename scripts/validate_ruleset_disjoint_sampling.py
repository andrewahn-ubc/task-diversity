#!/usr/bin/env python3
"""Sample-based uniqueness/disjointness validator for ruleset datasets.

This script:
1) Samples K tasks from a target depth in each dataset.
2) Verifies sampled tasks are unique within each dataset.
3) Verifies sampled tasks are disjoint across datasets.
4) Reports pool overlap from metadata as an additional sanity check.
"""

from __future__ import annotations

import argparse
import bz2
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from banyan_grid.environment.constants import (
    RULE_TYPE_COLLECT,
    RULE_TYPE_COMBINE,
    RULE_TYPE_DISTRACTOR_COMBINE,
    RULE_TYPE_PAD,
    RULE_TYPE_TERNARY_COMBINE,
    RULE_TYPE_TRANSFORM,
)
from banyan_grid.tasks.ruleset_codec import unpack_rules_uint32_np
from banyan_grid.tasks.ruleset_dataset_compact import load_meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Paths to .uint32.npy.bz2 dataset files.",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Task depth bucket to sample from (default: 3).",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of tasks to sample from each dataset (default: 100).",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for sampling.")
    p.add_argument(
        "--report-json",
        default="",
        help="Optional JSON report path.",
    )
    return p.parse_args()


def _meta_path_for_dataset(ds_path: Path) -> Path:
    name = ds_path.name
    if not name.endswith(".uint32.npy.bz2"):
        raise ValueError(f"Expected .uint32.npy.bz2 file, got: {ds_path}")
    base_name = name[: -len(".uint32.npy.bz2")]
    meta_path = ds_path.with_name(base_name + ".uint32_meta.json")
    if not meta_path.exists():
        alt_meta_path = ds_path.with_name(base_name + "_meta.json")
        if alt_meta_path.exists():
            return alt_meta_path
    return meta_path


def _load_dataset(ds_path: Path) -> np.ndarray:
    with bz2.BZ2File(ds_path, "rb") as f:
        arr = np.load(f, allow_pickle=False)
    if arr.dtype != np.uint32 or arr.ndim != 2:
        raise ValueError(f"{ds_path}: expected uint32 array [N, R], got {arr.dtype} {arr.shape}")
    return arr


def _depth_slice(meta: dict, depth: int) -> tuple[int, int]:
    structure = meta.get("structure", {})
    if not isinstance(structure, dict):
        raise ValueError("meta['structure'] missing or invalid")
    counts = {}
    for k, v in structure.items():
        counts[int(k)] = int(v)
    if depth not in counts:
        raise ValueError(f"Depth {depth} not present in structure={counts}")
    start = 0
    for d in sorted(counts.keys()):
        if d >= depth:
            break
        start += counts[d]
    end = start + counts[depth]
    return start, end


def _rule_signature_from_packed_rows(rule_rows_u32: np.ndarray) -> tuple:
    """Canonical task signature from packed rule rows, order-invariant."""
    entries = []
    for row in unpack_rules_uint32_np(rule_rows_u32):
        rt = int(row[0])
        if rt == RULE_TYPE_PAD:
            continue
        if rt == 0:
            entries.append((rt,))
            continue
        if rt == RULE_TYPE_COLLECT:
            entries.append((rt, int(row[1]), (int(row[2]), int(row[3]))))
            continue
        if rt not in (
            RULE_TYPE_COMBINE,
            RULE_TYPE_DISTRACTOR_COMBINE,
            RULE_TYPE_TRANSFORM,
            RULE_TYPE_TERNARY_COMBINE,
        ):
            raise ValueError(f"Unsupported decoded rule type: {rt}")

        colors = int(row[5])
        inputs = [(int(row[1]), colors & 0xF)]
        if rt != RULE_TYPE_TRANSFORM:
            inputs.append((int(row[2]), (colors >> 4) & 0xF))
        if rt == RULE_TYPE_TERNARY_COMBINE:
            inputs.append((int(row[4]) >> 1, (colors >> 8) & 0xF))
            cout = (colors >> 12) & 0xF
            adjacent = int(row[4]) & 0x1
        else:
            cout = (colors >> 8) & 0xF
            adjacent = int(row[4])
        out = (int(row[3]), cout)
        entries.append((rt, len(inputs), tuple(sorted(inputs)), out, adjacent))

    return tuple(sorted(entries))


def _sample_signatures(
    arr: np.ndarray, start: int, end: int, sample_size: int, rng: np.random.Generator
) -> tuple[list[int], list[tuple]]:
    n = end - start
    if n < sample_size:
        raise ValueError(f"Depth bucket has only {n} tasks, need sample_size={sample_size}")
    rel = rng.choice(n, size=sample_size, replace=False)
    idx = [start + int(i) for i in rel]
    sigs = [_rule_signature_from_packed_rows(arr[i]) for i in idx]
    return idx, sigs


def main() -> None:
    args = parse_args()
    ds_paths = [Path(p).expanduser().resolve() for p in args.datasets]
    for p in ds_paths:
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")

    rng = np.random.default_rng(args.seed)
    results: list[dict[str, Any]] = []
    sampled_sets = []

    for ds in ds_paths:
        meta_path = _meta_path_for_dataset(ds)
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found for dataset: {meta_path}")

        arr = _load_dataset(ds)
        meta = load_meta(str(ds.parent), ds.name)
        start, end = _depth_slice(meta, args.depth)
        d_count = end - start
        idx, sigs = _sample_signatures(arr, start, end, args.sample_size, rng)
        unique_count = len(set(sigs))
        sampled_sets.append(set(sigs))
        pool = meta.get("pool_uids")
        if pool is not None and not isinstance(pool, list):
            raise ValueError(f"{meta_path}: pool_uids must be a list or null.")
        pool_uids = [int(x) for x in pool] if pool is not None else None
        results.append(
            {
                "dataset": str(ds),
                "meta": str(meta_path),
                "shape": [int(arr.shape[0]), int(arr.shape[1])],
                "depth": int(args.depth),
                "depth_count": int(d_count),
                "sample_size": int(args.sample_size),
                "sample_indices": idx,
                "sample_unique_count": int(unique_count),
                "sample_has_duplicates": bool(unique_count != args.sample_size),
                "pool_uids_count": len(pool_uids) if pool_uids is not None else None,
                "pool_uids": pool_uids,
            }
        )

    pairwise_overlaps: list[dict[str, Any]] = []
    for i in range(len(sampled_sets)):
        for j in range(i + 1, len(sampled_sets)):
            overlap = sampled_sets[i].intersection(sampled_sets[j])
            pool_i, pool_j = results[i]["pool_uids"], results[j]["pool_uids"]
            pool_overlap = (
                len(set(pool_i).intersection(pool_j))
                if pool_i is not None and pool_j is not None
                else None
            )
            pairwise_overlaps.append(
                {
                    "i": i,
                    "j": j,
                    "dataset_i": results[i]["dataset"],
                    "dataset_j": results[j]["dataset"],
                    "sample_signature_overlap_count": int(len(overlap)),
                    "pool_uid_overlap_count": pool_overlap,
                }
            )

    within_ok = all(not r["sample_has_duplicates"] for r in results)
    cross_sample_ok = all(x["sample_signature_overlap_count"] == 0 for x in pairwise_overlaps)
    pool_counts = [x["pool_uid_overlap_count"] for x in pairwise_overlaps]
    cross_pool_ok = (
        False
        if any(count is not None and count > 0 for count in pool_counts)
        else None
        if None in pool_counts
        else True
    )
    all_ok = within_ok and cross_sample_ok and cross_pool_ok is True

    report = {
        "datasets": [str(p) for p in ds_paths],
        "depth": int(args.depth),
        "sample_size": int(args.sample_size),
        "seed": int(args.seed),
        "results": results,
        "pairwise": pairwise_overlaps,
        "within_ok": bool(within_ok),
        "cross_sample_ok": bool(cross_sample_ok),
        "cross_pool_ok": cross_pool_ok,
        "all_ok": bool(all_ok),
    }

    print("=" * 72)
    print("Ruleset Sampling Disjointness Report")
    print("=" * 72)
    print(f"Depth sampled: {args.depth}")
    print(f"Sample size: {args.sample_size}")
    print(f"Seed: {args.seed}")
    print("-" * 72)
    for i, r in enumerate(results):
        print(f"[{i}] {r['dataset']}")
        print(
            f"    depth_count={r['depth_count']} sample_unique={r['sample_unique_count']}/{r['sample_size']}"
        )
    print("-" * 72)
    for x in pairwise_overlaps:
        pool_overlap = x["pool_uid_overlap_count"]
        print(
            f"[{x['i']},{x['j']}] sample_overlap={x['sample_signature_overlap_count']} "
            f"pool_overlap={pool_overlap if pool_overlap is not None else 'unknown'}"
        )
    print("-" * 72)
    print(f"within_ok={within_ok} cross_sample_ok={cross_sample_ok} cross_pool_ok={cross_pool_ok}")
    if within_ok and cross_sample_ok and cross_pool_ok is None:
        print("INCONCLUSIVE: pool provenance is unknown; disjointness is not established.")
    else:
        print("PASS" if all_ok else "FAIL")

    if args.report_json:
        out = Path(args.report_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[write] {out}")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
