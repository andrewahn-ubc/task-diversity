#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import itertools
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MetaRow:
    name: str
    path: Path
    pool: set[int] | None
    topo: dict[int, set[str]]
    task: dict[int, set[str]]


def _load_row(path: Path) -> MetaRow:
    payload = json.loads(path.read_text(encoding="utf-8"))
    name = path.name
    pool_raw = payload.get("pool_uids")
    if pool_raw is not None and not isinstance(pool_raw, list):
        raise ValueError(f"{path}: pool_uids must be a list or null.")
    pool = set(int(x) for x in pool_raw) if pool_raw is not None else None
    topo_raw = payload.get("topology_signatures_per_depth") or {}
    task_raw = payload.get("task_signatures_per_depth") or {}
    topo = {int(d): set(str(x) for x in vals) for d, vals in topo_raw.items()}
    task = {int(d): set(str(x) for x in vals) for d, vals in task_raw.items()}
    return MetaRow(name=name, path=path, pool=pool, topo=topo, task=task)


def _print_depth_stats(rows: list[MetaRow], kind: str, depth: int) -> tuple[int, int, int]:
    values: list[int] = []
    getter = (
        (lambda r: r.topo.get(depth, set()))
        if kind == "topo"
        else (lambda r: r.task.get(depth, set()))
    )
    for a, b in itertools.combinations(rows, 2):
        values.append(len(getter(a) & getter(b)))
    if not values:
        return 0, 0, 0
    vmin = min(values)
    vmax = max(values)
    vavg = int(round(sum(values) / len(values)))
    print(f"{kind}_overlap_d{depth}: min={vmin} avg~={vavg} max={vmax}")
    return vmin, vavg, vmax


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Verify pool/topology/task overlap across dataset metas."
    )
    ap.add_argument(
        "--metas-glob",
        type=str,
        default="R??_*.uint32_meta.json",
        help="Glob (relative to --dir) used to select meta files.",
    )
    ap.add_argument(
        "--dir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Directory containing dataset meta files.",
    )
    ap.add_argument("--assert-zero-topo-d6", action="store_true")
    ap.add_argument("--assert-zero-task-d6", action="store_true")
    ap.add_argument("--assert-max-pool-overlap", type=int, default=-1)
    args = ap.parse_args()

    base = Path(args.dir).resolve()
    meta_paths = sorted(Path(p) for p in glob.glob(str(base / args.metas_glob)))
    if not meta_paths:
        raise SystemExit(f"No meta files matched: {base / args.metas_glob}")

    rows = [_load_row(p) for p in meta_paths]
    print(f"metas={len(rows)}")
    for r in rows:
        print(
            f"  {r.name}: pool={len(r.pool) if r.pool is not None else 'unknown'} "
            f"topo_d5={len(r.topo.get(5, set()))} topo_d6={len(r.topo.get(6, set()))} "
            f"task_d6={len(r.task.get(6, set()))}"
        )

    pool_overlaps: list[int] = []
    unknown_pool_pairs = 0
    for a, b in itertools.combinations(rows, 2):
        if a.pool is None or b.pool is None:
            unknown_pool_pairs += 1
        else:
            pool_overlaps.append(len(a.pool & b.pool))
    if pool_overlaps:
        print(
            f"pool_overlap: min={min(pool_overlaps)} "
            f"avg~={int(round(sum(pool_overlaps) / len(pool_overlaps)))} "
            f"max={max(pool_overlaps)}"
        )
    if unknown_pool_pairs:
        print(f"pool_overlap: unknown for {unknown_pool_pairs} pair(s); excluded from statistics")

    for d in [3, 4, 5, 6]:
        _print_depth_stats(rows, "topo", d)
    for d in [3, 4, 5, 6]:
        _print_depth_stats(rows, "task", d)

    d6_topo_union = set()
    d6_task_union = set()
    for r in rows:
        d6_topo_union |= r.topo.get(6, set())
        d6_task_union |= r.task.get(6, set())
    print(f"union_topo_d6={len(d6_topo_union)}")
    print(f"union_task_d6={len(d6_task_union)}")

    failed = False
    if args.assert_zero_topo_d6:
        for a, b in itertools.combinations(rows, 2):
            if len(a.topo.get(6, set()) & b.topo.get(6, set())) != 0:
                print(f"FAIL topo_d6 overlap: {a.name} vs {b.name}")
                failed = True
                break
    if args.assert_zero_task_d6:
        for a, b in itertools.combinations(rows, 2):
            if len(a.task.get(6, set()) & b.task.get(6, set())) != 0:
                print(f"FAIL task_d6 overlap: {a.name} vs {b.name}")
                failed = True
                break
    if args.assert_max_pool_overlap >= 0 and pool_overlaps:
        max_pool = max(pool_overlaps)
        if max_pool > int(args.assert_max_pool_overlap):
            print(
                f"FAIL pool overlap: max={max_pool} > allowed={int(args.assert_max_pool_overlap)}"
            )
            failed = True

    if args.assert_max_pool_overlap >= 0 and unknown_pool_pairs:
        print("FAIL pool overlap assertion: pool provenance is unknown.")
        failed = True

    if failed:
        raise SystemExit(1)
    print("PASS (pool overlap unknown)" if unknown_pool_pairs else "PASS")


if __name__ == "__main__":
    main()
