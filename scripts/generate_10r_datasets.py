#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

GEN_MODULE = "banyan_grid.tasks.make_ruleset_dataset"
TREE_TOPOLOGY = "mixed_u1_b2"


@dataclass
class DatasetResult:
    index: int
    name: str
    bench_seed: int
    meta_path: Path
    data_path: Path
    unique_d5: int
    unique_d6: int


def _build_low_overlap_pools() -> list[list[int]]:
    """Build 10 pools of size 10 over uid domain [0,79] with low pairwise overlap.

    Construction:
    - Base: 10 disjoint blocks of 8 uids each.
    - Extras: each dataset receives 2 additional uids copied from two other blocks.
    This yields pool-size 10 with max pairwise overlap typically <=1.
    """
    pools: list[list[int]] = []
    base_blocks = [list(range(8 * i, 8 * i + 8)) for i in range(10)]
    source_ptr = [0] * 10

    for i in range(10):
        pool = list(base_blocks[i])
        src1 = (i + 1) % 10
        src2 = (i + 2) % 10

        uid1 = base_blocks[src1][source_ptr[src1]]
        source_ptr[src1] += 1
        uid2 = base_blocks[src2][source_ptr[src2]]
        source_ptr[src2] += 1

        pool.extend([uid1, uid2])
        if len(pool) != 10:
            raise RuntimeError(f"Internal pool size error for pool {i}: {len(pool)}")
        if len(set(pool)) != 10:
            raise RuntimeError(f"Internal duplicate in pool {i}: {pool}")
        pools.append(pool)

    return pools


def _write_aggregate_meta(
    path: Path,
    topology_sigs_d6: set[str],
    task_sigs_d6: set[str],
    source_metas: Sequence[str],
) -> None:
    payload = {
        "tree_topology": TREE_TOPOLOGY,
        "topology_signatures_per_depth": {"6": sorted(topology_sigs_d6)},
        "task_signatures_per_depth": {"6": sorted(task_sigs_d6)},
        "source_metas": list(source_metas),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _remove_prefix_artifacts(
    out_dir: Path, prefix: str, keep_logs: bool = True
) -> None:
    for p in out_dir.glob(f"{prefix}*"):
        if not p.is_file():
            continue
        if keep_logs and p.suffix == ".log":
            continue
        p.unlink()


def _find_single(path_glob: str) -> Path:
    matches = sorted(glob.glob(path_glob))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly 1 match for {path_glob}, found {len(matches)}"
        )
    return Path(matches[0])


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rename_prefix_files(out_dir: Path, old_prefix: str, new_prefix: str) -> None:
    for p in sorted(out_dir.glob(f"{old_prefix}*")):
        if not p.is_file():
            continue
        if not p.name.startswith(old_prefix):
            continue
        new_name = new_prefix + p.name[len(old_prefix) :]
        p.rename(out_dir / new_name)


def _extract_dataset_path(meta_path: Path) -> Path:
    meta_name = meta_path.name
    if not meta_name.endswith("_meta.json"):
        raise RuntimeError(f"Unexpected meta name: {meta_name}")
    stem = meta_name[: -len("_meta.json")]
    data_path = meta_path.parent / f"{stem}.npy.bz2"
    if not data_path.exists():
        raise RuntimeError(f"Expected dataset file missing for meta: {data_path}")
    return data_path


def _run_cmd(
    cmd: list[str], env: dict[str, str], log_path: Path, timeout_s: int
) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            proc = subprocess.run(
                cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=int(timeout_s),
            )
            return int(proc.returncode)
        except subprocess.TimeoutExpired:
            return 124


def _tail_text(path: Path, max_lines: int = 12) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def _build_module_cmd(launcher: str, python_exe: str) -> list[str]:
    if launcher == "uv":
        return ["uv", "run", "-m", GEN_MODULE]
    return [python_exe, "-m", GEN_MODULE]


def _validate_required_flags(launcher: str, python_exe: str) -> None:
    help_cmd = _build_module_cmd(launcher, python_exe) + ["-h"]
    proc = subprocess.run(help_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to run ruleset generator help command. "
            f"cmd={' '.join(help_cmd)} rc={proc.returncode}"
        )
    help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    needed = [
        "--target-topologies-per-depth",
        "--target-topologies-depths",
        "--mixed-tasks-per-depth-ge3",
        "--exclude-topologies-from-meta",
        "--exclude-topology-depths",
        "--exclude-task-signatures-from-meta",
        "--exclude-task-depths",
    ]
    missing = [f for f in needed if f not in help_text]
    if missing:
        raise RuntimeError(
            "Current checkout of make_ruleset_dataset is missing required flags: "
            + ", ".join(missing)
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate 10 mixed datasets with d6 disjointness."
    )
    ap.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parent))
    ap.add_argument("--pool-size", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--seed-base", type=int, default=61000)
    ap.add_argument("--max-attempts", type=int, default=40)
    ap.add_argument("--python", type=str, default=shutil.which("python3") or "python3")
    ap.add_argument(
        "--launcher",
        type=str,
        choices=["python", "uv"],
        default="uv",
        help="How to invoke the generator module. 'uv' is recommended on cluster.",
    )
    ap.add_argument("--distractor-density", type=float, default=0.5)
    ap.add_argument("--tasks-per-depth-ge3", type=int, default=10000)
    ap.add_argument("--target-topologies", type=int, default=1000)
    ap.add_argument("--max-tries", type=int, default=8000)
    ap.add_argument("--strict-unique-tries", type=int, default=1000)
    ap.add_argument("--seed-block-size", type=int, default=64)
    ap.add_argument("--mixed-depth-workers", type=int, default=1)
    ap.add_argument("--attempt-timeout-sec", type=int, default=1800)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pool_size != 10:
        raise ValueError(
            "This generator currently uses a handcrafted low-overlap design for pool_size=10."
        )
    _validate_required_flags(args.launcher, args.python)

    pools = _build_low_overlap_pools()
    aggregate_meta = out_dir / "aggregate_exclusions_d6_meta.json"
    topology_sigs_d6: set[str] = set()
    task_sigs_d6: set[str] = set()
    source_metas: list[str] = []
    _write_aggregate_meta(aggregate_meta, topology_sigs_d6, task_sigs_d6, source_metas)
    print(f"[init] out_dir={out_dir}", flush=True)

    results: list[DatasetResult] = []

    for i in range(10):
        idx = i + 1
        final_prefix = f"R{idx:02d}"
        tmp_prefix = f"{final_prefix}_tmp"
        pool = pools[i]
        pool_indices_arg = " ".join(str(x) for x in pool)

        accepted = False
        for attempt in range(args.max_attempts):
            bench_seed = int(args.seed_base + idx * 1000 + attempt)
            _remove_prefix_artifacts(out_dir, tmp_prefix)

            log_path = out_dir / f"{tmp_prefix}.attempt{attempt:02d}.log"
            env = os.environ.copy()
            env["RULESET_GEN_MAX_TRIES"] = str(int(args.max_tries))
            env["RULESET_GEN_STRICT_UNIQUE_TRIES"] = str(int(args.strict_unique_tries))
            env["RULESET_GEN_SEED_BLOCK_SIZE"] = str(int(args.seed_block_size))
            env["RULESET_GEN_MIXED_DEPTH_WORKERS"] = str(int(args.mixed_depth_workers))
            env.setdefault("JAX_PLATFORMS", "cpu")
            env.setdefault("JAX_PLATFORM_NAME", "cpu")
            env.setdefault("PYTHONUNBUFFERED", "1")

            cmd = _build_module_cmd(args.launcher, args.python) + [
                "--max-depth",
                "6",
                "--pool-size",
                str(int(args.pool_size)),
                "--tree-topology",
                TREE_TOPOLOGY,
                "--mixed-tasks-per-depth-ge3",
                str(int(args.tasks_per_depth_ge3)),
                "--target-topologies-per-depth",
                str(int(args.target_topologies)),
                "--target-topologies-depths",
                "5,6",
                "--exclude-topologies-from-meta",
                str(aggregate_meta),
                "--exclude-topology-depths",
                "6",
                "--exclude-task-signatures-from-meta",
                str(aggregate_meta),
                "--exclude-task-depths",
                "6",
                "--distractor-density",
                str(float(args.distractor_density)),
                "--base-seed",
                str(int(args.base_seed)),
                "--bench-seed",
                str(bench_seed),
                "--pool-indices",
                pool_indices_arg,
                "--out-dir",
                str(out_dir),
                "--name",
                tmp_prefix,
            ]

            print(
                f"[attempt] {final_prefix} attempt={attempt:02d} bench_seed={bench_seed} "
                f"log={log_path.name}",
                flush=True,
            )
            rc = _run_cmd(
                cmd,
                env=env,
                log_path=log_path,
                timeout_s=int(args.attempt_timeout_sec),
            )
            if rc != 0:
                print(
                    f"[retry] {final_prefix} attempt={attempt:02d} rc={rc}",
                    flush=True,
                )
                tail = _tail_text(log_path, max_lines=12)
                if tail:
                    print(
                        f"[retry-log-tail] {final_prefix} attempt={attempt:02d}\n{tail}",
                        flush=True,
                    )
                continue

            try:
                meta_path = _find_single(str(out_dir / f"{tmp_prefix}_*_meta.json"))
                meta = _load_json(meta_path)
            except Exception:
                print(
                    f"[retry] {final_prefix} attempt={attempt:02d} missing/invalid meta",
                    flush=True,
                )
                _remove_prefix_artifacts(out_dir, tmp_prefix)
                continue

            unique_d5 = int((meta.get("topology_unique_per_depth") or {}).get("5", 0))
            unique_d6 = int((meta.get("topology_unique_per_depth") or {}).get("6", 0))
            shortfall = meta.get("target_topologies_per_depth_shortfall") or {}
            shortfall_d5 = int(shortfall.get("5", 0))
            shortfall_d6 = int(shortfall.get("6", 0))

            if unique_d5 < int(args.target_topologies) or unique_d6 < int(
                args.target_topologies
            ):
                print(
                    f"[retry] {final_prefix} attempt={attempt:02d} unique shortfall "
                    f"d5={unique_d5} d6={unique_d6}",
                    flush=True,
                )
                _remove_prefix_artifacts(out_dir, tmp_prefix)
                continue
            if shortfall_d5 != 0 or shortfall_d6 != 0:
                print(
                    f"[retry] {final_prefix} attempt={attempt:02d} target shortfall "
                    f"d5={shortfall_d5} d6={shortfall_d6}",
                    flush=True,
                )
                _remove_prefix_artifacts(out_dir, tmp_prefix)
                continue

            d6_topos = set(
                (meta.get("topology_signatures_per_depth") or {}).get("6", [])
            )
            d6_tasks = set((meta.get("task_signatures_per_depth") or {}).get("6", []))
            if d6_topos & topology_sigs_d6:
                print(
                    f"[retry] {final_prefix} attempt={attempt:02d} d6 topology overlap detected",
                    flush=True,
                )
                _remove_prefix_artifacts(out_dir, tmp_prefix)
                continue
            if d6_tasks & task_sigs_d6:
                print(
                    f"[retry] {final_prefix} attempt={attempt:02d} d6 task overlap detected",
                    flush=True,
                )
                _remove_prefix_artifacts(out_dir, tmp_prefix)
                continue

            _rename_prefix_files(out_dir, tmp_prefix, final_prefix)
            final_meta = _find_single(str(out_dir / f"{final_prefix}_*_meta.json"))
            final_data = _extract_dataset_path(final_meta)
            final_meta_json = _load_json(final_meta)
            d6_topos_final = set(
                (final_meta_json.get("topology_signatures_per_depth") or {}).get(
                    "6", []
                )
            )
            d6_tasks_final = set(
                (final_meta_json.get("task_signatures_per_depth") or {}).get("6", [])
            )
            topology_sigs_d6 |= d6_topos_final
            task_sigs_d6 |= d6_tasks_final
            source_metas.append(str(final_meta))
            _write_aggregate_meta(
                aggregate_meta, topology_sigs_d6, task_sigs_d6, source_metas
            )

            results.append(
                DatasetResult(
                    index=idx,
                    name=final_prefix,
                    bench_seed=bench_seed,
                    meta_path=final_meta,
                    data_path=final_data,
                    unique_d5=int(
                        (final_meta_json.get("topology_unique_per_depth") or {}).get(
                            "5", 0
                        )
                    ),
                    unique_d6=int(
                        (final_meta_json.get("topology_unique_per_depth") or {}).get(
                            "6", 0
                        )
                    ),
                )
            )
            accepted = True
            print(
                f"[accepted] {final_prefix}: bench_seed={bench_seed} "
                f"unique_d5={results[-1].unique_d5} unique_d6={results[-1].unique_d6}",
                flush=True,
            )
            break

        if not accepted:
            raise RuntimeError(
                f"Failed to generate {final_prefix} after {args.max_attempts} attempts"
            )

    summary = {
        "out_dir": str(out_dir),
        "pool_size": int(args.pool_size),
        "base_seed": int(args.base_seed),
        "target_topologies_d5_d6": int(args.target_topologies),
        "tasks_per_depth_ge3": int(args.tasks_per_depth_ge3),
        "d6_union_unique_topologies": int(len(topology_sigs_d6)),
        "d6_union_unique_task_signatures": int(len(task_sigs_d6)),
        "datasets": [
            {
                "index": r.index,
                "name": r.name,
                "bench_seed": r.bench_seed,
                "meta_path": str(r.meta_path),
                "data_path": str(r.data_path),
                "unique_d5": r.unique_d5,
                "unique_d6": r.unique_d6,
            }
            for r in results
        ],
        "pools": {f"R{i + 1:02d}": pools[i] for i in range(10)},
    }
    summary_path = out_dir / "generation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] Wrote summary: {summary_path}")
    print(f"[done] Aggregate exclusion meta: {aggregate_meta}")


if __name__ == "__main__":
    main()
