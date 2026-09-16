#!/usr/bin/env python3
"""Merge multiple ruleset datasets into one packed dataset + meta."""

from __future__ import annotations

import argparse
import bz2
import json
from pathlib import Path
from typing import Any

import numpy as np

from banyan_grid.environment.constants import NUM_COLORS, NUM_ITEMS, RULE_TYPE_PAD
from banyan_grid.tasks.ruleset_dataset_compact import load_meta


def _meta_path(dataset_path: Path) -> Path:
    if not dataset_path.name.endswith(".uint32.npy.bz2"):
        raise ValueError(f"Expected .uint32.npy.bz2 file: {dataset_path}")
    base = dataset_path.name[: -len(".uint32.npy.bz2")]
    return dataset_path.with_name(f"{base}.uint32_meta.json")


def _table_path(dataset_path: Path) -> Path:
    base = dataset_path.name[: -len(".uint32.npy.bz2")]
    return dataset_path.with_name(f"{base}.uint32_distractor_table.npy")


def _load_packed(path: Path) -> np.ndarray:
    with bz2.BZ2File(path, "rb") as f:
        arr = np.load(f, allow_pickle=False)
    if arr.dtype != np.uint32 or arr.ndim != 2:
        raise ValueError(f"Expected uint32 matrix for dataset: {path}")
    return arr


def _depth_ranges_from_structure(structure_raw: Any, total_rows: int) -> dict[int, tuple[int, int]]:
    if not isinstance(structure_raw, dict) or not structure_raw:
        raise ValueError("Input metadata missing nonempty dict field: structure")
    structure: dict[int, int] = {}
    for k, v in structure_raw.items():
        d = int(k)
        if d < 1 or d in structure:
            raise ValueError(f"Invalid or duplicate depth in structure: {k}")
        if isinstance(v, bool) or not isinstance(v, (int, str)) or int(v) < 0:
            raise ValueError(f"Invalid row count for depth {k}: {v}")
        structure[d] = int(v)

    cursor = 0
    out: dict[int, tuple[int, int]] = {}
    for d in sorted(structure):
        out[d] = (cursor, cursor + structure[d])
        cursor += structure[d]
    if cursor != total_rows:
        raise ValueError(
            "Structure counts do not match dataset rows: "
            f"sum(structure)={cursor}, rows={total_rows}"
        )
    return out


def _pad_rules(rules: np.ndarray, target_rules: int) -> np.ndarray:
    if rules.shape[1] == target_rules:
        return rules
    if rules.shape[1] > target_rules:
        raise ValueError("Cannot pad: source has more rules than target")
    pad_rows = target_rules - rules.shape[1]
    return np.pad(rules, ((0, 0), (0, pad_rows)), mode="constant", constant_values=RULE_TYPE_PAD)


def _sum_structure(metas: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for meta in metas:
        structure = meta.get("structure", {})
        for k, v in structure.items():
            key = str(int(k))
            out[key] = out.get(key, 0) + int(v)
    return dict(sorted(out.items(), key=lambda item: int(item[0])))


def _merge_rules_per_depth(metas: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for meta in metas:
        rules_per_depth = meta.get("rules_per_depth", {})
        for k, v in rules_per_depth.items():
            key = str(int(k))
            out[key] = max(out.get(key, 0), int(v))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True, help="Input dataset files.")
    ap.add_argument("--output", required=True, help="Output dataset .uint32.npy.bz2 path.")
    ap.add_argument(
        "--topology-label",
        default="merged",
        help="Metadata label for merged tree_topology.",
    )
    ap.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow overwriting output files, but never input files.",
    )
    args = ap.parse_args()

    input_paths = [Path(p).expanduser().resolve() for p in args.inputs]
    output_path = Path(args.output).expanduser().resolve()
    if not output_path.name.endswith(".uint32.npy.bz2"):
        raise ValueError("--output must end with .uint32.npy.bz2")

    meta_out = _meta_path(output_path)
    table_out = _table_path(output_path)
    source_paths = []
    for dataset_path in input_paths:
        meta_path = _meta_path(dataset_path)
        source_paths.extend(
            (
                dataset_path,
                meta_path,
                meta_path.with_name(meta_path.name.replace(".uint32_meta.json", "_meta.json")),
                _table_path(dataset_path),
            )
        )
    for p in (output_path, meta_out, table_out):
        if any(
            p.resolve() == source.resolve()
            or (p.exists() and source.exists() and p.samefile(source))
            for source in source_paths
        ):
            raise ValueError(f"Output would overwrite an input file: {p}")
        if p.exists() and not args.allow_overwrite:
            raise FileExistsError(f"Output exists (use --allow-overwrite): {p}")

    metas: list[dict[str, Any]] = []
    packed_sets: list[np.ndarray] = []
    depth_ranges: list[dict[int, tuple[int, int]]] = []
    distractor_tables: list[np.ndarray] = []
    distractor_density: float | None = None

    for dataset_path in input_paths:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Input dataset not found: {dataset_path}")
        meta = load_meta(str(dataset_path.parent), dataset_path.name)
        if not isinstance(meta, dict):
            raise ValueError(f"Expected metadata dictionary for: {dataset_path}")
        metas.append(meta)

        this_density = float(meta.get("distractor_density", 0.0))
        if distractor_density is None:
            distractor_density = this_density
        elif abs(this_density - distractor_density) > 1e-12:
            raise ValueError(
                "All inputs must share distractor_density. "
                f"Got {distractor_density} and {this_density}."
            )

        packed = _load_packed(dataset_path)
        depth_ranges.append(_depth_ranges_from_structure(meta.get("structure"), packed.shape[0]))
        packed_sets.append(packed)

        table_path = _table_path(dataset_path)
        if table_path.exists():
            table = np.load(table_path, allow_pickle=False)
            expected_shape = (NUM_ITEMS, NUM_COLORS, NUM_ITEMS, NUM_COLORS, 3)
            if table.shape != expected_shape:
                raise ValueError(
                    f"Expected global distractor table shape {expected_shape}: {table_path}"
                )
            if distractor_tables and (
                table.dtype != distractor_tables[0].dtype
                or not np.array_equal(table, distractor_tables[0])
            ):
                raise ValueError(
                    "Inputs have differing global distractor tables; merging would change task semantics."
                )
            distractor_tables.append(table)
        elif int(meta.get("distractor_pair_count", int(bool(meta.get("distractor_table"))))) > 0:
            raise FileNotFoundError(f"Metadata declares a missing distractor table: {table_path}")

    if distractor_tables and len(distractor_tables) != len(input_paths):
        raise ValueError("Cannot merge inputs with mixed missing/present distractor tables.")
    if not distractor_tables and table_out.exists():
        raise FileExistsError(
            f"Output has a stale distractor table; choose a new output path: {table_out}"
        )

    structure = _sum_structure(metas)
    max_rules = max(arr.shape[1] for arr in packed_sets)
    merged_packed = np.concatenate(
        [
            _pad_rules(packed[ranges[d][0] : ranges[d][1]], max_rules)
            for d in sorted(int(k) for k in structure)
            for packed, ranges in zip(packed_sets, depth_ranges)
            if d in ranges
        ],
        axis=0,
    )
    max_depth = max(int(d) for d in structure)
    rules_per_depth = _merge_rules_per_depth(metas)
    pool_uids: list[int] | None = []
    for meta in metas:
        pool = meta.get("pool_uids")
        if pool is not None and not isinstance(pool, list):
            raise ValueError("Input pool_uids must be a list or null.")
        if pool is None:
            pool_uids = None
        elif pool_uids is not None:
            pool_uids.extend(int(uid) for uid in pool)
    if pool_uids is not None:
        pool_uids = list(dict.fromkeys(pool_uids))

    merged_meta: dict[str, Any] = {
        "n": int(merged_packed.shape[0]),
        "structure": structure,
        "max_depth": max_depth,
        "tree_topology": args.topology_label,
        "rules_per_depth": {d: rules_per_depth[d] for d in structure if d in rules_per_depth},
        "pool_size": len(pool_uids) if pool_uids is not None else None,
        "pool_uids": pool_uids,
        "base_seed": None,
        "bench_seed": None,
        "base_seed_in_pool": False,
        "rule_shape": [int(max_rules), 6],
        "generation_mode": "merged",
        "description": f"Merged from {len(input_paths)} source datasets.",
        "distractor_density": float(distractor_density if distractor_density is not None else 0.0),
        "distractor_table": bool(distractor_tables),
        "merge_sources": [
            {"dataset": str(path), "metadata": meta} for path, meta in zip(input_paths, metas)
        ],
    }
    seed_keys = {k for meta in metas for k in meta if k == "seed" or k.endswith("_seed")}
    for k in sorted(seed_keys):
        value = metas[0].get(k)
        merged_meta[k] = value if all(meta.get(k) == value for meta in metas) else None
    merged_meta["base_seed_in_pool"] = merged_meta["base_seed"] is not None and all(
        meta.get("base_seed_in_pool") is True for meta in metas
    )
    if distractor_tables:
        merged_meta["distractor_table_shape"] = list(distractor_tables[0].shape)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with bz2.BZ2File(output_path, "wb" if args.allow_overwrite else "xb") as f:
        np.save(f, merged_packed, allow_pickle=False)
    if distractor_tables:
        with table_out.open("wb" if args.allow_overwrite else "xb") as f:
            np.save(f, distractor_tables[0], allow_pickle=False)
    with meta_out.open("w" if args.allow_overwrite else "x") as f:
        json.dump(merged_meta, f, indent=2)

    print(f"Merged dataset written to: {output_path}")
    print(f"Merged meta written to: {meta_out}")


if __name__ == "__main__":
    main()
