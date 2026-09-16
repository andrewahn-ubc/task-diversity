#!/usr/bin/env python3
"""Prune a packed ruleset dataset to a selected set of depths.

This keeps row order within selected depths and rewrites:
  - dataset (.uint32.npy.bz2)
  - metadata (.uint32_meta.json)
  - distractor table (.uint32_distractor_table.npy), if present
"""

from __future__ import annotations

import argparse
import bz2
import json
from pathlib import Path
from typing import Any

import numpy as np

from banyan_grid.tasks.ruleset_dataset_compact import load_meta


def _meta_path(dataset_path: Path) -> Path:
    if not dataset_path.name.endswith(".uint32.npy.bz2"):
        raise ValueError(f"Expected .uint32.npy.bz2 dataset path: {dataset_path}")
    base = dataset_path.name[: -len(".uint32.npy.bz2")]
    return dataset_path.with_name(f"{base}.uint32_meta.json")


def _table_path(dataset_path: Path) -> Path:
    base = dataset_path.name[: -len(".uint32.npy.bz2")]
    return dataset_path.with_name(f"{base}.uint32_distractor_table.npy")


def _load_packed(path: Path) -> np.ndarray:
    with bz2.BZ2File(path, "rb") as f:
        arr = np.load(f, allow_pickle=False)
    if arr.dtype != np.uint32 or arr.ndim != 2:
        raise ValueError(
            f"Expected uint32 matrix for dataset: {path} (got {arr.shape}, {arr.dtype})"
        )
    return arr


def _write_packed(path: Path, arr: np.ndarray, *, allow_overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with bz2.BZ2File(path, "wb" if allow_overwrite else "xb") as f:
        np.save(f, arr.astype(np.uint32, copy=False), allow_pickle=False)


def _parse_depths(depths_str: str) -> list[int]:
    vals: list[int] = []
    for part in depths_str.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("No depths provided.")
    return sorted(set(vals))


def _depth_ranges_from_structure(
    structure_raw: dict[str, Any], total_rows: int
) -> dict[int, tuple[int, int]]:
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
        cnt = int(structure[d])
        out[d] = (cursor, cursor + cnt)
        cursor += cnt

    if cursor != total_rows:
        raise ValueError(
            "Structure counts do not match dataset rows: "
            f"sum(structure)={cursor}, rows={total_rows}"
        )
    return out


def _filter_depth_dict(depth_dict: Any, keep_depths: set[int]) -> dict[str, Any]:
    if not isinstance(depth_dict, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in depth_dict.items():
        try:
            d = int(k)
        except Exception:
            continue
        if d in keep_depths:
            out[str(d)] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Input .uint32.npy.bz2 dataset path.")
    ap.add_argument("--output", required=True, help="Output .uint32.npy.bz2 dataset path.")
    ap.add_argument(
        "--keep-depths",
        required=True,
        help='Comma-separated depths to keep, e.g. "1,2,3" or "4,5,6".',
    )
    ap.add_argument(
        "--label",
        default="",
        help="Optional label appended to tree_topology in output metadata.",
    )
    ap.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow overwriting output files, but never input files.",
    )
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {in_path}")
    if not out_path.name.endswith(".uint32.npy.bz2"):
        raise ValueError("--output must end with .uint32.npy.bz2")

    out_meta_path = _meta_path(out_path)
    out_table_path = _table_path(out_path)
    in_meta_path = _meta_path(in_path)
    in_table_path = _table_path(in_path)
    source_paths = (
        in_path,
        in_meta_path,
        in_meta_path.with_name(in_meta_path.name.replace(".uint32_meta.json", "_meta.json")),
        in_table_path,
    )
    for p in (out_path, out_meta_path, out_table_path):
        if any(
            p.resolve() == source.resolve()
            or (p.exists() and source.exists() and p.samefile(source))
            for source in source_paths
        ):
            raise ValueError(f"Output would overwrite an input file: {p}")
        if p.exists() and not args.allow_overwrite:
            raise FileExistsError(f"Output exists (use --allow-overwrite): {p}")

    keep_depths = set(_parse_depths(args.keep_depths))

    packed = _load_packed(in_path)
    meta = load_meta(str(in_path.parent), in_path.name)
    if not isinstance(meta, dict):
        raise ValueError(f"Expected metadata dictionary for: {in_path}")

    structure_raw = meta.get("structure")
    if not isinstance(structure_raw, dict):
        raise ValueError("Input metadata missing dict field: structure")
    depth_ranges = _depth_ranges_from_structure(structure_raw, packed.shape[0])

    missing = sorted(d for d in keep_depths if d not in depth_ranges)
    if missing:
        raise ValueError(f"Requested keep depths not present in input dataset: {missing}")

    indices: list[np.ndarray] = []
    for d in sorted(keep_depths):
        lo, hi = depth_ranges[d]
        indices.append(np.arange(lo, hi, dtype=np.int64))
    keep_idx = np.concatenate(indices, axis=0)
    if keep_idx.size == 0:
        raise ValueError("No rows selected by --keep-depths.")

    pruned = packed[keep_idx]

    # Rewrite metadata
    new_meta = dict(meta)
    new_structure = {
        str(d): int(depth_ranges[d][1] - depth_ranges[d][0]) for d in sorted(keep_depths)
    }
    new_max_depth = max(keep_depths)
    new_meta["structure"] = new_structure
    new_meta["max_depth"] = int(new_max_depth)
    new_meta["n"] = int(pruned.shape[0])
    if "packed_shape" in new_meta:
        new_meta["packed_shape"] = list(pruned.shape)
    new_meta["depths"] = [int(d) for d in sorted(keep_depths)]
    new_meta["tree_topology"] = f"{meta.get('tree_topology', 'unknown')}" + (
        f"+{args.label}" if args.label else ""
    )

    # Keep per-depth metadata fields only for retained depths.
    per_depth_keys = [k for k, v in new_meta.items() if "_per_depth" in k and isinstance(v, dict)]
    for k in per_depth_keys:
        if k in new_meta:
            new_meta[k] = _filter_depth_dict(new_meta.get(k), keep_depths)

    # Depth-5 specific summaries are only meaningful if depth 5 is retained.
    if 5 not in keep_depths:
        for k in list(new_meta):
            if k.startswith("topology_depth5_"):
                del new_meta[k]

    prev_desc = str(meta.get("description", "")).strip()
    prune_note = f"Pruned to depths {','.join(str(d) for d in sorted(keep_depths))}."
    new_meta["description"] = f"{prev_desc} {prune_note}".strip()
    new_meta["pruned_from_dataset"] = str(in_path)
    new_meta["pruned_from_depths"] = sorted(int(d) for d in depth_ranges.keys())
    new_meta["pruned_keep_depths"] = sorted(int(d) for d in keep_depths)
    new_meta["pruned_row_count"] = int(pruned.shape[0])

    # Distractor table can be:
    # - per-ruleset (axis 0 == num_rulesets), or
    # - global lookup tensor (e.g., shape [9, 11, 9, 11, 3]).
    # Keep behavior robust for both.
    pruned_table = None
    if in_table_path.exists():
        table = np.load(in_table_path, allow_pickle=False)
        if table.ndim == 5 and table.shape[:2] == table.shape[2:4] and table.shape[-1] == 3:
            pruned_table = table
        elif table.ndim >= 1 and table.shape[0] == packed.shape[0]:
            pruned_table = table[keep_idx]
        else:
            pruned_table = table
        new_meta["distractor_table"] = True
        new_meta["distractor_table_shape"] = [int(x) for x in pruned_table.shape]
    else:
        if int(meta.get("distractor_pair_count", int(bool(meta.get("distractor_table"))))) > 0:
            raise FileNotFoundError(
                f"Metadata declares a missing distractor table: {in_table_path}"
            )
        if out_table_path.exists():
            raise FileExistsError(
                f"Output has a stale distractor table; choose a new output path: {out_table_path}"
            )
        new_meta["distractor_table"] = False
        new_meta.pop("distractor_table_shape", None)

    _write_packed(out_path, pruned, allow_overwrite=args.allow_overwrite)
    if pruned_table is not None:
        with out_table_path.open("wb" if args.allow_overwrite else "xb") as f:
            np.save(f, pruned_table, allow_pickle=False)
    with out_meta_path.open("w" if args.allow_overwrite else "x", encoding="utf-8") as f:
        json.dump(new_meta, f, indent=2)

    print(f"Wrote pruned dataset: {out_path}")
    print(f"Wrote pruned meta:    {out_meta_path}")
    if out_table_path.exists():
        print(f"Wrote pruned table:   {out_table_path}")
    print(f"Selected rows: {pruned.shape[0]} of {packed.shape[0]}")
    print(f"Kept depths: {sorted(keep_depths)}")


if __name__ == "__main__":
    main()
