from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from banyan_grid.environment.constants import NUM_COLORS, NUM_ITEMS
from banyan_grid.tasks.ruleset_dataset_compact import (
    load_distractor_table,
    load_meta,
    load_packed_u32_bz2,
)
from banyan_grid.tasks.ruleset_factory import (
    compute_depth_task_counts,
)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    max_depth: int
    n: int
    pool_size: int
    pool_partition: int | None = None  # For distribution sweep (disjoint pools)
    tree_topology: str = "balanced"


def _fmt_density_tag(value: float) -> str:
    clamped = max(0.0, min(1.0, float(value)))
    text = f"{clamped:.6f}".rstrip("0").rstrip(".")
    if text == "":
        text = "0"
    return text.replace(".", "p")


def _distractor_suffix(value: float) -> str:
    if float(value) <= 0.0:
        return ""
    return f"_dd{_fmt_density_tag(value)}"


def dataset_filenames(
    task_name: str,
    max_depth: int,
    n_highest: int,
    pool_size: int,
    base_seed: int,
    bench_seed: int,
    tree_topology: str = "balanced",
    distractor_density: float = 0.0,
) -> tuple[str, str]:
    if tree_topology != "balanced":
        raise ValueError(
            "Filename inference supports balanced datasets only. "
            "Use make_ruleset_dataset with explicit options for other topologies."
        )
    if max_depth < 1 or pool_size < 1 or n_highest < 1:
        raise ValueError("max_depth, pool_size, and n_highest must be positive.")
    structure = compute_depth_task_counts(max_depth, n_highest, pool_size)
    depths = sorted(structure.keys())
    depth_str = "-".join(str(d) for d in depths)
    total_n = sum(structure.values())
    file_name = (
        f"{task_name}_d{depth_str}_n{total_n}_ps{pool_size}"
        f"_bs{base_seed}_rs{bench_seed}{_distractor_suffix(distractor_density)}"
        ".uint32.npy.bz2"
    )
    stem = file_name
    if stem.endswith(".bz2"):
        stem = stem[:-4]
    if stem.endswith(".npy"):
        stem = stem[:-4]
    meta_name = f"{stem}_meta.json"
    return file_name, meta_name


def meta_matches(
    meta: dict[str, Any],
    spec: TaskSpec,
    base_seed: int,
    bench_seed: int,
    distractor_density: float = 0.0,
) -> bool:
    if meta.get("base_seed_in_pool") is not True:
        return False
    if meta.get("max_depth") != spec.max_depth:
        return False
    if meta.get("pool_size") != spec.pool_size:
        return False
    if meta.get("base_seed") != base_seed:
        return False
    if meta.get("bench_seed") != bench_seed:
        return False
    meta_density = float(meta.get("distractor_density", 0.0))
    if abs(meta_density - float(distractor_density)) > 1e-12:
        return False
    meta_topology = str(meta.get("tree_topology", "balanced"))
    if meta_topology != str(spec.tree_topology):
        return False
    structure = meta.get("structure", {})
    if not isinstance(structure, dict):
        return False
    if spec.tree_topology == "balanced":
        expected = compute_depth_task_counts(spec.max_depth, spec.n, spec.pool_size)
        actual = {int(k): int(v) for k, v in structure.items()}
        mode = meta.get("generation_mode", meta.get("curriculum", "strict"))
        return actual == expected and mode == "strict"
    n_high = structure.get(str(spec.max_depth), structure.get(spec.max_depth))
    return n_high is not None and int(n_high) == spec.n


_FILE_SUFFIX = ".uint32.npy.bz2"
_STEM_RE = re.compile(
    r"^(?P<name>.+)_d(?P<depths>\d+(?:-\d+)*)_n(?P<total_n>\d+)_ps(?P<pool_size>\d+)_bs(?P<base_seed>\d+)_rs(?P<bench_seed>\d+)(?:_dd(?P<dd>[0-9p]+))?$"
)


@dataclass(frozen=True)
class ParsedDatasetSpec:
    task_name: str
    max_depth: int
    n_highest: int
    total_n: int
    pool_size: int
    base_seed: int
    bench_seed: int
    distractor_density: float


def _strip_known_suffixes(dataset_file: str) -> str:
    if dataset_file.endswith(".uint32.npy.bz2"):
        return dataset_file[: -len(".uint32.npy.bz2")]
    if dataset_file.endswith(".npy.bz2"):
        return dataset_file[: -len(".npy.bz2")]
    return dataset_file


def _normalize_dataset_file(dataset_file: str) -> str:
    if dataset_file.endswith(_FILE_SUFFIX):
        return dataset_file
    return f"{_strip_known_suffixes(dataset_file)}{_FILE_SUFFIX}"


def _parse_density_tag(tag: str | None) -> float:
    if not tag:
        return 0.0
    return float(tag.replace("p", "."))


def _infer_n_highest(total_n: int, max_depth: int, pool_size: int) -> int:
    if total_n < 1 or max_depth < 1 or pool_size < 1:
        raise ValueError("Task counts, max_depth, and pool_size must be positive.")
    # compute_depth_task_counts is independent of n_highest for depth<=2.
    if max_depth <= 2:
        structure = compute_depth_task_counts(max_depth, 1, pool_size)
        if sum(structure.values()) != total_n or structure[max_depth] < 1:
            raise ValueError("Dataset counts do not match a balanced depth-1/2 dataset.")
        return structure[max_depth]

    base = pool_size + (pool_size * (pool_size - 1) // 2)
    denominator = (1 << (max_depth - 2)) - 1
    numerator = total_n - base
    if numerator <= 0 or denominator <= 0 or (numerator % denominator) != 0:
        raise ValueError(
            "Cannot infer balanced generation parameters from this filename. "
            "Supply metadata when parsing non-balanced datasets, or generate them "
            "with explicit make_ruleset_dataset options. "
            f"total_n={total_n}, max_depth={max_depth}, pool_size={pool_size}"
        )
    n_highest = numerator // denominator
    structure = compute_depth_task_counts(max_depth, n_highest, pool_size)
    if int(sum(structure.values())) != int(total_n):
        raise ValueError(
            "Inferred n_highest does not reproduce total_n. "
            f"inferred={n_highest}, expected_total={total_n}, got_total={sum(structure.values())}"
        )
    return int(n_highest)


def _metadata_depth_counts(metadata: dict[str, Any]) -> dict[int, int]:
    structure = metadata.get("structure")
    if not isinstance(structure, dict) or not structure:
        raise ValueError("Dataset metadata must contain per-depth task counts.")
    counts: dict[int, int] = {}
    for key, value in structure.items():
        depth = int(key)
        if depth < 1 or depth in counts:
            raise ValueError(f"Invalid or duplicate dataset depth: {key}")
        if isinstance(value, bool) or not isinstance(value, (int, str)) or int(value) < 0:
            raise ValueError(f"Invalid task count for depth {key}: {value}")
        counts[depth] = int(value)
    return counts


def parse_ruleset_dataset_file(
    dataset_file: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> ParsedDatasetSpec:
    stem = _strip_known_suffixes(dataset_file)
    match = _STEM_RE.fullmatch(stem)
    if match is None:
        raise ValueError(
            "Dataset file stem is not in launcher ruleset-dataset filename format. "
            "Expected: NAME_d..._n..._ps..._bs..._rs...[ _dd... ]. "
            f"Got: {dataset_file}"
        )

    task_name = match.group("name")
    depths = [int(x) for x in match.group("depths").split("-")]
    if depths != sorted(set(depths)) or depths[0] < 1:
        raise ValueError("Dataset depths must be positive, unique, and increasing.")
    max_depth = max(depths)
    total_n = int(match.group("total_n"))
    pool_size = int(match.group("pool_size"))
    base_seed = int(match.group("base_seed"))
    bench_seed = int(match.group("bench_seed"))
    distractor_density = _parse_density_tag(match.group("dd"))
    if metadata is None:
        if depths != list(range(1, max_depth + 1)):
            raise ValueError("Supply metadata to parse a dataset with non-contiguous depths.")
        n_highest = _infer_n_highest(total_n, max_depth, pool_size)
    else:
        structure = _metadata_depth_counts(metadata)
        if (
            sorted(structure) != depths
            or any(v < 0 for v in structure.values())
            or sum(structure.values()) != total_n
            or structure[max_depth] < 1
        ):
            raise ValueError("Dataset filename and metadata depth counts disagree.")
        expected = {
            "max_depth": max_depth,
            "pool_size": pool_size,
            "base_seed": base_seed,
            "bench_seed": bench_seed,
        }
        if any(k in metadata and metadata[k] != v for k, v in expected.items()):
            raise ValueError("Dataset filename and metadata generation parameters disagree.")
        if abs(float(metadata.get("distractor_density", 0.0)) - distractor_density) > 1e-6:
            raise ValueError("Dataset filename and metadata distractor density disagree.")
        n_highest = structure[max_depth]

    return ParsedDatasetSpec(
        task_name=task_name,
        max_depth=max_depth,
        n_highest=n_highest,
        total_n=total_n,
        pool_size=pool_size,
        base_seed=base_seed,
        bench_seed=bench_seed,
        distractor_density=distractor_density,
    )


def _validate_existing_dataset(dataset_dir: Path, dataset_file: str) -> None:
    metadata = load_meta(str(dataset_dir), dataset_file)
    packed = load_packed_u32_bz2(str(dataset_dir), dataset_file)
    counts = _metadata_depth_counts(metadata)
    if (
        packed.shape[0] < 1
        or packed.shape[1] < 1
        or sum(counts.values()) != packed.shape[0]
        or metadata.get("n", packed.shape[0]) != packed.shape[0]
        or int(metadata.get("max_depth", max(counts))) != max(counts)
    ):
        raise ValueError("Existing dataset metadata does not match its stored task counts.")
    if "rule_shape" in metadata and list(metadata["rule_shape"]) != [packed.shape[1], 6]:
        raise ValueError("Existing dataset metadata does not match its stored rule width.")
    table = load_distractor_table(str(dataset_dir), dataset_file)
    if (
        table is None
        and int(metadata.get("distractor_pair_count", int(bool(metadata.get("distractor_table")))))
        > 0
    ):
        raise FileNotFoundError("Existing dataset is missing its distractor table.")
    if table is not None and table.shape != (NUM_ITEMS, NUM_COLORS, NUM_ITEMS, NUM_COLORS, 3):
        raise ValueError("Existing dataset has an invalid global distractor table shape.")
    if _STEM_RE.fullmatch(_strip_known_suffixes(dataset_file)):
        parse_ruleset_dataset_file(dataset_file, metadata=metadata)


def ensure_ruleset_dataset(
    dataset_dir: str | Path,
    dataset_file: str,
    *,
    chunk: int = 200_000,
    python_executable: str | None = None,
    dry_run: bool = False,
) -> Path:
    """Reuse validated datasets; generate missing balanced datasets without overwriting files."""
    dataset_dir_path = Path(dataset_dir).resolve()
    requested_file = _normalize_dataset_file(dataset_file)
    if Path(requested_file).name != requested_file:
        raise ValueError("dataset_file must be a filename relative to dataset_dir.")
    dataset_path = dataset_dir_path / requested_file
    stem = _strip_known_suffixes(requested_file)
    meta_paths = (
        dataset_dir_path / f"{stem}.uint32_meta.json",
        dataset_dir_path / f"{stem}_meta.json",
    )
    table_path = dataset_dir_path / f"{stem}.uint32_distractor_table.npy"
    if any(p.exists() for p in (dataset_path, *meta_paths, table_path)):
        if not dataset_path.is_file() or not any(p.is_file() for p in meta_paths):
            raise FileNotFoundError(
                f"Incomplete existing dataset {dataset_path}; refusing to overwrite its files."
            )
        _validate_existing_dataset(dataset_dir_path, requested_file)
        return dataset_path

    spec = parse_ruleset_dataset_file(requested_file)
    expected_file, expected_meta = dataset_filenames(
        spec.task_name,
        spec.max_depth,
        spec.n_highest,
        spec.pool_size,
        spec.base_seed,
        spec.bench_seed,
        distractor_density=spec.distractor_density,
    )
    if requested_file != expected_file:
        raise ValueError(
            "Dataset file does not match inferred launcher filename. "
            f"requested={requested_file}, expected={expected_file}"
        )
    cmd = [
        python_executable or sys.executable,
        "-m",
        "banyan_grid.tasks.make_ruleset_dataset",
        "--n",
        str(spec.n_highest),
        "--max-depth",
        str(spec.max_depth),
        "--pool-size",
        str(spec.pool_size),
        "--distractor-density",
        str(spec.distractor_density),
        "--base-seed",
        str(spec.base_seed),
        "--bench-seed",
        str(spec.bench_seed),
        "--tree-topology",
        "balanced",
        "--out-dir",
        str(dataset_dir_path),
        "--name",
        spec.task_name,
        "--chunk",
        str(chunk),
    ]
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return dataset_path
    dataset_dir_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    meta_path = dataset_dir_path / expected_meta
    if not dataset_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Dataset generation did not produce expected files: {dataset_path} and {meta_path}"
        )
    _validate_existing_dataset(dataset_dir_path, requested_file)
    return dataset_path


# Backwards-compatible aliases for the pre-release function names.
parse_curriculum_dataset_file = parse_ruleset_dataset_file
ensure_curriculum_dataset = ensure_ruleset_dataset
