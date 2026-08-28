from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import load_config


ALGORITHMS = ("ppo", "ppo_cbp")
CATALOG_SEEDS = (260600880, 260600881)
COLORS = {1: "#D55E00", 4: "#0072B2"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def collect(
    root: Path,
    config_path: Path,
    catalog_seeds: tuple[int, ...] = CATALOG_SEEDS,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    expected_steps = config.experiment.steps_per_distribution
    expected_fingerprint = config.fingerprint()
    rows: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        for catalog_seed in catalog_seeds:
            for diversity in config.experiment.diversities:
                for training_seed in config.experiment.seeds:
                    run_dir = (
                        root
                        / algorithm
                        / f"catalog_{catalog_seed}"
                        / f"n{diversity}"
                        / f"seed_{training_seed}"
                    )
                    completed_path = run_dir / "completed.json"
                    if not completed_path.exists():
                        raise RuntimeError(f"Missing completed run: {run_dir}")
                    completed = _read_json(completed_path)
                    metadata = _read_json(run_dir / "run.json")
                    if completed.get("status") != "complete":
                        raise RuntimeError(f"Incomplete run marker: {completed_path}")
                    if completed.get("total_env_steps") != expected_steps:
                        raise RuntimeError(f"Unexpected training budget: {completed_path}")
                    if metadata.get("config_fingerprint") != expected_fingerprint:
                        raise RuntimeError(f"Configuration mismatch: {run_dir}")
                    expected_identity = {
                        "algorithm": algorithm,
                        "catalog_seed": catalog_seed,
                        "vary_layout": True,
                        "layout_count": diversity,
                    }
                    if metadata.get("run_identity") != expected_identity:
                        raise RuntimeError(
                            f"Run identity mismatch in {run_dir}: "
                            f"expected {expected_identity}, found {metadata.get('run_identity')}"
                        )
                    if metadata.get("unique_layout_topology_pairs") != diversity**2:
                        raise RuntimeError(f"Layout/topology cross product is incomplete: {run_dir}")
                    evaluations = [
                        row
                        for row in _read_jsonl(run_dir / "metrics.jsonl")
                        if row.get("event") == "evaluation"
                        and row.get("kind") in {"phase_start", "periodic"}
                    ]
                    by_step = {int(row["phase_env_steps"]): row for row in evaluations}
                    if expected_steps not in by_step:
                        phase_end = [
                            row
                            for row in _read_jsonl(run_dir / "metrics.jsonl")
                            if row.get("event") == "evaluation"
                            and row.get("kind") == "phase_end"
                        ]
                        if len(phase_end) != 1:
                            raise RuntimeError(f"Missing unique phase-end evaluation: {run_dir}")
                        by_step[expected_steps] = phase_end[0]
                    for step, evaluation in sorted(by_step.items()):
                        expected_per_depth = config.experiment.eval_episodes // 6
                        if any(
                            int(evaluation.get(f"episodes_depth_{depth}", -1))
                            != expected_per_depth
                            for depth in range(1, 7)
                        ):
                            raise RuntimeError(
                                f"Evaluation is not evenly allocated across depths in "
                                f"{run_dir} at {step}"
                            )
                        depth_mean = sum(
                            float(evaluation[f"success_depth_{depth}"])
                            for depth in range(1, 7)
                        ) / 6.0
                        if not math.isclose(
                            float(evaluation["success_rate"]), depth_mean, abs_tol=1e-10
                        ):
                            raise RuntimeError(
                                f"Overall success is not equal-depth averaged in {run_dir} at {step}"
                            )
                        rows.append(
                            {
                                **evaluation,
                                "algorithm": algorithm,
                                "catalog_seed": catalog_seed,
                                "training_seed": training_seed,
                                "diversity": diversity,
                                "layout_count": diversity,
                                "topology_count": diversity,
                                "unique_layout_topology_pairs": diversity**2,
                                "run_cluster": f"{catalog_seed}:{training_seed}",
                            }
                        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _series(
    rows: list[dict[str, Any]], algorithm: str, diversity: int, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    selected = [
        row for row in rows
        if row["algorithm"] == algorithm and int(row["diversity"]) == diversity
    ]
    steps = np.array(sorted({int(row["phase_env_steps"]) for row in selected}))
    clusters = sorted({str(row["run_cluster"]) for row in selected})
    trajectories: list[np.ndarray] = []
    for cluster in clusters:
        lookup = {
            int(row["phase_env_steps"]): float(row[metric])
            for row in selected if row["run_cluster"] == cluster
        }
        if set(lookup) != set(steps.tolist()):
            raise RuntimeError(f"Misaligned curve for {algorithm}, n={diversity}, {cluster}")
        trajectories.append(np.array([lookup[int(step)] for step in steps]))
    matrix = np.stack(trajectories)
    mean = matrix.mean(axis=0)
    ci = 1.96 * matrix.std(axis=0, ddof=1) / math.sqrt(matrix.shape[0])
    return steps, mean, ci, trajectories


def _plot_metric(
    rows: list[dict[str, Any]],
    diversities: tuple[int, ...],
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    *,
    ylim: tuple[float, float] = (-0.02, 1.02),
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharex=True, sharey=True)
    for axis, algorithm in zip(axes, ALGORITHMS):
        for diversity in diversities:
            steps, mean, ci, trajectories = _series(rows, algorithm, diversity, metric)
            x = steps / 1_000_000.0
            for trajectory in trajectories:
                axis.plot(x, trajectory, color=COLORS[diversity], alpha=0.16, linewidth=0.8)
            axis.plot(
                x,
                mean,
                color=COLORS[diversity],
                linewidth=2.4,
                label=(
                    f"n={diversity} ({diversity**2} "
                    f"{'pair' if diversity == 1 else 'pairs'})"
                ),
            )
            axis.fill_between(x, mean - ci, mean + ci, color=COLORS[diversity], alpha=0.18)
        axis.set_title("Plain PPO" if algorithm == "ppo" else "PPO + CBP")
        axis.set_xlabel("Environment steps (millions)")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
    axes[0].set_ylabel(ylabel)
    axes[0].set_ylim(*ylim)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize(
    rows: list[dict[str, Any]],
    diversities: tuple[int, ...],
    target_steps: int,
    minimum_depth6: float,
) -> dict[str, Any]:
    endpoints = [row for row in rows if int(row["phase_env_steps"]) == target_steps]
    conditions: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        for diversity in diversities:
            selected = [
                row for row in endpoints
                if row["algorithm"] == algorithm and int(row["diversity"]) == diversity
            ]
            depth6 = np.array([float(row["success_depth_6"]) for row in selected])
            conditions.append(
                {
                    "algorithm": algorithm,
                    "diversity": diversity,
                    "unique_layout_topology_pairs": diversity**2,
                    "runs": len(selected),
                    "mean_overall_success": float(
                        np.mean([float(row["success_rate"]) for row in selected])
                    ),
                    "mean_depth6_success": float(depth6.mean()),
                    "depth6_success_by_run": depth6.tolist(),
                    "fraction_depth6_at_least_0.05": float(np.mean(depth6 >= 0.05)),
                    "mean_timeout_rate": float(
                        np.mean([float(row["timeout_rate"]) for row in selected])
                    ),
                    "mean_effective_manipulation_rate": float(
                        np.mean(
                            [float(row["effective_manipulation_rate"]) for row in selected]
                        )
                    ),
                }
            )
    primary = [row for row in conditions if row["algorithm"] == "ppo_cbp"]
    passed = all(
        row["mean_depth6_success"] >= minimum_depth6
        and row["fraction_depth6_at_least_0.05"] >= 0.75
        for row in primary
    )
    return {
        "status": "pass" if passed else "fail",
        "purpose": "single-distribution depth-6 learnability gate; launches no continual jobs",
        "criterion": (
            f"For PPO+CBP at every diversity, mean endpoint depth-6 success >= "
            f"{minimum_depth6:.2f} and at least 3/4 runs have depth-6 success >= 0.05"
        ),
        "minimum_mean_depth6_success": minimum_depth6,
        "conditions": conditions,
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Layout-topology learnability diagnostic",
        "",
        f"Gate status: **{summary['status'].upper()}**",
        "",
        summary["criterion"],
        "",
        "The gate is diagnostic only. No continual sweep is submitted regardless of its result.",
        "",
        "| Algorithm | n | Layout-topology pairs | Overall | Depth 6 | Timeout | Effective manipulation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["conditions"]:
        label = "PPO" if row["algorithm"] == "ppo" else "PPO + CBP"
        lines.append(
            f"| {label} | {row['diversity']} | {row['unique_layout_topology_pairs']} | "
            f"{row['mean_overall_success']:.3f} | {row['mean_depth6_success']:.3f} | "
            f"{row['mean_timeout_rate']:.3f} | {row['mean_effective_manipulation_rate']:.4f} |"
        )
    lines.extend(
        (
            "",
            "Overall success is the arithmetic mean of the six depth-specific success rates. "
            "Each curve point uses an equal episode allocation across depths, so shallow tasks "
            "cannot dominate by terminating faster.",
            "",
            "Individual faint trajectories are the four catalog-seed/training-seed run clusters; "
            "solid lines show their mean and bands show normal-approximation 95% intervals.",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/learnability/raw")
    parser.add_argument("--output", default="results/learnability/analysis")
    parser.add_argument("--config", default="configs/learnability.toml")
    parser.add_argument("--catalog-seeds", default=",".join(map(str, CATALOG_SEEDS)))
    parser.add_argument("--minimum-depth6", type=float, default=0.10)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    catalog_seeds = tuple(int(value) for value in args.catalog_seeds.split(","))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = collect(Path(args.root), config_path, catalog_seeds)
    _write_csv(output / "learning_curves.csv", rows)
    diversities = config.experiment.diversities
    _plot_metric(
        rows,
        diversities,
        "success_rate",
        "Success rate (equal mean over depths 1-6)",
        "Single-distribution learning over co-varied layouts and topologies",
        output / "learning_curves_overall.png",
    )
    _plot_metric(
        rows,
        diversities,
        "success_depth_6",
        "Depth-6 success rate",
        "Depth-6 learnability",
        output / "learning_curves_depth6.png",
    )
    _plot_metric(
        rows,
        diversities,
        "timeout_rate",
        "Timeout rate (equal mean over depths 1-6)",
        "Timeout behavior during learning",
        output / "learning_curves_timeout.png",
    )
    _plot_metric(
        rows,
        diversities,
        "effective_manipulation_rate",
        "Effective manipulation actions / environment step",
        "Effective pickup, drop, merge, and toggle behavior",
        output / "learning_curves_effective_manipulation.png",
        ylim=(-0.001, 0.2),
    )
    summary = summarize(
        rows,
        diversities,
        config.experiment.steps_per_distribution,
        args.minimum_depth6,
    )
    (output / "learnability_gate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(output / "report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
