from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DIVERSITIES = (1, 16, 256)
SEEDS = (0, 1, 2, 3, 4)
COLORS = {1: "#D55E00", 16: "#0072B2", 256: "#009E73"}
MARKERS = {1: "o", 16: "s", 256: "^"}
SEED_MARKERS = {0: "o", 1: "s", 2: "^", 3: "D", 4: "P"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty table {path}")
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean_ci(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan")
    if len(array) == 1:
        return float(array[0]), 0.0
    # Exact 0.975 Student-t critical values for the seed counts used here.
    critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(len(array) - 1, 1.96)
    return float(array.mean()), float(critical * array.std(ddof=1) / math.sqrt(len(array)))


def _single(records: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    matches = [
        record for record in records
        if all(record.get(key) == value for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one record for {criteria}, found {len(matches)}")
    return matches[0]


def collect(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    phase_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for diversity in DIVERSITIES:
        for seed in SEEDS:
            run_dir = root / f"n{diversity}" / f"seed_{seed}"
            completed_path = run_dir / "completed.json"
            if not completed_path.exists():
                raise RuntimeError(f"Missing completed run: {run_dir}")
            completed = json.loads(completed_path.read_text(encoding="utf-8"))
            replacement_totals = completed.get("cbp_total_replacements")
            expected_layers = {"conv1", "conv2", "pre_gru", "gru"}
            if completed.get("algorithm") != "ppo_cbp" or (
                not isinstance(replacement_totals, dict)
                or set(replacement_totals) != expected_layers
                or any(int(replacement_totals[layer]) <= 0 for layer in expected_layers)
            ):
                raise RuntimeError(f"Run is not a verified PPO + CBP result: {run_dir}")
            metrics = read_jsonl(run_dir / "metrics.jsonl")
            evaluations = [record for record in metrics if record.get("event") == "evaluation"]
            backward = [record for record in metrics if record.get("event") == "backward_evaluation"]
            phase_ends: dict[int, float] = {}
            for phase in range(1, 5):
                start = _single(evaluations, phase=phase, kind="phase_start")
                end = _single(evaluations, phase=phase, kind="phase_end")
                phase_ends[phase] = end["success_rate"]
                transfer = float("nan") if phase == 1 else phase_ends[phase - 1] - start["success_rate"]
                phase_rows.append(
                    {
                        "diversity": diversity,
                        "seed": seed,
                        "phase": phase,
                        "start_success": start["success_rate"],
                        "end_success": end["success_rate"],
                        "transfer_gap": transfer,
                        "specialization_gain": end["success_rate"] - start["success_rate"],
                    }
                )
            for record in evaluations:
                curve_rows.append(
                    {
                        "diversity": diversity,
                        "seed": seed,
                        "phase": record["phase"],
                        "kind": record["kind"],
                        "phase_env_steps": record["phase_env_steps"],
                        "total_env_steps": record["total_env_steps"],
                        "success_rate": record["success_rate"],
                    }
                )
            backward_by_phase = {
                record["after_phase"]: record["success_rate"] for record in backward
            }
            baseline = backward_by_phase[1]
            for phase in range(1, 5):
                phase_rows[(len(phase_rows) - 4) + phase - 1]["d1_success_after_phase"] = backward_by_phase[phase]
                phase_rows[(len(phase_rows) - 4) + phase - 1]["backward_d1"] = (
                    backward_by_phase[phase] - baseline
                )
            diagnostics = read_jsonl(run_dir / "diagnostics.jsonl")
            for record in diagnostics:
                row = dict(record)
                row["subsequent_specialization_gain"] = (
                    phase_ends[record["phase"]] - record["checkpoint_success_rate"]
                )
                diagnostic_rows.append(row)
    return phase_rows, diagnostic_rows, curve_rows


def summarize(
    phase_rows: list[dict[str, Any]], diagnostic_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for diversity in DIVERSITIES:
        later = [
            row for row in phase_rows if row["diversity"] == diversity and row["phase"] >= 2
        ]
        per_seed_transfer = [
            np.mean([row["transfer_gap"] for row in later if row["seed"] == seed])
            for seed in SEEDS
        ]
        per_seed_specialization = [
            np.mean([row["specialization_gain"] for row in later if row["seed"] == seed])
            for seed in SEEDS
        ]
        transfer_mean, transfer_ci = mean_ci(per_seed_transfer)
        spec_mean, spec_ci = mean_ci(per_seed_specialization)
        cosines = [
            np.mean(
                [
                    row["cosine_all"] for row in diagnostic_rows
                    if row["diversity"] == diversity and row["seed"] == seed
                ]
            )
            for seed in SEEDS
        ]
        cosine_mean, cosine_ci = mean_ci(cosines)
        conditions[str(diversity)] = {
            "mean_transfer_gap_d2_d4": transfer_mean,
            "ci95_transfer_gap_d2_d4": transfer_ci,
            "mean_specialization_gain_d2_d4": spec_mean,
            "ci95_specialization_gain_d2_d4": spec_ci,
            "mean_gradient_cosine": cosine_mean,
            "ci95_gradient_cosine": cosine_ci,
        }
    cosine = np.asarray([row["cosine_all"] for row in diagnostic_rows], dtype=float)
    subsequent = np.asarray(
        [row["subsequent_specialization_gain"] for row in diagnostic_rows], dtype=float
    )
    valid = np.isfinite(cosine) & np.isfinite(subsequent)
    correlation = float(np.corrcoef(cosine[valid], subsequent[valid])[0, 1])
    design = np.column_stack(
        (
            np.ones(valid.sum()),
            cosine[valid],
            np.log2(
                np.asarray([row["diversity"] for row in diagnostic_rows], dtype=float)[valid]
            ),
        )
    )
    coefficients = np.linalg.lstsq(design, subsequent[valid], rcond=None)[0]
    low, middle, high = (conditions[str(value)] for value in DIVERSITIES)
    transfer_order = high["mean_transfer_gap_d2_d4"] < low["mean_transfer_gap_d2_d4"]
    specialization_plateau = (
        high["mean_specialization_gain_d2_d4"]
        < middle["mean_specialization_gain_d2_d4"]
    )
    interference_order = high["mean_gradient_cosine"] < middle["mean_gradient_cosine"]
    return {
        "conditions": conditions,
        "interference_specialization_correlation": correlation,
        "regression_subsequent_gain_on_cosine_and_log2_diversity": {
            "intercept": float(coefficients[0]),
            "cosine_coefficient": float(coefficients[1]),
            "log2_diversity_coefficient": float(coefficients[2]),
            "n": int(valid.sum()),
        },
        "criteria": {
            "higher_diversity_improves_transfer": bool(transfer_order),
            "n256_specializes_less_than_n16": bool(specialization_plateau),
            "n256_has_lower_cosine_than_n16": bool(interference_order),
            "reduced_tradeoff_reproduced": bool(transfer_order and specialization_plateau),
            "interference_association_supported": bool(interference_order and correlation > 0),
        },
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, figures: Path, stem: str) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / f"{stem}.png")
    fig.savefig(figures / f"{stem}.pdf")
    plt.close(fig)


def make_figures(
    phase_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
    figures: Path,
) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for diversity in DIVERSITIES:
        points: list[tuple[int, int, str]] = []
        for phase in range(1, 5):
            phase_rows_for_curve = [
                row for row in curve_rows
                if row["diversity"] == diversity and row["phase"] == phase
            ]
            start = next(row for row in phase_rows_for_curve if row["kind"] == "phase_start")
            end = next(row for row in phase_rows_for_curve if row["kind"] == "phase_end")
            points.append((phase, start["total_env_steps"], "phase_start"))
            periodic_steps = sorted(
                {
                    row["total_env_steps"] for row in phase_rows_for_curve
                    if row["kind"] == "periodic"
                    and row["phase_env_steps"] < end["phase_env_steps"]
                }
            )
            points.extend((phase, step, "periodic") for step in periodic_steps)
            points.append((phase, end["total_env_steps"], "phase_end"))
        means, cis, steps = [], [], []
        for phase, step, kind in points:
            values = [
                row["success_rate"] for row in curve_rows
                if row["diversity"] == diversity and row["phase"] == phase
                and row["total_env_steps"] == step and row["kind"] == kind
            ]
            mean, ci = mean_ci(values)
            means.append(mean)
            cis.append(ci)
            steps.append(step)
        x = np.asarray(steps) / 1e6
        means_array, ci_array = np.asarray(means), np.asarray(cis)
        ax.plot(x, means_array, color=COLORS[diversity], label=f"n={diversity}", lw=2)
        ax.fill_between(x, means_array - ci_array, means_array + ci_array, color=COLORS[diversity], alpha=0.16)
    phase_budget = max(row["total_env_steps"] for row in curve_rows) / 4.0
    for boundary in range(1, 4):
        ax.axvline(boundary * phase_budget / 1e6, color="0.35", ls="--", lw=1)
    ax.set(xlabel="Environment steps (millions)", ylabel="Evaluation success rate", ylim=(-0.02, 1.02))
    ax.legend(frameon=False, ncol=3)
    ax.set_title("Continual learning across four topology distributions")
    _save(fig, figures, "figure1_learning_curves")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.7), sharex=True)
    for axis, metric, title in zip(
        axes,
        ("transfer_gap", "specialization_gain"),
        ("Forward-transfer gap (d2-d4)", "Within-distribution gain (d2-d4)"),
    ):
        means, cis = [], []
        for diversity in DIVERSITIES:
            per_seed = [
                np.mean(
                    [row[metric] for row in phase_rows if row["diversity"] == diversity and row["seed"] == seed and row["phase"] >= 2]
                )
                for seed in SEEDS
            ]
            mean, ci = mean_ci(per_seed)
            means.append(mean)
            cis.append(ci)
        axis.bar(range(3), means, yerr=cis, capsize=4, color=[COLORS[n] for n in DIVERSITIES], alpha=0.85)
        axis.axhline(0, color="0.25", lw=0.8)
        axis.set_xticks(range(3), [str(n) for n in DIVERSITIES])
        axis.set_title(title)
        axis.set_xlabel("Topology diversity n")
    axes[0].set_ylabel("Success-rate difference (mean and 95% CI)")
    _save(fig, figures, "figure2_transfer_specialization")

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), sharey=True)
    for phase, axis in zip((2, 3, 4), axes):
        for diversity in DIVERSITIES:
            rows = [row for row in diagnostic_rows if row["phase"] == phase and row["diversity"] == diversity]
            for fraction in (0.0, 0.5, 1.0):
                values = [row["cosine_all"] for row in rows if math.isclose(row["phase_fraction"], fraction)]
                mean, ci = mean_ci(values)
                x = fraction + {1: -0.025, 16: 0.0, 256: 0.025}[diversity]
                axis.errorbar(x, mean, yerr=ci, color=COLORS[diversity], marker=MARKERS[diversity], capsize=3, label=f"n={diversity}")
        axis.axhline(0, color="0.25", lw=0.8)
        axis.set_xticks((0.0, 0.5, 1.0), ("start", "mid", "end"))
        axis.set_title(f"Distribution d{phase}")
    axes[0].set_ylabel("Current-vs-previous gradient cosine")
    handles, labels = axes[-1].get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    fig.legend(
        [unique[f"n={diversity}"] for diversity in DIVERSITIES],
        [f"n={diversity}" for diversity in DIVERSITIES],
        loc="upper center", ncol=3, frameon=False,
    )
    fig.subplots_adjust(top=0.78)
    _save(fig, figures, "figure3_gradient_interference")

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for diversity in DIVERSITIES:
        for seed in SEEDS:
            rows = [
                row for row in diagnostic_rows
                if row["diversity"] == diversity and row["seed"] == seed
            ]
            ax.scatter(
                [row["cosine_all"] for row in rows],
                [row["subsequent_specialization_gain"] for row in rows],
                c=COLORS[diversity], marker=SEED_MARKERS[seed], alpha=0.75,
            )
    x = np.asarray([row["cosine_all"] for row in diagnostic_rows], dtype=float)
    y = np.asarray([row["subsequent_specialization_gain"] for row in diagnostic_rows], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    slope, intercept = np.polyfit(x[valid], y[valid], 1)
    line = np.linspace(x[valid].min(), x[valid].max(), 100)
    ax.plot(line, intercept + slope * line, color="0.15", lw=1.5, ls="--")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set(xlabel="Gradient cosine at checkpoint", ylabel="Subsequent phase specialization gain")
    diversity_handles = [
        plt.Line2D([], [], color=COLORS[diversity], marker="o", linestyle="", label=f"n={diversity}")
        for diversity in DIVERSITIES
    ]
    seed_handles = [
        plt.Line2D([], [], color="0.35", marker=SEED_MARKERS[seed], linestyle="", label=str(seed))
        for seed in SEEDS
    ]
    first_legend = ax.legend(handles=diversity_handles, title="Diversity", frameon=False, loc="upper left")
    ax.add_artist(first_legend)
    ax.legend(handles=seed_handles, title="Seed", frameon=False, loc="lower right", ncol=2)
    _save(fig, figures, "figure4_interference_specialization")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for diversity in DIVERSITIES:
        means, cis = [], []
        for phase in range(1, 5):
            values = [row["backward_d1"] for row in phase_rows if row["diversity"] == diversity and row["phase"] == phase]
            mean, ci = mean_ci(values)
            means.append(mean)
            cis.append(ci)
        ax.errorbar(range(1, 5), means, yerr=cis, color=COLORS[diversity], marker=MARKERS[diversity], capsize=3, lw=2, label=f"n={diversity}")
    ax.axhline(0, color="0.25", lw=0.8)
    ax.set_xticks(range(1, 5))
    ax.set(xlabel="After training distribution", ylabel="B(i,1): change in d1 success")
    ax.legend(frameon=False)
    _save(fig, figures, "figure5_backward_d1")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    conditions = summary["conditions"]
    criteria = summary["criteria"]
    tradeoff = "replicated" if criteria["reduced_tradeoff_reproduced"] else "did not replicate"
    interference = "supported" if criteria["interference_association_supported"] else "not supported"
    lines = [
        "# Reduced Banyan topology-diversity pilot report",
        "",
        "## Result",
        "",
        f"Under the preregistered qualitative criteria, the reduced topology-only tradeoff **{tradeoff}**. "
        f"The gradient-interference association was **{interference}**. These are diagnostic associations, not causal claims.",
        "",
        "## Exact setup",
        "",
        "A recurrent PPO + Continual Backprop (CBP) agent trained sequentially on four disjoint topology distributions at n = 1, 16, and 256, with five matched seeds. Each phase used 50,331,648 environment steps. The layout, deterministic object grounding procedure, architecture, optimizer, CBP configuration, evaluation budget, and diagnostic schedule were held fixed. Success was measured on independent evaluation episodes.",
        "",
        "The official Banyan code was not public when this repository was created. The CBP generate-and-test rule and element-wise Adam-state reset are adapted from Dohare et al.'s official implementation. This repository applies them to every learned feature layer and adds a documented GRU extension. A plateau that persists under this intervention is less consistent with conventional loss of plasticity, although CBP cannot logically eliminate every plasticity-related explanation.",
        "",
        "## Effect sizes (mean across seeds, 95% CI)",
        "",
        "| n | Transfer gap d2-d4 | Specialization gain d2-d4 | Gradient cosine |",
        "|---:|---:|---:|---:|",
    ]
    for diversity in DIVERSITIES:
        item = conditions[str(diversity)]
        lines.append(
            f"| {diversity} | {item['mean_transfer_gap_d2_d4']:.3f} +/- {item['ci95_transfer_gap_d2_d4']:.3f} | "
            f"{item['mean_specialization_gain_d2_d4']:.3f} +/- {item['ci95_specialization_gain_d2_d4']:.3f} | "
            f"{item['mean_gradient_cosine']:.3f} +/- {item['ci95_gradient_cosine']:.3f} |"
        )
    regression = summary["regression_subsequent_gain_on_cosine_and_log2_diversity"]
    lines.extend(
        [
            "",
            "## Interference diagnostic",
            "",
            f"The pooled correlation between gradient cosine and subsequent specialization gain was {summary['interference_specialization_correlation']:.3f}. "
            f"In the simple regression controlling for log2 diversity, the cosine coefficient was {regression['cosine_coefficient']:.3f} (n = {regression['n']} checkpoints). No inferential p-value is reported because checkpoints within a run are not independent.",
            "",
            "## Backward performance",
            "",
            "Figure 5 reports B(i,1) for every condition. Positive values mean later training improved performance on d1; negative values mean forgetting. This comparison distinguishes a current-distribution specialization plateau from a total halt in learning.",
            "",
            "## Limitations",
            "",
            "- The environment is a documented reduced reconstruction because the authors' implementation and full hyperparameters were unavailable.",
            "- Only topology diversity is varied, the sequence has four rather than ten distributions, and the phase budget is approximately half the paper's 100M steps.",
            "- The official CBP code supports feed-forward networks; the GRU feature-block extension here is necessary for the recurrent policy but has not been validated by the Banyan or CBP authors.",
            "- CBP actively mitigates loss of plasticity but cannot prove that every residual plateau is caused only by gradient interference.",
            "- Gradient conflict is observational; it cannot establish that interference causes stalled specialization.",
            "",
            "## Recommendation",
            "",
            "If the tradeoff is present and stable across seeds, next run one targeted causal intervention that reduces measured conflict without changing task diversity. If it is absent, first test a longer sequence or jointly vary layouts and topologies; do not tune the current pilot after seeing this result.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/raw-cbp-50m")
    parser.add_argument("--output", default="results/cbp-50m")
    args = parser.parse_args()
    output = Path(args.output)
    phase_rows, diagnostic_rows, curve_rows = collect(Path(args.root))
    write_csv(output / "aggregated" / "phase_metrics.csv", phase_rows)
    write_csv(output / "aggregated" / "interference_metrics.csv", diagnostic_rows)
    write_csv(output / "aggregated" / "learning_curves.csv", curve_rows)
    summary = summarize(phase_rows, diagnostic_rows)
    summary_path = output / "aggregated" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, summary_path)
    make_figures(phase_rows, diagnostic_rows, curve_rows, output / "figures")
    write_report(output / "report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
