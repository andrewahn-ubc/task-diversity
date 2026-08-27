from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .aggregate import COLORS, DIVERSITIES, SEEDS, mean_ci, read_jsonl, write_csv


GROUPS = ("shared", "policy", "value")
GROUP_LABELS = {
    "shared": "Shared conv + pre-GRU + GRU",
    "policy": "Policy head",
    "value": "Value head",
}
GROUP_COLORS = {
    "shared": "#0072B2",
    "policy": "#D55E00",
    "value": "#009E73",
}
SEED_COLORS = plt.get_cmap("tab10").colors[: len(SEEDS)]


def _single(records: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    matches = [
        record
        for record in records
        if all(record.get(key) == value for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one record for {criteria}, found {len(matches)}")
    return matches[0]


def collect_reanalysis(
    root: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    curve_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    for diversity in DIVERSITIES:
        for seed in SEEDS:
            run_dir = root / f"n{diversity}" / f"seed_{seed}"
            completed_path = run_dir / "completed.json"
            if not completed_path.exists():
                raise RuntimeError(f"Missing completed run: {run_dir}")
            completed = json.loads(completed_path.read_text(encoding="utf-8"))
            if completed.get("status") != "complete":
                raise RuntimeError(f"Run is not complete: {run_dir}")

            metrics = read_jsonl(run_dir / "metrics.jsonl")
            evaluations = [record for record in metrics if record.get("event") == "evaluation"]
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
                        **{
                            f"success_depth_{depth}": record[f"success_depth_{depth}"]
                            for depth in range(1, 7)
                        },
                    }
                )

            phase_evaluations: dict[int, dict[str, dict[str, Any]]] = {}
            phase_budget: int | None = None
            for phase in range(1, 5):
                start = _single(evaluations, phase=phase, kind="phase_start")
                end = _single(evaluations, phase=phase, kind="phase_end")
                if phase_budget is None:
                    phase_budget = int(end["phase_env_steps"])
                elif phase_budget != int(end["phase_env_steps"]):
                    raise RuntimeError(f"Inconsistent phase budget in {run_dir}")
                midpoint = _single(
                    evaluations,
                    phase=phase,
                    phase_env_steps=phase_budget // 2,
                    kind="periodic",
                )
                phase_evaluations[phase] = {
                    "start": start,
                    "mid": midpoint,
                    "end": end,
                }
                if phase >= 2:
                    previous_end = phase_evaluations[phase - 1]["end"]
                    transition_rows.append(
                        {
                            "diversity": diversity,
                            "seed": seed,
                            "target_phase": phase,
                            "transition": f"d{phase - 1}->d{phase}",
                            "previous_end_success": previous_end["success_rate"],
                            "start_success": start["success_rate"],
                            "end_success": end["success_rate"],
                            "transfer_gap": previous_end["success_rate"]
                            - start["success_rate"],
                            "specialization_gain": end["success_rate"]
                            - start["success_rate"],
                            "previous_end_depth6": previous_end["success_depth_6"],
                            "start_depth6": start["success_depth_6"],
                            "end_depth6": end["success_depth_6"],
                            "transfer_gap_depth6": previous_end["success_depth_6"]
                            - start["success_depth_6"],
                            "specialization_gain_depth6": end["success_depth_6"]
                            - start["success_depth_6"],
                        }
                    )

            diagnostics = read_jsonl(run_dir / "diagnostics.jsonl")
            for phase in range(2, 5):
                checkpoints = phase_evaluations[phase]
                for fraction, interval, start_key, end_key in (
                    (0.0, "start_to_mid", "start", "mid"),
                    (0.5, "mid_to_end", "mid", "end"),
                ):
                    diagnostic = _single(
                        diagnostics,
                        phase=phase,
                        phase_fraction=fraction,
                    )
                    start_eval = checkpoints[start_key]
                    end_eval = checkpoints[end_key]
                    interval_rows.append(
                        {
                            "diversity": diversity,
                            "seed": seed,
                            "run": f"n{diversity}_seed{seed}",
                            "phase": phase,
                            "interval": interval,
                            "predictor_fraction": fraction,
                            "start_success": start_eval["success_rate"],
                            "end_success": end_eval["success_rate"],
                            "interval_gain": end_eval["success_rate"]
                            - start_eval["success_rate"],
                            "start_depth6": start_eval["success_depth_6"],
                            "end_depth6": end_eval["success_depth_6"],
                            "interval_gain_depth6": end_eval["success_depth_6"]
                            - start_eval["success_depth_6"],
                            **{
                                key: diagnostic[key]
                                for group in ("all", *GROUPS)
                                for key in (
                                    f"cosine_{group}",
                                    f"norm_current_{group}",
                                    f"norm_previous_{group}",
                                )
                            },
                        }
                    )
    return curve_rows, transition_rows, interval_rows


def _clustered_ols(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    names: tuple[str, ...],
    *,
    intercept: bool,
) -> dict[str, Any]:
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Invalid regression arrays")
    if x.shape[1] != len(names):
        raise ValueError("Coefficient names do not match the design")
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("Regression design is rank deficient")
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    bread = np.linalg.inv(x.T @ x)
    unique_clusters = np.unique(clusters)
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    for cluster in unique_clusters:
        mask = clusters == cluster
        score = x[mask].T @ residual[mask]
        meat += np.outer(score, score)
    n, k = x.shape
    cluster_count = len(unique_clusters)
    correction = (cluster_count / (cluster_count - 1)) * ((n - 1) / (n - k))
    covariance = correction * bread @ meat @ bread
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    # Student-t(24) 0.975 quantile; every requested model has 25 run clusters.
    critical = 2.064 if cluster_count == 25 else 1.96
    coefficients = {}
    for index, name in enumerate(names):
        coefficients[name] = {
            "estimate": float(beta[index]),
            "cluster_robust_se": float(standard_error[index]),
            "ci95_low": float(beta[index] - critical * standard_error[index]),
            "ci95_high": float(beta[index] + critical * standard_error[index]),
        }
    return {
        "n_observations": int(n),
        "n_run_clusters": int(cluster_count),
        "run_clustered_cr1": True,
        "intercept": intercept,
        "coefficients": coefficients,
    }


def _regression_arrays(
    interval_rows: list[dict[str, Any]], group: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.column_stack(
        (
            np.ones(len(interval_rows)),
            np.asarray([row[f"cosine_{group}"] for row in interval_rows]),
            np.log2(np.asarray([row["diversity"] for row in interval_rows])),
            np.asarray([row["phase"] == 3 for row in interval_rows], dtype=float),
            np.asarray([row["phase"] == 4 for row in interval_rows], dtype=float),
            np.asarray(
                [row["interval"] == "mid_to_end" for row in interval_rows],
                dtype=float,
            ),
        )
    )
    y = np.asarray([row["interval_gain"] for row in interval_rows], dtype=float)
    clusters = np.asarray([row["run"] for row in interval_rows])
    return x, y, clusters


def _within_run_arrays(
    interval_rows: list[dict[str, Any]], group: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = np.column_stack(
        (
            np.asarray([row[f"cosine_{group}"] for row in interval_rows]),
            np.asarray([row["phase"] == 3 for row in interval_rows], dtype=float),
            np.asarray([row["phase"] == 4 for row in interval_rows], dtype=float),
            np.asarray(
                [row["interval"] == "mid_to_end" for row in interval_rows],
                dtype=float,
            ),
        )
    )
    y = np.asarray([row["interval_gain"] for row in interval_rows], dtype=float)
    clusters = np.asarray([row["run"] for row in interval_rows])
    centered_x = columns.copy()
    centered_y = y.copy()
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        centered_x[mask] -= centered_x[mask].mean(axis=0)
        centered_y[mask] -= centered_y[mask].mean()
    return centered_x, centered_y, clusters


def _partial_residuals(
    interval_rows: list[dict[str, Any]], group: str
) -> tuple[np.ndarray, np.ndarray]:
    clusters = [row["run"] for row in interval_rows]
    unique_runs = sorted(set(clusters))
    nuisance = np.column_stack(
        (
            *(
                np.asarray([run == candidate for run in clusters], dtype=float)
                for candidate in unique_runs
            ),
            np.asarray([row["phase"] == 3 for row in interval_rows], dtype=float),
            np.asarray([row["phase"] == 4 for row in interval_rows], dtype=float),
            np.asarray(
                [row["interval"] == "mid_to_end" for row in interval_rows],
                dtype=float,
            ),
        )
    )
    cosine = np.asarray([row[f"cosine_{group}"] for row in interval_rows], dtype=float)
    gain = np.asarray([row["interval_gain"] for row in interval_rows], dtype=float)
    cosine_residual = cosine - nuisance @ np.linalg.lstsq(nuisance, cosine, rcond=None)[0]
    gain_residual = gain - nuisance @ np.linalg.lstsq(nuisance, gain, rcond=None)[0]
    return cosine_residual, gain_residual


def summarize_reanalysis(
    transition_rows: list[dict[str, Any]], interval_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    transitions: dict[str, Any] = {}
    for phase in range(2, 5):
        phase_summary: dict[str, Any] = {}
        for diversity in DIVERSITIES:
            rows = [
                row
                for row in transition_rows
                if row["target_phase"] == phase and row["diversity"] == diversity
            ]
            transfer_mean, transfer_ci = mean_ci(row["transfer_gap"] for row in rows)
            gain_mean, gain_ci = mean_ci(row["specialization_gain"] for row in rows)
            phase_summary[str(diversity)] = {
                "mean_transfer_gap": transfer_mean,
                "ci95_transfer_gap": transfer_ci,
                "mean_specialization_gain": gain_mean,
                "ci95_specialization_gain": gain_ci,
            }
        transitions[f"d{phase - 1}->d{phase}"] = phase_summary

    gradient_groups: dict[str, Any] = {}
    for group in GROUPS:
        per_diversity: dict[str, Any] = {}
        for diversity in DIVERSITIES:
            per_seed = [
                np.mean(
                    [
                        row[f"cosine_{group}"]
                        for row in interval_rows
                        if row["diversity"] == diversity and row["seed"] == seed
                    ]
                )
                for seed in SEEDS
            ]
            mean, ci = mean_ci(per_seed)
            per_seed_conflict_fraction = [
                np.mean(
                    [
                        row[f"cosine_{group}"] < 0
                        for row in interval_rows
                        if row["diversity"] == diversity and row["seed"] == seed
                    ]
                )
                for seed in SEEDS
            ]
            conflict_mean, conflict_ci = mean_ci(per_seed_conflict_fraction)
            per_seed_dot_product = [
                np.mean(
                    [
                        row[f"cosine_{group}"]
                        * row[f"norm_current_{group}"]
                        * row[f"norm_previous_{group}"]
                        for row in interval_rows
                        if row["diversity"] == diversity and row["seed"] == seed
                    ]
                )
                for seed in SEEDS
            ]
            dot_mean, dot_ci = mean_ci(per_seed_dot_product)
            per_diversity[str(diversity)] = {
                "mean_cosine": mean,
                "ci95_cosine": ci,
                "mean_conflict_fraction": conflict_mean,
                "ci95_conflict_fraction": conflict_ci,
                "mean_signed_dot_product": dot_mean,
                "ci95_signed_dot_product": dot_ci,
            }

        pooled_x, pooled_y, clusters = _regression_arrays(interval_rows, group)
        pooled = _clustered_ols(
            pooled_x,
            pooled_y,
            clusters,
            (
                "intercept",
                "cosine",
                "log2_diversity",
                "phase_d3",
                "phase_d4",
                "mid_to_end",
            ),
            intercept=True,
        )
        within_x, within_y, within_clusters = _within_run_arrays(interval_rows, group)
        fixed_effects = _clustered_ols(
            within_x,
            within_y,
            within_clusters,
            ("cosine", "phase_d3", "phase_d4", "mid_to_end"),
            intercept=False,
        )
        gradient_groups[group] = {
            "label": GROUP_LABELS[group],
            "cosine_by_diversity": per_diversity,
            "pooled_adjusted_regression": pooled,
            "within_run_fixed_effects_regression": fixed_effects,
        }
    return {
        "analysis_type": "post_hoc_reanalysis",
        "outcome": "main evaluation success-rate change over the following half phase",
        "predictor_checkpoints": ["phase_start", "phase_midpoint"],
        "endpoint_predictors_excluded": True,
        "n_runs": len(DIVERSITIES) * len(SEEDS),
        "n_intervals": len(interval_rows),
        "transition_effects": transitions,
        "gradient_groups": gradient_groups,
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


def _curve_points(
    curve_rows: list[dict[str, Any]], diversity: int, seed: int, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    steps: list[int] = []
    values: list[float] = []
    for phase in range(1, 5):
        rows = [
            row
            for row in curve_rows
            if row["diversity"] == diversity
            and row["seed"] == seed
            and row["phase"] == phase
        ]
        start = _single(rows, kind="phase_start")
        end = _single(rows, kind="phase_end")
        ordered = [start] + sorted(
            [
                row
                for row in rows
                if row["kind"] == "periodic"
                and row["phase_env_steps"] < end["phase_env_steps"]
            ],
            key=lambda row: row["phase_env_steps"],
        ) + [end]
        steps.extend(int(row["total_env_steps"]) for row in ordered)
        values.extend(float(row[metric]) for row in ordered)
    return np.asarray(steps, dtype=float) / 1e6, np.asarray(values)


def make_figures(
    curve_rows: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
    interval_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    figures: Path,
) -> None:
    _style()
    phase_budget = max(row["total_env_steps"] for row in curve_rows) / 4.0

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    depth6_upper_bounds: list[float] = []
    for diversity in DIVERSITIES:
        seed_curves = [
            _curve_points(curve_rows, diversity, seed, "success_depth_6")
            for seed in SEEDS
        ]
        steps = seed_curves[0][0]
        if any(not np.array_equal(candidate[0], steps) for candidate in seed_curves[1:]):
            raise RuntimeError(f"Depth-6 curve steps are not aligned for n={diversity}")
        stacked = np.vstack([candidate[1] for candidate in seed_curves])
        means = stacked.mean(axis=0)
        cis = np.asarray([mean_ci(stacked[:, index])[1] for index in range(stacked.shape[1])])
        depth6_upper_bounds.extend((means + cis).tolist())
        ax.plot(steps, means, color=COLORS[diversity], label=f"n={diversity}", lw=2)
        ax.fill_between(steps, means - cis, means + cis, color=COLORS[diversity], alpha=0.16)
    for boundary in range(1, 4):
        ax.axvline(boundary * phase_budget / 1e6, color="0.35", ls="--", lw=1)
    depth6_upper = max(0.06, 1.1 * max(depth6_upper_bounds))
    ax.set(
        xlabel="Environment steps (millions)",
        ylabel="Depth-6 evaluation success rate",
        ylim=(-0.002, depth6_upper),
        title="Continual learning on depth-6 tasks only (zoomed y-axis)",
    )
    ax.legend(frameon=False, ncol=len(DIVERSITIES))
    _save(fig, figures, "figure6_learning_curves_depth6")

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0), sharex=True)
    for column, phase in enumerate(range(2, 5)):
        for row_index, (metric, label) in enumerate(
            (
                ("transfer_gap", "Transfer gap (lower is better)"),
                ("specialization_gain", "Within-distribution gain"),
            )
        ):
            axis = axes[row_index, column]
            means, cis = [], []
            for diversity in DIVERSITIES:
                values = [
                    row[metric]
                    for row in transition_rows
                    if row["target_phase"] == phase and row["diversity"] == diversity
                ]
                mean, ci = mean_ci(values)
                means.append(mean)
                cis.append(ci)
            positions = np.arange(len(DIVERSITIES))
            axis.bar(
                positions,
                means,
                yerr=cis,
                capsize=3,
                color=[COLORS[diversity] for diversity in DIVERSITIES],
                alpha=0.85,
            )
            axis.axhline(0, color="0.25", lw=0.8)
            axis.set_xticks(positions, [str(diversity) for diversity in DIVERSITIES])
            axis.set_title(f"d{phase - 1} -> d{phase}")
            if column == 0:
                axis.set_ylabel(label)
            if row_index == 1:
                axis.set_xlabel("Topology diversity n")
    fig.suptitle("Each distribution transition shown separately (mean across seeds, 95% CI)")
    fig.subplots_adjust(top=0.90)
    _save(fig, figures, "figure7_transition_metrics")

    fig, axes = plt.subplots(3, 2, figsize=(10.0, 9.0), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    for panel, diversity in enumerate(DIVERSITIES):
        axis = axes_flat[panel]
        for seed, color in zip(SEEDS, SEED_COLORS):
            steps, values = _curve_points(curve_rows, diversity, seed, "success_rate")
            axis.plot(steps, values, color=color, lw=1.4, alpha=0.9, label=f"seed {seed}")
        for boundary in range(1, 4):
            axis.axvline(boundary * phase_budget / 1e6, color="0.45", ls="--", lw=0.8)
        axis.set_title(f"n={diversity}")
        axis.set_ylim(-0.02, 1.02)
    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.93, 0.08), frameon=False)
    fig.supxlabel("Environment steps (millions)")
    fig.supylabel("Overall evaluation success rate")
    fig.suptitle("Individual training trajectories")
    fig.subplots_adjust(top=0.94, bottom=0.08)
    _save(fig, figures, "figure8_individual_seed_learning_curves")

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
    positions = np.arange(len(DIVERSITIES))
    offsets = dict(zip(GROUPS, (-0.12, 0.0, 0.12)))
    for axis, metric, ci_metric, title in zip(
        axes,
        ("mean_cosine", "mean_conflict_fraction", "mean_signed_dot_product"),
        ("ci95_cosine", "ci95_conflict_fraction", "ci95_signed_dot_product"),
        ("Mean cosine", "Fraction with cosine < 0", "Mean signed gradient dot product"),
    ):
        for group in GROUPS:
            group_summary = summary["gradient_groups"][group]["cosine_by_diversity"]
            means = [group_summary[str(diversity)][metric] for diversity in DIVERSITIES]
            cis = [group_summary[str(diversity)][ci_metric] for diversity in DIVERSITIES]
            axis.errorbar(
                positions + offsets[group],
                means,
                yerr=cis,
                marker="o",
                color=GROUP_COLORS[group],
                capsize=3,
                lw=1.8,
                label=GROUP_LABELS[group],
            )
        axis.axhline(0, color="0.25", lw=0.8)
        axis.set_xticks(positions, [str(diversity) for diversity in DIVERSITIES])
        axis.set_xlabel("Topology diversity n")
        axis.set_title(title)
    axes[0].set_ylabel("Alignment / conflict metric")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=3, frameon=False)
    fig.suptitle("Gradient conflict by parameter group (start/mid predictors only)", y=0.995)
    fig.subplots_adjust(top=0.76)
    _save(fig, figures, "figure9_gradient_cosine_by_group")

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), sharey=True)
    for axis, group in zip(axes, GROUPS):
        cosine_residual, gain_residual = _partial_residuals(interval_rows, group)
        for diversity in DIVERSITIES:
            mask = np.asarray([row["diversity"] == diversity for row in interval_rows])
            axis.scatter(
                cosine_residual[mask],
                gain_residual[mask],
                color=COLORS[diversity],
                alpha=0.55,
                s=20,
                label=f"n={diversity}",
            )
        fixed_effect = summary["gradient_groups"][group][
            "within_run_fixed_effects_regression"
        ]["coefficients"]["cosine"]
        line = np.linspace(cosine_residual.min(), cosine_residual.max(), 100)
        axis.plot(line, fixed_effect["estimate"] * line, color="0.15", ls="--", lw=1.5)
        axis.axhline(0, color="0.4", lw=0.8)
        axis.axvline(0, color="0.4", lw=0.8)
        axis.set_title(GROUP_LABELS[group])
        axis.set_xlabel("Residualized gradient cosine")
        axis.text(
            0.03,
            0.97,
            f"within-run slope={fixed_effect['estimate']:.3f}\n"
            f"95% CI [{fixed_effect['ci95_low']:.3f}, {fixed_effect['ci95_high']:.3f}]",
            transform=axis.transAxes,
            va="top",
            fontsize=8,
        )
    axes[0].set_ylabel("Residualized following half-phase gain")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=len(DIVERSITIES),
        frameon=False,
    )
    fig.suptitle(
        "Run-fixed-effects association after adjusting for phase and interval",
        y=0.995,
    )
    fig.subplots_adjust(top=0.78)
    _save(fig, figures, "figure10_group_conflict_interval_gain")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Post-hoc corrected analysis",
        "",
        "This analysis uses the already completed 25 runs; no trajectory was retrained. The original overall learning curve pooled completed evaluation episodes from depths 1 through 6. Figure 6 instead conditions on depth 6 only.",
        "",
        "## Corrected temporal diagnostic",
        "",
        "Only start and midpoint gradient measurements are predictors. Start cosine predicts start-to-midpoint success-rate change, and midpoint cosine predicts midpoint-to-end change. Endpoint predictors are excluded. Outcomes use the main 1,024-episode evaluations rather than the smaller diagnostic evaluations.",
        "",
        "Each pooled model adjusts for log2 diversity, distribution phase, and half-phase interval and uses CR1 standard errors clustered by the 25 training runs. The fixed-effects model additionally removes every run's mean, testing whether within-run changes in cosine covary with within-run changes in subsequent learning.",
        "",
        "## Gradient-group results",
        "",
        "| Parameter group | Pooled adjusted cosine slope (95% clustered CI) | Within-run cosine slope (95% clustered CI) |",
        "|---|---:|---:|",
    ]
    for group in GROUPS:
        item = summary["gradient_groups"][group]
        pooled = item["pooled_adjusted_regression"]["coefficients"]["cosine"]
        fixed = item["within_run_fixed_effects_regression"]["coefficients"]["cosine"]
        lines.append(
            f"| {item['label']} | {pooled['estimate']:.3f} "
            f"[{pooled['ci95_low']:.3f}, {pooled['ci95_high']:.3f}] | "
            f"{fixed['estimate']:.3f} [{fixed['ci95_low']:.3f}, {fixed['ci95_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Conflict frequency by diversity",
            "",
            "| Group | n | Mean cosine | Fraction cosine < 0 | Mean signed dot product |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for group in GROUPS:
        item = summary["gradient_groups"][group]
        for diversity in DIVERSITIES:
            condition = item["cosine_by_diversity"][str(diversity)]
            lines.append(
                f"| {item['label']} | {diversity} | {condition['mean_cosine']:.3f} | "
                f"{condition['mean_conflict_fraction']:.3f} | "
                f"{condition['mean_signed_dot_product']:.6f} |"
            )
    lines.extend(
        [
            "",
            "A positive slope means more aligned current-versus-previous gradients predict greater subsequent learning. A negative slope means greater alignment predicts less subsequent learning. Confidence intervals spanning zero do not provide a stable directional association.",
            "",
            "## Generated artifacts",
            "",
            "- `figure6_learning_curves_depth6`: depth-6-only mean learning curves.",
            "- `figure7_transition_metrics`: every switch and following within-distribution gain separately.",
            "- `figure8_individual_seed_learning_curves`: all 25 trajectories without seed averaging.",
            "- `figure9_gradient_cosine_by_group`: shared, policy-head, and value-head cosine by diversity using valid predictor checkpoints only.",
            "- `figure10_group_conflict_interval_gain`: run-fixed-effects, temporally aligned association plots.",
            "",
            "This is a post-hoc reanalysis. It improves temporal alignment and dependence handling but remains observational and cannot establish causality.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/raw-cbp")
    parser.add_argument("--output", default="results/cbp/reanalysis")
    args = parser.parse_args()
    output = Path(args.output)
    curve_rows, transition_rows, interval_rows = collect_reanalysis(Path(args.root))
    summary = summarize_reanalysis(transition_rows, interval_rows)
    write_csv(output / "aggregated" / "learning_curves_by_depth.csv", curve_rows)
    write_csv(output / "aggregated" / "transition_metrics.csv", transition_rows)
    write_csv(output / "aggregated" / "aligned_interval_metrics.csv", interval_rows)
    summary_path = output / "aggregated" / "corrected_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_figures(curve_rows, transition_rows, interval_rows, summary, output / "figures")
    write_report(output / "report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
