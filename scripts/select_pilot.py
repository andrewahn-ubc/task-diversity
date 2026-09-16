"""Select pilot hyperparameters from all complete pilot trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from continual_config import DEFAULTS, VARIANTS


def score(records: list[dict]) -> dict:
    by_phase = {phase: sorted((r for r in records if r["phase"] == phase),
                              key=lambda r: r["phase_steps"]) for phase in (1, 2, 3)}
    if any(not rows for rows in by_phase.values()):
        raise ValueError("Pilot trajectory lacks one or more phases")
    terminal = by_phase[3][-1]
    auc_values = []
    for rows in by_phase.values():
        xs = np.asarray([r["phase_steps"] for r in rows], dtype=float)
        ys = np.asarray([r["all_depths"] for r in rows], dtype=float)
        auc_values.append(float(np.sum((ys[1:] + ys[:-1]) * np.diff(xs) / 2) /
                                max(1, xs[-1])))
    auc = np.mean(auc_values)
    return {"terminal_all": terminal["all_depths"],
            "terminal_depth6": terminal["depth6"], "phase_auc": float(auc),
            "score": .5 * terminal["all_depths"] + .3 * terminal["depth6"] + .2 * auc}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs/continual"))
    p.add_argument("--seeds", type=str, default="0,1")
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    seeds = [int(x) for x in args.seeds.split(",")]
    results = {}
    missing = []
    for variant in VARIANTS:
        runs = []
        for n in (4, 64):
            for seed in seeds:
                run_dir = root / "runs" / "pilot" / f"n{n:04d}" / variant / f"seed{seed}"
                progress_file = run_dir / "progress.json"
                metrics_file = run_dir / "metrics.jsonl"
                if not progress_file.exists() or not json.loads(progress_file.read_text())["complete"]:
                    missing.append(str(run_dir))
                    continue
                rows = [json.loads(line) for line in metrics_file.read_text().splitlines()]
                runs.append({"n": n, "seed": seed, **score(rows)})
        if runs:
            results[variant] = {"mean_score": float(np.mean([r["score"] for r in runs])),
                                "runs": runs}
    if missing and not args.allow_incomplete:
        raise SystemExit(f"Pilot incomplete ({len(missing)} runs); first missing: {missing[0]}")
    if not results:
        raise SystemExit("No complete pilot runs")
    best = max(results, key=lambda variant: results[variant]["mean_score"])
    hyperparameters = {key: DEFAULTS[key] for key in
                       ("learning_rate", "entropy_coef", "cbp_rate", "cbp_decay",
                        "cbp_maturity", "num_envs", "rollout_steps", "minibatches",
                        "update_epochs", "width", "max_steps")}
    hyperparameters.update(VARIANTS[best])
    output = {"selected_variant": best, "hyperparameters": hyperparameters,
              "selection_metric": ".5 terminal all-depth + .3 terminal depth-6 + .2 phase AUC",
              "scores": results, "missing": missing}
    path = root / "pilot_best.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Selected {best}; wrote {path}")


if __name__ == "__main__":
    main()
