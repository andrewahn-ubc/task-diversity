from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .config import load_config


def _records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/raw-cbp-50m")
    parser.add_argument("--output", default="results/raw-cbp-50m/smoke-gate.json")
    parser.add_argument("--config", default="configs/main.toml")
    parser.add_argument("--minimum-gain", type=float, default=0.03)
    args = parser.parse_args()
    config = load_config(args.config)
    expected_fingerprint = config.fingerprint()
    if not config.cbp.enabled:
        raise RuntimeError("The production smoke gate requires CBP to be enabled")
    root = Path(args.root)
    details: list[dict[str, Any]] = []
    for diversity in (1, 16, 256):
        run_dir = root / f"n{diversity}" / "seed_0"
        marker_path = run_dir / "pilot_complete.json"
        if not marker_path.exists():
            raise RuntimeError(f"First-phase pilot is incomplete: {run_dir}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        run_metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if (
            marker.get("config_fingerprint") != expected_fingerprint
            or run_metadata.get("config_fingerprint") != expected_fingerprint
        ):
            raise RuntimeError(f"Stale or mismatched pilot configuration in {run_dir}")
        if marker.get("algorithm") != "ppo_cbp":
            raise RuntimeError(f"Expected PPO + CBP pilot metadata in {marker_path}")
        if marker.get("completed_distributions") != 1:
            raise RuntimeError(f"Expected exactly one completed pilot phase in {marker_path}")
        if marker.get("total_env_steps") != config.experiment.steps_per_distribution:
            raise RuntimeError(f"Pilot did not use the fixed full-phase budget in {marker_path}")
        replacement_totals = marker.get("cbp_total_replacements")
        expected_layers = {"conv1", "conv2", "pre_gru", "gru"}
        if (
            not isinstance(replacement_totals, dict)
            or set(replacement_totals) != expected_layers
            or any(int(replacement_totals[layer]) <= 0 for layer in expected_layers)
        ):
            raise RuntimeError(
                f"CBP did not replace features in every controlled layer: {replacement_totals}"
            )
        records = _records(run_dir / "metrics.jsonl")
        evaluations = [record for record in records if record.get("event") == "evaluation"]
        starts = [
            record for record in evaluations
            if record["phase"] == 1 and record["kind"] == "phase_start"
        ]
        ends = [
            record for record in evaluations
            if record["phase"] == 1 and record["kind"] == "phase_end"
        ]
        if len(starts) != 1 or len(ends) != 1:
            raise RuntimeError(f"Expected one phase-1 start/end evaluation for n={diversity}")
        phase_gain = ends[0]["success_rate"] - starts[0]["success_rate"]
        details.append(
            {
                "diversity": diversity,
                "phase_1_gain": phase_gain,
                "cbp_total_replacements": replacement_totals,
            }
        )
    required_conditions = [item for item in details if item["diversity"] in (1, 16)]
    passed = all(item["phase_1_gain"] >= args.minimum_gain for item in required_conditions)
    payload = {
        "status": "pass" if passed else "fail",
        "criterion": (
            f"Both n=1 and n=16 improve by at least {args.minimum_gain:.3f} "
            "during the full-budget first phase, with verified CBP replacements"
        ),
        "minimum_gain": args.minimum_gain,
        "conditions": details,
        "budget_was_changed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        print("Smoke gate failed; the fixed main sweep was not launched.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
