from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/smoke")
    parser.add_argument("--output", default="results/smoke/gate.json")
    parser.add_argument("--minimum-gain", type=float, default=0.03)
    args = parser.parse_args()
    root = Path(args.root)
    details: list[dict[str, Any]] = []
    for diversity in (1, 16, 256):
        run_dir = root / f"n{diversity}" / "seed_0"
        if not (run_dir / "completed.json").exists():
            raise RuntimeError(f"Smoke run is incomplete: {run_dir}")
        records = _records(run_dir / "metrics.jsonl")
        evaluations = [record for record in records if record.get("event") == "evaluation"]
        phase_gains: list[float] = []
        for phase in range(1, 5):
            starts = [
                record for record in evaluations
                if record["phase"] == phase and record["kind"] == "phase_start"
            ]
            ends = [
                record for record in evaluations
                if record["phase"] == phase and record["kind"] == "phase_end"
            ]
            if len(starts) != 1 or len(ends) != 1:
                raise RuntimeError(f"Expected one start/end evaluation for n={diversity}, phase={phase}")
            phase_gains.append(ends[0]["success_rate"] - starts[0]["success_rate"])
        details.append(
            {
                "diversity": diversity,
                "phase_gains": phase_gains,
                "maximum_gain": max(phase_gains),
            }
        )
    maximum_gain = max(item["maximum_gain"] for item in details)
    required_conditions = [item for item in details if item["diversity"] in (1, 16)]
    passed = all(item["maximum_gain"] >= args.minimum_gain for item in required_conditions)
    payload = {
        "status": "pass" if passed else "fail",
        "criterion": (
            f"Both n=1 and n=16 improve by at least {args.minimum_gain:.3f} in one phase"
        ),
        "minimum_gain": args.minimum_gain,
        "observed_maximum_gain": maximum_gain,
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
