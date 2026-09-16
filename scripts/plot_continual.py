"""Plot success trajectories in the style of paper Figure 6."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "banyan-mpl-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from continual_config import VARIANTS


COLORS = {1: "#c8372a", 4: "#f58b4b", 16: "#f4d35e",
          64: "#88c765", 256: "#078641"}


def load_run(root: Path, n: int, seed: int, mode: str, variant: str) -> list[dict]:
    folder = root / "runs" / mode / f"n{n:04d}" / variant / f"seed{seed}"
    path = folder / "metrics.jsonl"
    if not path.exists():
        return []
    dedup = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        dedup[(row["phase"], row["phase_steps"])] = row
    return sorted(dedup.values(), key=lambda r: (r["env_steps"], r["phase"]))


def plot(root: Path, mode: str, seeds: list[int], output: Path) -> None:
    n_values = (1, 4, 16, 64, 256) if mode == "full" else (4, 64)
    rounds = 7 if mode == "full" else 3
    target = 100_000_000 if mode == "full" else 10_000_000
    actual_phase_steps = math.ceil(target / (128 * 128)) * (128 * 128)
    fig, axs = plt.subplots(2, 1, figsize=(15, 8.5), sharex=True,
                            constrained_layout=True)
    any_data = False
    max_x = 0
    variants = ("selected",) if mode == "full" else tuple(VARIANTS)
    styles = {"base": "-", "lower_lr": "--", "more_entropy": ":", "more_cbp": "-."}
    for n in n_values:
        for variant in variants:
            grouped = defaultdict(list)
            for seed in seeds:
                for row in load_run(root, n, seed, mode, variant):
                    grouped[(row["phase"], row["phase_steps"])].append(row)
            if not grouped:
                continue
            any_data = True
            for ax, field in zip(axs, ("all_depths", "depth6")):
                xs, mean, std = [], [], []
                for key in sorted(grouped, key=lambda pair: ((pair[0]-1) * actual_phase_steps + pair[1], pair[0])):
                    rows = grouped[key]
                    x = rows[0]["env_steps"]
                    values = np.asarray([r[field] for r in rows], dtype=float)
                    xs.append(x / 1e6)
                    mean.append(values.mean())
                    std.append(values.std(ddof=1) if len(values) > 1 else 0)
                max_x = max(max_x, max(xs))
                label = rf"${n}^2$" if mode == "full" else rf"${n}^2$ {variant}"
                ax.plot(xs, mean, lw=2.6, color=COLORS[n],
                        linestyle=styles.get(variant, "-"), label=label)
                ax.fill_between(xs, np.maximum(0, np.asarray(mean) - std),
                                np.minimum(1, np.asarray(mean) + std),
                                color=COLORS[n], alpha=.12, linewidth=0)
    if not any_data:
        raise SystemExit("No evaluation metrics found. Run training first.")
    for ax, title in zip(axs, (f"Success rate over {rounds} task distributions",
                               f"Depth 6 success rate over {rounds} task distributions")):
        for phase in range(1, rounds):
            ax.axvline(phase * actual_phase_steps / 1e6, color=".5", ls="--", lw=1.4)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, max(rounds * actual_phase_steps / 1e6, max_x))
        ax.set_ylabel("PPO + CBP\nSuccess rate", fontsize=15)
        ax.set_title(title, fontsize=20, weight="bold", pad=12)
        ax.grid(axis="y", alpha=.2)
    axs[0].legend(title="Layouts × topologies" if mode == "full" else "Diversity and variant",
                  ncol=len(n_values) if mode == "full" else 2,
                  loc="lower right", fontsize=11, framealpha=.9)
    axs[1].set_xlabel("Environment steps (millions)", fontsize=16)
    fig.suptitle("Lines: seed mean   ·   shading: one standard deviation",
                 fontsize=11, y=1.01)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(output)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("pilot", "full"), default="full")
    p.add_argument("--root", type=Path, default=Path("outputs/continual"))
    p.add_argument("--seeds", default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    seeds = [int(x) for x in (args.seeds or ("0,1,2" if args.mode == "full" else "0,1")).split(",")]
    output = args.output or args.root / f"{args.mode}_figure6.png"
    plot(args.root.resolve(), args.mode, seeds, output.resolve())


if __name__ == "__main__":
    main()
