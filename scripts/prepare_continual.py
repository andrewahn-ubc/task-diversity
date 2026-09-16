"""Create round-disjoint task datasets and globally distinct layout banks.

Run once per n. The generated artifacts are shared by all seeds and PPO variants.
The task generator is the repository's own depth-6 round-disjoint generator.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def connected(mask: np.ndarray) -> bool:
    """Require every free tile to be reachable from the protected spawn area."""
    free = ~mask
    seen = {(0, 0)}
    pending = [(0, 0)]
    while pending:
        y, x = pending.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                    and free[yy, xx] and (yy, xx) not in seen):
                seen.add((yy, xx))
                pending.append((yy, xx))
    return len(seen) == int(free.sum())


def make_layouts(n: int, rounds: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1009 * n)
    masks = []
    keys = set()
    protected = np.zeros((size, size), dtype=bool)
    protected[:3, :3] = True
    attempts = 0
    while len(masks) < n * rounds:
        attempts += 1
        if attempts > 1_000_000:
            raise RuntimeError("Could not sample enough distinct connected layouts")
        candidate = (rng.random((size, size)) < rng.uniform(0.08, 0.18)) & ~protected
        key = candidate.tobytes()
        if key in keys or not connected(candidate):
            continue
        keys.add(key)
        masks.append(candidate)
    return np.asarray(masks, dtype=bool).reshape(rounds, n, size, size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--root", type=Path, default=Path("outputs/continual"))
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    rounds = 3 if args.mode == "pilot" else 7
    root = args.root.resolve() / "datasets" / args.mode
    task_dir = root / f"n{args.n:04d}"
    layout_file = task_dir / "layouts.npz"
    summary = task_dir / f"continual_n{args.n}_summary.json"
    task_dir.mkdir(parents=True, exist_ok=True)
    if not summary.exists():
        subprocess.run(
            [sys.executable, "-m", "banyan_grid.tasks.make_depth6_round_disjoint_dataset",
             "--n", str(args.n), "--rounds", str(rounds),
             "--unique-topologies-within-round",
             "--out-dir", str(root), "--name", "continual"],
            cwd=ROOT, check=True,
        )
    with summary.open() as stream:
        meta = json.load(stream)
    guarantees = meta.get("cross_round_guarantees", {})
    if not (guarantees.get("depth6_topologies_disjoint_across_rounds")
            and guarantees.get("producer_signatures_disjoint_across_rounds")
            and not guarantees.get("same_round_topology_reuse_allowed")):
        raise RuntimeError("Task generator did not confirm cross-round disjointness")
    if any(row["unique_depth6_topologies"] != args.n
           for row in meta["round_meta_summaries"]):
        raise RuntimeError("A distribution has fewer than n unique depth-6 topologies")
    if meta.get("issues"):
        print(f"Generator advisory for n={args.n}: {meta['issues']}")
    for phase in range(rounds):
        stem = task_dir / f"continual_n{args.n}_r{phase:02d}"
        if not Path(str(stem) + ".uint32.npy.bz2").exists():
            raise FileNotFoundError(str(stem) + ".uint32.npy.bz2")
    if not layout_file.exists():
        masks = make_layouts(args.n, rounds, 8, 1729)
        np.savez_compressed(layout_file, obstacle_mask=masks,
                            seed=np.asarray(1729), grid_size=np.asarray(8))
    print(f"Prepared {rounds} distributions for n={args.n}: {task_dir}")


if __name__ == "__main__":
    main()
