#!/usr/bin/env python3
"""Validate map generation correctness and symbolic solvability for ruleset datasets.

This script samples tasks from a compact ruleset dataset, generates maps using the
same reset path as training (`banyan_grid.utils.banyan.get_reset_params`),
and checks:

1) Leaf placement correctness:
   - all required leaf tokens exist on the map with exact counts
   - no extra collectible tokens are present
   - required leaves are reachable from spawn through walkable cells

2) Symbolic solvability:
   - goal token can be derived from map tokens via transform/combine rules
     under the same token semantics used by Banyan rulesets.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from banyan_grid.environment.constants import (
    COLLECTIBLE_TILES_ARRAY,
    ITEM_TO_TILE,
    RULE_TYPE_COMBINE,
    RULE_TYPE_TERNARY_COMBINE,
    RULE_TYPE_TRANSFORM,
    TILE_TO_ITEM,
    WALKABLE_MASK,
)
from banyan_grid.tasks.ruleset_codec import unpack_rules_uint32_np
from banyan_grid.tasks.ruleset_dataset_compact import (
    load_meta,
    load_packed_u32_bz2,
)
from banyan_grid.utils.banyan import get_reset_params

Token = tuple[int, int]  # (item, color)


def _resolve_dataset_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.exists():
        return path.resolve()
    if path.suffixes[-2:] == [".npy", ".bz2"] and not path.name.endswith(".uint32.npy.bz2"):
        alt = path.with_name(path.name.replace(".npy.bz2", ".uint32.npy.bz2"))
        if alt.exists():
            return alt.resolve()
    if not path.name.endswith(".uint32.npy.bz2"):
        alt = path.with_name(path.name + ".uint32.npy.bz2")
        if alt.exists():
            return alt.resolve()
    raise FileNotFoundError(f"Dataset not found: {path_str}")


def _decode_colors(packed: int) -> tuple[int, int, int]:
    c1 = packed & 0xF
    c2 = (packed >> 4) & 0xF
    cout = (packed >> 8) & 0xF
    return int(c1), int(c2), int(cout)


def _extract_collect_tokens(ruleset: np.ndarray) -> list[Token]:
    out: list[Token] = []
    for row in ruleset:
        if int(row[0]) == 1:
            out.append((int(row[2]), int(row[3])))
    return out


def _extract_producer_rules(ruleset: np.ndarray) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for row in ruleset:
        rt = int(row[0])
        if rt not in (RULE_TYPE_COMBINE, RULE_TYPE_TRANSFORM, RULE_TYPE_TERNARY_COMBINE):
            continue
        colors = int(row[5])
        c1, c2, cout = _decode_colors(colors)
        inputs = ((int(row[1]), c1),)
        if rt != RULE_TYPE_TRANSFORM:
            inputs += ((int(row[2]), c2),)
        if rt == RULE_TYPE_TERNARY_COMBINE:
            inputs += ((int(row[4]) >> 1, (colors >> 8) & 0xF),)
            cout = (colors >> 12) & 0xF
        rules.append(
            {
                "kind": "transform" if rt == RULE_TYPE_TRANSFORM else "combine",
                "inputs": inputs,
                "output": (int(row[3]), cout),
            }
        )
    return rules


def _build_tree_requirements(
    producer_rules: list[dict[str, Any]],
    collect_tokens: list[Token],
) -> tuple[bool, Token | None, Counter[Token], str | None]:
    """Infer root and required leaves when producer graph is tree-like."""
    if not producer_rules:
        if not collect_tokens:
            return False, None, Counter(), "no_producers_no_collect"
        # Depth-1 collect task.
        goal = collect_tokens[0]
        return True, goal, Counter({goal: 1}), None

    output_to_rule: dict[Token, dict[str, Any]] = {}
    consumer_count: Counter[Token] = Counter()
    for rule in producer_rules:
        out_tok = rule["output"]
        if out_tok in output_to_rule:
            return False, None, Counter(), "duplicate_output_token"
        output_to_rule[out_tok] = rule

    for rule in producer_rules:
        for tok in rule["inputs"]:
            if tok in output_to_rule:
                consumer_count[tok] += 1

    roots = [tok for tok in output_to_rule if consumer_count[tok] == 0]
    if len(roots) != 1:
        return False, None, Counter(), "invalid_root_count"
    root = roots[0]

    if any(consumer_count[tok] > 1 for tok in output_to_rule):
        return False, None, Counter(), "producer_output_reused"

    required: Counter[Token] = Counter()
    visiting: set[Token] = set()

    def expand(tok: Token) -> bool:
        if tok not in output_to_rule:
            required[tok] += 1
            return True
        if tok in visiting:
            return False
        visiting.add(tok)
        rule = output_to_rule[tok]
        ok = True
        for child in rule["inputs"]:
            ok = ok and expand(child)
            if not ok:
                break
        visiting.remove(tok)
        return ok

    if not expand(root):
        return False, None, Counter(), "cycle_detected"
    return True, root, required, None


def _token_counts_on_map(map_array: np.ndarray, color_map: np.ndarray) -> Counter[Token]:
    collectible_mask = np.isin(map_array, np.asarray(COLLECTIBLE_TILES_ARRAY))
    ys, xs = np.where(collectible_mask)
    counts: Counter[Token] = Counter()
    tile_to_item = np.asarray(TILE_TO_ITEM)
    for y, x in zip(ys.tolist(), xs.tolist()):
        tile = int(map_array[y, x])
        item = int(tile_to_item[tile]) if tile < tile_to_item.shape[0] else -1
        if item < 0:
            continue
        color = int(color_map[y, x])
        counts[(item, color)] += 1
    return counts


def _reachable_mask(map_array: np.ndarray) -> np.ndarray:
    walkable_lut = np.asarray(WALKABLE_MASK)
    walkable = walkable_lut[map_array].astype(np.bool_)
    h, w = walkable.shape
    visited = np.zeros_like(walkable, dtype=np.bool_)
    frontier: list[tuple[int, int]] = []
    for sy, sx in ((0, 0), (1, 0)):
        if 0 <= sy < h and 0 <= sx < w and walkable[sy, sx] and not visited[sy, sx]:
            visited[sy, sx] = True
            frontier.append((sy, sx))

    while frontier:
        y, x = frontier.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if visited[ny, nx] or not walkable[ny, nx]:
                continue
            visited[ny, nx] = True
            frontier.append((ny, nx))
    return visited


def _leaf_reachability_ok(
    required_leaves: Counter[Token],
    map_array: np.ndarray,
    color_map: np.ndarray,
    reachable: np.ndarray,
) -> bool:
    item_to_tile = np.asarray(ITEM_TO_TILE)
    for (item, color), need in required_leaves.items():
        tile = int(item_to_tile[item])
        pos = np.argwhere((map_array == tile) & (color_map == color))
        if pos.shape[0] < need:
            return False
        reachable_count = int(np.sum(reachable[pos[:, 0], pos[:, 1]]))
        if reachable_count < need:
            return False
    return True


def _symbolic_solvable_fallback(
    producer_rules: list[dict[str, Any]],
    map_counts: Counter[Token],
    goal: Token | None,
) -> bool:
    if goal is None:
        return False
    if map_counts.get(goal, 0) > 0:
        return True

    universe: list[Token] = sorted(
        set(map_counts.keys())
        | {r["output"] for r in producer_rules}
        | {tok for r in producer_rules for tok in r["inputs"]}
        | {goal}
    )
    tok_to_idx = {tok: i for i, tok in enumerate(universe)}
    goal_idx = tok_to_idx[goal]

    init = [0] * len(universe)
    for tok, c in map_counts.items():
        init[tok_to_idx[tok]] = int(c)
    init_state = tuple(init)

    encoded_rules: list[tuple[tuple[tuple[int, int], ...], int]] = []
    for r in producer_rules:
        out_i = tok_to_idx[r["output"]]
        input_counts = Counter(tok_to_idx[tok] for tok in r["inputs"])
        encoded_rules.append((tuple(sorted(input_counts.items())), out_i))

    @lru_cache(maxsize=100000)
    def dfs(state: tuple[int, ...]) -> bool:
        if state[goal_idx] > 0:
            return True

        for input_counts, out in encoded_rules:
            if any(state[in_i] < need for in_i, need in input_counts):
                continue
            cur = list(state)
            for in_i, need in input_counts:
                cur[in_i] -= need
            cur[out] += 1
            if dfs(tuple(cur)):
                return True
        return False

    return dfs(init_state)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="Path to .uint32.npy.bz2 dataset")
    ap.add_argument("--sample-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid-size", type=int, default=6)
    ap.add_argument("--replace", action="store_true", help="Sample with replacement")
    ap.add_argument(
        "--report-json",
        type=str,
        default="",
        help="Optional path to write summary JSON report.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = _resolve_dataset_path(args.dataset)
    dataset_dir = str(dataset_path.parent)
    dataset_file = dataset_path.name

    meta = load_meta(dataset_dir, dataset_file)
    packed = load_packed_u32_bz2(dataset_dir, dataset_file)
    total_tasks = int(packed.shape[0])
    if args.sample_size > total_tasks and not args.replace:
        raise ValueError(
            f"sample-size={args.sample_size} exceeds dataset size {total_tasks} without --replace"
        )

    rng_np = np.random.default_rng(args.seed)
    sampled_idx = rng_np.choice(total_tasks, size=args.sample_size, replace=args.replace)
    sampled_idx = np.asarray(sampled_idx, dtype=np.int32)

    packed_sel = packed[sampled_idx]
    rulesets = unpack_rules_uint32_np(packed_sel)

    map_keys = jax.random.split(jax.random.PRNGKey(args.seed + 1234), args.sample_size)
    reset_params = get_reset_params(
        map_keys,
        args.grid_size,
        jnp.asarray(rulesets, dtype=jnp.int32),
    )
    maps = np.asarray(reset_params["map_array"], dtype=np.int32)
    colors = np.asarray(reset_params["color_map"], dtype=np.int32)

    leaf_ok = 0
    reachable_ok = 0
    symbolic_ok = 0
    all_ok = 0

    failure_examples: list[dict[str, Any]] = []
    max_fail_examples = 30

    for i in range(args.sample_size):
        rs = rulesets[i]
        map_i = maps[i]
        color_i = colors[i]
        ds_idx = int(sampled_idx[i])

        collect_tokens = _extract_collect_tokens(rs)
        prod_rules = _extract_producer_rules(rs)
        is_tree, goal, required_leaves, tree_err = _build_tree_requirements(
            prod_rules, collect_tokens
        )

        map_counts = _token_counts_on_map(map_i, color_i)
        leaf_match = map_counts == required_leaves
        if leaf_match:
            leaf_ok += 1

        reach_mask = _reachable_mask(map_i)
        leaf_reachable = _leaf_reachability_ok(required_leaves, map_i, color_i, reach_mask)
        if leaf_reachable:
            reachable_ok += 1

        if is_tree:
            solvable = all(map_counts[tok] >= need for tok, need in required_leaves.items())
        else:
            solvable = _symbolic_solvable_fallback(prod_rules, map_counts, goal)
        if solvable:
            symbolic_ok += 1

        row_ok = leaf_match and leaf_reachable and solvable
        if row_ok:
            all_ok += 1
        elif len(failure_examples) < max_fail_examples:
            failure_examples.append(
                {
                    "sample_pos": int(i),
                    "dataset_index": ds_idx,
                    "tree_error": tree_err,
                    "goal": list(goal) if goal is not None else None,
                    "required_leaves": {
                        f"{k[0]}:{k[1]}": int(v) for k, v in required_leaves.items()
                    },
                    "map_collectibles": {f"{k[0]}:{k[1]}": int(v) for k, v in map_counts.items()},
                    "leaf_match": bool(leaf_match),
                    "leaf_reachable": bool(leaf_reachable),
                    "symbolic_solvable": bool(solvable),
                }
            )

        if ((i + 1) % 100) == 0 or (i + 1) == args.sample_size:
            print(
                f"checked {i + 1}/{args.sample_size} | "
                f"leaf_ok={leaf_ok} reachable_ok={reachable_ok} "
                f"symbolic_ok={symbolic_ok} all_ok={all_ok}"
            )

    summary = {
        "dataset": str(dataset_path),
        "meta_n": int(meta.get("n", total_tasks)),
        "dataset_rows": total_tasks,
        "sample_size": int(args.sample_size),
        "seed": int(args.seed),
        "grid_size": int(args.grid_size),
        "leaf_exact_ok": int(leaf_ok),
        "leaf_reachable_ok": int(reachable_ok),
        "symbolic_solvable_ok": int(symbolic_ok),
        "all_checks_ok": int(all_ok),
        "num_failures": int(args.sample_size - all_ok),
        "failure_examples": failure_examples,
    }

    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "failure_examples"}, indent=2))
    if failure_examples:
        print("\nFirst failure examples:")
        for ex in failure_examples[:10]:
            print(json.dumps(ex, indent=2))

    if args.report_json:
        out = Path(args.report_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote report: {out}")


if __name__ == "__main__":
    main()
