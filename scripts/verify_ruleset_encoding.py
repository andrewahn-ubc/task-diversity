#!/usr/bin/env python
import argparse
import bz2
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

from banyan_grid.environment.constants import (
    TileType,
    Colors,
    NUM_ITEMS,
    NUM_COLORS,
    NUM_TILE_TYPES,
)
from banyan_grid.environment.banyan import (
    Banyan,
)
from banyan_grid.tasks.ruleset_codec import (
    unpack_rules_uint32_np,
)


def _resolve_dataset_path(dataset_dir: str, dataset_file: str | None) -> str:
    if dataset_file is None:
        candidates = [
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.endswith(".uint32.npy.bz2")
        ]
        if not candidates:
            raise FileNotFoundError(f"No .uint32.npy.bz2 files in {dataset_dir}")
        return sorted(candidates)[0]

    path = dataset_file
    if not os.path.isabs(path):
        path = os.path.join(dataset_dir, path)

    if os.path.exists(path):
        return path

    alt = f"{path}.uint32.npy.bz2"
    if os.path.exists(alt):
        return alt

    raise FileNotFoundError(f"Could not find dataset file: {dataset_file}")


def _load_packed(path: str) -> np.ndarray:
    if path.endswith(".bz2"):
        with bz2.open(path, "rb") as f:
            return np.load(f)
    return np.load(path)


def _decode_colors(packed_cols: int) -> tuple[int, int, int]:
    c1 = packed_cols & 0xF
    c2 = (packed_cols >> 4) & 0xF
    cout = (packed_cols >> 8) & 0xF
    return int(c1), int(c2), int(cout)


def _expected_goal_from_ruleset(ruleset: np.ndarray) -> tuple[int, int, int]:
    comb_rows = []
    collect_rows = []
    for row in ruleset:
        rtype = int(row[0])
        if rtype == 2:
            c1, c2, cout = _decode_colors(int(row[5]))
            comb_rows.append(
                {
                    "in1": int(row[1]),
                    "c1": c1,
                    "in2": int(row[2]),
                    "c2": c2,
                    "out": int(row[3]),
                    "cout": cout,
                }
            )
        elif rtype == 1:
            collect_rows.append((int(row[2]), int(row[3])))

    if comb_rows:
        used_inputs = {(r["in1"], r["c1"]) for r in comb_rows} | {
            (r["in2"], r["c2"]) for r in comb_rows
        }
        for r in comb_rows:
            if (r["out"], r["cout"]) not in used_inputs:
                return r["out"], r["cout"], 1
        return comb_rows[0]["out"], comb_rows[0]["cout"], 1

    if not collect_rows:
        return 0, 0, 0
    return collect_rows[0][0], collect_rows[0][1], 0


def _find_rule_index_by_output(
    ruleset: np.ndarray, out_item: int, out_color: int
) -> int | None:
    for idx, row in enumerate(ruleset):
        if int(row[0]) != 2:
            continue
        c1, c2, cout = _decode_colors(int(row[5]))
        if int(row[3]) == out_item and cout == out_color:
            return idx
    return None


def _obs_slices(env: Banyan) -> tuple[int, int, int, int]:
    grid_size = env.grid_size
    num_tile_types = NUM_TILE_TYPES
    spatial_dim = grid_size * grid_size * (num_tile_types + 1 + NUM_COLORS)
    inv_dim = NUM_ITEMS
    inv_color_dim = NUM_ITEMS * NUM_COLORS
    goal_dim = 2 + NUM_ITEMS + NUM_COLORS
    return spatial_dim, inv_dim, inv_color_dim, goal_dim


def _check_obs_encoding(env: Banyan, ruleset: np.ndarray) -> None:
    grid_size = env.grid_size
    map_array = jnp.full((grid_size, grid_size), TileType.OPEN_FAST, dtype=jnp.int32)
    color_map = jnp.full((grid_size, grid_size), Colors.BLACK, dtype=jnp.int32)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(
        key, {"map_array": map_array, "color_map": color_map}, jnp.array(ruleset)
    )
    spatial_dim, inv_dim, inv_color_dim, goal_dim = _obs_slices(env)
    goal_offset = spatial_dim + inv_dim + inv_color_dim
    goal_slice = obs[goal_offset : goal_offset + goal_dim]
    rules_slice = obs[goal_offset + goal_dim :]

    if rules_slice.shape[0] != env.R_max * env.rule_oh_dim:
        raise AssertionError("Ruleset tail length mismatch in observation.")

    np.testing.assert_allclose(
        np.array(goal_slice), np.array(state.goal_vec), atol=1e-6
    )

    rules_tail = rules_slice.reshape(env.R_max, env.rule_oh_dim)
    if ruleset.shape[0] < env.R_max:
        pad = rules_tail[ruleset.shape[0] :]
        if not np.allclose(np.array(pad), 0.0):
            raise AssertionError("Ruleset padding is not zero.")

    combine_idxs = np.where(ruleset[:, 0] == 2)[0]
    if combine_idxs.size == 0:
        return

    out_item, out_color, _ = _expected_goal_from_ruleset(ruleset)
    root_idx = _find_rule_index_by_output(ruleset, out_item, out_color)
    if root_idx is None:
        raise AssertionError("Could not find root combine rule by output.")

    is_root_flags = np.array(rules_tail[: ruleset.shape[0], -1])
    if is_root_flags[root_idx] != 1.0:
        raise AssertionError("Root combine rule not marked as root in obs encoding.")

    if np.sum(is_root_flags[combine_idxs]) != 1.0:
        raise AssertionError("Expected exactly one root combine rule.")

    test_idx = int(combine_idxs[0])
    row = ruleset[test_idx]
    in1, in2, out = int(row[1]), int(row[2]), int(row[3])
    c1, c2, cout = _decode_colors(int(row[5]))

    inv = np.zeros((NUM_ITEMS,), dtype=bool)
    cols = np.full((NUM_ITEMS,), Colors.BLACK, dtype=np.int32)
    inv[in1] = True
    cols[in1] = c1
    inv[in2] = True
    cols[in2] = c2
    state_inputs = state.replace(
        inventories=jnp.array(inv), inventory_colors=jnp.array(cols)
    )
    obs_inputs = env.get_obs(state_inputs)
    rules_inputs = obs_inputs[goal_offset + goal_dim :].reshape(
        env.R_max, env.rule_oh_dim
    )
    inputs_ready = float(rules_inputs[test_idx, -3])
    if inputs_ready != 1.0:
        raise AssertionError("inputs_ready not set for combine rule.")

    inv = np.zeros((NUM_ITEMS,), dtype=bool)
    cols = np.full((NUM_ITEMS,), Colors.BLACK, dtype=np.int32)
    inv[out] = True
    cols[out] = cout
    state_out = state.replace(
        inventories=jnp.array(inv), inventory_colors=jnp.array(cols)
    )
    obs_out = env.get_obs(state_out)
    rules_out = obs_out[goal_offset + goal_dim :].reshape(env.R_max, env.rule_oh_dim)
    output_owned = float(rules_out[test_idx, -2])
    if output_owned != 1.0:
        raise AssertionError("output_owned not set for combine rule.")


def _check_goal_consistency(
    env: Banyan, rulesets: np.ndarray, max_checks: int
) -> int:
    mismatches = 0
    checks = min(max_checks, rulesets.shape[0])
    for i in range(checks):
        rs = rulesets[i]
        exp_item, exp_color, exp_type = _expected_goal_from_ruleset(rs)
        gi, gc, gt, goal_vec, _, _ = env.compile_goal_from_ruleset(
            jnp.array(rs), NUM_ITEMS, NUM_COLORS
        )
        if (int(gi), int(gc), int(gt)) != (exp_item, exp_color, exp_type):
            mismatches += 1
        else:
            # Goal vector should match the expected one-hot
            goal_type_oh = jax.nn.one_hot(exp_type, num_classes=2).astype(jnp.float32)
            goal_item_oh = jax.nn.one_hot(exp_item, num_classes=NUM_ITEMS).astype(
                jnp.float32
            )
            goal_color_oh = jax.nn.one_hot(exp_color, num_classes=NUM_COLORS).astype(
                jnp.float32
            )
            expected_goal_vec = jnp.concatenate(
                [goal_type_oh, goal_item_oh, goal_color_oh], axis=0
            )
            np.testing.assert_allclose(
                np.array(goal_vec), np.array(expected_goal_vec), atol=1e-6
            )
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify ruleset encoding and goal/task embedding."
    )
    parser.add_argument(
        "--dataset-dir", default="CUR_DATASETS", help="Directory of packed rulesets."
    )
    parser.add_argument(
        "--dataset-file",
        default=None,
        help="Dataset file basename (with or without .uint32.npy.bz2).",
    )
    parser.add_argument(
        "--index", type=int, default=0, help="Ruleset index to inspect."
    )
    parser.add_argument(
        "--max-checks",
        type=int,
        default=64,
        help="Number of rulesets to check for goal consistency.",
    )
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()

    path = _resolve_dataset_path(args.dataset_dir, args.dataset_file)
    packed = _load_packed(path)
    rulesets = unpack_rules_uint32_np(packed)
    if rulesets.ndim != 3 or rulesets.shape[-1] != 6:
        raise ValueError(f"Unexpected ruleset shape: {rulesets.shape}")

    if not (0 <= args.index < rulesets.shape[0]):
        raise IndexError(f"index {args.index} out of bounds for {rulesets.shape[0]}")

    map_array = jnp.full(
        (args.grid_size, args.grid_size), TileType.OPEN_FAST, dtype=jnp.int32
    )
    color_map = jnp.full(
        (args.grid_size, args.grid_size), Colors.BLACK, dtype=jnp.int32
    )
    env = Banyan(
        grid_size=args.grid_size,
        max_steps=10,
        map_array=map_array,
        color_map=color_map,
        max_depth=args.max_depth,
    )

    mismatches = _check_goal_consistency(env, rulesets, args.max_checks)
    if mismatches:
        print(f"Goal mismatch count (first {args.max_checks}): {mismatches}")
    else:
        print(f"Goal consistency OK for first {args.max_checks} rulesets.")

    _check_obs_encoding(env, rulesets[args.index])
    print("Observation encoding checks OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
