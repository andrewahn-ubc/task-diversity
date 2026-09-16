"""Validation script for distractor-table combines in Banyan.

This script verifies that two adjacent items combine when:
1) The pair is legal in the distractor lookup table.
2) The pair is NOT part of the current task-tree ruleset.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from banyan_grid.environment import Banyan
from banyan_grid.environment.constants import (
    Action,
    Colors,
    ItemType,
    NUM_COLORS,
    NUM_ITEMS,
    TileType,
)


def _pack_colors(c1: int, c2: int, cout: int) -> int:
    return ((cout & 0xF) << 8) | ((c2 & 0xF) << 4) | (c1 & 0xF)


def _make_task_tree_ruleset() -> jnp.ndarray:
    """Build a tiny task-tree ruleset that excludes KEY+BALL combine."""
    return jnp.array(
        [
            # Collect KEY[RED]
            [1, int(TileType.KEY), int(ItemType.KEY), int(Colors.RED), 0, 0],
            # Collect MAP[BLUE]
            [1, int(TileType.MAP), int(ItemType.MAP), int(Colors.BLUE), 0, 0],
            # In-tree combine is KEY[RED] + MAP[BLUE] -> STAR[GREEN]
            [
                2,
                int(ItemType.KEY),
                int(ItemType.MAP),
                int(ItemType.STAR),
                1,
                _pack_colors(
                    int(Colors.RED),
                    int(Colors.BLUE),
                    int(Colors.GREEN),
                ),
            ],
        ],
        dtype=jnp.int32,
    )


def _combine_pairs_from_ruleset(ruleset: np.ndarray) -> set[tuple[int, int, int, int]]:
    pairs: set[tuple[int, int, int, int]] = set()
    for row in ruleset:
        if int(row[0]) != 2:
            continue
        packed = int(row[5])
        c1 = packed & 0xF
        c2 = (packed >> 4) & 0xF
        pair = (int(row[1]), c1, int(row[2]), c2)
        pairs.add(pair)
    return pairs


def main() -> None:
    grid_size = 5
    key_item = int(ItemType.KEY)
    ball_item = int(ItemType.BALL)
    out_item = int(ItemType.MAP)
    key_tile = int(TileType.KEY)
    ball_tile = int(TileType.BALL)
    out_tile = int(TileType.MAP)
    c_key = int(Colors.RED)
    c_ball = int(Colors.GREEN)
    c_out = int(Colors.BLUE)

    # Agent will drop KEY[RED] adjacent to BALL[GREEN].
    map_array = jnp.full((grid_size, grid_size), TileType.OPEN_FAST, dtype=jnp.int32)
    color_map = jnp.full((grid_size, grid_size), Colors.BLACK, dtype=jnp.int32)
    map_array = map_array.at[1, 2].set(ball_tile)
    color_map = color_map.at[1, 2].set(c_ball)

    ruleset = _make_task_tree_ruleset()
    ruleset_np = np.array(ruleset, dtype=np.int32)

    # Assert this adjacent pair is NOT part of the task tree combine rules.
    in_tree_pairs = _combine_pairs_from_ruleset(ruleset_np)
    off_tree_pair = (key_item, c_key, ball_item, c_ball)
    assert off_tree_pair not in in_tree_pairs, (
        "Validation setup error: test pair unexpectedly appears in task tree."
    )

    # Distractor lookup table marks KEY[RED] + BALL[GREEN] as legal -> MAP[BLUE].
    distractor_table = np.zeros(
        (NUM_ITEMS, NUM_COLORS, NUM_ITEMS, NUM_COLORS, 3),
        dtype=np.int32,
    )
    distractor_table[key_item, c_key, ball_item, c_ball] = [out_item, c_out, 1]
    distractor_table[ball_item, c_ball, key_item, c_key] = [out_item, c_out, 1]

    env = Banyan(
        grid_size=grid_size,
        max_steps=8,
        map_array=map_array,
        color_map=color_map,
        max_depth=3,
        max_rules=ruleset.shape[0],
        distractor_table=jnp.asarray(distractor_table, dtype=jnp.int32),
    )

    key = jax.random.PRNGKey(0)
    _, state = env.reset(key, {"map_array": map_array, "color_map": color_map}, ruleset)
    inventories = state.inventories.at[key_item].set(True)
    inventory_colors = state.inventory_colors.at[key_item].set(c_key)
    positions = jnp.array([1, 1], dtype=jnp.int32)
    state = state.replace(
        positions=positions,
        inventories=inventories,
        inventory_colors=inventory_colors,
    )
    action = jnp.asarray(Action.DROP, dtype=jnp.int32)
    _, next_state, _, _, _ = env.step(key, state, action)

    map_np = np.array(next_state.map_array)
    color_np = np.array(next_state.color_map)

    # Verify merge happened: one MAP[BLUE], no KEY or BALL left.
    map_positions = np.argwhere(map_np == out_tile)
    assert map_positions.shape[0] == 1, (
        f"Expected exactly one output MAP tile, found {map_positions.shape[0]}"
    )
    out_y, out_x = map_positions[0]
    assert int(color_np[out_y, out_x]) == c_out, (
        f"Expected output color {c_out}, found {int(color_np[out_y, out_x])}"
    )
    assert int(np.sum(map_np == key_tile)) == 0, "Expected KEY to be consumed"
    assert int(np.sum(map_np == ball_tile)) == 0, "Expected BALL to be consumed"

    # Sanity check around the original pair positions.
    pair_tiles = {int(map_np[1, 1]), int(map_np[1, 2])}
    assert pair_tiles == {int(TileType.OPEN_FAST), out_tile}, (
        "Expected one source cell to become OPEN_FAST and one to hold output tile."
    )

    print("PASS: Off-tree but legal distractor pair merged successfully.")
    print("  Pair tested: KEY[RED] + BALL[GREEN] -> MAP[BLUE]")
    print(
        "  Pair is absent from task-tree combine rules and present in distractor table."
    )


if __name__ == "__main__":
    main()
