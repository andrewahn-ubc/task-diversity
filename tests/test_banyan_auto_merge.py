import numpy as np
import jax
import jax.numpy as jnp

from banyan_grid.environment import Banyan
from banyan_grid.tasks.ruleset_factory import build_ruleset
from banyan_grid.environment.constants import (
    Action,
    Colors,
    TileType,
    ITEM_TO_TILE,
)


def _base_maps(grid_size: int):
    map_array = jnp.full((grid_size, grid_size), TileType.OPEN_FAST, dtype=jnp.int32)
    color_map = jnp.full((grid_size, grid_size), Colors.BLACK, dtype=jnp.int32)
    return map_array, color_map


def _first_combine_rule(ruleset):
    rules_np = np.array(ruleset)
    comb_idx = int(np.where(rules_np[:, 0] == 2)[0][0])
    enc = rules_np[comb_idx]
    packed = int(enc[5])
    c1 = packed & 0xF
    c2 = (packed >> 4) & 0xF
    cout = (packed >> 8) & 0xF
    return int(enc[1]), int(enc[2]), int(enc[3]), int(c1), int(c2), int(cout)


def test_adjacent_items_auto_merge():
    key = jax.random.PRNGKey(0)
    ruleset = build_ruleset(key, depth=2, base_seed=0, required_adjacent=True)
    in1, in2, out, c1, c2, cout = _first_combine_rule(ruleset)

    map_array, color_map = _base_maps(5)
    pos1 = (2, 2)
    pos2 = (2, 3)
    map_array = map_array.at[pos1].set(ITEM_TO_TILE[in1])
    color_map = color_map.at[pos1].set(c1)
    map_array = map_array.at[pos2].set(ITEM_TO_TILE[in2])
    color_map = color_map.at[pos2].set(c2)

    env = Banyan(
        grid_size=5,
        max_steps=10,
        map_array=map_array,
        color_map=color_map,
        max_depth=2,
    )
    _, state = env.reset(key, {"map_array": map_array, "color_map": color_map}, ruleset)
    actions = jnp.asarray(Action.STAY, dtype=jnp.int32)
    _, next_state, _, _, _ = env.step(key, state, actions)

    out_tile = int(ITEM_TO_TILE[out])
    np.testing.assert_equal(np.array(next_state.map_array[pos1]), out_tile)
    np.testing.assert_equal(np.array(next_state.color_map[pos1]), cout)
    np.testing.assert_equal(np.array(next_state.map_array[pos2]), TileType.OPEN_FAST)
    np.testing.assert_equal(np.array(next_state.color_map[pos2]), Colors.BLACK)


def test_non_adjacent_items_do_not_merge():
    key = jax.random.PRNGKey(1)
    ruleset = build_ruleset(key, depth=2, base_seed=0, required_adjacent=True)
    in1, in2, out, c1, c2, cout = _first_combine_rule(ruleset)

    map_array, color_map = _base_maps(5)
    pos1 = (2, 2)
    pos2 = (2, 4)
    map_array = map_array.at[pos1].set(ITEM_TO_TILE[in1])
    color_map = color_map.at[pos1].set(c1)
    map_array = map_array.at[pos2].set(ITEM_TO_TILE[in2])
    color_map = color_map.at[pos2].set(c2)

    env = Banyan(
        grid_size=5,
        max_steps=10,
        map_array=map_array,
        color_map=color_map,
        max_depth=2,
    )
    _, state = env.reset(key, {"map_array": map_array, "color_map": color_map}, ruleset)
    actions = jnp.asarray(Action.STAY, dtype=jnp.int32)
    _, next_state, _, _, _ = env.step(key, state, actions)

    np.testing.assert_equal(np.array(next_state.map_array[pos1]), ITEM_TO_TILE[in1])
    np.testing.assert_equal(np.array(next_state.color_map[pos1]), c1)
    np.testing.assert_equal(np.array(next_state.map_array[pos2]), ITEM_TO_TILE[in2])
    np.testing.assert_equal(np.array(next_state.color_map[pos2]), c2)
