import jax
import jax.numpy as jnp
import numpy as np

from banyan_grid.utils.banyan import compute_depth, success_from_state
from banyan_grid.environment import Banyan
from banyan_grid.environment.constants import (
    Action,
    Colors,
    ITEM_TO_TILE,
    ItemType,
    TileType,
)
from banyan_grid.environment.goals import AgentHasItemFromRulesetGoal
from banyan_grid.tasks.ruleset_codec import (
    pack_rules_uint32_np,
    unpack_rules_uint32_jit,
)


def _pack_colors(c1: int, c2: int, cout: int) -> int:
    return ((cout & 0xF) << 8) | ((c2 & 0xF) << 4) | (c1 & 0xF)


def _assert_tree_equal(left, right):
    left_leaves, left_def = jax.tree_util.tree_flatten(jax.device_get(left))
    right_leaves, right_def = jax.tree_util.tree_flatten(jax.device_get(right))
    assert left_def == right_def
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def test_ruleset_codec_roundtrip_transform():
    rules = np.array(
        [
            [
                [4, int(ItemType.KEY), 0, int(ItemType.MAP), 1, _pack_colors(int(Colors.RED), 0, int(Colors.GREEN))],
                [
                    2,
                    int(ItemType.MAP),
                    int(ItemType.BALL),
                    int(ItemType.TRIANGLE),
                    1,
                    _pack_colors(int(Colors.GREEN), int(Colors.BLUE), int(Colors.YELLOW)),
                ],
            ]
        ],
        dtype=np.int32,
    )
    packed = pack_rules_uint32_np(rules)
    decoded = np.array(unpack_rules_uint32_jit(jnp.array(packed, dtype=jnp.uint32)))
    np.testing.assert_array_equal(decoded, rules)


def test_transform_rule_executes_on_toggle_adjacent():
    grid_size = 5
    map_array = jnp.full((grid_size, grid_size), TileType.OPEN_FAST, dtype=jnp.int32)
    color_map = jnp.full((grid_size, grid_size), Colors.BLACK, dtype=jnp.int32)
    map_array = map_array.at[0, 1].set(ITEM_TO_TILE[int(ItemType.KEY)])
    color_map = color_map.at[0, 1].set(Colors.RED)

    ruleset = jnp.array(
        [
            [
                4,
                int(ItemType.KEY),
                0,
                int(ItemType.BALL),
                1,
                _pack_colors(int(Colors.RED), 0, int(Colors.BLUE)),
            ]
        ],
        dtype=jnp.int32,
    )

    env = Banyan(
        grid_size=grid_size,
        max_steps=8,
        map_array=map_array,
        color_map=color_map,
        goal=AgentHasItemFromRulesetGoal(),
        max_rules=1,
    )
    key = jax.random.PRNGKey(0)
    _, state = env.reset(
        key, {"map_array": map_array, "color_map": color_map}, ruleset
    )
    state = state.replace(
        positions=jnp.array([0, 0], dtype=jnp.int32),
        directions=jnp.asarray(1, dtype=jnp.int32),
    )

    actions = jnp.asarray(Action.TOGGLE, dtype=jnp.int32)
    _, next_state, _, _, _ = env.step(
        key, state, actions
    )

    assert int(next_state.map_array[0, 1]) == int(ITEM_TO_TILE[int(ItemType.BALL)])
    assert int(next_state.color_map[0, 1]) == int(Colors.BLUE)


def test_transform_rule_noops_without_toggle_even_when_adjacent():
    grid_size = 5
    map_array = jnp.full((grid_size, grid_size), TileType.OPEN_FAST, dtype=jnp.int32)
    color_map = jnp.full((grid_size, grid_size), Colors.BLACK, dtype=jnp.int32)
    map_array = map_array.at[0, 1].set(ITEM_TO_TILE[int(ItemType.KEY)])
    color_map = color_map.at[0, 1].set(Colors.RED)

    ruleset = jnp.array(
        [
            [
                4,
                int(ItemType.KEY),
                0,
                int(ItemType.BALL),
                1,
                _pack_colors(int(Colors.RED), 0, int(Colors.BLUE)),
            ]
        ],
        dtype=jnp.int32,
    )

    env = Banyan(
        grid_size=grid_size,
        max_steps=8,
        map_array=map_array,
        color_map=color_map,
        goal=AgentHasItemFromRulesetGoal(),
        max_rules=1,
    )
    key = jax.random.PRNGKey(0)
    _, state = env.reset(
        key, {"map_array": map_array, "color_map": color_map}, ruleset
    )
    state = state.replace(
        positions=jnp.array([0, 0], dtype=jnp.int32),
        directions=jnp.asarray(1, dtype=jnp.int32),
    )

    actions = jnp.asarray(Action.STAY, dtype=jnp.int32)
    _, next_state, _, _, info = env.step(
        key, state, actions
    )

    assert int(next_state.map_array[0, 1]) == int(ITEM_TO_TILE[int(ItemType.KEY)])
    assert int(next_state.color_map[0, 1]) == int(Colors.RED)
    np.testing.assert_allclose(
        np.array(info["toggle_transform_success"]),
        np.float32(0.0),
        atol=1e-6,
    )


def test_step_outputs_do_not_depend_on_prng_key():
    grid_size = 5
    map_array = jnp.full((grid_size, grid_size), TileType.OPEN_FAST, dtype=jnp.int32)
    color_map = jnp.full((grid_size, grid_size), Colors.BLACK, dtype=jnp.int32)
    map_array = map_array.at[0, 1].set(ITEM_TO_TILE[int(ItemType.KEY)])
    color_map = color_map.at[0, 1].set(Colors.RED)

    ruleset = jnp.array(
        [
            [
                4,
                int(ItemType.KEY),
                0,
                int(ItemType.BALL),
                1,
                _pack_colors(int(Colors.RED), 0, int(Colors.BLUE)),
            ]
        ],
        dtype=jnp.int32,
    )
    env = Banyan(
        grid_size=grid_size,
        max_steps=8,
        map_array=map_array,
        color_map=color_map,
        goal=AgentHasItemFromRulesetGoal(),
        max_rules=1,
    )
    _, state = env.reset(
        jax.random.PRNGKey(0),
        {"map_array": map_array, "color_map": color_map},
        ruleset,
    )
    state = state.replace(
        positions=jnp.array([0, 0], dtype=jnp.int32),
        directions=jnp.asarray(1, dtype=jnp.int32),
    )
    actions = jnp.asarray(Action.TOGGLE, dtype=jnp.int32)

    out_a = env.step(jax.random.PRNGKey(1), state, actions)
    out_b = env.step(jax.random.PRNGKey(999), state, actions)
    _assert_tree_equal(out_a, out_b)


def test_transform_chain_goal_depth_and_success():
    grid_size = 6
    map_array = jnp.full((grid_size, grid_size), TileType.OPEN_FAST, dtype=jnp.int32)
    color_map = jnp.full((grid_size, grid_size), Colors.BLACK, dtype=jnp.int32)

    # KEY[RED] --toggle--> MAP[GREEN]
    # MAP[GREEN] + BALL[BLUE] -> TRIANGLE[YELLOW]
    ruleset = jnp.array(
        [
            [
                2,
                int(ItemType.MAP),
                int(ItemType.BALL),
                int(ItemType.TRIANGLE),
                1,
                _pack_colors(int(Colors.GREEN), int(Colors.BLUE), int(Colors.YELLOW)),
            ],
            [
                4,
                int(ItemType.KEY),
                0,
                int(ItemType.MAP),
                1,
                _pack_colors(int(Colors.RED), 0, int(Colors.GREEN)),
            ],
        ],
        dtype=jnp.int32,
    )

    env = Banyan(
        grid_size=grid_size,
        max_steps=16,
        map_array=map_array,
        color_map=color_map,
        goal=AgentHasItemFromRulesetGoal(),
        max_rules=2,
    )
    key = jax.random.PRNGKey(1)
    _, state = env.reset(
        key, {"map_array": map_array, "color_map": color_map}, ruleset
    )

    assert int(state.depth) == 3
    assert int(state.goal_item) == int(ItemType.TRIANGLE)
    assert int(state.goal_color) == int(Colors.YELLOW)

    state_success = state.replace(
        inventories=state.inventories.at[int(ItemType.TRIANGLE)].set(True),
        inventory_colors=state.inventory_colors.at[int(ItemType.TRIANGLE)].set(
            int(Colors.YELLOW)
        ),
    )
    assert bool(success_from_state(state_success))
    assert int(compute_depth(state_success)) == 3
