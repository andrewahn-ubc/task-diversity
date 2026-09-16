import jax
import jax.numpy as jnp
import numpy as np

from banyan_grid.utils.dataset_loader import (
    compute_rule_execution_signatures_host as _compute_rule_execution_signatures_host,
    global_rule_program_bounds_from_signatures as _global_rule_program_bounds_from_signatures,
)
from banyan_grid.environment import Banyan
from banyan_grid.environment.constants import (
    Action,
    Colors,
    ITEM_TO_TILE,
    ItemType,
    RULE_TYPE_COMBINE,
    RULE_TYPE_TRANSFORM,
    TileType,
)
from banyan_grid.environment.goals import AgentHasItemFromRulesetGoal


def _pack_colors(c1: int, c2: int, cout: int) -> int:
    return ((cout & 0xF) << 8) | ((c2 & 0xF) << 4) | (c1 & 0xF)


def _combine_row(in1, in2, out, c1, c2, cout, *, required_adjacent=True):
    return [
        RULE_TYPE_COMBINE,
        int(in1),
        int(in2),
        int(out),
        1 if required_adjacent else 0,
        _pack_colors(int(c1), int(c2), int(cout)),
    ]


def _assert_tree_allclose(a, b):
    leaves_a, treedef_a = jax.tree_util.tree_flatten(jax.device_get(a))
    leaves_b, treedef_b = jax.tree_util.tree_flatten(jax.device_get(b))
    assert treedef_a == treedef_b
    for left, right in zip(leaves_a, leaves_b):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def _assert_state_equivalent(left, right):
    for field in (
        "positions",
        "directions",
        "map_array",
        "color_map",
        "token_grid",
        "token_counts",
        "inventory_colors",
        "inventories",
        "items_ever_picked",
        "time_step",
        "done",
        "rules_dirty",
        "rule_encodings",
        "goal_item",
        "goal_color",
        "goal_type",
        "relevant_item_mask",
        "pickup_reward_weights",
        "depth",
        "goal_vec",
        "rules_oh_static",
        "is_root",
        "is_comb",
    ):
        _assert_tree_allclose(getattr(left, field), getattr(right, field))


def test_compact_rule_program_bounds_match_default_step_behavior():
    grid_size = 5
    map_array = jnp.full((grid_size, grid_size), TileType.OPEN_FAST, dtype=jnp.int32)
    color_map = jnp.full((grid_size, grid_size), Colors.BLACK, dtype=jnp.int32)
    map_array = map_array.at[0, 1].set(ITEM_TO_TILE[int(ItemType.KEY)])
    color_map = color_map.at[0, 1].set(Colors.RED)
    map_array = map_array.at[0, 2].set(ITEM_TO_TILE[int(ItemType.BALL)])
    color_map = color_map.at[0, 2].set(Colors.BLUE)

    ruleset = jnp.array(
        [
            [
                RULE_TYPE_TRANSFORM,
                int(ItemType.KEY),
                0,
                int(ItemType.MAP),
                1,
                _pack_colors(int(Colors.RED), 0, int(Colors.GREEN)),
            ],
            [
                RULE_TYPE_COMBINE,
                int(ItemType.MAP),
                int(ItemType.BALL),
                int(ItemType.TRIANGLE),
                1,
                _pack_colors(
                    int(Colors.GREEN), int(Colors.BLUE), int(Colors.YELLOW)
                ),
            ],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=jnp.int32,
    )

    env = Banyan(
        grid_size=grid_size,
        max_steps=8,
        map_array=map_array,
        color_map=color_map,
        goal=AgentHasItemFromRulesetGoal(),
        max_rules=4,
    )
    signatures = _compute_rule_execution_signatures_host(
        np.asarray(ruleset[None, ...], dtype=np.int32)
    )
    compact_bounds = _global_rule_program_bounds_from_signatures(signatures)
    assert compact_bounds == {
        "pre_move_program_size": 2,
        "post_move_program_size": 0,
        "transform_program_size": 1,
        "pre_move_combine_steps": 1,
        "post_move_combine_steps": 0,
        "pre_move_transform_steps": 1,
        "post_move_transform_steps": 0,
        "all_transform_steps": 1,
    }

    metadata_kwargs = dict(
        include_rules_in_obs=env.include_rules_in_obs,
        depth_weighted_pickup_shaping=env.depth_weighted_pickup_shaping,
        pickup_shaping_leaf_reward=env.pickup_shaping_leaf_reward,
        pickup_shaping_root_reward=env.pickup_shaping_root_reward,
    )
    default_meta = env.compile_task_reset_metadata(ruleset, **metadata_kwargs)
    compact_meta = env.compile_task_reset_metadata(
        ruleset,
        **metadata_kwargs,
        **compact_bounds,
    )

    key = jax.random.PRNGKey(0)
    reset_params = {"map_array": map_array, "color_map": color_map}
    _, default_state = env.reset(key, {**reset_params, **default_meta}, ruleset)
    _, compact_state = env.reset(key, {**reset_params, **compact_meta}, ruleset)
    default_state = default_state.replace(
        positions=jnp.array([0, 0], dtype=jnp.int32),
        directions=jnp.asarray(1, dtype=jnp.int32),
    )
    compact_state = compact_state.replace(
        positions=jnp.array([0, 0], dtype=jnp.int32),
        directions=jnp.asarray(1, dtype=jnp.int32),
    )

    for actions in (
        jnp.asarray(Action.TOGGLE, dtype=jnp.int32),
        jnp.asarray(Action.STAY, dtype=jnp.int32),
        jnp.asarray(Action.PICKUP, dtype=jnp.int32),
    ):
        default_out = env.step(key, default_state, actions)
        compact_out = env.step(key, compact_state, actions)
        _assert_tree_allclose(default_out[0], compact_out[0])
        _assert_tree_allclose(default_out[2], compact_out[2])
        _assert_tree_allclose(default_out[3], compact_out[3])
        _assert_tree_allclose(default_out[4], compact_out[4])
        default_state = default_out[1]
        compact_state = compact_out[1]
        _assert_state_equivalent(default_state, compact_state)

    assert int(default_state.map_array[0, 1]) == int(ITEM_TO_TILE[int(ItemType.TRIANGLE)])
    assert int(compact_state.map_array[0, 1]) == int(ITEM_TO_TILE[int(ItemType.TRIANGLE)])
