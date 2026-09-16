import numpy as np
import jax
import jax.numpy as jnp

from banyan_grid.environment import Banyan
from banyan_grid.tasks.ruleset_factory import build_ruleset
from banyan_grid.environment.constants import (
    Action,
    Colors,
    ITEM_TO_TILE,
    NUM_ITEMS,
    TileType,
)


def _base_maps(grid_size: int):
    map_array = jnp.full((grid_size, grid_size), TileType.OPEN_FAST, dtype=jnp.int32)
    color_map = jnp.full((grid_size, grid_size), Colors.BLACK, dtype=jnp.int32)
    return map_array, color_map


def test_teamwide_shaping_reward():
    key = jax.random.PRNGKey(0)
    ruleset = build_ruleset(key, depth=1, base_seed=0)
    rules_np = np.array(ruleset)
    collect_idx = int(np.where(rules_np[:, 0] == 1)[0][0])
    tile_type = int(rules_np[collect_idx, 1])
    color = int(rules_np[collect_idx, 3])

    map_array, color_map = _base_maps(5)
    map_array = map_array.at[0, 0].set(tile_type)
    color_map = color_map.at[0, 0].set(color)

    env = Banyan(
        grid_size=5,
        max_steps=10,
        map_array=map_array,
        color_map=color_map,
        max_depth=3,
    )
    _, state = env.reset(key, {"map_array": map_array, "color_map": color_map}, ruleset)
    state = state.replace(positions=jnp.array([0, 0], dtype=jnp.int32))
    actions = jnp.asarray(Action.PICKUP, dtype=jnp.int32)
    _, _, _, _, info = env.step(key, state, actions)

    np.testing.assert_allclose(
        np.array(info["reward_shape"]),
        np.float32(0.1),
        atol=1e-6,
    )


def test_no_drop_bonus_for_intermediate():
    key = jax.random.PRNGKey(1)
    ruleset = build_ruleset(key, depth=3, base_seed=0)
    rules_np = np.array(ruleset)

    map_array, color_map = _base_maps(5)
    env = Banyan(
        grid_size=5,
        max_steps=10,
        map_array=map_array,
        color_map=color_map,
        max_depth=3,
    )
    _, state = env.reset(key, {"map_array": map_array, "color_map": color_map}, ruleset)

    goal_item = int(state.goal_item)
    goal_color = int(state.goal_color)
    packed = rules_np[:, 5]
    out_item = rules_np[:, 3]
    out_color = (packed >> 8) & 0xF
    is_combine = rules_np[:, 0] == 2
    is_intermediate = is_combine & ~(
        (out_item == goal_item) & (out_color == goal_color)
    )
    assert np.any(is_intermediate), "No intermediate combine outputs found"
    idx = int(np.where(is_intermediate)[0][0])
    inter_item = int(out_item[idx])
    inter_color = int(out_color[idx])

    inv = np.zeros((NUM_ITEMS,), dtype=bool)
    cols = np.full((NUM_ITEMS,), Colors.BLACK, dtype=np.int32)
    inv[inter_item] = True
    cols[inter_item] = inter_color
    state = state.replace(inventories=jnp.array(inv), inventory_colors=jnp.array(cols))

    actions = jnp.asarray(Action.DROP, dtype=jnp.int32)
    _, _, rewards, _, _ = env.step(key, state, actions)

    np.testing.assert_allclose(
        np.array(rewards),
        np.float32(-0.001),
        atol=1e-6,
    )


def test_goal_reward_scale_applied():
    key = jax.random.PRNGKey(2)
    ruleset = build_ruleset(key, depth=1, base_seed=0)
    map_array, color_map = _base_maps(5)

    env = Banyan(
        grid_size=5,
        max_steps=10,
        map_array=map_array,
        color_map=color_map,
        max_depth=3,
        goal_reward_scale=2.0,
    )
    _, state = env.reset(key, {"map_array": map_array, "color_map": color_map}, ruleset)

    inv = np.zeros((NUM_ITEMS,), dtype=bool)
    cols = np.full((NUM_ITEMS,), Colors.BLACK, dtype=np.int32)
    goal_item = int(state.goal_item)
    goal_color = int(state.goal_color)
    inv[goal_item] = True
    cols[goal_item] = goal_color
    state = state.replace(inventories=jnp.array(inv), inventory_colors=jnp.array(cols))

    actions = jnp.asarray(Action.STAY, dtype=jnp.int32)
    _, _, rewards, done, info = env.step(key, state, actions)

    # Goal reward (2.0) plus per-step penalty (-0.001), no shaping on STAY.
    np.testing.assert_allclose(
        np.array(rewards),
        np.float32(1.999),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(info["goal_achieved"]),
        np.float32(1.0),
        atol=1e-6,
    )
    assert bool(np.array(done)) is True


def test_pickup_shaping_requires_matching_color_token():
    key = jax.random.PRNGKey(3)

    # Single collect rule requiring item 0 with RED color.
    item_type = 0
    required_color = int(Colors.RED)
    wrong_color = int(Colors.BLUE)
    tile_type = int(ITEM_TO_TILE[item_type])
    ruleset = jnp.array(
        [[1, tile_type, item_type, required_color, 0, 0]],
        dtype=jnp.int32,
    )

    map_array, color_map = _base_maps(5)
    map_array = map_array.at[0, 0].set(tile_type)
    color_map = color_map.at[0, 0].set(wrong_color)

    env = Banyan(
        grid_size=5,
        max_steps=10,
        map_array=map_array,
        color_map=color_map,
        max_depth=3,
    )
    _, state = env.reset(key, {"map_array": map_array, "color_map": color_map}, ruleset)
    state = state.replace(positions=jnp.array([0, 0], dtype=jnp.int32))

    actions = jnp.asarray(Action.PICKUP, dtype=jnp.int32)
    _, _, rewards, _, info = env.step(key, state, actions)

    # Wrong-color pickup should not count as relevant shaping.
    np.testing.assert_allclose(
        np.array(info["reward_shape"]),
        np.float32(0.0),
        atol=1e-6,
    )
    # Base per-step penalty still applies.
    np.testing.assert_allclose(
        np.array(rewards),
        np.float32(-0.001),
        atol=1e-6,
    )
