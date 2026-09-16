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
from banyan_grid.tasks.ruleset_codec import (
    pack_rules_uint32_np,
    unpack_rules_uint32_jit,
)


def _pack_colors(c1: int, c2: int, cout: int) -> int:
    return ((cout & 0xF) << 8) | ((c2 & 0xF) << 4) | (c1 & 0xF)


def test_ruleset_codec_roundtrip_distractor_combine():
    rules = np.array(
        [
            [
                [2, 0, 1, 2, 1, _pack_colors(0, 1, 2)],
                [3, 2, 3, 4, 1, _pack_colors(2, 3, 4)],
            ]
        ],
        dtype=np.int32,
    )
    packed = pack_rules_uint32_np(rules)
    decoded = np.array(
        unpack_rules_uint32_jit(jnp.array(packed, dtype=jnp.uint32)), dtype=np.int32
    )
    np.testing.assert_array_equal(decoded, rules)


def test_distractor_combine_rule_executes():
    grid_size = 4
    map_array = jnp.full((grid_size, grid_size), TileType.OPEN_FAST, dtype=jnp.int32)
    color_map = jnp.full((grid_size, grid_size), Colors.BLACK, dtype=jnp.int32)

    # Place required inputs adjacent.
    map_array = map_array.at[1, 1].set(TileType.KEY)
    color_map = color_map.at[1, 1].set(Colors.RED)
    map_array = map_array.at[1, 2].set(TileType.BALL)
    color_map = color_map.at[1, 2].set(Colors.GREEN)

    # One collect row (goal scaffold) + one distractor combine row.
    ruleset = jnp.array(
        [
            [1, int(TileType.KEY), 0, int(Colors.RED), 0, 0],
            [
                3,
                0,
                1,
                2,
                1,
                _pack_colors(int(Colors.RED), int(Colors.GREEN), int(Colors.BLUE)),
            ],
        ],
        dtype=jnp.int32,
    )

    env = Banyan(
        grid_size=grid_size,
        max_steps=8,
        map_array=map_array,
        color_map=color_map,
        max_depth=2,
        max_rules=2,
    )
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key, {"map_array": map_array, "color_map": color_map}, ruleset)

    actions = jnp.asarray(Action.STAY, dtype=jnp.int32)
    _, next_state, _, _, _ = env.step(key, state, actions)

    key_tile = int(TileType.KEY)
    ball_tile = int(TileType.BALL)
    map_tile = int(TileType.MAP)

    map_np = np.array(next_state.map_array)
    color_np = np.array(next_state.color_map)
    map_positions = np.argwhere(map_np == map_tile)
    assert map_positions.shape[0] == 1
    y, x = map_positions[0]
    assert color_np[y, x] == int(Colors.BLUE)
    assert int(np.sum(map_np == key_tile)) == 0
    assert int(np.sum(map_np == ball_tile)) == 0


def test_distractor_table_drop_dead_end_ends_episode_with_strong_penalty():
    grid_size = 4
    map_array = jnp.full((grid_size, grid_size), TileType.OPEN_FAST, dtype=jnp.int32)
    color_map = jnp.full((grid_size, grid_size), Colors.BLACK, dtype=jnp.int32)

    # Fixed distractor pair on board: BALL[GREEN] adjacent to where agent 0 will drop KEY[RED].
    map_array = map_array.at[1, 2].set(TileType.BALL)
    color_map = color_map.at[1, 2].set(Colors.GREEN)

    # Minimal scaffold ruleset for reset/goal compilation.
    ruleset = jnp.array(
        [[1, int(TileType.KEY), 0, int(Colors.RED), 0, 0]], dtype=jnp.int32
    )

    # Build distractor lookup table entry:
    # KEY[RED] + BALL[GREEN] -> MAP[BLUE]
    table = np.zeros((NUM_ITEMS, NUM_COLORS, NUM_ITEMS, NUM_COLORS, 3), dtype=np.int32)
    table[int(ItemType.KEY), int(Colors.RED), int(ItemType.BALL), int(Colors.GREEN)] = (
        int(ItemType.MAP),
        int(Colors.BLUE),
        1,
    )
    table[int(ItemType.BALL), int(Colors.GREEN), int(ItemType.KEY), int(Colors.RED)] = (
        int(ItemType.MAP),
        int(Colors.BLUE),
        1,
    )

    env = Banyan(
        grid_size=grid_size,
        max_steps=8,
        map_array=map_array,
        color_map=color_map,
        max_depth=1,
        max_rules=1,
        distractor_table=jnp.array(table, dtype=jnp.int32),
        distractor_combine_penalty=-1.0,
    )
    key = jax.random.PRNGKey(0)
    _, state = env.reset(
        key,
        {"map_array": map_array, "color_map": color_map},
        ruleset,
    )

    # Force deterministic setup for the drop-triggered distractor merge.
    positions = jnp.array([1, 1], dtype=jnp.int32)
    inv = np.zeros((NUM_ITEMS,), dtype=bool)
    inv_cols = np.full((NUM_ITEMS,), int(Colors.BLACK), dtype=np.int32)
    inv[int(ItemType.KEY)] = True
    inv_cols[int(ItemType.KEY)] = int(Colors.RED)
    state = state.replace(
        positions=positions,
        inventories=jnp.array(inv),
        inventory_colors=jnp.array(inv_cols),
    )

    actions = jnp.asarray(Action.DROP, dtype=jnp.int32)
    _, next_state, rewards, done, info = env.step(
        key, state, actions
    )

    # Merge result appears at the drop tile (scan merges right-neighbor into left cell).
    assert int(next_state.map_array[1, 1]) == int(TileType.MAP)
    assert int(next_state.color_map[1, 1]) == int(Colors.BLUE)
    assert int(next_state.map_array[1, 2]) == int(TileType.OPEN_FAST)
    assert int(next_state.color_map[1, 2]) == int(Colors.BLACK)

    np.testing.assert_allclose(
        np.array(info["reward_distractor_penalty"]),
        np.float32(-1.0),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(rewards),
        np.float32(-1.0),
        atol=1e-6,
    )
    assert bool(np.array(done))
    np.testing.assert_allclose(
        np.array(info["dead_end"]),
        np.float32(1.0),
        atol=1e-6,
    )


def test_distractor_drop_attempt_without_actual_drop_does_not_dead_end():
    grid_size = 4
    map_array = jnp.full((grid_size, grid_size), TileType.OPEN_FAST, dtype=jnp.int32)
    color_map = jnp.full((grid_size, grid_size), Colors.BLACK, dtype=jnp.int32)

    # Distractor-valid adjacent pair already on board.
    map_array = map_array.at[1, 1].set(TileType.KEY)
    color_map = color_map.at[1, 1].set(Colors.RED)
    map_array = map_array.at[1, 2].set(TileType.BALL)
    color_map = color_map.at[1, 2].set(Colors.GREEN)

    ruleset = jnp.array(
        [[1, int(TileType.KEY), 0, int(Colors.RED), 0, 0]], dtype=jnp.int32
    )

    table = np.zeros((NUM_ITEMS, NUM_COLORS, NUM_ITEMS, NUM_COLORS, 3), dtype=np.int32)
    table[int(ItemType.KEY), int(Colors.RED), int(ItemType.BALL), int(Colors.GREEN)] = (
        int(ItemType.MAP),
        int(Colors.BLUE),
        1,
    )
    table[int(ItemType.BALL), int(Colors.GREEN), int(ItemType.KEY), int(Colors.RED)] = (
        int(ItemType.MAP),
        int(Colors.BLUE),
        1,
    )

    env = Banyan(
        grid_size=grid_size,
        max_steps=8,
        map_array=map_array,
        color_map=color_map,
        max_depth=1,
        max_rules=1,
        distractor_table=jnp.array(table, dtype=jnp.int32),
        distractor_combine_penalty=-1.0,
    )
    key = jax.random.PRNGKey(0)
    _, state = env.reset(
        key,
        {"map_array": map_array, "color_map": color_map},
        ruleset,
    )

    # Agent 0 stands on KEY[RED] but has empty inventory, so DROP is an attempt only.
    positions = jnp.array([1, 1], dtype=jnp.int32)
    inv = np.zeros((NUM_ITEMS,), dtype=bool)
    inv_cols = np.full((NUM_ITEMS,), int(Colors.BLACK), dtype=np.int32)
    state = state.replace(
        positions=positions,
        inventories=jnp.array(inv),
        inventory_colors=jnp.array(inv_cols),
    )

    actions = jnp.asarray(Action.DROP, dtype=jnp.int32)
    _, next_state, rewards, done, info = env.step(
        key, state, actions
    )

    # No successful drop => distractor merge must not trigger.
    assert int(next_state.map_array[1, 1]) == int(TileType.KEY)
    assert int(next_state.color_map[1, 1]) == int(Colors.RED)
    assert int(next_state.map_array[1, 2]) == int(TileType.BALL)
    assert int(next_state.color_map[1, 2]) == int(Colors.GREEN)

    np.testing.assert_allclose(
        np.array(info["action_drop"]),
        np.float32(1.0),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(info["action_drop_success"]),
        np.float32(0.0),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(info["action_drop_failed"]),
        np.float32(1.0),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(info["reward_distractor_penalty"]),
        np.float32(0.0),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(info["dead_end"]),
        np.float32(0.0),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(rewards),
        np.float32(-0.001),
        atol=1e-6,
    )
    assert bool(np.array(done)) is False


def test_distractor_dead_end_requires_pair_to_include_dropped_token():
    grid_size = 4
    map_array = jnp.full((grid_size, grid_size), TileType.OPEN_FAST, dtype=jnp.int32)
    color_map = jnp.full((grid_size, grid_size), Colors.BLACK, dtype=jnp.int32)

    # Agent drops KEY[RED] at (1,1), combine with BALL[GREEN] at (1,2) -> MAP[BLUE].
    # STAR[YELLOW] at (2,1) is adjacent afterward.
    # Distractor table contains MAP[BLUE] + STAR[YELLOW], which should NOT dead-end
    # because the dropped token was KEY[RED], not MAP[BLUE].
    map_array = map_array.at[1, 2].set(TileType.BALL)
    color_map = color_map.at[1, 2].set(Colors.GREEN)
    map_array = map_array.at[2, 1].set(TileType.STAR)
    color_map = color_map.at[2, 1].set(Colors.YELLOW)

    combine_key_ball_to_map = jnp.array(
        [
            2,  # combine
            int(ItemType.KEY),
            int(ItemType.BALL),
            int(ItemType.MAP),
            1,  # required adjacent
            _pack_colors(int(Colors.RED), int(Colors.GREEN), int(Colors.BLUE)),
        ],
        dtype=jnp.int32,
    )
    ruleset = jnp.stack(
        [
            jnp.array([1, int(TileType.KEY), 0, int(Colors.RED), 0, 0], dtype=jnp.int32),
            combine_key_ball_to_map,
        ],
        axis=0,
    )

    table = np.zeros((NUM_ITEMS, NUM_COLORS, NUM_ITEMS, NUM_COLORS, 3), dtype=np.int32)
    table[int(ItemType.MAP), int(Colors.BLUE), int(ItemType.STAR), int(Colors.YELLOW)] = (
        int(ItemType.TRIANGLE),
        int(Colors.PURPLE),
        1,
    )
    table[int(ItemType.STAR), int(Colors.YELLOW), int(ItemType.MAP), int(Colors.BLUE)] = (
        int(ItemType.TRIANGLE),
        int(Colors.PURPLE),
        1,
    )

    env = Banyan(
        grid_size=grid_size,
        max_steps=8,
        map_array=map_array,
        color_map=color_map,
        max_depth=2,
        max_rules=2,
        distractor_table=jnp.array(table, dtype=jnp.int32),
        distractor_combine_penalty=-1.0,
    )
    key = jax.random.PRNGKey(0)
    _, state = env.reset(
        key,
        {"map_array": map_array, "color_map": color_map},
        ruleset,
    )

    positions = jnp.array([1, 1], dtype=jnp.int32)
    inv = np.zeros((NUM_ITEMS,), dtype=bool)
    inv_cols = np.full((NUM_ITEMS,), int(Colors.BLACK), dtype=np.int32)
    inv[int(ItemType.KEY)] = True
    inv_cols[int(ItemType.KEY)] = int(Colors.RED)
    state = state.replace(
        positions=positions,
        inventories=jnp.array(inv),
        inventory_colors=jnp.array(inv_cols),
    )

    actions = jnp.asarray(Action.DROP, dtype=jnp.int32)
    _, next_state, rewards, done, info = env.step(
        key, state, actions
    )

    # Core combine happened.
    assert int(next_state.map_array[1, 1]) == int(TileType.MAP)
    assert int(next_state.color_map[1, 1]) == int(Colors.BLUE)
    # But no distractor dead-end because dropped token KEY is not in MAP+STAR pair.
    np.testing.assert_allclose(
        np.array(info["dead_end"]),
        np.float32(0.0),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(info["reward_distractor_penalty"]),
        np.float32(0.0),
        atol=1e-6,
    )
    # Normal step reward path (not terminal override).
    assert float(np.array(rewards)) > -1.0
    assert bool(np.array(done)) is False
