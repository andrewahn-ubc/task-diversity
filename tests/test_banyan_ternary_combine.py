import numpy as np
import jax
import jax.numpy as jnp

from banyan_grid.utils.banyan import get_map
from banyan_grid.environment import Banyan
from banyan_grid.environment.constants import (
    Action,
    Colors,
    ITEM_TO_TILE,
    RULE_TYPE_COMBINE,
    RULE_TYPE_TERNARY_COMBINE,
    TileType,
)
from banyan_grid.tasks.ruleset_codec import (
    pack_rules_uint32_np,
    unpack_rules_uint32,
)
from banyan_grid.tasks.ruleset_factory import (
    _pair_to_output_uid_global_np,
    _triple_to_output_uid_global_np,
    _uid_to_item_color_np,
    _unary_to_output_uid_global_np,
)


def _base_maps(grid_size: int):
    map_array = jnp.full((grid_size, grid_size), TileType.OPEN_FAST, dtype=jnp.int32)
    color_map = jnp.full((grid_size, grid_size), Colors.BLACK, dtype=jnp.int32)
    return map_array, color_map


def _ternary_rule(
    in1: int,
    in2: int,
    in3: int,
    out: int,
    c1: int,
    c2: int,
    c3: int,
    cout: int,
    *,
    required_adjacent: bool = True,
) -> jnp.ndarray:
    meta = ((int(in3) & 0xF) << 1) | (1 if required_adjacent else 0)
    packed = (
        ((int(cout) & 0xF) << 12)
        | ((int(c3) & 0xF) << 8)
        | ((int(c2) & 0xF) << 4)
        | (int(c1) & 0xF)
    )
    return jnp.array(
        [[RULE_TYPE_TERNARY_COMBINE, in1, in2, out, meta, packed]],
        dtype=jnp.int32,
    )


def _combine_rule(
    in1: int,
    in2: int,
    out: int,
    c1: int,
    c2: int,
    cout: int,
    *,
    required_adjacent: bool = True,
) -> jnp.ndarray:
    packed = (
        ((int(cout) & 0xF) << 8)
        | ((int(c2) & 0xF) << 4)
        | (int(c1) & 0xF)
    )
    return jnp.array(
        [
            RULE_TYPE_COMBINE,
            in1,
            in2,
            out,
            1 if required_adjacent else 0,
            packed,
        ],
        dtype=jnp.int32,
    )


def test_ternary_codec_roundtrip():
    ruleset = _ternary_rule(
        0,
        1,
        2,
        3,
        int(Colors.RED),
        int(Colors.GREEN),
        int(Colors.BLUE),
        int(Colors.YELLOW),
    )
    packed = pack_rules_uint32_np(np.asarray(ruleset, dtype=np.int32))
    decoded = np.asarray(unpack_rules_uint32(jnp.asarray(packed, dtype=jnp.uint32)))
    np.testing.assert_array_equal(decoded, np.asarray(ruleset, dtype=np.int32))


def test_compact_codec_roundtrip_with_new_item_ids():
    uid_a = 8 * int(Colors.BLACK) + int(Colors.CYAN)
    uid_b = 9 * int(Colors.BLACK) + int(Colors.LIME)
    uid_c = 8 * int(Colors.BLACK) + int(Colors.ORANGE)

    uid_pair_out = _pair_to_output_uid_global_np(uid_a, uid_b, base_seed=0)
    uid_unary_out = _unary_to_output_uid_global_np(uid_b, base_seed=0)
    uid_ternary_out = _triple_to_output_uid_global_np(uid_a, uid_b, uid_c, base_seed=0)

    pair_item, pair_color = _uid_to_item_color_np(uid_pair_out)
    unary_item, unary_color = _uid_to_item_color_np(uid_unary_out)
    ternary_item, ternary_color = _uid_to_item_color_np(uid_ternary_out)

    rows = np.array(
        [
            [
                2,
                8,
                9,
                pair_item,
                1,
                ((int(pair_color) & 0xF) << 8)
                | ((int(Colors.LIME) & 0xF) << 4)
                | (int(Colors.CYAN) & 0xF),
            ],
            [
                4,
                9,
                0,
                unary_item,
                1,
                ((int(unary_color) & 0xF) << 8) | (int(Colors.LIME) & 0xF),
            ],
            [
                int(RULE_TYPE_TERNARY_COMBINE),
                8,
                9,
                ternary_item,
                ((8 & 0xF) << 1) | 1,
                ((int(ternary_color) & 0xF) << 12)
                | ((int(Colors.ORANGE) & 0xF) << 8)
                | ((int(Colors.LIME) & 0xF) << 4)
                | (int(Colors.CYAN) & 0xF),
            ],
        ],
        dtype=np.int32,
    )

    packed = pack_rules_uint32_np(rows)
    decoded = np.asarray(unpack_rules_uint32(jnp.asarray(packed, dtype=jnp.uint32)))
    np.testing.assert_array_equal(decoded, rows)


def test_adjacent_items_auto_merge_ternary():
    key = jax.random.PRNGKey(0)
    ruleset = _ternary_rule(
        0,
        1,
        2,
        3,
        int(Colors.RED),
        int(Colors.GREEN),
        int(Colors.BLUE),
        int(Colors.YELLOW),
    )

    map_array, color_map = _base_maps(5)
    pos1 = (2, 2)
    pos2 = (2, 3)
    pos3 = (3, 2)
    map_array = map_array.at[pos1].set(ITEM_TO_TILE[0])
    color_map = color_map.at[pos1].set(Colors.RED)
    map_array = map_array.at[pos2].set(ITEM_TO_TILE[1])
    color_map = color_map.at[pos2].set(Colors.GREEN)
    map_array = map_array.at[pos3].set(ITEM_TO_TILE[2])
    color_map = color_map.at[pos3].set(Colors.BLUE)

    env = Banyan(
        grid_size=5,
        max_steps=10,
        map_array=map_array,
        color_map=color_map,
        max_rules=1,
    )
    _, state = env.reset(key, {"map_array": map_array, "color_map": color_map}, ruleset)
    actions = jnp.asarray(Action.STAY, dtype=jnp.int32)
    _, next_state, _, _, _ = env.step(key, state, actions)

    np.testing.assert_equal(np.array(next_state.map_array[pos1]), int(ITEM_TO_TILE[3]))
    np.testing.assert_equal(np.array(next_state.color_map[pos1]), int(Colors.YELLOW))
    np.testing.assert_equal(np.array(next_state.map_array[pos2]), int(TileType.OPEN_FAST))
    np.testing.assert_equal(np.array(next_state.color_map[pos2]), int(Colors.BLACK))
    np.testing.assert_equal(np.array(next_state.map_array[pos3]), int(TileType.OPEN_FAST))
    np.testing.assert_equal(np.array(next_state.color_map[pos3]), int(Colors.BLACK))


def test_ternary_merge_uses_lexicographic_first_valid_triple():
    key = jax.random.PRNGKey(4)
    ruleset = _ternary_rule(
        0,
        1,
        2,
        3,
        int(Colors.RED),
        int(Colors.GREEN),
        int(Colors.BLUE),
        int(Colors.YELLOW),
    )

    map_array, color_map = _base_maps(5)
    pos1 = (0, 0)
    pos2 = (0, 1)
    pos3 = (0, 2)
    later_pos1 = (1, 1)
    later_pos3 = (2, 2)
    for pos, item, color in (
        (pos1, 0, Colors.RED),
        (pos2, 1, Colors.GREEN),
        (pos3, 2, Colors.BLUE),
        (later_pos1, 0, Colors.RED),
        (later_pos3, 2, Colors.BLUE),
    ):
        map_array = map_array.at[pos].set(ITEM_TO_TILE[item])
        color_map = color_map.at[pos].set(color)

    env = Banyan(
        grid_size=5,
        max_steps=10,
        map_array=map_array,
        color_map=color_map,
        max_rules=1,
    )
    _, state = env.reset(key, {"map_array": map_array, "color_map": color_map}, ruleset)
    actions = jnp.asarray(Action.STAY, dtype=jnp.int32)
    _, next_state, _, _, _ = env.step(key, state, actions)

    np.testing.assert_equal(np.array(next_state.map_array[pos1]), int(ITEM_TO_TILE[3]))
    np.testing.assert_equal(np.array(next_state.color_map[pos1]), int(Colors.YELLOW))
    np.testing.assert_equal(np.array(next_state.map_array[pos2]), int(TileType.OPEN_FAST))
    np.testing.assert_equal(np.array(next_state.color_map[pos2]), int(Colors.BLACK))
    np.testing.assert_equal(np.array(next_state.map_array[pos3]), int(TileType.OPEN_FAST))
    np.testing.assert_equal(np.array(next_state.color_map[pos3]), int(Colors.BLACK))
    np.testing.assert_equal(
        np.array(next_state.map_array[later_pos1]), int(ITEM_TO_TILE[0])
    )
    np.testing.assert_equal(
        np.array(next_state.map_array[later_pos3]), int(ITEM_TO_TILE[2])
    )


def test_disconnected_items_do_not_merge_ternary():
    key = jax.random.PRNGKey(1)
    ruleset = _ternary_rule(
        0,
        1,
        2,
        3,
        int(Colors.RED),
        int(Colors.GREEN),
        int(Colors.BLUE),
        int(Colors.YELLOW),
    )

    map_array, color_map = _base_maps(5)
    pos1 = (2, 2)
    pos2 = (2, 4)
    pos3 = (4, 2)
    map_array = map_array.at[pos1].set(ITEM_TO_TILE[0])
    color_map = color_map.at[pos1].set(Colors.RED)
    map_array = map_array.at[pos2].set(ITEM_TO_TILE[1])
    color_map = color_map.at[pos2].set(Colors.GREEN)
    map_array = map_array.at[pos3].set(ITEM_TO_TILE[2])
    color_map = color_map.at[pos3].set(Colors.BLUE)

    env = Banyan(
        grid_size=5,
        max_steps=10,
        map_array=map_array,
        color_map=color_map,
        max_rules=1,
    )
    _, state = env.reset(key, {"map_array": map_array, "color_map": color_map}, ruleset)
    actions = jnp.asarray(Action.STAY, dtype=jnp.int32)
    _, next_state, _, _, _ = env.step(key, state, actions)

    np.testing.assert_equal(np.array(next_state.map_array[pos1]), int(ITEM_TO_TILE[0]))
    np.testing.assert_equal(np.array(next_state.color_map[pos1]), int(Colors.RED))
    np.testing.assert_equal(np.array(next_state.map_array[pos2]), int(ITEM_TO_TILE[1]))
    np.testing.assert_equal(np.array(next_state.color_map[pos2]), int(Colors.GREEN))
    np.testing.assert_equal(np.array(next_state.map_array[pos3]), int(ITEM_TO_TILE[2]))
    np.testing.assert_equal(np.array(next_state.color_map[pos3]), int(Colors.BLUE))


def test_unrestricted_ternary_merges_disconnected_repeated_tokens():
    key = jax.random.PRNGKey(9)
    ruleset = _ternary_rule(
        0,
        0,
        1,
        3,
        int(Colors.RED),
        int(Colors.RED),
        int(Colors.BLUE),
        int(Colors.YELLOW),
        required_adjacent=False,
    )

    map_array, color_map = _base_maps(5)
    first_key = (0, 0)
    second_key = (3, 4)
    ball = (4, 1)
    for pos, item, color in (
        (first_key, 0, Colors.RED),
        (second_key, 0, Colors.RED),
        (ball, 1, Colors.BLUE),
    ):
        map_array = map_array.at[pos].set(ITEM_TO_TILE[item])
        color_map = color_map.at[pos].set(color)

    env = Banyan(
        grid_size=5,
        max_steps=10,
        map_array=map_array,
        color_map=color_map,
        max_rules=1,
    )
    _, state = env.reset(key, {"map_array": map_array, "color_map": color_map}, ruleset)
    actions = jnp.asarray(Action.STAY, dtype=jnp.int32)
    _, next_state, _, _, _ = env.step(key, state, actions)

    np.testing.assert_equal(np.array(next_state.map_array[first_key]), int(ITEM_TO_TILE[3]))
    np.testing.assert_equal(np.array(next_state.color_map[first_key]), int(Colors.YELLOW))
    np.testing.assert_equal(
        np.array(next_state.map_array[second_key]), int(TileType.OPEN_FAST)
    )
    np.testing.assert_equal(np.array(next_state.color_map[second_key]), int(Colors.BLACK))
    np.testing.assert_equal(np.array(next_state.map_array[ball]), int(TileType.OPEN_FAST))
    np.testing.assert_equal(np.array(next_state.color_map[ball]), int(Colors.BLACK))


def test_unrestricted_ternary_all_same_token_requires_three_distinct_cells():
    key = jax.random.PRNGKey(10)
    ruleset = _ternary_rule(
        0,
        0,
        0,
        3,
        int(Colors.RED),
        int(Colors.RED),
        int(Colors.RED),
        int(Colors.YELLOW),
        required_adjacent=False,
    )

    two_item_map, two_item_colors = _base_maps(5)
    first_key = (0, 0)
    second_key = (4, 4)
    for pos in (first_key, second_key):
        two_item_map = two_item_map.at[pos].set(ITEM_TO_TILE[0])
        two_item_colors = two_item_colors.at[pos].set(Colors.RED)

    env = Banyan(
        grid_size=5,
        max_steps=10,
        map_array=two_item_map,
        color_map=two_item_colors,
        max_rules=1,
    )
    _, state = env.reset(
        key,
        {"map_array": two_item_map, "color_map": two_item_colors},
        ruleset,
    )
    actions = jnp.asarray(Action.STAY, dtype=jnp.int32)
    _, unchanged_state, _, _, _ = env.step(key, state, actions)
    np.testing.assert_array_equal(
        np.asarray(unchanged_state.map_array), np.asarray(two_item_map)
    )
    np.testing.assert_array_equal(
        np.asarray(unchanged_state.color_map), np.asarray(two_item_colors)
    )

    three_item_map = two_item_map.at[2, 2].set(ITEM_TO_TILE[0])
    three_item_colors = two_item_colors.at[2, 2].set(Colors.RED)
    _, state = env.reset(
        key,
        {"map_array": three_item_map, "color_map": three_item_colors},
        ruleset,
    )
    _, next_state, _, _, _ = env.step(key, state, actions)

    np.testing.assert_equal(np.array(next_state.map_array[first_key]), int(ITEM_TO_TILE[3]))
    np.testing.assert_equal(np.array(next_state.color_map[first_key]), int(Colors.YELLOW))
    np.testing.assert_equal(
        np.array(next_state.map_array[2, 2]), int(TileType.OPEN_FAST)
    )
    np.testing.assert_equal(np.array(next_state.color_map[2, 2]), int(Colors.BLACK))
    np.testing.assert_equal(
        np.array(next_state.map_array[second_key]), int(TileType.OPEN_FAST)
    )
    np.testing.assert_equal(np.array(next_state.color_map[second_key]), int(Colors.BLACK))


def test_ternary_requires_third_item_adjacent_to_connected_pair():
    key = jax.random.PRNGKey(6)
    ruleset = _ternary_rule(
        0,
        1,
        2,
        3,
        int(Colors.RED),
        int(Colors.GREEN),
        int(Colors.BLUE),
        int(Colors.YELLOW),
    )

    map_array, color_map = _base_maps(5)
    pos1 = (0, 0)
    pos2 = (0, 1)
    pos3 = (4, 4)
    map_array = map_array.at[pos1].set(ITEM_TO_TILE[0])
    color_map = color_map.at[pos1].set(Colors.RED)
    map_array = map_array.at[pos2].set(ITEM_TO_TILE[1])
    color_map = color_map.at[pos2].set(Colors.GREEN)
    map_array = map_array.at[pos3].set(ITEM_TO_TILE[2])
    color_map = color_map.at[pos3].set(Colors.BLUE)

    env = Banyan(
        grid_size=5,
        max_steps=10,
        map_array=map_array,
        color_map=color_map,
        max_rules=1,
    )
    _, state = env.reset(key, {"map_array": map_array, "color_map": color_map}, ruleset)
    actions = jnp.asarray(Action.STAY, dtype=jnp.int32)
    _, next_state, _, _, _ = env.step(key, state, actions)

    np.testing.assert_equal(np.array(next_state.map_array[pos1]), int(ITEM_TO_TILE[0]))
    np.testing.assert_equal(np.array(next_state.color_map[pos1]), int(Colors.RED))
    np.testing.assert_equal(np.array(next_state.map_array[pos2]), int(ITEM_TO_TILE[1]))
    np.testing.assert_equal(np.array(next_state.color_map[pos2]), int(Colors.GREEN))
    np.testing.assert_equal(np.array(next_state.map_array[pos3]), int(ITEM_TO_TILE[2]))
    np.testing.assert_equal(np.array(next_state.color_map[pos3]), int(Colors.BLUE))


def test_ternary_allows_common_neighbor_when_first_pair_is_not_adjacent():
    key = jax.random.PRNGKey(7)
    ruleset = _ternary_rule(
        0,
        1,
        2,
        3,
        int(Colors.RED),
        int(Colors.GREEN),
        int(Colors.BLUE),
        int(Colors.YELLOW),
    )

    map_array, color_map = _base_maps(5)
    pos1 = (0, 0)
    pos2 = (0, 2)
    pos3 = (0, 1)
    map_array = map_array.at[pos1].set(ITEM_TO_TILE[0])
    color_map = color_map.at[pos1].set(Colors.RED)
    map_array = map_array.at[pos2].set(ITEM_TO_TILE[1])
    color_map = color_map.at[pos2].set(Colors.GREEN)
    map_array = map_array.at[pos3].set(ITEM_TO_TILE[2])
    color_map = color_map.at[pos3].set(Colors.BLUE)

    env = Banyan(
        grid_size=5,
        max_steps=10,
        map_array=map_array,
        color_map=color_map,
        max_rules=1,
    )
    _, state = env.reset(key, {"map_array": map_array, "color_map": color_map}, ruleset)
    actions = jnp.asarray(Action.STAY, dtype=jnp.int32)
    _, next_state, _, _, _ = env.step(key, state, actions)

    np.testing.assert_equal(np.array(next_state.map_array[pos1]), int(ITEM_TO_TILE[3]))
    np.testing.assert_equal(np.array(next_state.color_map[pos1]), int(Colors.YELLOW))
    np.testing.assert_equal(np.array(next_state.map_array[pos2]), int(TileType.OPEN_FAST))
    np.testing.assert_equal(np.array(next_state.color_map[pos2]), int(Colors.BLACK))
    np.testing.assert_equal(np.array(next_state.map_array[pos3]), int(TileType.OPEN_FAST))
    np.testing.assert_equal(np.array(next_state.color_map[pos3]), int(Colors.BLACK))


def test_reset_goal_and_depth_use_ternary_output():
    key = jax.random.PRNGKey(2)
    ruleset = _ternary_rule(
        0,
        1,
        2,
        3,
        int(Colors.RED),
        int(Colors.GREEN),
        int(Colors.BLUE),
        int(Colors.YELLOW),
    )
    map_array, color_map = _base_maps(5)
    env = Banyan(
        grid_size=5,
        max_steps=10,
        map_array=map_array,
        color_map=color_map,
        max_rules=1,
    )
    _, state = env.reset(key, {"map_array": map_array, "color_map": color_map}, ruleset)

    assert int(np.array(state.goal_item)) == 3
    assert int(np.array(state.goal_color)) == int(Colors.YELLOW)
    assert int(np.array(state.depth)) == 2
    assert bool(np.array(state.relevant_item_mask[2, int(Colors.BLUE)]))
    assert bool(np.array(state.relevant_item_mask[3, int(Colors.YELLOW)]))


def test_banyan_map_generation_places_all_ternary_leaves():
    key = jax.random.PRNGKey(3)
    ruleset = _ternary_rule(
        0,
        1,
        2,
        3,
        int(Colors.RED),
        int(Colors.GREEN),
        int(Colors.BLUE),
        int(Colors.YELLOW),
    )

    map_array, color_map = get_map(key, 5, ruleset)
    map_np = np.array(map_array)
    color_np = np.array(color_map)

    for item, color in [(0, Colors.RED), (1, Colors.GREEN), (2, Colors.BLUE)]:
        positions = np.argwhere(
            (map_np == int(ITEM_TO_TILE[item])) & (color_np == int(color))
        )
        assert positions.shape[0] == 1


def test_combine_dispatch_skips_count_match_without_grid_match():
    key = jax.random.PRNGKey(5)
    # Rule 0 has enough tokens by count, but its inputs are not adjacent.
    invalid_first = _combine_rule(
        0,
        1,
        3,
        int(Colors.RED),
        int(Colors.GREEN),
        int(Colors.YELLOW),
    )
    # Rule 1 is the first rule that can actually merge on the grid.
    valid_second = _combine_rule(
        0,
        2,
        4,
        int(Colors.RED),
        int(Colors.BLUE),
        int(Colors.PURPLE),
    )
    ruleset = jnp.stack([invalid_first, valid_second], axis=0)

    map_array, color_map = _base_maps(5)
    pos_key = (0, 0)
    pos_ball = (4, 4)
    pos_map = (0, 1)
    for pos, item, color in (
        (pos_key, 0, Colors.RED),
        (pos_ball, 1, Colors.GREEN),
        (pos_map, 2, Colors.BLUE),
    ):
        map_array = map_array.at[pos].set(ITEM_TO_TILE[item])
        color_map = color_map.at[pos].set(color)

    env = Banyan(
        grid_size=5,
        max_steps=10,
        map_array=map_array,
        color_map=color_map,
        max_rules=2,
    )
    _, state = env.reset(key, {"map_array": map_array, "color_map": color_map}, ruleset)
    actions = jnp.asarray(Action.STAY, dtype=jnp.int32)
    _, next_state, _, _, _ = env.step(key, state, actions)

    np.testing.assert_equal(np.array(next_state.map_array[pos_key]), int(ITEM_TO_TILE[4]))
    np.testing.assert_equal(np.array(next_state.color_map[pos_key]), int(Colors.PURPLE))
    np.testing.assert_equal(
        np.array(next_state.map_array[pos_map]), int(TileType.OPEN_FAST)
    )
    np.testing.assert_equal(np.array(next_state.map_array[pos_ball]), int(ITEM_TO_TILE[1]))
    np.testing.assert_equal(np.array(next_state.color_map[pos_ball]), int(Colors.GREEN))


def test_ternary_dispatch_skips_count_match_without_grid_match():
    key = jax.random.PRNGKey(8)
    # Rule 0 has the right token counts, but only one adjacency edge.
    invalid_first = _ternary_rule(
        0,
        1,
        2,
        3,
        int(Colors.RED),
        int(Colors.GREEN),
        int(Colors.BLUE),
        int(Colors.YELLOW),
    )[0]
    # Rule 1 is the first rule with two adjacency edges on the grid.
    valid_second = _ternary_rule(
        0,
        1,
        4,
        5,
        int(Colors.RED),
        int(Colors.GREEN),
        int(Colors.PURPLE),
        int(Colors.ORANGE),
    )[0]
    ruleset = jnp.stack([invalid_first, valid_second], axis=0)

    map_array, color_map = _base_maps(5)
    pos_key = (0, 0)
    pos_ball = (0, 1)
    pos_map_far = (4, 4)
    pos_star = (1, 0)
    for pos, item, color in (
        (pos_key, 0, Colors.RED),
        (pos_ball, 1, Colors.GREEN),
        (pos_map_far, 2, Colors.BLUE),
        (pos_star, 4, Colors.PURPLE),
    ):
        map_array = map_array.at[pos].set(ITEM_TO_TILE[item])
        color_map = color_map.at[pos].set(color)

    env = Banyan(
        grid_size=5,
        max_steps=10,
        map_array=map_array,
        color_map=color_map,
        max_rules=2,
    )
    _, state = env.reset(key, {"map_array": map_array, "color_map": color_map}, ruleset)
    actions = jnp.asarray(Action.STAY, dtype=jnp.int32)
    _, next_state, _, _, _ = env.step(key, state, actions)

    np.testing.assert_equal(np.array(next_state.map_array[pos_key]), int(ITEM_TO_TILE[5]))
    np.testing.assert_equal(np.array(next_state.color_map[pos_key]), int(Colors.ORANGE))
    np.testing.assert_equal(
        np.array(next_state.map_array[pos_ball]), int(TileType.OPEN_FAST)
    )
    np.testing.assert_equal(np.array(next_state.color_map[pos_ball]), int(Colors.BLACK))
    np.testing.assert_equal(
        np.array(next_state.map_array[pos_star]), int(TileType.OPEN_FAST)
    )
    np.testing.assert_equal(np.array(next_state.color_map[pos_star]), int(Colors.BLACK))
    np.testing.assert_equal(
        np.array(next_state.map_array[pos_map_far]), int(ITEM_TO_TILE[2])
    )
    np.testing.assert_equal(np.array(next_state.color_map[pos_map_far]), int(Colors.BLUE))
