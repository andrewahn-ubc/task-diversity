# ruff: noqa: F403, F405

import abc
from functools import lru_cache
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from banyan_grid.environment.constants import *

"""
rule encodings are 6 elements long
[rule_type, param1, param2, param3, param4, param5]
"""


class BaseRule(struct.PyTreeNode):
    """Base class for all rules with common JAX patterns."""

    @abc.abstractmethod
    def __call__(self, state, actions, env):
        pass

    @abc.abstractmethod
    def encode(self) -> jax.Array:
        pass

    def get_adjacent_positions(
        self, position: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Get positions in all four directions."""
        y, x = position
        up = jnp.array([y - 1, x])
        right = jnp.array([y, x + 1])
        down = jnp.array([y + 1, x])
        left = jnp.array([y, x - 1])
        return up, right, down, left

    def get_tile_at(self, position: jax.Array, map_array: jax.Array) -> jax.Array:
        """Safely get tile at position with boundary checking."""
        valid_pos = jnp.all((position >= 0) & (position < map_array.shape[0]))
        return jax.lax.select(
            valid_pos, map_array[position[0], position[1]], TileType.BLOCK
        )


COMPILED_OP_NONE = 0
COMPILED_OP_COMBINE = 1
COMPILED_OP_TRANSFORM = 2
MAX_COMPILED_INPUTS = 3
NUM_TOKENS = NUM_ITEMS * NUM_COLORS
EMPTY_TOKEN_ID = jnp.int16(-1)
DEFAULT_MAX_COMBINE_CASCADES = 4
DEFAULT_MAX_TRANSFORM_APPLIES = 4


@lru_cache(maxsize=None)
def _grid_rule_match_indices(
    height: int,
    width: int,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Return static directed pair/triple indices in old rule-search order."""

    height = int(height)
    width = int(width)
    num_cells = height * width

    adjacent_left: list[int] = []
    adjacent_right: list[int] = []
    for left in range(num_cells):
        y, x = divmod(left, width)
        for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            ny = y + dy
            nx = x + dx
            if 0 <= ny < height and 0 <= nx < width:
                adjacent_left.append(left)
                adjacent_right.append(ny * width + nx)

    ternary_required_i: list[int] = []
    ternary_required_j: list[int] = []
    ternary_required_k: list[int] = []
    coords = [divmod(cell, width) for cell in range(num_cells)]
    for i in range(num_cells):
        yi, xi = coords[i]
        for j in range(num_cells):
            if i == j:
                continue
            yj, xj = coords[j]
            ij_adjacent = abs(yi - yj) + abs(xi - xj) == 1
            for k in range(num_cells):
                if i == k or j == k:
                    continue
                yk, xk = coords[k]
                adjacency_edges = (
                    int(ij_adjacent)
                    + int(abs(yi - yk) + abs(xi - xk) == 1)
                    + int(abs(yj - yk) + abs(xj - xk) == 1)
                )
                if adjacency_edges >= 2:
                    ternary_required_i.append(i)
                    ternary_required_j.append(j)
                    ternary_required_k.append(k)

    return (
        tuple(adjacent_left),
        tuple(adjacent_right),
        tuple(ternary_required_i),
        tuple(ternary_required_j),
        tuple(ternary_required_k),
    )


@struct.dataclass
class CompiledOpProgram:
    """Ordered runtime program for rule-like map operations."""

    kind: jax.Array
    input_count: jax.Array
    input_tokens: jax.Array
    output_tile: jax.Array
    output_color: jax.Array
    required_adjacent: jax.Array
    combine_scan_steps: jax.Array
    transform_scan_steps: jax.Array
    count: jax.Array


@struct.dataclass
class RuleRuntimeCache:
    """Rule tensors compiled once at reset and reused every step."""

    row_type: jax.Array
    input_count: jax.Array
    col1: jax.Array
    col2: jax.Array
    col3: jax.Array
    packed: jax.Array
    is_move: jax.Array
    is_collect: jax.Array
    is_combine: jax.Array
    is_transform: jax.Array
    required_adjacent: jax.Array
    collect_tile: jax.Array
    collect_item: jax.Array
    collect_color: jax.Array
    input_item_a: jax.Array
    input_item_b: jax.Array
    input_item_c: jax.Array
    output_item: jax.Array
    input_color_a: jax.Array
    input_color_b: jax.Array
    input_color_c: jax.Array
    output_color: jax.Array
    input_tile_a: jax.Array
    input_tile_b: jax.Array
    input_tile_c: jax.Array
    output_tile: jax.Array
    input_token_a: jax.Array
    input_token_b: jax.Array
    input_token_c: jax.Array
    output_token: jax.Array
    first_move_row: jax.Array
    pre_move_ops: CompiledOpProgram
    post_move_ops: CompiledOpProgram
    transform_ops: CompiledOpProgram
    has_explicit_movement_row: jax.Array
    has_transform_rules: jax.Array


def _pack_compiled_program(
    mask: jax.Array,
    kind_by_row: jax.Array,
    input_count_by_row: jax.Array,
    input_tokens_by_row: jax.Array,
    output_tile_by_row: jax.Array,
    output_color_by_row: jax.Array,
    required_adjacent_by_row: jax.Array,
    *,
    out_size: int | None = None,
    combine_scan_length: int | None = None,
    transform_scan_length: int | None = None,
) -> CompiledOpProgram:
    row_count = int(kind_by_row.shape[0])
    row_index = jnp.arange(row_count, dtype=jnp.int32)
    sort_key = jnp.where(mask, row_index, row_index + row_count)
    if out_size is None:
        out_size = row_count
    order = jnp.argsort(sort_key)[: int(out_size)]
    return CompiledOpProgram(
        kind=kind_by_row[order],
        input_count=input_count_by_row[order],
        input_tokens=input_tokens_by_row[order],
        output_tile=output_tile_by_row[order],
        output_color=output_color_by_row[order],
        required_adjacent=required_adjacent_by_row[order],
        combine_scan_steps=jnp.arange(
            max(0, int(combine_scan_length or 0)), dtype=jnp.int32
        ),
        transform_scan_steps=jnp.arange(
            max(0, int(transform_scan_length or 0)), dtype=jnp.int32
        ),
        count=jnp.sum(mask.astype(jnp.int32)),
    )


def compute_rule_execution_signature(codes: jax.Array) -> jax.Array:
    """Return a compact execution signature for bucketing compatible rulesets.

    Layout:
      [row_count,
       pre_binary, pre_ternary, pre_transform,
       post_binary, post_ternary, post_transform]
    """

    codes = jnp.asarray(codes, dtype=jnp.int32)
    row_count = int(codes.shape[0])
    row_type = codes[:, 0]
    is_move = row_type == 0
    is_binary_combine = (row_type == RULE_TYPE_COMBINE) | (
        row_type == RULE_TYPE_DISTRACTOR_COMBINE
    )
    is_ternary_combine = row_type == RULE_TYPE_TERNARY_COMBINE
    is_transform = row_type == RULE_TYPE_TRANSFORM
    row_index = jnp.arange(row_count, dtype=jnp.int32)
    first_move_row = jax.lax.cond(
        jnp.any(is_move),
        lambda _: jnp.argmax(is_move.astype(jnp.int32)),
        lambda _: jnp.asarray(row_count, dtype=jnp.int32),
        operand=None,
    )
    pre_mask = row_index < first_move_row
    post_mask = row_index > first_move_row
    return jnp.asarray(
        [
            row_count,
            jnp.sum((is_binary_combine & pre_mask).astype(jnp.int32)),
            jnp.sum((is_ternary_combine & pre_mask).astype(jnp.int32)),
            jnp.sum((is_transform & pre_mask).astype(jnp.int32)),
            jnp.sum((is_binary_combine & post_mask).astype(jnp.int32)),
            jnp.sum((is_ternary_combine & post_mask).astype(jnp.int32)),
            jnp.sum((is_transform & post_mask).astype(jnp.int32)),
        ],
        dtype=jnp.int32,
    )


def compile_rule_runtime(
    codes: jax.Array,
    *,
    pre_move_program_size: int | None = None,
    post_move_program_size: int | None = None,
    transform_program_size: int | None = None,
    pre_move_combine_steps: int | None = None,
    post_move_combine_steps: int | None = None,
    pre_move_transform_steps: int | None = None,
    post_move_transform_steps: int | None = None,
    all_transform_steps: int | None = None,
) -> RuleRuntimeCache:
    """Compile a raw encoded ruleset into typed tensors for the step kernel."""

    if pre_move_combine_steps is None:
        pre_move_combine_steps = DEFAULT_MAX_COMBINE_CASCADES
    if post_move_combine_steps is None:
        post_move_combine_steps = DEFAULT_MAX_COMBINE_CASCADES
    if pre_move_transform_steps is None:
        pre_move_transform_steps = DEFAULT_MAX_TRANSFORM_APPLIES
    if post_move_transform_steps is None:
        post_move_transform_steps = DEFAULT_MAX_TRANSFORM_APPLIES
    if all_transform_steps is None:
        all_transform_steps = DEFAULT_MAX_TRANSFORM_APPLIES

    codes = jnp.asarray(codes, dtype=jnp.int32)
    row_count = int(codes.shape[0])
    row_type = codes[:, 0]
    col1 = codes[:, 1]
    col2 = codes[:, 2]
    col3 = codes[:, 3]
    packed = codes[:, 5]

    is_move = row_type == 0
    is_collect = row_type == RULE_TYPE_COLLECT
    is_binary_combine = (row_type == RULE_TYPE_COMBINE) | (
        row_type == RULE_TYPE_DISTRACTOR_COMBINE
    )
    is_ternary_combine = row_type == RULE_TYPE_TERNARY_COMBINE
    is_combine = is_binary_combine | is_ternary_combine
    is_transform = row_type == RULE_TYPE_TRANSFORM

    input_count = jnp.where(
        is_ternary_combine,
        3,
        jnp.where(is_combine, 2, jnp.where(is_transform, 1, 0)),
    ).astype(jnp.int32)
    required_adjacent = jnp.where(
        is_ternary_combine,
        (codes[:, 4] & 0x1) == 1,
        codes[:, 4] == 1,
    )

    input_color_a = jnp.clip(packed & 0xF, 0, NUM_COLORS - 1)
    input_color_b = jnp.clip((packed >> 4) & 0xF, 0, NUM_COLORS - 1)
    input_color_c = jnp.where(
        is_ternary_combine,
        jnp.clip((packed >> 8) & 0xF, 0, NUM_COLORS - 1),
        jnp.zeros_like(col1),
    )
    output_color = jnp.where(
        is_ternary_combine,
        jnp.clip((packed >> 12) & 0xF, 0, NUM_COLORS - 1),
        jnp.clip((packed >> 8) & 0xF, 0, NUM_COLORS - 1),
    )

    collect_tile = jnp.clip(col1, 0, NUM_TILE_TYPES - 1)
    collect_item = jnp.clip(col2, 0, NUM_ITEMS - 1)
    collect_color = jnp.clip(col3, 0, NUM_COLORS - 1)

    input_item_a = jnp.clip(col1, 0, NUM_ITEMS - 1)
    input_item_b = jnp.clip(col2, 0, NUM_ITEMS - 1)
    input_item_c = jnp.where(
        is_ternary_combine,
        jnp.clip(codes[:, 4] >> 1, 0, NUM_ITEMS - 1),
        jnp.zeros_like(col1),
    )
    output_item = jnp.clip(col3, 0, NUM_ITEMS - 1)

    input_tile_a = ITEM_TO_TILE[input_item_a]
    input_tile_b = ITEM_TO_TILE[input_item_b]
    input_tile_c = ITEM_TO_TILE[input_item_c]
    output_tile = ITEM_TO_TILE[output_item]

    input_token_a = input_item_a * NUM_COLORS + input_color_a
    input_token_b = input_item_b * NUM_COLORS + input_color_b
    input_token_c = input_item_c * NUM_COLORS + input_color_c
    output_token = output_item * NUM_COLORS + output_color
    row_index = jnp.arange(row_count, dtype=jnp.int32)
    first_move_row = jax.lax.cond(
        jnp.any(is_move),
        lambda _: jnp.argmax(is_move.astype(jnp.int32)),
        lambda _: jnp.asarray(row_count, dtype=jnp.int32),
        operand=None,
    )

    exec_kind_by_row = jnp.where(
        is_combine,
        COMPILED_OP_COMBINE,
        jnp.where(
            is_transform,
            COMPILED_OP_TRANSFORM,
            COMPILED_OP_NONE,
        ),
    ).astype(jnp.int32)
    exec_input_count_by_row = input_count
    exec_input_tokens_by_row = jnp.full(
        (row_count, MAX_COMPILED_INPUTS), -1, dtype=jnp.int32
    )
    exec_input_tokens_by_row = exec_input_tokens_by_row.at[:, 0].set(input_token_a)
    exec_input_tokens_by_row = exec_input_tokens_by_row.at[:, 1].set(input_token_b)
    exec_input_tokens_by_row = exec_input_tokens_by_row.at[:, 2].set(input_token_c)
    exec_output_tile_by_row = jnp.where(
        is_combine | is_transform, output_tile, jnp.zeros_like(output_tile)
    )
    exec_output_color_by_row = jnp.where(
        is_combine | is_transform, output_color, jnp.zeros_like(output_color)
    )
    exec_mask = exec_kind_by_row != COMPILED_OP_NONE
    pre_move_mask = exec_mask & (row_index < first_move_row)
    post_move_mask = exec_mask & (row_index > first_move_row)
    pre_move_ops = _pack_compiled_program(
        pre_move_mask,
        exec_kind_by_row,
        exec_input_count_by_row,
        exec_input_tokens_by_row,
        exec_output_tile_by_row,
        exec_output_color_by_row,
        required_adjacent,
        out_size=pre_move_program_size,
        combine_scan_length=pre_move_combine_steps,
        transform_scan_length=pre_move_transform_steps,
    )
    post_move_ops = _pack_compiled_program(
        post_move_mask,
        exec_kind_by_row,
        exec_input_count_by_row,
        exec_input_tokens_by_row,
        exec_output_tile_by_row,
        exec_output_color_by_row,
        required_adjacent,
        out_size=post_move_program_size,
        combine_scan_length=post_move_combine_steps,
        transform_scan_length=post_move_transform_steps,
    )
    transform_ops = _pack_compiled_program(
        is_transform,
        exec_kind_by_row,
        exec_input_count_by_row,
        exec_input_tokens_by_row,
        exec_output_tile_by_row,
        exec_output_color_by_row,
        required_adjacent,
        out_size=transform_program_size,
        combine_scan_length=0,
        transform_scan_length=all_transform_steps,
    )

    return RuleRuntimeCache(
        row_type=row_type,
        input_count=input_count,
        col1=col1,
        col2=col2,
        col3=col3,
        packed=packed,
        is_move=is_move,
        is_collect=is_collect,
        is_combine=is_combine,
        is_transform=is_transform,
        required_adjacent=required_adjacent,
        collect_tile=collect_tile,
        collect_item=collect_item,
        collect_color=collect_color,
        input_item_a=input_item_a,
        input_item_b=input_item_b,
        input_item_c=input_item_c,
        output_item=output_item,
        input_color_a=input_color_a,
        input_color_b=input_color_b,
        input_color_c=input_color_c,
        output_color=output_color,
        input_tile_a=input_tile_a,
        input_tile_b=input_tile_b,
        input_tile_c=input_tile_c,
        output_tile=output_tile,
        input_token_a=input_token_a,
        input_token_b=input_token_b,
        input_token_c=input_token_c,
        output_token=output_token,
        first_move_row=first_move_row,
        pre_move_ops=pre_move_ops,
        post_move_ops=post_move_ops,
        transform_ops=transform_ops,
        has_explicit_movement_row=jnp.any(is_move),
        has_transform_rules=jnp.any(is_transform),
    )


class MovementRule(BaseRule):
    """
    Movement + direction updates.
    Actions:
      - RIGHT=0 => direction=1 => move forward
      - LEFT=1  => direction=3 => move forward
      - UP=2    => direction=0 => move forward
      - DOWN=3  => direction=2 => move forward
      - STAY=4  => no movement
      - DROP=5, PICKUP=6 => no movement
    """

    # Lookup table for action -> direction mapping (much faster than lax.switch)
    ACTION_TO_DIR = jnp.array([1, 3, 0, 2, -1, -1, -1, -1, -1], dtype=jnp.int32)

    def __call__(self, state, action, env):
        action = jnp.asarray(action, dtype=jnp.int32)

        # 1) Possibly update direction using lookup table
        old_dir = state.directions
        new_dir = self.ACTION_TO_DIR[action]
        # if new_dir == -1 => keep old_dir
        final_dir = jax.lax.select(new_dir == -1, old_dir, new_dir)

        # 2) Move forward if one of [UP,RIGHT,DOWN,LEFT], i.e. new_dir != -1
        def do_movement(pos, ddir):
            move_vec = DIR_TO_VEC[ddir]
            intended = pos + move_vec
            clipped = jnp.clip(intended, 0, state.map_array.shape[0] - 1)
            tile = state.map_array[clipped[0], clipped[1]]
            walkable = WALKABLE_MASK[tile]
            return jax.lax.select(walkable, clipped, pos)

        old_pos = state.positions
        new_pos = jax.lax.cond(
            new_dir == -1,
            lambda _: old_pos,
            lambda _: do_movement(old_pos, final_dir),
            operand=None,
        )

        return state.replace(positions=new_pos, directions=final_dir)

    @classmethod
    def decode(cls, encoding: jax.Array) -> "MovementRule":
        # The encoding format is:
        # [0, 0, 0, 0, 0, 0, 0]
        # No extra parameters are needed.
        return cls()

    def encode(self) -> jax.Array:
        # Encode as a vector of zeros with the first element (rule type) = 0.
        return jnp.zeros(MAX_RULE_ENCODING_LEN, dtype=jnp.int32)


class DropRule(BaseRule):
    """Drop one item from inventory onto the agent's tile.

    With multi-item inventory (MAX_INVENTORY_SIZE=3), this drops
    only the first item found in the inventory, not all items.
    """

    def __call__(self, state, action, env):
        action = jnp.asarray(action, dtype=jnp.int32)

        inv = state.inventories
        has_item = jnp.any(inv)
        is_drop = action == Action.DROP
        pos = state.positions
        tile = state.map_array[pos[0], pos[1]]

        can_drop_tile = (
            (tile == TileType.OPEN_FAST)
            | (tile == TileType.OPEN_MEDIUM)
        )
        can_drop = is_drop & has_item & can_drop_tile

        def do_drop(s):
            # Find the first item in inventory (lowest index with True)
            item_idx = jnp.argmax(inv.astype(jnp.int32))
            drop_tile = ITEM_TO_TILE[item_idx]
            drop_color = s.inventory_colors[item_idx]

            # Only clear this specific item slot, not the whole inventory
            new_inv = s.inventories.at[item_idx].set(False)
            new_inv_colors = s.inventory_colors.at[item_idx].set(Colors.BLACK)

            s2 = s.replace(
                inventories=new_inv,
                inventory_colors=new_inv_colors,
                map_array=s.map_array.at[pos[0], pos[1]].set(drop_tile),
                color_map=s.color_map.at[pos[0], pos[1]].set(drop_color),
            )
            token_grid, token_counts = _sync_cell_token_views(
                s.token_grid,
                s.token_counts,
                s2.map_array,
                s2.color_map,
                pos[0],
                pos[1],
            )
            return s2.replace(token_grid=token_grid, token_counts=token_counts)

        return jax.lax.cond(can_drop, do_drop, lambda s: s, state)


class GenericPickupRule(BaseRule):
    """Generic pickup rule - picks up ANY collectible item the agent is standing on.

    This eliminates the need for item-specific collect-rule instances.
    Benefits:
    - No need for intermediate collect rules for depth-3+ tasks
    - Constant rule count regardless of tree structure
    - Simpler ruleset encoding

    The agent can pick up any item if:
    1. Agent is standing on a collectible tile
    2. Agent's inventory has space (< MAX_INVENTORY_SIZE items)
    3. Agent uses PICKUP action
    """

    def __call__(self, state, action, env):
        action = jnp.asarray(action, dtype=jnp.int32)

        inv = state.inventories
        pos = state.positions
        tile = state.map_array[pos[0], pos[1]]
        tile_color = state.color_map[pos[0], pos[1]]

        # Check if this tile is collectible (TILE_TO_ITEM returns item type or -1)
        item_type = TILE_TO_ITEM[tile]
        is_collectible = item_type >= 0

        # Check pickup conditions
        is_pickup = action == Action.PICKUP
        # Allow pickup if inventory has space (fewer than MAX_INVENTORY_SIZE items)
        inventory_count = jnp.sum(inv.astype(jnp.int32))
        has_space = inventory_count < MAX_INVENTORY_SIZE
        can_collect = is_collectible & is_pickup & has_space

        def do_pickup(s):
            # Set the item in inventory
            new_inv = s.inventories.at[item_type].set(True)
            # Record the color
            new_inv_colors = s.inventory_colors.at[item_type].set(tile_color)
            # Clear the tile
            s2 = s.replace(
                inventories=new_inv,
                inventory_colors=new_inv_colors,
                map_array=s.map_array.at[pos[0], pos[1]].set(TileType.OPEN_FAST),
                color_map=s.color_map.at[pos[0], pos[1]].set(Colors.BLACK),
            )
            token_grid, token_counts = _sync_cell_token_views(
                s.token_grid,
                s.token_counts,
                s2.map_array,
                s2.color_map,
                pos[0],
                pos[1],
            )
            return s2.replace(token_grid=token_grid, token_counts=token_counts)

        return jax.lax.cond(can_collect, do_pickup, lambda s: s, state)


def build_token_grid(map_array: jax.Array, color_map: jax.Array) -> jax.Array:
    """Return per-cell token ids for collectible items, else -1."""
    item_ids = TILE_TO_ITEM[map_array]
    valid = item_ids >= 0
    safe_items = jnp.clip(item_ids, 0, NUM_ITEMS - 1)
    safe_colors = jnp.clip(color_map, 0, NUM_COLORS - 1)
    token_ids = safe_items * NUM_COLORS + safe_colors
    return jnp.where(valid, token_ids, EMPTY_TOKEN_ID).astype(jnp.int16)


def build_token_counts(token_grid: jax.Array) -> jax.Array:
    """Return token multiplicities from a per-cell token id grid."""

    valid = token_grid >= 0
    safe_token_ids = jnp.clip(token_grid.astype(jnp.int32), 0, NUM_TOKENS - 1)
    token_oh = jax.nn.one_hot(safe_token_ids, num_classes=NUM_TOKENS).astype(
        jnp.int32
    )
    counts = jnp.sum(token_oh * valid[..., None].astype(jnp.int32), axis=(0, 1))
    return counts.astype(jnp.int16)


def build_token_occupancy(map_array: jax.Array, color_map: jax.Array) -> jax.Array:
    """Backward-compatible occupancy builder from the compact token grid."""

    token_grid = build_token_grid(map_array, color_map)
    valid = token_grid >= 0
    safe_token_ids = jnp.clip(token_grid.astype(jnp.int32), 0, NUM_TOKENS - 1)
    one_hot = jax.nn.one_hot(safe_token_ids, num_classes=NUM_TOKENS).astype(jnp.bool_)
    return jnp.moveaxis(one_hot & valid[..., None], -1, 0)


def build_token_state(
    map_array: jax.Array,
    color_map: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return compact token state views derived from the map/color grids."""

    token_grid = build_token_grid(map_array, color_map)
    token_counts = build_token_counts(token_grid)
    return token_grid, token_counts


def _sync_cell_token_views(
    token_grid: jax.Array,
    token_counts: jax.Array,
    map_array: jax.Array,
    color_map: jax.Array,
    y: jax.Array,
    x: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Synchronize one compact token cell with the underlying map/color grids."""

    old_token = token_grid[y, x].astype(jnp.int32)
    token_grid = token_grid.at[y, x].set(EMPTY_TOKEN_ID)
    token_counts = jax.lax.cond(
        old_token >= 0,
        lambda counts: counts.at[old_token].add(
            jnp.asarray(-1, dtype=counts.dtype)
        ),
        lambda counts: counts,
        token_counts,
    )
    item = TILE_TO_ITEM[map_array[y, x]]
    color = jnp.clip(color_map[y, x], 0, NUM_COLORS - 1)
    token_id = jnp.clip(item, 0, NUM_ITEMS - 1) * NUM_COLORS + color

    return jax.lax.cond(
        item >= 0,
        lambda carry: (
            carry[0].at[y, x].set(token_id.astype(carry[0].dtype)),
            carry[1].at[token_id].add(jnp.asarray(1, dtype=carry[1].dtype)),
        ),
        lambda carry: carry,
        (token_grid, token_counts),
    )


def _apply_compiled_combine_op(
    state,
    program: CompiledOpProgram,
    op_idx: jax.Array,
):
    input_count = program.input_count[op_idx]
    token1 = program.input_tokens[op_idx, 0]
    token2 = program.input_tokens[op_idx, 1]
    token3 = program.input_tokens[op_idx, 2]
    out_tile = program.output_tile[op_idx]
    out_color = program.output_color[op_idx]
    required_adjacent = program.required_adjacent[op_idx]
    h, w = state.map_array.shape

    def do_binary(st):
        mask1 = st.token_grid == token1
        mask2 = st.token_grid == token2

        def find_adjacent_pair():
            def shift_up(mask):
                return jnp.pad(mask, ((1, 0), (0, 0)), constant_values=False)[:-1, :]

            def shift_down(mask):
                return jnp.pad(mask, ((0, 1), (0, 0)), constant_values=False)[1:, :]

            def shift_left(mask):
                return jnp.pad(mask, ((0, 0), (1, 0)), constant_values=False)[:, :-1]

            def shift_right(mask):
                return jnp.pad(mask, ((0, 0), (0, 1)), constant_values=False)[:, 1:]

            adj_up = mask1 & shift_up(mask2)
            adj_right = mask1 & shift_right(mask2)
            adj_down = mask1 & shift_down(mask2)
            adj_left = mask1 & shift_left(mask2)
            adj = jnp.stack([adj_up, adj_right, adj_down, adj_left], axis=0)

            any_adj = jnp.any(adj)
            flat_idx = jnp.argmax(adj.reshape(-1).astype(jnp.int32))
            dir_idx = flat_idx // (h * w)
            pos_idx = flat_idx % (h * w)
            y = pos_idx // w
            x = pos_idx % w
            dy = jnp.array([-1, 0, 1, 0], dtype=jnp.int32)
            dx = jnp.array([0, 1, 0, -1], dtype=jnp.int32)
            ny = jnp.clip(y + dy[dir_idx], 0, h - 1)
            nx = jnp.clip(x + dx[dir_idx], 0, w - 1)
            return any_adj, y, x, ny, nx

        def find_any_pair():
            flat1 = mask1.reshape(-1)
            flat2 = mask2.reshape(-1)
            idx = jnp.arange(h * w, dtype=jnp.int32)

            def find_repeated_token_pair():
                idx1 = jnp.argmax(flat1.astype(jnp.int32))
                second_mask = flat1 & (idx != idx1)
                idx2 = jnp.argmax(second_mask.astype(jnp.int32))
                return jnp.any(second_mask), idx1, idx2

            def find_distinct_token_pair():
                idx1 = jnp.argmax(flat1.astype(jnp.int32))
                idx2 = jnp.argmax(flat2.astype(jnp.int32))
                return jnp.any(flat1) & jnp.any(flat2), idx1, idx2

            any_match, idx1, idx2 = jax.lax.cond(
                token1 == token2,
                lambda _: find_repeated_token_pair(),
                lambda _: find_distinct_token_pair(),
                operand=None,
            )
            y1 = idx1 // w
            x1 = idx1 % w
            y2 = idx2 // w
            x2 = idx2 % w
            return any_match, y1, x1, y2, x2

        can_merge, y, x, ny, nx = jax.lax.cond(
            required_adjacent,
            lambda _: find_adjacent_pair(),
            lambda _: find_any_pair(),
            operand=None,
        )

        def do_merge(carry_merge):
            st2 = carry_merge
            map_array = st2.map_array.at[y, x].set(out_tile)
            color_map = st2.color_map.at[y, x].set(out_color)
            map_array = map_array.at[ny, nx].set(TileType.OPEN_FAST)
            color_map = color_map.at[ny, nx].set(Colors.BLACK)
            st2 = st2.replace(map_array=map_array, color_map=color_map)
            token_grid, token_counts = _sync_cell_token_views(
                st2.token_grid,
                st2.token_counts,
                st2.map_array,
                st2.color_map,
                y,
                x,
            )
            token_grid, token_counts = _sync_cell_token_views(
                token_grid,
                token_counts,
                st2.map_array,
                st2.color_map,
                ny,
                nx,
            )
            return st2.replace(token_grid=token_grid, token_counts=token_counts)

        return jax.lax.cond(can_merge, do_merge, lambda st2: st2, st)

    def do_ternary(st):
        flat1 = (st.token_grid == token1).reshape(-1)
        flat2 = (st.token_grid == token2).reshape(-1)
        flat3 = (st.token_grid == token3).reshape(-1)
        (
            _adj_left,
            _adj_right,
            req_i,
            req_j,
            req_k,
        ) = _grid_rule_match_indices(int(h), int(w))
        req_i = jnp.asarray(req_i, dtype=jnp.int32)
        req_j = jnp.asarray(req_j, dtype=jnp.int32)
        req_k = jnp.asarray(req_k, dtype=jnp.int32)

        def find_required():
            valid = flat1[req_i] & flat2[req_j] & flat3[req_k]
            any_match = jnp.any(valid)
            match_idx = jnp.argmax(valid.astype(jnp.int32))
            return any_match, req_i[match_idx], req_j[match_idx], req_k[match_idx]

        def find_unrestricted():
            num_cells = h * w
            idx = jnp.arange(num_cells, dtype=jnp.int32)
            pair_distinct = idx[:, None] != idx[None, :]
            pair_present = flat1[:, None] & flat2[None, :] & pair_distinct
            token3_count = jnp.sum(flat3.astype(jnp.int32))
            any_k_distinct = (
                token3_count
                - flat3[:, None].astype(jnp.int32)
                - flat3[None, :].astype(jnp.int32)
            ) > 0
            valid_pair = pair_present & any_k_distinct
            any_match = jnp.any(valid_pair)
            pair_idx = jnp.argmax(valid_pair.reshape(-1).astype(jnp.int32))
            idx1 = pair_idx // num_cells
            idx2 = pair_idx % num_cells
            k_allowed = (idx != idx1) & (idx != idx2)
            k_mask = flat3 & k_allowed
            idx3 = jnp.argmax(k_mask.astype(jnp.int32))
            return any_match, idx1, idx2, idx3

        any_valid, idx1, idx2, idx3 = jax.lax.cond(
            required_adjacent,
            lambda _: find_required(),
            lambda _: find_unrestricted(),
            operand=None,
        )
        y1, x1 = idx1 // w, idx1 % w
        y2, x2 = idx2 // w, idx2 % w
        y3, x3 = idx3 // w, idx3 % w

        def do_merge(carry_merge):
            st2 = carry_merge
            map_array = st2.map_array.at[y1, x1].set(out_tile)
            color_map = st2.color_map.at[y1, x1].set(out_color)
            map_array = map_array.at[y2, x2].set(TileType.OPEN_FAST)
            color_map = color_map.at[y2, x2].set(Colors.BLACK)
            map_array = map_array.at[y3, x3].set(TileType.OPEN_FAST)
            color_map = color_map.at[y3, x3].set(Colors.BLACK)
            st2 = st2.replace(map_array=map_array, color_map=color_map)
            token_grid, token_counts = _sync_cell_token_views(
                st2.token_grid,
                st2.token_counts,
                st2.map_array,
                st2.color_map,
                y1,
                x1,
            )
            token_grid, token_counts = _sync_cell_token_views(
                token_grid,
                token_counts,
                st2.map_array,
                st2.color_map,
                y2,
                x2,
            )
            token_grid, token_counts = _sync_cell_token_views(
                token_grid,
                token_counts,
                st2.map_array,
                st2.color_map,
                y3,
                x3,
            )
            return st2.replace(token_grid=token_grid, token_counts=token_counts)

        return jax.lax.cond(any_valid, do_merge, lambda st2: st2, st)

    return jax.lax.cond(
        input_count == 3,
        do_ternary,
        lambda c: jax.lax.cond(input_count == 2, do_binary, lambda x: x, c),
        state,
    )


def _compiled_combine_match_mask(
    state,
    program: CompiledOpProgram,
    count1: jax.Array,
    count2: jax.Array,
    count3: jax.Array,
) -> jax.Array:
    """Return rules that have an actual grid merge, preserving program order."""
    h, w = state.map_array.shape
    flat_tokens = state.token_grid.reshape(-1)
    (
        adj_left,
        adj_right,
        req_i,
        req_j,
        req_k,
    ) = _grid_rule_match_indices(int(h), int(w))
    adj_left = jnp.asarray(adj_left, dtype=jnp.int32)
    adj_right = jnp.asarray(adj_right, dtype=jnp.int32)
    req_i = jnp.asarray(req_i, dtype=jnp.int32)
    req_j = jnp.asarray(req_j, dtype=jnp.int32)
    req_k = jnp.asarray(req_k, dtype=jnp.int32)

    token1 = program.input_tokens[:, 0]
    token2 = program.input_tokens[:, 1]
    token3 = program.input_tokens[:, 2]

    binary_any = jnp.where(
        token1 == token2,
        count1 >= 2,
        (count1 >= 1) & (count2 >= 1),
    )
    binary_adjacent = jnp.any(
        (flat_tokens[adj_left][None, :] == token1[:, None])
        & (flat_tokens[adj_right][None, :] == token2[:, None]),
        axis=1,
    )
    binary_match = jnp.where(program.required_adjacent, binary_adjacent, binary_any)

    ternary_adjacent = jnp.any(
        (flat_tokens[req_i][None, :] == token1[:, None])
        & (flat_tokens[req_j][None, :] == token2[:, None])
        & (flat_tokens[req_k][None, :] == token3[:, None]),
        axis=1,
    )
    needs_unrestricted_ternary = jnp.any(
        (program.input_count == 3) & (~program.required_adjacent)
    )

    def with_unrestricted_ternary(_unused):
        same12 = token1 == token2
        same13 = token1 == token3
        same23 = token2 == token3
        need1 = (
            jnp.ones_like(count1)
            + same12.astype(jnp.int32)
            + same13.astype(jnp.int32)
        )
        need2 = jnp.where(
            same12,
            jnp.zeros_like(count2),
            jnp.ones_like(count2) + same23.astype(jnp.int32),
        )
        need3 = jnp.where(
            same13 | same23,
            jnp.zeros_like(count3),
            jnp.ones_like(count3),
        )
        ternary_any = (count1 >= need1) & (count2 >= need2) & (count3 >= need3)
        return jnp.where(program.required_adjacent, ternary_adjacent, ternary_any)

    ternary_match = jax.lax.cond(
        needs_unrestricted_ternary,
        with_unrestricted_ternary,
        lambda _: ternary_adjacent,
        operand=None,
    )

    return jnp.where(
        program.input_count == 3,
        ternary_match,
        jnp.where(program.input_count == 2, binary_match, False),
    )


def _apply_compiled_transform_op(
    state,
    action: jax.Array,
    program: CompiledOpProgram,
    op_idx: jax.Array,
):
    input_count = program.input_count[op_idx]
    input_token = program.input_tokens[op_idx, 0]
    output_tile = program.output_tile[op_idx]
    output_color = program.output_color[op_idx]
    required_adjacent = program.required_adjacent[op_idx]
    toggle = action == Action.TOGGLE
    allowed_local = jnp.array([True, False, False, False, False], dtype=jnp.bool_)

    def do_unary(st):

        def _process_toggle_agent(st2):
            pos = st2.positions
            adj_positions = jnp.stack(
                (
                    pos + jnp.array([-1, 0], dtype=jnp.int32),
                    pos + jnp.array([0, 1], dtype=jnp.int32),
                    pos + jnp.array([1, 0], dtype=jnp.int32),
                    pos + jnp.array([0, -1], dtype=jnp.int32),
                ),
                axis=0,
            )
            positions_to_check = jnp.concatenate([pos[None, :], adj_positions], axis=0)
            ys = jnp.clip(positions_to_check[:, 0], 0, st2.map_array.shape[0] - 1)
            xs = jnp.clip(positions_to_check[:, 1], 0, st2.map_array.shape[1] - 1)
            in_bounds = (
                (positions_to_check[:, 0] >= 0)
                & (positions_to_check[:, 0] < st2.map_array.shape[0])
                & (positions_to_check[:, 1] >= 0)
                & (positions_to_check[:, 1] < st2.map_array.shape[1])
            )
            allowed_positions = jnp.where(required_adjacent, in_bounds, allowed_local)
            local_tokens = st2.token_grid[ys, xs].astype(jnp.int32)
            matches = allowed_positions & (local_tokens == input_token)
            any_match = jnp.any(matches)
            match_idx = jnp.argmax(matches.astype(jnp.int32))
            target_y = ys[match_idx]
            target_x = xs[match_idx]

            def do_transform(op_carry):
                st3 = op_carry
                map_array = st3.map_array.at[target_y, target_x].set(output_tile)
                color_map = st3.color_map.at[target_y, target_x].set(output_color)
                st3 = st3.replace(map_array=map_array, color_map=color_map)
                token_grid, token_counts = _sync_cell_token_views(
                    st3.token_grid,
                    st3.token_counts,
                    st3.map_array,
                    st3.color_map,
                    target_y,
                    target_x,
                )
                return st3.replace(token_grid=token_grid, token_counts=token_counts)

            return jax.lax.cond(
                any_match,
                do_transform,
                lambda st3: st3,
                st2,
            )

        return jax.lax.cond(
            toggle,
            _process_toggle_agent,
            lambda st2: st2,
            st,
        )

    return jax.lax.cond(input_count == 1, do_unary, lambda st: st, state)


def _run_compiled_program(
    state,
    action: jax.Array,
    program: CompiledOpProgram,
):
    """Execute a compiled rule program using vectorized rule matching.

    Instead of scanning R_max slots with a 3-way ``lax.switch`` (which traces
    every branch body on every iteration), this:
    1. Checks all combine rules in parallel via ``token_counts``, then applies
       only the first match using local ``token_grid`` scans. Repeats for the
       program's fixed bucket-local combine bound.
    2. Checks all transform rules in parallel, applies matching ones.

    No ``lax.switch`` is used — combines and transforms have separate code
    paths, so JAX never traces one type's body when executing the other.
    """
    R = program.kind.shape[0]
    if int(R) == 0:
        return state
    row_idx = jnp.arange(R, dtype=jnp.int32)
    active = row_idx < program.count
    is_combine = active & (program.kind == COMPILED_OP_COMBINE)
    is_transform = active & (program.kind == COMPILED_OP_TRANSFORM)
    any_combines = jnp.any(is_combine)
    any_transforms = jnp.any(is_transform)

    # Pre-extract token ids for all rules (used in vectorized checks).
    token_a = program.input_tokens[:, 0]  # (R,)
    token_b = program.input_tokens[:, 1]  # (R,)
    token_c = program.input_tokens[:, 2]  # (R,)
    needs_b = program.input_count >= 2  # (R,) bool
    needs_c = program.input_count >= 3  # (R,) bool
    need_a = (
        jnp.ones_like(token_a, dtype=jnp.int32)
        + (needs_b & (token_b == token_a)).astype(jnp.int32)
        + (needs_c & (token_c == token_a)).astype(jnp.int32)
    )
    need_b = jnp.where(
        needs_b,
        1 + (needs_c & (token_c == token_b)).astype(jnp.int32),
        0,
    )

    # ------------------------------------------------------------------
    # Combines: vectorized check → apply first match, repeat for cascades
    # ------------------------------------------------------------------
    def _run_combines(st):
        def _combine_iter(inner_st, _unused):
            counts = inner_st.token_counts.astype(jnp.int32)

            # Vectorized pre-check: which combine rules have enough inputs on the map?
            has_a = counts[token_a] >= need_a  # (R,)
            has_b = jnp.where(needs_b, counts[token_b] >= need_b, True)  # (R,)
            has_c = jnp.where(needs_c, counts[token_c] >= 1, True)  # (R,)
            can_fire = is_combine & has_a & has_b & has_c  # (R,)

            count_a = counts[token_a]
            count_b = counts[token_b]
            count_c = counts[token_c]

            actual_match = jax.lax.cond(
                jnp.any(can_fire),
                lambda _: can_fire
                & _compiled_combine_match_mask(
                    inner_st,
                    program,
                    count_a,
                    count_b,
                    count_c,
                ),
                lambda _: jnp.zeros_like(can_fire, dtype=jnp.bool_),
                operand=None,
            )
            has_any = jnp.any(actual_match)
            first_idx = jnp.argmax(actual_match.astype(jnp.int32))

            next_st = jax.lax.cond(
                has_any,
                lambda st2: _apply_compiled_combine_op(st2, program, first_idx),
                lambda st2: st2,
                inner_st,
            )
            return next_st, None

        return jax.lax.scan(_combine_iter, st, program.combine_scan_steps)[0]

    state = jax.lax.cond(any_combines, _run_combines, lambda st: st, state)

    # ------------------------------------------------------------------
    # Transforms: vectorized check → apply matches
    # ------------------------------------------------------------------
    def _run_transforms(st):
        # Pre-check: is the agent toggling?
        any_toggle = action == Action.TOGGLE

        def _with_toggles(st_inner):
            def _transform_iter(inner_carry, _unused):
                st2, remaining = inner_carry
                # remaining: (R,) bool mask of not-yet-applied transforms
                has_any = jnp.any(remaining)
                first_idx = jnp.argmax(remaining.astype(jnp.int32))

                st2 = jax.lax.cond(
                    has_any,
                    lambda st3: _apply_compiled_transform_op(
                        st3, action, program, first_idx
                    ),
                    lambda st3: st3,
                    st2,
                )
                remaining = remaining.at[first_idx].set(
                    remaining[first_idx] & (~has_any)
                )
                return (st2, remaining), None

            (st_out, _), _ = jax.lax.scan(
                _transform_iter,
                (st_inner, is_transform),
                program.transform_scan_steps,
            )
            return st_out

        return jax.lax.cond(any_toggle, _with_toggles, lambda st2: st2, st)

    state = jax.lax.cond(any_transforms, _run_transforms, lambda st: st, state)

    return state


def _run_compiled_program_if_enabled(
    state,
    action: jax.Array,
    program: CompiledOpProgram,
    *,
    enabled: jax.Array,
):
    def _run(st):
        st2 = _run_compiled_program(st, action, program)
        changed = jnp.any(st2.token_grid != st.token_grid)
        return st2, changed

    def _skip(st):
        return st, jnp.asarray(False, dtype=jnp.bool_)

    return jax.lax.cond(enabled, _run, _skip, state)


def check_rule(rule_data: Any, state, action: jax.Array, env) -> Any:
    action = jnp.asarray(action, dtype=jnp.int32)
    runtime = (
        rule_data
        if isinstance(rule_data, RuleRuntimeCache)
        else compile_rule_runtime(rule_data)
    )

    token_grid_before_actions = state.token_grid
    prior_rules_dirty = getattr(
        state, "rules_dirty", jnp.asarray(True, dtype=jnp.bool_)
    )

    state = DropRule()(state, action, env)
    state = GenericPickupRule()(state, action, env)

    token_changed = jnp.any(state.token_grid != token_grid_before_actions)
    any_toggle = action == Action.TOGGLE
    should_run_program = prior_rules_dirty | token_changed | any_toggle

    state, pre_changed = _run_compiled_program_if_enabled(
        state,
        action,
        runtime.pre_move_ops,
        enabled=should_run_program,
    )
    state = MovementRule()(state, action, env)
    state, post_changed = _run_compiled_program_if_enabled(
        state,
        action,
        runtime.post_move_ops,
        enabled=should_run_program | pre_changed,
    )
    return state.replace(rules_dirty=pre_changed | post_changed)


def apply_distractor_combines(
    state, table: jax.Array, agent_position: jax.Array, did_drop: jax.Array
):
    """Apply distractor combines via a dense lookup table.

    Instead of scanning individual distractor rule rows, this checks all
    adjacent item pairs on the map against a precomputed table.

    table: (NUM_ITEMS, NUM_COLORS, NUM_ITEMS, NUM_COLORS, 3) int32
           table[i1, c1, i2, c2] = [out_item, out_color, valid]
           Table is symmetric (both orderings populated at gen time).

    Returns:
      state_after_merges: environment state after applying all distractor merges.
      penalized: scalar bool indicating whether the dropping agent was
        involved in at least one distractor merge this step.
    """
    H, W = state.map_array.shape

    def check_and_merge(carry, y1, x1, y2, x2):
        st, penalized = carry
        tile1 = st.map_array[y1, x1]
        color1 = st.color_map[y1, x1]
        item1 = TILE_TO_ITEM[tile1]

        tile2 = st.map_array[y2, x2]
        color2 = st.color_map[y2, x2]
        item2 = TILE_TO_ITEM[tile2]

        # Clamp to valid range for safe indexing
        i1 = jnp.clip(item1, 0, NUM_ITEMS - 1)
        c1 = jnp.clip(color1, 0, NUM_COLORS - 1)
        i2 = jnp.clip(item2, 0, NUM_ITEMS - 1)
        c2 = jnp.clip(color2, 0, NUM_COLORS - 1)

        entry = table[i1, c1, i2, c2]
        out_item, out_color, valid = entry[0], entry[1], entry[2]

        # Only merge if both cells actually hold item tiles and lookup is valid
        can_merge = (valid == 1) & (item1 >= 0) & (item2 >= 0)

        out_tile = ITEM_TO_TILE[jnp.clip(out_item, 0, NUM_ITEMS - 1)]

        def do_merge(carry_merge):
            s, p = carry_merge
            m = s.map_array.at[y1, x1].set(out_tile).at[y2, x2].set(TileType.OPEN_FAST)
            c = s.color_map.at[y1, x1].set(out_color).at[y2, x2].set(Colors.BLACK)
            token_grid, token_counts = _sync_cell_token_views(
                s.token_grid, s.token_counts, m, c, y1, x1
            )
            token_grid, token_counts = _sync_cell_token_views(
                token_grid, token_counts, m, c, y2, x2
            )
            s2 = s.replace(
                map_array=m,
                color_map=c,
                token_grid=token_grid,
                token_counts=token_counts,
            )

            pos1_match = jnp.logical_and(
                agent_position[0] == y1, agent_position[1] == x1
            )
            pos2_match = jnp.logical_and(
                agent_position[0] == y2, agent_position[1] == x2
            )
            involved = jnp.logical_and(
                jnp.logical_or(pos1_match, pos2_match), did_drop
            )
            p2 = jnp.logical_or(p, involved)
            return s2, p2

        return jax.lax.cond(can_merge, do_merge, lambda c: c, (st, penalized))

    def process_cell(carry, pos_flat):
        st, penalized = carry
        y = pos_flat // W
        x = pos_flat % W

        # Check right neighbor
        has_right = x + 1 < W
        rx = jnp.clip(x + 1, 0, W - 1)
        st, penalized = jax.lax.cond(
            has_right,
            lambda c: check_and_merge(c, y, x, y, rx),
            lambda c: c,
            (st, penalized),
        )

        # Check down neighbor
        has_down = y + 1 < H
        dy = jnp.clip(y + 1, 0, H - 1)
        st, penalized = jax.lax.cond(
            has_down,
            lambda c: check_and_merge(c, y, x, dy, x),
            lambda c: c,
            (st, penalized),
        )

        return (st, penalized), None

    positions = jnp.arange(H * W)
    init_penalized = jnp.asarray(False, dtype=jnp.bool_)
    (state, penalized), _ = jax.lax.scan(
        process_cell, (state, init_penalized), positions
    )
    return state, penalized


def apply_distractor_combines_near_drops(
    state,
    table: jax.Array,
    agent_position: jax.Array,
    did_drop: jax.Array,
    dropped_item: jax.Array,
    dropped_color: jax.Array,
):
    """Apply distractor combines only around newly dropped items.

    This checks 4-neighborhood pairs for the dropping agent tile and applies
    merges in a deterministic offsets order.
    """
    H, W = state.map_array.shape
    did_drop = jnp.asarray(did_drop, dtype=jnp.bool_)
    dropped_item = jnp.asarray(dropped_item, dtype=jnp.int32)
    dropped_color = jnp.asarray(dropped_color, dtype=jnp.int32)
    offsets = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32)

    def check_and_merge(carry, y1, x1, y2, x2, enabled, dropped_item, dropped_color):
        st, penalized = carry
        tile1 = st.map_array[y1, x1]
        color1 = st.color_map[y1, x1]
        item1 = TILE_TO_ITEM[tile1]

        tile2 = st.map_array[y2, x2]
        color2 = st.color_map[y2, x2]
        item2 = TILE_TO_ITEM[tile2]

        # Clamp to valid range for safe indexing
        i1 = jnp.clip(item1, 0, NUM_ITEMS - 1)
        c1 = jnp.clip(color1, 0, NUM_COLORS - 1)
        i2 = jnp.clip(item2, 0, NUM_ITEMS - 1)
        c2 = jnp.clip(color2, 0, NUM_COLORS - 1)

        entry = table[i1, c1, i2, c2]
        out_item, out_color, valid = entry[0], entry[1], entry[2]

        # Only merge if this pair is enabled, both cells hold item tiles, and
        # one side of the pair is the token that was actually dropped.
        dropped_token_in_pair = (
            ((item1 == dropped_item) & (color1 == dropped_color))
            | ((item2 == dropped_item) & (color2 == dropped_color))
        )
        can_merge = (
            enabled
            & dropped_token_in_pair
            & (valid == 1)
            & (item1 >= 0)
            & (item2 >= 0)
        )
        out_tile = ITEM_TO_TILE[jnp.clip(out_item, 0, NUM_ITEMS - 1)]

        def do_merge(carry_merge):
            s, p = carry_merge
            m = s.map_array.at[y1, x1].set(out_tile).at[y2, x2].set(TileType.OPEN_FAST)
            c = s.color_map.at[y1, x1].set(out_color).at[y2, x2].set(Colors.BLACK)
            token_grid, token_counts = _sync_cell_token_views(
                s.token_grid, s.token_counts, m, c, y1, x1
            )
            token_grid, token_counts = _sync_cell_token_views(
                token_grid, token_counts, m, c, y2, x2
            )
            s2 = s.replace(
                map_array=m,
                color_map=c,
                token_grid=token_grid,
                token_counts=token_counts,
            )

            pos1_match = jnp.logical_and(
                agent_position[0] == y1, agent_position[1] == x1
            )
            pos2_match = jnp.logical_and(
                agent_position[0] == y2, agent_position[1] == x2
            )
            involved = jnp.logical_and(
                jnp.logical_or(pos1_match, pos2_match), did_drop
            )
            p2 = jnp.logical_or(p, involved)
            return s2, p2

        return jax.lax.cond(can_merge, do_merge, lambda c: c, (st, penalized))

    y1 = agent_position[0]
    x1 = agent_position[1]
    has_dropped_token = (dropped_item >= 0) & (dropped_color >= 0)

    def process_offset(carry2, off):
        st2, penalized2 = carry2
        y2 = y1 + off[0]
        x2 = x1 + off[1]
        in_bounds = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
        y2_safe = jnp.clip(y2, 0, H - 1)
        x2_safe = jnp.clip(x2, 0, W - 1)
        enabled = did_drop & has_dropped_token & in_bounds
        out = check_and_merge(
            (st2, penalized2),
            y1,
            x1,
            y2_safe,
            x2_safe,
            enabled,
            dropped_item,
            dropped_color,
        )
        return out, None

    init_penalized = jnp.asarray(False, dtype=jnp.bool_)
    (state, penalized), _ = jax.lax.scan(
        process_offset, (state, init_penalized), offsets
    )
    return state, penalized
