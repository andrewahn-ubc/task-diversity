# ruff: noqa: F403, F405

import collections.abc as c_abc
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct
from flax.core.frozen_dict import FrozenDict

from banyan_grid.environment.constants import *
from banyan_grid.environment.goals import (
    AgentHasItemFromRulesetGoal,
    BaseGridGoal,
)
from banyan_grid.environment.rules import (
    BaseRule,
    MovementRule,
    RuleRuntimeCache,
    apply_distractor_combines_near_drops,
    build_token_state,
    check_rule,
    compile_rule_runtime,
)
from banyan_grid.environment.spaces import Box, Discrete
from banyan_grid.tasks.ruleset_factory import _rule_count


@struct.dataclass
class RuleFieldView:
    rule_type: jax.Array
    is_collect: jax.Array
    is_binary_combine: jax.Array
    is_ternary_combine: jax.Array
    is_combine: jax.Array
    is_transform: jax.Array
    is_prod: jax.Array
    required_adjacent: jax.Array
    input_item_a: jax.Array
    input_item_b: jax.Array
    input_item_c: jax.Array
    output_item: jax.Array
    input_color_a: jax.Array
    input_color_b: jax.Array
    input_color_c: jax.Array
    output_color: jax.Array


def _decode_rule_fields(codes: jnp.ndarray) -> RuleFieldView:
    codes = jnp.asarray(codes, dtype=jnp.int32)
    rule_type = codes[:, 0]
    is_collect = rule_type == RULE_TYPE_COLLECT
    is_binary_combine = rule_type == RULE_TYPE_COMBINE
    is_ternary_combine = rule_type == RULE_TYPE_TERNARY_COMBINE
    is_combine = is_binary_combine | is_ternary_combine
    is_transform = rule_type == RULE_TYPE_TRANSFORM
    is_prod = is_combine | is_transform
    packed = codes[:, 5]
    input_color_a = packed & 0xF
    input_color_b = (packed >> 4) & 0xF
    input_color_c = jnp.where(is_ternary_combine, (packed >> 8) & 0xF, -1)
    output_color = jnp.where(is_ternary_combine, (packed >> 12) & 0xF, (packed >> 8) & 0xF)
    required_adjacent = jnp.where(
        is_ternary_combine,
        (codes[:, 4] & 0x1) == 1,
        codes[:, 4] == 1,
    )
    return RuleFieldView(
        rule_type=rule_type,
        is_collect=is_collect,
        is_binary_combine=is_binary_combine,
        is_ternary_combine=is_ternary_combine,
        is_combine=is_combine,
        is_transform=is_transform,
        is_prod=is_prod,
        required_adjacent=required_adjacent,
        input_item_a=codes[:, 1],
        input_item_b=jnp.where(is_combine, codes[:, 2], -2),
        input_item_c=jnp.where(is_ternary_combine, codes[:, 4] >> 1, -2),
        output_item=jnp.where(is_prod, codes[:, 3], -1),
        input_color_a=jnp.where(is_prod, input_color_a, -2),
        input_color_b=jnp.where(is_combine, input_color_b, -2),
        input_color_c=jnp.where(is_ternary_combine, input_color_c, -2),
        output_color=jnp.where(is_prod, output_color, -1),
    )


def _producer_usage_matrix(view: RuleFieldView, row_count: int) -> jax.Array:
    uses_out_as_in1 = (view.input_item_a[:, None] == view.output_item[None, :]) & (
        view.input_color_a[:, None] == view.output_color[None, :]
    )
    uses_out_as_in2 = (view.input_item_b[:, None] == view.output_item[None, :]) & (
        view.input_color_b[:, None] == view.output_color[None, :]
    )
    uses_out_as_in3 = (view.input_item_c[:, None] == view.output_item[None, :]) & (
        view.input_color_c[:, None] == view.output_color[None, :]
    )
    return (
        (uses_out_as_in1 | uses_out_as_in2 | uses_out_as_in3)
        & view.is_prod[:, None]
        & view.is_prod[None, :]
        & (~jnp.eye(row_count, dtype=bool))
    )


RULE_OBS_PROGRESS_DIM = 3
RULE_OBS_STATIC_DIM = NUM_RULE_TYPES + 4 * NUM_ITEMS + 1 + 4 * NUM_COLORS


class Banyan:
    def __init__(
        self,
        grid_size: int = 5,
        max_steps: int = 20,
        map_array: Optional[chex.Array] = None,
        color_map: Optional[chex.Array] = None,
        goal: Optional[BaseGridGoal] = None,
        rules: Optional[List[BaseRule]] = None,
        ruleset: Optional[chex.Array] = None,
        max_depth: int = 4,
        max_rules: Optional[int] = None,
        include_rules_in_obs: bool = True,
        distractor_table: Optional[jax.Array] = None,
        distractor_combine_penalty: float = -0.1,
        timeout_penalty: float = 0.0,
        depth_weighted_pickup_shaping: bool = False,
        pickup_shaping_leaf_reward: float = 0.05,
        pickup_shaping_root_reward: float = 0.2,
        goal_reward_scale: float = 1.0,
    ):
        if rules is not None:
            raise ValueError(
                "Custom rules are not supported by the compiled rule engine. "
                "Leave rules=None and supply encoded rules via ruleset instead."
            )
        self.ruleset = None if ruleset is None else jnp.array(ruleset, dtype=jnp.int32)
        if self.ruleset is not None and (
            self.ruleset.ndim != 2
            or self.ruleset.shape[0] < 1
            or self.ruleset.shape[1] != MAX_RULE_ENCODING_LEN
        ):
            raise ValueError(
                f"ruleset must have shape (num_rules, {MAX_RULE_ENCODING_LEN}) "
                "with at least one rule."
            )

        self.grid_size = grid_size
        self.distractor_table = distractor_table
        self.distractor_combine_penalty = float(distractor_combine_penalty)
        self.timeout_penalty = float(timeout_penalty)
        self.depth_weighted_pickup_shaping = bool(depth_weighted_pickup_shaping)
        self.pickup_shaping_leaf_reward = float(pickup_shaping_leaf_reward)
        self.pickup_shaping_root_reward = float(pickup_shaping_root_reward)
        self.goal_reward_scale = float(goal_reward_scale)
        self.max_steps = max_steps

        self._action_space = Discrete(NUM_ACTIONS)
        max_tile_value = NUM_TILE_TYPES - 1
        num_tile_types = NUM_TILE_TYPES
        # spatial: tiles one-hot + self position + color one-hot planes
        # inventories: item bits + item-color one-hots
        # goal: goal_type (2) + goal_item (NUM_ITEMS) + goal_color (NUM_COLORS)
        # R_max = number of rules for the maximum depth supported
        # With GenericPickupRule, agents can pick up any item so no special intermediate rules needed
        if max_rules is not None:
            R_max = int(max_rules)
            if R_max < 1:
                raise ValueError("max_rules must be >= 1 when provided.")
        else:
            R_max = _rule_count(max_depth)
        if self.ruleset is not None:
            num_rules = self.ruleset.shape[0]
            if max_rules is not None and num_rules > R_max:
                raise ValueError(
                    f"Constructor ruleset has {num_rules} rows, but max_rules={R_max}. "
                    f"Set max_rules >= {num_rules} or omit max_rules to size automatically."
                )
            R_max = max(R_max, num_rules)
        self.R_max = R_max
        goal_dim = 2 + NUM_ITEMS + NUM_COLORS
        # Per-rule one-hot:
        # rule_type(NUM_RULE_TYPES) + in1(NUM_ITEMS) + in2(NUM_ITEMS) +
        # out(NUM_ITEMS) + in3(NUM_ITEMS) + adjacent(1) + 4 color one-hots + progress(3)
        # All item columns now use NUM_ITEMS for consistency (enables transfer learning)
        rule_oh_dim_per_rule = RULE_OBS_STATIC_DIM + RULE_OBS_PROGRESS_DIM
        self.rule_oh_dim = rule_oh_dim_per_rule
        self.include_rules_in_obs = include_rules_in_obs
        rules_dim = R_max * rule_oh_dim_per_rule
        inventory_obs_agents = 1
        position_channels = 1

        flat_dim = (
            grid_size * grid_size * (num_tile_types + position_channels + NUM_COLORS)
            + inventory_obs_agents * NUM_ITEMS * (1 + NUM_COLORS)
            + goal_dim
            + (rules_dim if include_rules_in_obs else 0)
        )

        self._observation_space = Box(
            low=0.0,
            high=float(max_tile_value),
            shape=(flat_dim,),
            dtype=jnp.float32,
        )

        self.map_array = (
            jnp.full((grid_size, grid_size), TileType.OPEN_FAST, dtype=jnp.int32)
            if map_array is None
            else jnp.array(map_array, dtype=jnp.int32)
        )
        chex.assert_shape(self.map_array, (grid_size, grid_size))

        self.color_map = (
            jnp.full((grid_size, grid_size), Colors.BLACK, dtype=jnp.int32)
            if color_map is None
            else jnp.array(color_map, dtype=jnp.int32)
        )
        chex.assert_shape(self.color_map, (grid_size, grid_size))

        # if no rules provided, Banyan uses default rules
        self.goal = goal if goal is not None else AgentHasItemFromRulesetGoal(True)
        self.rules = (
            rules
            if rules is not None
            else [
                MovementRule(),
            ]
        )

    @struct.dataclass
    class State:
        positions: jax.Array  # Shape: (2,)
        directions: jax.Array  # Shape: () in {0,1,2,3}
        map_array: jax.Array  # Shape: (grid_size, grid_size)
        color_map: jax.Array  # Shape: (grid_size, grid_size)
        token_grid: jax.Array  # Shape: (grid_size, grid_size), -1 for empty else token_id
        token_counts: jax.Array  # Shape: (NUM_ITEMS * NUM_COLORS,), multiplicity per token
        inventory_colors: jax.Array  # (NUM_ITEMS,), ints in [0..Colors.BLACK] (BLACK = no color)
        inventories: jax.Array  # Shape: (NUM_ITEMS,)
        items_ever_picked: (
            jax.Array
        )  # Shape: (NUM_ITEMS, NUM_COLORS) - token-level first-pickup tracking
        time_step: int | jax.Array
        done: bool | jax.Array
        rules_dirty: (
            jax.Array
        )  # bool scalar - rerun compiled rules next step if prior pass changed map tokens
        rule_encodings: jax.Array
        rule_runtime: RuleRuntimeCache
        # Precomputed goal information (computed once at reset)
        goal_item: int | jax.Array  # Target item type
        goal_color: int | jax.Array  # Target color
        goal_type: int | jax.Array  # 0=collect-only, 1=combine-tree
        relevant_item_mask: jax.Array  # bool (NUM_ITEMS, NUM_COLORS) - tokens in ruleset
        pickup_reward_weights: (
            jax.Array
        )  # Shape: (NUM_ITEMS, NUM_COLORS) - per-token first-pickup shaping reward
        depth: int | jax.Array  # Ruleset depth (tree depth)
        goal_vec: (
            jax.Array
        )  # Precomputed goal one-hot: [goal_type(2), goal_item(NUM_ITEMS), goal_color(NUM_COLORS)]
        # Precomputed rule observation components (computed once at reset, static per episode)
        rules_oh_static: jax.Array  # (R, rule_oh_dim - 3) static one-hot part of rules
        is_root: jax.Array  # (R,) bool - whether each producer rule is a root/sink
        is_comb: jax.Array  # (R,) bool - whether each rule is a producer

    @staticmethod
    def compile_goal_from_ruleset(codes: jnp.ndarray, num_items: int, num_colors: int):
        """Efficiently compute goal item/color/type from ruleset"""
        view = _decode_rule_fields(codes)
        any_prod = jnp.any(view.is_prod)

        def from_first_producer(rs):
            used_as_input = _producer_usage_matrix(view, rs.shape[0])
            out_is_used = jnp.any(used_as_input, axis=0)
            sink_mask = view.is_prod & (~out_is_used)
            have_sink = jnp.any(sink_mask)
            idx = jax.lax.select(
                have_sink,
                jnp.argmax(sink_mask.astype(jnp.int32)),
                jnp.argmax(view.is_prod.astype(jnp.int32)),
            )
            item = view.output_item[idx]
            color = view.output_color[idx]
            return item, color, jnp.int32(1)

        def from_first_collect(rs):
            idx = jnp.argmax(view.is_collect.astype(jnp.int32))
            item = rs[idx, 2]
            color = rs[idx, 3]
            return item, color, jnp.int32(0)

        item, color, gtype = jax.lax.cond(any_prod, from_first_producer, from_first_collect, codes)

        # Safety clamps for one_hot indexing
        item = jnp.clip(item, 0, num_items - 1)
        color = jnp.clip(color, 0, num_colors - 1)

        # Build goal_vec: [goal_type_oh(2), goal_item_oh(NUM_ITEMS), goal_color_oh(NUM_COLORS)]
        goal_type_oh = jax.nn.one_hot(gtype, num_classes=2).astype(jnp.float32)
        goal_item_oh = jax.nn.one_hot(item, num_classes=num_items).astype(jnp.float32)
        goal_color_oh = jax.nn.one_hot(color, num_classes=num_colors).astype(jnp.float32)
        goal_vec = jnp.concatenate([goal_type_oh, goal_item_oh, goal_color_oh], axis=0)

        # Compute relevant token mask (item+color) for collect + producer inputs/outputs.
        collect_item = jnp.where(view.is_collect, codes[:, 2], -1)
        collect_color = jnp.where(view.is_collect, codes[:, 3], -1)
        in1_item_rel = jnp.where(view.is_prod, view.input_item_a, -1)
        in1_color_rel = jnp.where(view.is_prod, view.input_color_a, -1)
        in2_item_rel = jnp.where(view.is_combine, view.input_item_b, -1)
        in2_color_rel = jnp.where(view.is_combine, view.input_color_b, -1)
        in3_item_rel = jnp.where(view.is_ternary_combine, view.input_item_c, -1)
        in3_color_rel = jnp.where(view.is_ternary_combine, view.input_color_c, -1)
        out_item_rel = jnp.where(view.is_prod, view.output_item, -1)
        out_color_rel = jnp.where(view.is_prod, view.output_color, -1)

        rel_items = jnp.concatenate(
            (collect_item, in1_item_rel, in2_item_rel, in3_item_rel, out_item_rel),
            axis=0,
        )
        rel_colors = jnp.concatenate(
            (collect_color, in1_color_rel, in2_color_rel, in3_color_rel, out_color_rel),
            axis=0,
        )
        rel_valid = (
            (rel_items >= 0)
            & (rel_items < num_items)
            & (rel_colors >= 0)
            & (rel_colors < num_colors)
        )
        rel_token_ids = jnp.clip(rel_items, 0, num_items - 1) * num_colors + jnp.clip(
            rel_colors, 0, num_colors - 1
        )
        rel_token_oh = (
            jax.nn.one_hot(
                rel_token_ids,
                num_classes=num_items * num_colors,
            ).astype(jnp.bool_)
            & rel_valid[:, None]
        )
        relevant_item_mask = jnp.any(rel_token_oh, axis=0).reshape(num_items, num_colors)

        # Compute depth from producer DAG longest path:
        # op_depth(leaf-level producer)=1, task depth = 1 + max(op_depth).
        dep = _producer_usage_matrix(view, codes.shape[0])
        op_depth = view.is_prod.astype(jnp.int32)

        def relax_depth(_i, cur_depth):
            parent_max = jnp.max(jnp.where(dep, cur_depth[None, :], 0), axis=1)
            candidate = 1 + parent_max
            return jnp.where(view.is_prod, jnp.maximum(cur_depth, candidate), 0)

        op_depth = jax.lax.fori_loop(0, codes.shape[0], relax_depth, op_depth)
        depth = jnp.where(any_prod, jnp.int32(1) + jnp.max(op_depth), jnp.int32(1))

        return item, color, gtype, goal_vec, relevant_item_mask, depth

    @staticmethod
    def compute_pickup_reward_weights(
        codes: jnp.ndarray,
        num_items: int,
        leaf_reward: float,
        root_reward: float,
    ) -> jax.Array:
        """Compute per-token (item+color) first-pickup shaping weights by producer depth."""
        view = _decode_rule_fields(codes)

        collect_item = jnp.where(view.is_collect, codes[:, 2], -1)
        collect_color = jnp.where(view.is_collect, codes[:, 3], -1)
        in1_item_rel = jnp.where(view.is_prod, view.input_item_a, -1)
        in1_color_rel = jnp.where(view.is_prod, view.input_color_a, -1)
        in2_item_rel = jnp.where(view.is_combine, view.input_item_b, -1)
        in2_color_rel = jnp.where(view.is_combine, view.input_color_b, -1)
        in3_item_rel = jnp.where(view.is_ternary_combine, view.input_item_c, -1)
        in3_color_rel = jnp.where(view.is_ternary_combine, view.input_color_c, -1)
        out_item_rel = jnp.where(view.is_prod, view.output_item, -1)
        out_color_rel = jnp.where(view.is_prod, view.output_color, -1)

        rel_items = jnp.concatenate(
            (collect_item, in1_item_rel, in2_item_rel, in3_item_rel, out_item_rel),
            axis=0,
        )
        rel_colors = jnp.concatenate(
            (collect_color, in1_color_rel, in2_color_rel, in3_color_rel, out_color_rel),
            axis=0,
        )
        rel_valid = (
            (rel_items >= 0)
            & (rel_items < num_items)
            & (rel_colors >= 0)
            & (rel_colors < NUM_COLORS)
        )
        rel_token_ids = jnp.clip(rel_items, 0, num_items - 1) * NUM_COLORS + jnp.clip(
            rel_colors, 0, NUM_COLORS - 1
        )
        rel_token_oh = (
            jax.nn.one_hot(
                rel_token_ids,
                num_classes=num_items * NUM_COLORS,
            ).astype(jnp.bool_)
            & rel_valid[:, None]
        )
        relevant_item_mask = jnp.any(rel_token_oh, axis=0)

        dep = _producer_usage_matrix(view, codes.shape[0])

        op_depth = view.is_prod.astype(jnp.int32)

        def relax_depth(_i, cur_depth):
            parent_max = jnp.max(jnp.where(dep, cur_depth[None, :], 0), axis=1)
            candidate = 1 + parent_max
            return jnp.where(view.is_prod, jnp.maximum(cur_depth, candidate), 0)

        op_depth = jax.lax.fori_loop(0, codes.shape[0], relax_depth, op_depth)
        max_op_depth = jnp.max(jnp.where(view.is_prod, op_depth, 0))

        leaf_v = jnp.asarray(leaf_reward, dtype=jnp.float32)
        root_v = jnp.asarray(root_reward, dtype=jnp.float32)
        denom = jnp.maximum(max_op_depth - 1, 1).astype(jnp.float32)
        depth_t = (op_depth.astype(jnp.float32) - 1.0) / denom
        producer_reward = leaf_v + depth_t * (root_v - leaf_v)
        producer_reward = jnp.where(view.is_prod, producer_reward, 0.0)

        out_item_safe = jnp.clip(view.output_item, 0, num_items - 1)
        out_color_safe = jnp.clip(view.output_color, 0, NUM_COLORS - 1)
        out_token_ids = out_item_safe * NUM_COLORS + out_color_safe
        token_ids = jnp.arange(num_items * NUM_COLORS, dtype=jnp.int32)
        out_mask = view.is_prod[None, :] & (out_token_ids[None, :] == token_ids[:, None])
        has_output = jnp.any(out_mask, axis=1)
        max_output_reward = jnp.max(jnp.where(out_mask, producer_reward[None, :], 0.0), axis=1)

        reward_weights_flat = jnp.where(
            relevant_item_mask,
            jnp.where(has_output, max_output_reward, leaf_v),
            0.0,
        )
        return reward_weights_flat.reshape(num_items, NUM_COLORS).astype(jnp.float32)

    @staticmethod
    def compute_toggle_transform_success(
        state: State,
        action: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Detect toggle attempts and transform-triggering toggles.

        A toggle is marked successful if, at pre-step state, at least one transform rule
        has a matching input tile/color on the agent's current or adjacent tile.
        """
        action = jnp.asarray(action, dtype=jnp.int32)
        toggle_attempt = action == Action.TOGGLE

        runtime = state.rule_runtime
        any_transform = runtime.has_transform_rules

        def _compute_success(_):
            transform_ops = runtime.transform_ops
            input_tokens = transform_ops.input_tokens[:, 0]
            active_transform = jnp.arange(input_tokens.shape[0]) < transform_ops.count
            requires_adjacent = transform_ops.required_adjacent
            pos = state.positions  # (2,)
            positions_to_check = jnp.concatenate(
                [pos[None, :], pos[None, :] + DIR_TO_VEC], axis=0
            )  # (5, 2)
            ys = jnp.clip(positions_to_check[:, 0], 0, state.map_array.shape[0] - 1)
            xs = jnp.clip(positions_to_check[:, 1], 0, state.map_array.shape[1] - 1)
            in_bounds = (
                (positions_to_check[:, 0] >= 0)
                & (positions_to_check[:, 0] < state.map_array.shape[0])
                & (positions_to_check[:, 1] >= 0)
                & (positions_to_check[:, 1] < state.map_array.shape[1])
            )

            local_tokens = state.token_grid[ys, xs].astype(jnp.int32)
            valid_positions = jnp.where(
                requires_adjacent[None, :],
                in_bounds[:, None],
                in_bounds[:1, None],
            )
            rule_match = (
                active_transform[None, :]
                & valid_positions
                & (local_tokens[:, None] == input_tokens[None, :])
            )
            has_rule_match = jnp.any(rule_match)
            return toggle_attempt & has_rule_match

        toggle_success = jax.lax.cond(
            any_transform & toggle_attempt,
            _compute_success,
            lambda _: jnp.zeros_like(toggle_attempt, dtype=jnp.bool_),
            operand=None,
        )
        return toggle_success.astype(jnp.float32), toggle_attempt.astype(jnp.float32)

    @staticmethod
    def precompute_rule_obs(codes: jnp.ndarray):
        """Precompute static rule observation components (one-hot + is_root).

        These depend only on rule_encodings and don't change within an episode.
        Returns (rules_oh_static, is_root, is_prod).
        """
        view = _decode_rule_fields(codes)
        input_item_c_safe = jnp.clip(view.input_item_c, 0, NUM_ITEMS - 1)
        input_color_c_safe = jnp.clip(view.input_color_c, 0, NUM_COLORS - 1)

        # One-hot encode each column
        rule_type_oh = jax.nn.one_hot(codes[:, 0], num_classes=NUM_RULE_TYPES)
        col1_oh = jax.nn.one_hot(codes[:, 1], num_classes=NUM_ITEMS)
        col2_oh = jax.nn.one_hot(codes[:, 2], num_classes=NUM_ITEMS)
        col3_oh = jax.nn.one_hot(codes[:, 3], num_classes=NUM_ITEMS)
        col4_oh = (
            jax.nn.one_hot(input_item_c_safe, num_classes=NUM_ITEMS)
            * view.is_ternary_combine[:, None]
        )
        required_adjacent = view.required_adjacent[:, None].astype(jnp.float32)
        c1_oh = jax.nn.one_hot(
            jnp.clip(view.input_color_a, 0, NUM_COLORS - 1), num_classes=NUM_COLORS
        )
        c2_oh = jax.nn.one_hot(
            jnp.clip(view.input_color_b, 0, NUM_COLORS - 1), num_classes=NUM_COLORS
        )
        c3_oh = (
            jax.nn.one_hot(input_color_c_safe, num_classes=NUM_COLORS)
            * view.is_ternary_combine[:, None]
        )
        cout_oh = jax.nn.one_hot(
            jnp.clip(view.output_color, 0, NUM_COLORS - 1), num_classes=NUM_COLORS
        )

        rules_oh_static = jnp.concatenate(
            [
                rule_type_oh,
                col1_oh,
                col2_oh,
                col3_oh,
                col4_oh,
                required_adjacent,
                c1_oh,
                c2_oh,
                c3_oh,
                cout_oh,
            ],
            axis=-1,
        ).astype(jnp.float32)

        used_as_input = _producer_usage_matrix(view, codes.shape[0])
        out_is_used = jnp.any(used_as_input, axis=0)
        is_root = view.is_prod & (~out_is_used)

        return rules_oh_static, is_root, view.is_prod

    @staticmethod
    def compile_task_reset_metadata(
        codes: jnp.ndarray,
        *,
        include_rules_in_obs: bool,
        depth_weighted_pickup_shaping: bool,
        pickup_shaping_leaf_reward: float,
        pickup_shaping_root_reward: float,
        pre_move_program_size: int | None = None,
        post_move_program_size: int | None = None,
        transform_program_size: int | None = None,
        pre_move_combine_steps: int | None = None,
        post_move_combine_steps: int | None = None,
        pre_move_transform_steps: int | None = None,
        post_move_transform_steps: int | None = None,
        all_transform_steps: int | None = None,
    ) -> dict[str, Any]:
        """Compile all task-only reset metadata once for later lookup."""
        rule_runtime = compile_rule_runtime(
            codes,
            pre_move_program_size=pre_move_program_size,
            post_move_program_size=post_move_program_size,
            transform_program_size=transform_program_size,
            pre_move_combine_steps=pre_move_combine_steps,
            post_move_combine_steps=post_move_combine_steps,
            pre_move_transform_steps=pre_move_transform_steps,
            post_move_transform_steps=post_move_transform_steps,
            all_transform_steps=all_transform_steps,
        )
        goal_item, goal_color, goal_type, goal_vec, relevant_item_mask, depth = (
            Banyan.compile_goal_from_ruleset(codes, NUM_ITEMS, NUM_COLORS)
        )
        if include_rules_in_obs:
            rules_oh_static, is_root, is_comb = Banyan.precompute_rule_obs(codes)
        else:
            rules_oh_static = jnp.zeros((0, RULE_OBS_STATIC_DIM), dtype=jnp.float32)
            is_root = jnp.zeros((0,), dtype=jnp.bool_)
            is_comb = jnp.zeros((0,), dtype=jnp.bool_)
        pickup_reward_weights = jax.lax.cond(
            jnp.asarray(depth_weighted_pickup_shaping, dtype=jnp.bool_),
            lambda _: Banyan.compute_pickup_reward_weights(
                codes,
                NUM_ITEMS,
                pickup_shaping_leaf_reward,
                pickup_shaping_root_reward,
            ),
            lambda _: jnp.zeros((NUM_ITEMS, NUM_COLORS), dtype=jnp.float32),
            operand=None,
        )
        return {
            "rule_runtime": rule_runtime,
            "goal_item": goal_item,
            "goal_color": goal_color,
            "goal_type": goal_type,
            "goal_vec": goal_vec,
            "relevant_item_mask": relevant_item_mask,
            "pickup_reward_weights": pickup_reward_weights,
            "depth": depth,
            "rules_oh_static": rules_oh_static,
            "is_root": is_root,
            "is_comb": is_comb,
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey,
        initial_state: Optional[State] = None,
        ruleset: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, Any]:
        # 1) Treat dict/FrozenDict as PARAMS, not as a full State
        params: Optional[c_abc.Mapping[str, object]] = None
        if isinstance(initial_state, (dict, FrozenDict, c_abc.Mapping)):
            params = cast(c_abc.Mapping[str, object], initial_state)
            initial_state = None  # don't treat it as State

        # 2) If a real State was provided, use it; else construct a new State
        if initial_state is not None:
            state = initial_state
            updates = {}
            if getattr(state, "rule_runtime", None) is None:
                updates["rule_runtime"] = compile_rule_runtime(state.rule_encodings)
            if (
                getattr(state, "token_grid", None) is None
                or getattr(state, "token_counts", None) is None
            ):
                token_grid, token_counts = build_token_state(state.map_array, state.color_map)
                updates["token_grid"] = token_grid
                updates["token_counts"] = token_counts
            if getattr(state, "rules_dirty", None) is None:
                updates["rules_dirty"] = jnp.asarray(True, dtype=jnp.bool_)
            if updates:
                state = state.replace(**updates)  # ty:ignore[unresolved-attribute]
        else:
            # Choose map for this reset (override if provided)
            map_array = (
                self.map_array
                if (params is None or "map_array" not in params)
                else jnp.array(params["map_array"], dtype=jnp.int32)
            )

            color_map = (
                self.color_map
                if (params is None or "color_map" not in params)
                else jnp.array(params["color_map"], dtype=jnp.int32)
            )

            # Prefer spawn tiles in the seed-reachable component to avoid
            # disconnected-layout starts that cannot reach required items.
            flat_map = map_array.flatten()
            spawnable_mask = jnp.isin(flat_map, INITIAL_AGENT_POSITION_TILES)
            walkable = WALKABLE_MASK[map_array]
            walkable_flat = walkable.flatten()

            if params is not None and "spawn_reachable_mask" in params:
                reachable = jnp.asarray(params["spawn_reachable_mask"], dtype=jnp.bool_)
            else:
                seeds = jnp.zeros_like(walkable, dtype=jnp.bool_)
                seeds = seeds.at[0, 0].set(walkable[0, 0])
                if self.grid_size > 1:
                    seeds = seeds.at[1, 0].set(walkable[1, 0])

                def _flood_step(_i, reach):
                    up = jnp.pad(reach[1:, :], ((0, 1), (0, 0)))
                    down = jnp.pad(reach[:-1, :], ((1, 0), (0, 0)))
                    left = jnp.pad(reach[:, 1:], ((0, 0), (0, 1)))
                    right = jnp.pad(reach[:, :-1], ((0, 0), (1, 0)))
                    nbr = reach | up | down | left | right
                    return nbr & walkable

                reachable = jax.lax.fori_loop(
                    0, self.grid_size * self.grid_size, _flood_step, seeds
                )
            preferred_mask = spawnable_mask & reachable.flatten()

            preferred_count = jnp.sum(preferred_mask.astype(jnp.int32))
            spawnable_count = jnp.sum(spawnable_mask.astype(jnp.int32))
            valid_mask = jax.lax.cond(
                preferred_count >= 1,
                lambda _: preferred_mask,
                lambda _: jax.lax.cond(
                    spawnable_count >= 1,
                    lambda __: spawnable_mask,
                    lambda __: walkable_flat,
                    operand=None,
                ),
                operand=None,
            )

            key, pos_key = jax.random.split(key)
            probs = valid_mask.astype(jnp.float32)
            prob_sum = jnp.sum(probs)
            probs = jax.lax.cond(
                prob_sum > 0.0,
                lambda _: probs / prob_sum,
                lambda _: jnp.full_like(probs, 1.0 / probs.shape[0]),
                operand=None,
            )
            sampled_flat_position = jax.random.choice(
                pos_key,
                valid_mask.size,
                shape=(),
                replace=False,
                p=probs,
            )
            positions = jnp.stack(
                [
                    sampled_flat_position // self.grid_size,  # row
                    sampled_flat_position % self.grid_size,  # col
                ],
                axis=-1,
            ).astype(jnp.int32)

            # Randomly sample starting direction
            key, dir_key = jax.random.split(key)
            directions = jax.random.randint(dir_key, (), 0, 4)

            # 2. initialize inventory
            inventories = jnp.zeros((NUM_ITEMS,), dtype=jnp.bool_)
            inventory_colors = jnp.full((NUM_ITEMS,), Colors.BLACK, dtype=jnp.int32)
            items_ever_picked = jnp.zeros((NUM_ITEMS, NUM_COLORS), dtype=jnp.bool_)
            if ruleset is None:
                if params is not None and "ruleset" in params:
                    ruleset = jnp.asarray(params["ruleset"], dtype=jnp.int32)
                elif params is not None and "rule_encodings" in params:
                    ruleset = jnp.asarray(params["rule_encodings"], dtype=jnp.int32)
                else:
                    ruleset = self.ruleset
            if ruleset is None:
                ruleset = jnp.zeros((1, 6), dtype=jnp.int32)
            token_grid, token_counts = build_token_state(map_array, color_map)
            if params is not None and "rule_runtime" in params:
                task_metadata = {
                    "rule_runtime": cast(RuleRuntimeCache, params["rule_runtime"]),
                    "goal_item": jnp.asarray(params["goal_item"], dtype=jnp.int32),
                    "goal_color": jnp.asarray(params["goal_color"], dtype=jnp.int32),
                    "goal_type": jnp.asarray(params["goal_type"], dtype=jnp.int32),
                    "goal_vec": jnp.asarray(params["goal_vec"], dtype=jnp.float32),
                    "relevant_item_mask": jnp.asarray(
                        params["relevant_item_mask"], dtype=jnp.bool_
                    ),
                    "pickup_reward_weights": jnp.asarray(
                        params["pickup_reward_weights"], dtype=jnp.float32
                    ),
                    "depth": jnp.asarray(params["depth"], dtype=jnp.int32),
                    "rules_oh_static": jnp.asarray(params["rules_oh_static"], dtype=jnp.float32),
                    "is_root": jnp.asarray(params["is_root"], dtype=jnp.bool_),
                    "is_comb": jnp.asarray(params["is_comb"], dtype=jnp.bool_),
                }
            else:
                task_metadata = self.compile_task_reset_metadata(
                    ruleset,
                    include_rules_in_obs=self.include_rules_in_obs,
                    depth_weighted_pickup_shaping=self.depth_weighted_pickup_shaping,
                    pickup_shaping_leaf_reward=self.pickup_shaping_leaf_reward,
                    pickup_shaping_root_reward=self.pickup_shaping_root_reward,
                )

            # 3. initialize state
            state = self.State(
                positions=positions,
                directions=directions,
                inventories=inventories,
                inventory_colors=inventory_colors,
                map_array=map_array,
                color_map=color_map,
                token_grid=token_grid,
                token_counts=token_counts,
                items_ever_picked=items_ever_picked,
                time_step=0,
                rules_dirty=jnp.asarray(True, dtype=jnp.bool_),
                done=False,
                rule_encodings=ruleset,
                rule_runtime=cast(RuleRuntimeCache, task_metadata["rule_runtime"]),
                goal_item=cast(jax.Array, task_metadata["goal_item"]),
                goal_color=cast(jax.Array, task_metadata["goal_color"]),
                goal_type=cast(jax.Array, task_metadata["goal_type"]),
                goal_vec=cast(jax.Array, task_metadata["goal_vec"]),
                relevant_item_mask=cast(jax.Array, task_metadata["relevant_item_mask"]),
                pickup_reward_weights=cast(jax.Array, task_metadata["pickup_reward_weights"]),
                depth=cast(jax.Array, task_metadata["depth"]),
                rules_oh_static=cast(jax.Array, task_metadata["rules_oh_static"]),
                is_root=cast(jax.Array, task_metadata["is_root"]),
                is_comb=cast(jax.Array, task_metadata["is_comb"]),
            )

        obs = self.get_obs(state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: State,
        action: jax.Array,
    ) -> Tuple[jax.Array, Any, jax.Array, jax.Array, Dict[str, Any]]:
        action = jnp.asarray(action, dtype=jnp.int32).reshape(())

        state = state.replace(time_step=state.time_step + 1)  # ty:ignore[unresolved-attribute]
        old_inventories = state.inventories
        old_inventory_colors = state.inventory_colors
        old_ever_picked = state.items_ever_picked
        toggle_transform_success, toggle_attempt = self.compute_toggle_transform_success(
            state, action
        )
        state = check_rule(state.rule_runtime, state, action, self)

        drop_attempt = action == Action.DROP
        dropped_item_mask = jnp.logical_and(old_inventories, jnp.logical_not(state.inventories))
        dropped_any = jnp.any(dropped_item_mask)
        dropped_item_idx = jnp.argmax(dropped_item_mask.astype(jnp.int32))
        dropped_color = old_inventory_colors[dropped_item_idx]
        drop_success = drop_attempt & dropped_any
        drop_failed = drop_attempt & (~drop_success)
        dropped_item_idx = jnp.where(drop_success, dropped_item_idx, -1)
        dropped_color = jnp.where(drop_success, dropped_color, -1)

        # Apply distractor combines via lookup table (if present)
        distractor_penalty = jnp.asarray(0.0, dtype=jnp.float32)
        dead_end = jnp.array(False, dtype=jnp.bool_)
        penalty_enabled = jnp.asarray(self.distractor_combine_penalty != 0.0, dtype=jnp.bool_)
        if self.distractor_table is not None:
            state, penalized = apply_distractor_combines_near_drops(
                state,
                self.distractor_table,
                state.positions,
                drop_success,
                dropped_item_idx,
                dropped_color,
            )
            distractor_penalty = self.distractor_combine_penalty * penalized.astype(jnp.float32)
            dead_end = penalty_enabled & penalized

        # check goal conditions
        done_goal, reward = self.goal(state)
        reward = reward * self.goal_reward_scale

        # check termination
        is_timeout = state.time_step >= self.max_steps
        done = jnp.logical_or(jnp.logical_or(done_goal, is_timeout), dead_end)
        state = state.replace(done=done)

        obs = self.get_obs(state)
        # Intermediate shaping: +0.1 when the agent newly gains a relevant item for the FIRST TIME
        # Only reward first-time pickups to prevent pick/drop reward hacking
        new_inv = state.inventories
        newly_picked = jnp.logical_and(new_inv, jnp.logical_not(old_inventories))
        picked_color = jnp.clip(state.inventory_colors, 0, NUM_COLORS - 1)
        newly_picked_token = newly_picked[:, None] & jax.nn.one_hot(
            picked_color, num_classes=NUM_COLORS
        ).astype(jnp.bool_)
        first_time_pickup = jnp.logical_and(newly_picked_token, jnp.logical_not(old_ever_picked))
        # Update ever-picked tracking
        state = state.replace(items_ever_picked=jnp.logical_or(old_ever_picked, first_time_pickup))
        gained_relevant = jnp.logical_and(first_time_pickup, state.relevant_item_mask)
        if self.depth_weighted_pickup_shaping:
            shaping_reward = jnp.sum(
                gained_relevant.astype(jnp.float32) * state.pickup_reward_weights
            ).astype(jnp.float32)
        else:
            shaping_reward = 0.1 * jnp.any(gained_relevant).astype(jnp.float32)
        # a small per-step time penalty
        base_reward = reward + shaping_reward + distractor_penalty + (-0.001)
        timeout_only = is_timeout & (~done_goal) & (~dead_end)
        timeout_penalty = jnp.where(
            timeout_only,
            jnp.asarray(self.timeout_penalty, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        reward = jnp.where(
            dead_end,
            jnp.asarray(self.distractor_combine_penalty, dtype=jnp.float32),
            jnp.where(timeout_only, timeout_penalty, base_reward),
        ).astype(jnp.float32)
        info = {
            "reward_shape": shaping_reward,
            "reward_distractor_penalty": distractor_penalty,
            "reward_timeout_penalty": timeout_penalty,
            "reward_total": reward,
            "goal_achieved": done_goal.astype(jnp.float32),
            "timeout": is_timeout.astype(jnp.float32),
            "dead_end": dead_end.astype(jnp.float32),
            "action_pickup": (action == Action.PICKUP).astype(jnp.float32),
            "action_drop": drop_attempt.astype(jnp.float32),
            "action_drop_success": drop_success.astype(jnp.float32),
            "action_drop_failed": drop_failed.astype(jnp.float32),
            "action_toggle": (action == Action.TOGGLE).astype(jnp.float32),
            "action_move": (action <= Action.DOWN).astype(jnp.float32),
            "toggle_attempt": toggle_attempt,
            "toggle_transform_success": toggle_transform_success,
        }

        return obs, state, reward, done, info

    def action_space(self) -> Discrete:
        return self._action_space

    def observation_space(self) -> Box:
        return self._observation_space

    def get_obs(self, state: State) -> jax.Array:
        # ---- constants / shapes ----
        H = self.grid_size
        W = self.grid_size
        num_tile_types = NUM_TILE_TYPES

        # ---------------------------------------------------------------------
        # 1) Grid encodings
        # ---------------------------------------------------------------------
        # (H, W, num_tile_types)
        map_grid_one_hot = jax.nn.one_hot(state.map_array, num_classes=num_tile_types).astype(
            jnp.float32
        )

        # (H, W, NUM_COLORS)
        color_oh = jax.nn.one_hot(state.color_map, num_classes=NUM_COLORS).astype(jnp.float32)

        # ---------------------------------------------------------------------
        # 2) Goal encoding (precomputed at reset, no expensive sink-detection here!)
        # ---------------------------------------------------------------------
        # (goal_dim,) where goal_dim = 2 + NUM_ITEMS + NUM_COLORS
        goal_vec = state.goal_vec

        # ---------------------------------------------------------------------
        # 3) Inventories + masked inventory-color onehots
        # ---------------------------------------------------------------------
        # (NUM_ITEMS,) float32
        inventories = state.inventories.astype(jnp.float32)

        # (NUM_ITEMS, NUM_COLORS) float32
        inv_color_oh = jax.nn.one_hot(state.inventory_colors, num_classes=NUM_COLORS).astype(
            jnp.float32
        )
        inv_color_oh = inv_color_oh * inventories[:, None]  # mask by possession

        # ---------------------------------------------------------------------
        # 4) Self-position grid
        # ---------------------------------------------------------------------
        pos = state.positions
        pos_idx = (pos[0] * W + pos[1]).astype(jnp.int32)

        # self_grid[y, x] = 1 at the agent position
        # Shape: (H, W)
        self_grid = jax.nn.one_hot(pos_idx, num_classes=H * W).astype(jnp.float32).reshape(H, W)

        # ---------------------------------------------------------------------
        # 5) Spatial obs
        # ---------------------------------------------------------------------
        spatial = jnp.concatenate(
            [map_grid_one_hot, self_grid[..., None], color_oh],
            axis=-1,
        )

        spatial_flat = spatial.reshape(-1)  # (H*W*(num_tile_types+1+NUM_COLORS),)

        # ---------------------------------------------------------------------
        # 6) Inventory view
        # ---------------------------------------------------------------------
        inv_flat = inventories
        invc_flat = inv_color_oh.reshape(-1)

        if not self.include_rules_in_obs:
            return jnp.concatenate([spatial_flat, inv_flat, invc_flat, goal_vec], axis=-1).astype(
                jnp.float32
            )

        # ---------------------------------------------------------------------
        # 8) Ruleset encoding (uses precomputed static components from reset)
        # ---------------------------------------------------------------------
        num_rules = state.rules_oh_static.shape[0]
        if num_rules > self.R_max:
            raise ValueError(
                f"Ruleset has {num_rules} rows, but max_rules={self.R_max} cannot fit "
                "them in the observation. Construct Banyan with "
                f"max_rules >= {num_rules}, or set include_rules_in_obs=False. "
                "Rules are never truncated."
            )
        runtime = state.rule_runtime
        is_prod = state.is_comb  # precomputed producer mask at reset

        inv_item_ids = jnp.arange(NUM_ITEMS, dtype=jnp.int32)
        inv_token_ids = inv_item_ids * NUM_COLORS + jnp.clip(
            state.inventory_colors, 0, NUM_COLORS - 1
        )
        inv_token_oh = jax.nn.one_hot(inv_token_ids, num_classes=NUM_ITEMS * NUM_COLORS).astype(
            jnp.bool_
        )
        inventory_token_present = jnp.any(inv_token_oh & state.inventories[:, None], axis=0)

        map_token_present = state.token_counts > 0

        has_in1 = inventory_token_present[runtime.input_token_a]
        has_in2 = jnp.where(
            runtime.input_count >= 2,
            inventory_token_present[runtime.input_token_b],
            jnp.ones_like(is_prod, dtype=jnp.bool_),
        )
        has_in3 = jnp.where(
            runtime.input_count >= 3,
            inventory_token_present[runtime.input_token_c],
            jnp.ones_like(is_prod, dtype=jnp.bool_),
        )
        has_out = inventory_token_present[runtime.output_token]
        inputs_ready = has_in1 & has_in2 & has_in3 & is_prod
        map_has_out = map_token_present[runtime.output_token]
        output_exists = (has_out | map_has_out) & is_prod

        # is_root is precomputed at reset (static per episode)
        is_root = state.is_root
        progress = jnp.stack([inputs_ready, output_exists, is_root], axis=-1).astype(jnp.float32)

        # Concat precomputed static one-hots with dynamic progress
        rules_oh = jnp.concatenate(
            [state.rules_oh_static, progress],
            axis=-1,
        ).astype(jnp.float32)  # (R, rule_oh_dim)

        # Pad to R_max rules for consistent observation shape
        rule_oh_dim = self.rule_oh_dim
        rules_flat_target_dim = self.R_max * rule_oh_dim
        rules_flat = rules_oh.reshape(-1)
        rules_flat = jnp.pad(rules_flat, (0, rules_flat_target_dim - rules_flat.shape[0]))

        obs = jnp.concatenate(
            [spatial_flat, inv_flat, invc_flat, goal_vec, rules_flat], axis=-1
        ).astype(jnp.float32)
        return obs
