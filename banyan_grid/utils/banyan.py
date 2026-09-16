from typing import Any

import jax
import jax.numpy as jnp

from banyan_grid.environment.constants import (
    ITEM_TO_TILE,
    RULE_TYPE_COLLECT,
    RULE_TYPE_COMBINE,
    RULE_TYPE_TERNARY_COMBINE,
    RULE_TYPE_TRANSFORM,
    Colors,
    TileType,
)
from banyan_grid.environment.goals import AgentHasItemFromRulesetGoal
from banyan_grid.environment.banyan import (
    Banyan,
)


def create_env(
    size: int,
    steps: int,
    include_rules_in_obs: bool = True,
    max_rules: int | None = None,
    distractor_table: jax.Array | None = None,
    distractor_combine_penalty: float = -0.1,
    timeout_penalty: float = 0.0,
    depth_weighted_pickup_shaping: bool = False,
    pickup_shaping_leaf_reward: float = 0.05,
    pickup_shaping_root_reward: float = 0.2,
    goal_reward_scale: float = 1.0,
) -> Banyan:
    """Create a minimal env for shape inference; real maps/rulesets come at reset."""
    grid = jnp.full((size, size), TileType.OPEN_FAST, jnp.int32)
    colors = jnp.full((size, size), Colors.BLACK, jnp.int32)

    return Banyan(
        size,
        steps,
        grid,
        colors,
        AgentHasItemFromRulesetGoal(True),
        max_rules=max_rules,
        include_rules_in_obs=include_rules_in_obs,
        distractor_table=distractor_table,
        distractor_combine_penalty=distractor_combine_penalty,
        timeout_penalty=timeout_penalty,
        depth_weighted_pickup_shaping=depth_weighted_pickup_shaping,
        pickup_shaping_leaf_reward=pickup_shaping_leaf_reward,
        pickup_shaping_root_reward=pickup_shaping_root_reward,
        goal_reward_scale=goal_reward_scale,
    )


def get_map(
    rng: jax.Array,
    size: int,
    ruleset: jax.Array,
    reserved_cells: jax.Array | None = None,
    reserved_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Generate map/color arrays from a ruleset."""
    rng, rng_perm = jax.random.split(rng)
    num_cells = size * size
    perm = jax.random.permutation(rng_perm, num_cells)

    if reserved_mask is not None:
        allowed = ~reserved_mask.reshape((num_cells,)).astype(jnp.bool_)
        reserved = jnp.empty((0, 2), dtype=jnp.int32)
    else:
        allowed = jnp.ones((num_cells,), dtype=bool)
        reserved = (
            reserved_cells
            if reserved_cells is not None
            else jnp.empty((0, 2), dtype=jnp.int32)
        )

    def block_one(a: jax.Array, coord: jax.Array) -> jax.Array:
        y, x = coord[0], coord[1]
        is_in_bounds = (y >= 0) & (x >= 0) & (y < size) & (x < size)
        idx = y * size + x
        return jax.lax.cond(
            is_in_bounds, lambda arr: arr.at[idx].set(False), lambda arr: arr, a
        )

    allowed = jax.lax.scan(
        lambda a, coord: (block_one(a, coord), None), allowed, reserved
    )[0]

    grid = jnp.full((size, size), TileType.OPEN_FAST, dtype=jnp.int32)
    colors = jnp.full((size, size), Colors.BLACK, dtype=jnp.int32)

    rule_types = ruleset[:, 0]
    is_collect = rule_types == RULE_TYPE_COLLECT
    is_binary_combine = rule_types == RULE_TYPE_COMBINE
    is_ternary_combine = rule_types == RULE_TYPE_TERNARY_COMBINE
    is_combine = is_binary_combine | is_ternary_combine
    is_transform = rule_types == RULE_TYPE_TRANSFORM
    is_producer = is_combine | is_transform
    any_collect = jnp.any(is_collect)
    any_producer = jnp.any(is_producer)

    packed = ruleset[:, 5]
    c1 = packed & 0xF
    c2 = (packed >> 4) & 0xF
    c3 = jnp.where(is_ternary_combine, (packed >> 8) & 0xF, -1)
    cout = jnp.where(
        is_ternary_combine,
        (packed >> 12) & 0xF,
        (packed >> 8) & 0xF,
    )
    in1_item = ruleset[:, 1]
    in2_item = ruleset[:, 2]
    in3_item = jnp.where(is_ternary_combine, ruleset[:, 4] >> 1, -1)
    out_item = ruleset[:, 3]
    out_item_comb = jnp.where(is_producer, out_item, -1)
    out_color_comb = jnp.where(is_producer, cout, -1)

    def is_leaf_input(item, color):
        return ~jnp.any((out_item_comb == item) & (out_color_comb == color))

    leaf1_mask = jax.vmap(is_leaf_input)(in1_item, c1)
    leaf2_mask = jax.vmap(is_leaf_input)(in2_item, c2)
    leaf3_mask = jax.vmap(is_leaf_input)(in3_item, c3)

    def place_item(
        carry2: tuple[jax.Array, jax.Array, jax.Array, int],
        tile_type: jax.Array,
        color_req: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, int]:
        grid_p, colors_p, allowed_p, ptr_p = carry2

        def cond_fn(ptr_prime: int) -> jax.Array:
            idx = perm[jnp.minimum(ptr_prime, num_cells - 1)]
            return (ptr_prime < num_cells) & (~allowed_p[idx])

        ptr2 = jax.lax.while_loop(cond_fn, lambda ptr_prime: ptr_prime + 1, ptr_p)
        carry_in = (grid_p, colors_p, allowed_p, ptr2)

        def on_found(
            carry3: tuple[jax.Array, jax.Array, jax.Array, int],
        ) -> tuple[jax.Array, jax.Array, jax.Array, int]:
            grid2, colors2, allowed2, ptr3 = carry3
            flat_idx = perm[ptr3]
            r, col = flat_idx // size, flat_idx % size
            grid2 = grid2.at[r, col].set(tile_type)
            colors2 = colors2.at[r, col].set(color_req)
            allowed2 = allowed2.at[flat_idx].set(False)
            return (grid2, colors2, allowed2, ptr3 + 1)

        return jax.lax.cond(ptr2 < num_cells, on_found, lambda c: c, carry_in)

    def body(
        i: int, carry: tuple[jax.Array, jax.Array, jax.Array, int]
    ) -> tuple[jax.Array, jax.Array, jax.Array, int]:
        grid_acc, colors_acc, allowed_mask, ptr = carry

        def place_inputs(
            carry_in: tuple[jax.Array, jax.Array, jax.Array, int],
        ) -> tuple[jax.Array, jax.Array, jax.Array, int]:
            grid_p, colors_p, allowed_p, ptr_p = carry_in
            leaf1 = is_leaf_input(in1_item[i], c1[i]) & is_producer[i]
            carry_out = jax.lax.cond(
                leaf1,
                lambda carry: place_item(carry, ITEM_TO_TILE[in1_item[i]], c1[i]),
                lambda carry: carry,
                (grid_p, colors_p, allowed_p, ptr_p),
            )
            leaf2 = leaf2_mask[i] & is_combine[i]
            carry_out = jax.lax.cond(
                leaf2,
                lambda carry: place_item(carry, ITEM_TO_TILE[in2_item[i]], c2[i]),
                lambda carry: carry,
                carry_out,
            )
            leaf3 = leaf3_mask[i] & is_ternary_combine[i]
            carry_out = jax.lax.cond(
                leaf3,
                lambda carry: place_item(carry, ITEM_TO_TILE[in3_item[i]], c3[i]),
                lambda carry: carry,
                carry_out,
            )
            return carry_out

        return jax.lax.cond(
            is_producer[i],
            place_inputs,
            lambda carry_: carry_,
            (grid_acc, colors_acc, allowed_mask, ptr),
        )

    def place_from_collect(
        carry: tuple[jax.Array, jax.Array, jax.Array, int],
    ) -> tuple[jax.Array, jax.Array, jax.Array, int]:
        grid_acc, colors_acc, allowed_mask, ptr = carry
        idx = jnp.argmax(is_collect.astype(jnp.int32))
        tile_type = ruleset[idx, 1]
        color_req = ruleset[idx, 3]
        return jax.lax.cond(
            any_collect,
            lambda c2: place_item(c2, tile_type, color_req),
            lambda c2: c2,
            (grid_acc, colors_acc, allowed_mask, ptr),
        )

    num_rules = ruleset.shape[0]
    result = jax.lax.cond(
        any_producer,
        lambda carry: jax.lax.fori_loop(0, num_rules, body, carry),
        place_from_collect,
        (grid, colors, allowed, 0),
    )
    return result[0], result[1]


def get_reset_params(
    rng: jax.Array,
    size: int,
    rulesets: jax.Array,
    layout_obstacle_masks: jax.Array | None = None,
    layout_reserved_cells: jax.Array | None = None,
    layout_reserved_masks: jax.Array | None = None,
    layout_spawn_reachable_masks: jax.Array | None = None,
    layout_indices: jax.Array | None = None,
    ruleset_indices: jax.Array | None = None,
    precomputed_maps: jax.Array | None = None,
    precomputed_colors: jax.Array | None = None,
    precomputed_task_layout_indices: jax.Array | None = None,
    precomputed_task_layout_counts: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """Batched map generation for vectorized environment resets."""

    def _select_precomputed(key: jax.Array, task_idx: jax.Array) -> tuple[jax.Array, jax.Array]:
        if (
            precomputed_maps is None
            or precomputed_colors is None
            or precomputed_task_layout_indices is None
            or precomputed_task_layout_counts is None
        ):
            raise ValueError("precomputed reset bank tensors must all be provided together.")

        num_tasks = precomputed_task_layout_counts.shape[0]
        max_choices = precomputed_task_layout_indices.shape[1]
        valid_task = (task_idx >= 0) & (task_idx < num_tasks)
        safe_task = jnp.clip(task_idx, 0, num_tasks - 1)
        count = jnp.where(valid_task, precomputed_task_layout_counts[safe_task], 0)

        def _use_bank(_: None) -> tuple[jax.Array, jax.Array]:
            raw_slot = jax.random.randint(
                key,
                (),
                0,
                max(max_choices, 1),
                dtype=jnp.int32,
            )
            slot = jnp.mod(raw_slot, jnp.maximum(count, 1))
            layout_idx = precomputed_task_layout_indices[safe_task, slot]
            safe_layout_idx = jnp.clip(layout_idx, 0, precomputed_maps.shape[0] - 1)
            return (
                precomputed_maps[safe_layout_idx],
                precomputed_colors[safe_layout_idx],
            )

        return jax.lax.cond(
            count > 0,
            _use_bank,
            lambda _: (
                jnp.full((size, size), jnp.int32(TileType.OPEN_FAST), dtype=jnp.int32),
                jnp.full((size, size), jnp.int32(Colors.BLACK), dtype=jnp.int32),
            ),
            operand=None,
        )

    def single(
        key: jax.Array,
        ruleset: jax.Array,
        reserved_cells: jax.Array | None,
        reserved_mask: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        return get_map(
            key,
            size,
            ruleset,
            reserved_cells=reserved_cells,
            reserved_mask=reserved_mask,
        )

    def single_with_optional_bank(
        key: jax.Array,
        ruleset: jax.Array,
        task_idx: jax.Array | None,
        reserved_cells: jax.Array | None,
        reserved_mask: jax.Array | None,
        obstacle_mask: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        if task_idx is not None and precomputed_task_layout_counts is not None:
            num_tasks = precomputed_task_layout_counts.shape[0]
            valid_task = (task_idx >= 0) & (task_idx < num_tasks)
            safe_task = jnp.clip(task_idx, 0, num_tasks - 1)
            count = jnp.where(valid_task, precomputed_task_layout_counts[safe_task], 0)

            def _from_bank(_: None) -> tuple[jax.Array, jax.Array]:
                return _select_precomputed(key, task_idx)

            def _online(_: None) -> tuple[jax.Array, jax.Array]:
                map_arr, color_arr = single(
                    key,
                    ruleset,
                    reserved_cells,
                    reserved_mask,
                )
                if obstacle_mask is not None:
                    map_arr = jnp.where(obstacle_mask, jnp.int32(TileType.BLOCK), map_arr)
                    color_arr = jnp.where(
                        obstacle_mask, jnp.int32(Colors.BLACK), color_arr
                    )
                return map_arr, color_arr

            return jax.lax.cond(count > 0, _from_bank, _online, operand=None)

        map_arr, color_arr = single(key, ruleset, reserved_cells, reserved_mask)
        if obstacle_mask is not None:
            map_arr = jnp.where(obstacle_mask, jnp.int32(TileType.BLOCK), map_arr)
            color_arr = jnp.where(obstacle_mask, jnp.int32(Colors.BLACK), color_arr)
        return map_arr, color_arr

    if layout_obstacle_masks is None:
        if ruleset_indices is None:
            map_array, color_map = jax.vmap(
                lambda key, ruleset: single_with_optional_bank(
                    key, ruleset, None, None, None, None
                )
            )(rng, rulesets)
        else:
            map_array, color_map = jax.vmap(
                lambda key, ruleset, task_idx: single_with_optional_bank(
                    key, ruleset, task_idx, None, None, None
                )
            )(rng, rulesets, ruleset_indices)
        return {"map_array": map_array, "color_map": color_map}

    if layout_reserved_cells is None:
        raise ValueError(
            "layout_reserved_cells must be provided when layout_obstacle_masks is set."
        )

    if layout_indices is None:
        batch_size = rulesets.shape[0]
        num_layouts = layout_obstacle_masks.shape[0]
        layout_indices = jnp.arange(batch_size, dtype=jnp.int32) % num_layouts

    selected_masks = layout_obstacle_masks[layout_indices]
    selected_reserved = layout_reserved_cells[layout_indices]
    use_layout_reserved_mask = layout_reserved_masks is not None
    selected_reserved_mask = (
        layout_reserved_masks[layout_indices]
        if use_layout_reserved_mask
        else jnp.zeros_like(selected_masks)
    )
    selected_spawn_reachable = (
        layout_spawn_reachable_masks[layout_indices]
        if (
            layout_spawn_reachable_masks is not None
            and precomputed_task_layout_counts is None
        )
        else None
    )

    if ruleset_indices is None:
        map_array, color_map = jax.vmap(
            lambda key, ruleset, reserved_cells, reserved_mask, obstacle_mask: single_with_optional_bank(
                key,
                ruleset,
                None,
                reserved_cells,
                reserved_mask if use_layout_reserved_mask else None,
                obstacle_mask,
            )
        )(rng, rulesets, selected_reserved, selected_reserved_mask, selected_masks)
    else:
        map_array, color_map = jax.vmap(
            lambda key, ruleset, task_idx, reserved_cells, reserved_mask, obstacle_mask: single_with_optional_bank(
                key,
                ruleset,
                task_idx,
                reserved_cells,
                reserved_mask if use_layout_reserved_mask else None,
                obstacle_mask,
            )
        )(
            rng,
            rulesets,
            ruleset_indices,
            selected_reserved,
            selected_reserved_mask,
            selected_masks,
        )
    reset_params = {"map_array": map_array, "color_map": color_map}
    if selected_spawn_reachable is not None:
        reset_params["spawn_reachable_mask"] = selected_spawn_reachable
    return reset_params


def success_from_state(state: Banyan.State) -> jax.Array:
    """Return True if the goal is achieved for collect/compositional rules."""
    codes = state.rule_encodings
    rule_types = codes[:, 0]
    is_collect = rule_types == RULE_TYPE_COLLECT
    is_binary_combine = rule_types == RULE_TYPE_COMBINE
    is_ternary_combine = rule_types == RULE_TYPE_TERNARY_COMBINE
    is_comb = is_binary_combine | is_ternary_combine
    is_transform = rule_types == RULE_TYPE_TRANSFORM
    is_prod = is_comb | is_transform
    any_prod = jnp.any(is_prod)

    def from_collect(ruleset: jax.Array) -> tuple[jax.Array, jax.Array]:
        idx = jnp.argmax(is_collect.astype(jnp.int32))
        return ruleset[idx, 2], ruleset[idx, 3]

    def from_producer(ruleset: jax.Array) -> tuple[jax.Array, jax.Array]:
        packed = ruleset[:, 5]
        c1 = packed & 0xF
        c2 = (packed >> 4) & 0xF
        c3 = jnp.where(is_ternary_combine, (packed >> 8) & 0xF, -1)
        cout = jnp.where(
            is_ternary_combine,
            (packed >> 12) & 0xF,
            (packed >> 8) & 0xF,
        )
        out_vec = jnp.where(is_prod, ruleset[:, 3], -1)
        out_col = jnp.where(is_prod, cout, -1)
        in1_vec = jnp.where(is_prod, ruleset[:, 1], -2)
        in1_col = jnp.where(is_prod, c1, -2)
        in2_vec = jnp.where(is_comb, ruleset[:, 2], -2)
        in2_col = jnp.where(is_comb, c2, -2)
        in3_vec = jnp.where(is_ternary_combine, ruleset[:, 4] >> 1, -2)
        in3_col = jnp.where(is_ternary_combine, c3, -2)

        used_as_input = (
            ((in1_vec[:, None] == out_vec[None, :]) & (in1_col[:, None] == out_col[None, :]))
            | ((in2_vec[:, None] == out_vec[None, :]) & (in2_col[:, None] == out_col[None, :]))
            | ((in3_vec[:, None] == out_vec[None, :]) & (in3_col[:, None] == out_col[None, :]))
        )
        used_as_input = (
            used_as_input
            & is_prod[:, None]
            & is_prod[None, :]
            & (~jnp.eye(ruleset.shape[0], dtype=bool))
        )
        out_is_used = jnp.any(used_as_input, axis=0)

        sink_mask = is_prod & (~out_is_used)
        have_sink = jnp.any(sink_mask)
        target_idx = jnp.argmax(
            jax.lax.select(have_sink, sink_mask, is_prod).astype(jnp.int32)
        )
        target_item = out_vec[target_idx]
        target_color = out_col[target_idx]
        return target_item, target_color

    target_item, target_color = jax.lax.cond(
        any_prod, from_producer, from_collect, codes
    )

    inventories = state.inventories
    inventory_colors = state.inventory_colors
    has_item = inventories[target_item]
    color_ok = inventory_colors[target_item] == target_color
    return has_item & color_ok


def compute_depth(state: Banyan.State) -> jax.Array:
    """Compute ruleset depth from longest producer-chain in the rule DAG."""
    codes = state.rule_encodings
    rule_types = codes[:, 0]
    combine_count = jnp.sum(
        (rule_types == RULE_TYPE_COMBINE) | (rule_types == RULE_TYPE_TERNARY_COMBINE)
    )
    transform_count = jnp.sum(rule_types == RULE_TYPE_TRANSFORM)
    producer_count = combine_count + transform_count

    is_binary_combine = rule_types == RULE_TYPE_COMBINE
    is_ternary_combine = rule_types == RULE_TYPE_TERNARY_COMBINE
    is_comb = is_binary_combine | is_ternary_combine
    is_prod = is_comb | (rule_types == RULE_TYPE_TRANSFORM)
    packed = codes[:, 5]
    c1 = packed & 0xF
    c2 = (packed >> 4) & 0xF
    c3 = jnp.where(is_ternary_combine, (packed >> 8) & 0xF, -1)
    cout = jnp.where(
        is_ternary_combine,
        (packed >> 12) & 0xF,
        (packed >> 8) & 0xF,
    )

    out_item = jnp.where(is_prod, codes[:, 3], -1)
    out_color = jnp.where(is_prod, cout, -1)
    in1_item = jnp.where(is_prod, codes[:, 1], -2)
    in1_color = jnp.where(is_prod, c1, -2)
    in2_item = jnp.where(is_comb, codes[:, 2], -2)
    in2_color = jnp.where(is_comb, c2, -2)
    in3_item = jnp.where(is_ternary_combine, codes[:, 4] >> 1, -2)
    in3_color = jnp.where(is_ternary_combine, c3, -2)

    uses_out_as_in1 = (in1_item[:, None] == out_item[None, :]) & (
        in1_color[:, None] == out_color[None, :]
    )
    uses_out_as_in2 = (in2_item[:, None] == out_item[None, :]) & (
        in2_color[:, None] == out_color[None, :]
    )
    uses_out_as_in3 = (in3_item[:, None] == out_item[None, :]) & (
        in3_color[:, None] == out_color[None, :]
    )
    dep = (
        (uses_out_as_in1 | uses_out_as_in2 | uses_out_as_in3)
        & is_prod[:, None]
        & is_prod[None, :]
        & (~jnp.eye(codes.shape[0], dtype=bool))
    )

    op_depth = is_prod.astype(jnp.int32)

    def relax(_i, cur):
        parent_max = jnp.max(jnp.where(dep, cur[None, :], 0), axis=1)
        candidate = 1 + parent_max
        return jnp.where(is_prod, jnp.maximum(cur, candidate), 0)

    op_depth = jax.lax.fori_loop(0, codes.shape[0], relax, op_depth)
    return jnp.where(
        producer_count == 0,
        jnp.int32(1),
        jnp.int32(1) + jnp.max(op_depth),
    )


def build_depth_lookup_from_meta(
    meta: dict[str, Any] | None,
) -> tuple[jax.Array, jax.Array] | None:
    """Build combine_count->depth lookup arrays from dataset metadata."""
    if not isinstance(meta, dict):
        return None

    if str(meta.get("tree_topology", "")) == "mixed_u1_b2":
        return None

    topo_rule_count_dist = meta.get("topology_rule_count_distribution_per_depth")
    if isinstance(topo_rule_count_dist, dict):
        count_to_depth: dict[int, int] = {0: 1}
        ambiguous = False
        for depth_key, dist in topo_rule_count_dist.items():
            try:
                depth = int(depth_key)
            except (TypeError, ValueError):
                continue
            if depth <= 0 or not isinstance(dist, dict):
                continue
            for count_key, freq_val in dist.items():
                try:
                    producer_count = int(count_key)
                    freq = int(freq_val)
                except (TypeError, ValueError):
                    continue
                if producer_count < 0 or freq <= 0:
                    continue
                prev_depth = count_to_depth.get(producer_count)
                if prev_depth is None:
                    count_to_depth[producer_count] = depth
                elif prev_depth != depth:
                    ambiguous = True
                    break
            if ambiguous:
                break

        if ambiguous:
            return None
        if count_to_depth:
            keys_sorted = sorted(count_to_depth.keys())
            combine_counts = jnp.asarray(keys_sorted, dtype=jnp.int32)
            depths = jnp.asarray(
                [count_to_depth[k] for k in keys_sorted], dtype=jnp.int32
            )
            return combine_counts, depths

    rules_per_depth = meta.get("rules_per_depth")
    if not isinstance(rules_per_depth, dict):
        return None

    combine_to_depth: dict[int, int] = {}
    for depth_key, rule_count_val in rules_per_depth.items():
        try:
            depth = int(depth_key)
            rule_count = int(rule_count_val)
        except (TypeError, ValueError):
            continue

        if depth <= 0:
            continue
        producer_count = 0 if depth == 1 else max(0, rule_count)
        if producer_count in combine_to_depth:
            combine_to_depth[producer_count] = max(combine_to_depth[producer_count], depth)
        else:
            combine_to_depth[producer_count] = depth

    if not combine_to_depth:
        return None

    keys_sorted = sorted(combine_to_depth.keys())
    combine_counts = jnp.asarray(keys_sorted, dtype=jnp.int32)
    depths = jnp.asarray([combine_to_depth[k] for k in keys_sorted], dtype=jnp.int32)
    return combine_counts, depths


def compute_depth_with_lookup(
    state: Banyan.State,
    depth_lookup: tuple[jax.Array, jax.Array] | None,
) -> jax.Array:
    """Compute depth using optional metadata-derived lookup, with formula fallback."""
    depth_formula = compute_depth(state)
    if depth_lookup is None:
        return depth_formula

    combine_counts, depth_values = depth_lookup
    codes = state.rule_encodings
    producer_count = jnp.sum(
        (codes[:, 0] == RULE_TYPE_COMBINE)
        | (codes[:, 0] == RULE_TYPE_TERNARY_COMBINE)
        | (codes[:, 0] == RULE_TYPE_TRANSFORM)
    )
    matches = combine_counts == producer_count
    mapped_depth = jnp.max(jnp.where(matches, depth_values, jnp.int32(0)))
    return jnp.where(jnp.any(matches), mapped_depth, depth_formula)
