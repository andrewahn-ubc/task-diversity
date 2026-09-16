# banyan_grid/tasks/ruleset_factory.py
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np

from banyan_grid.environment.constants import (
    NUM_RULESET_ITEMS,
    RULE_TYPE_COMBINE,
    RULE_TYPE_TERNARY_COMBINE,
    Colors,
    TileType,
)

# ---------------------------------------------------------------------
# Utilities for encoding/decoding and deterministic, globally-consistent
# mapping from (unique item A, unique item B) -> (output item, output color).
# A "unique item" is (item_type, color), with color in [0..Colors.BLACK-1].
# ---------------------------------------------------------------------

_NUM_REAL_COLORS = int(Colors.BLACK)  # 12 (BLACK is sentinel, excluded)
_DOMAIN_SIZE = NUM_RULESET_ITEMS * _NUM_REAL_COLORS  # 10 * 12 = 120

# Consistent tile mapping for collectibles, index by item type (0..7)
_ITEM_TO_TILE = jnp.array(
    [
        TileType.KEY,  # ItemType.KEY
        TileType.BALL,  # ItemType.BALL
        TileType.MAP,  # ItemType.MAP
        TileType.TRIANGLE,  # ItemType.TRIANGLE
        TileType.STAR,  # ItemType.STAR
        TileType.HEX,  # ItemType.HEX
        TileType.SQUARE,  # ItemType.SQUARE
        TileType.PYRAMID,  # ItemType.PYRAMID
        TileType.DIAMOND,  # ItemType.DIAMOND
        TileType.CRESCENT,  # ItemType.CRESCENT
    ],
    dtype=jnp.int32,
)

def _enc_collect(tile_type, item_type, color) -> jnp.ndarray:
    # [1, tile, item, color, 0, 0]
    return jnp.array([1, tile_type, item_type, color, 0, 0], dtype=jnp.int32)


def _pack_colors(c1: int, c2: int, c_out: int) -> int:
    # 3×4-bit packing: c1 + (c2<<4) + (c_out<<8)
    return ((c_out & 0xF) << 8) | ((c2 & 0xF) << 4) | (c1 & 0xF)


def _pack_ternary_colors(c1: int, c2: int, c3: int, c_out: int) -> int:
    return (
        ((c_out & 0xF) << 12)
        | ((c3 & 0xF) << 8)
        | ((c2 & 0xF) << 4)
        | (c1 & 0xF)
    )


def _enc_combine(
    a, b, out, c1, c2, c_out, required_adjacent: bool = True
) -> jnp.ndarray:
    # [2, a, b, out, required_adjacent(0/1), packed_colors]
    return jnp.array(
        [
            RULE_TYPE_COMBINE,
            a,
            b,
            out,
            1 if required_adjacent else 0,
            _pack_colors(c1, c2, c_out),
        ],
        dtype=jnp.int32,
    )


def _enc_ternary_combine(
    a,
    b,
    c,
    out,
    c1,
    c2,
    c3,
    c_out,
    required_adjacent: bool = True,
) -> jnp.ndarray:
    meta = ((int(c) & 0xF) << 1) | (1 if required_adjacent else 0)
    return jnp.array(
        [
            RULE_TYPE_TERNARY_COMBINE,
            a,
            b,
            out,
            meta,
            _pack_ternary_colors(c1, c2, c3, c_out),
        ],
        dtype=jnp.int32,
    )


# -------- Unique-item <-> integer id helpers (for hashing) --------------------


def _uid(item_type, color):
    """Map (item_type, color) -> unique id in [0..NUM_RULESET_ITEMS*_NUM_REAL_COLORS-1]"""
    return jnp.asarray(item_type, dtype=jnp.int32) * _NUM_REAL_COLORS + jnp.asarray(
        color, dtype=jnp.int32
    )


def _uid_to_tuple(uid):
    u = jnp.asarray(uid, dtype=jnp.int32)
    item = (u // _NUM_REAL_COLORS).astype(jnp.int32)
    col = (u % _NUM_REAL_COLORS).astype(jnp.int32)
    return item, col


# -------- Deterministic word-size mixer --------------------------------------
#
# JAX commonly runs here with x64 disabled, so "uint64" arithmetic silently
# truncates to uint32. Make the mixer explicitly 32-bit so NumPy dataset
# generation and JAX-side decoding/runtime agree exactly.

_MIX_ADD = 0x7F4A7C15
_MIX_MUL1 = 0x1CE4E5B9
_MIX_MUL2 = 0x133111EB
_PAIR_MUL_A = 0x7F4A7C15
_PAIR_MUL_B = 0xB1CE6E93
_PAIR_MUL_C = 0x133111EB


def _splitmix_word(x: jax.Array) -> jax.Array:
    x = jnp.asarray(x, dtype=jnp.uint32) + jnp.uint32(_MIX_ADD)
    z = (x ^ (x >> jnp.uint32(30))) * jnp.uint32(_MIX_MUL1)
    z = (z ^ (z >> jnp.uint32(27))) * jnp.uint32(_MIX_MUL2)
    z = z ^ (z >> jnp.uint32(31))
    return z.astype(jnp.uint32)


def _splitmix_word_np(x: np.ndarray | np.uint32 | int) -> np.uint32:
    with np.errstate(over="ignore", invalid="ignore"):
        x = np.uint32(x) + np.uint32(_MIX_ADD)
        z = (x ^ (x >> np.uint32(30))) * np.uint32(_MIX_MUL1)
        z = (z ^ (z >> np.uint32(27))) * np.uint32(_MIX_MUL2)
        z = z ^ (z >> np.uint32(31))
    return np.uint32(z)


def _pair_to_output_uid_global_np(
    uid_a: int, uid_b: int, *, base_seed: int = 0, enforce_commutative: bool = True
) -> int:
    a = min(uid_a, uid_b) if enforce_commutative else uid_a
    b = max(uid_a, uid_b) if enforce_commutative else uid_b
    s = np.uint32(base_seed)
    with np.errstate(over="ignore", invalid="ignore"):
        mixed = (
            s
            ^ (np.uint32(a) * np.uint32(_PAIR_MUL_A))
            ^ (np.uint32(b) * np.uint32(_PAIR_MUL_B))
        )
    z = _splitmix_word_np(mixed)
    return int(z % np.uint32(_DOMAIN_SIZE))


def _pair_to_output_uid_global(
    uid_a: jax.Array,
    uid_b: jax.Array,
    *,
    base_seed: int = 0,
    enforce_commutative: bool = True,
) -> jax.Array:
    a = jnp.minimum(uid_a, uid_b) if enforce_commutative else uid_a
    b = jnp.maximum(uid_a, uid_b) if enforce_commutative else uid_b
    s = jnp.uint32(base_seed)
    mixed = (
        s
        ^ (jnp.asarray(a, dtype=jnp.uint32) * jnp.uint32(_PAIR_MUL_A))
        ^ (jnp.asarray(b, dtype=jnp.uint32) * jnp.uint32(_PAIR_MUL_B))
    )
    z = _splitmix_word(mixed)
    return (z % jnp.uint32(_DOMAIN_SIZE)).astype(jnp.int32)


def _triple_to_output_uid_global_np(
    uid_a: int,
    uid_b: int,
    uid_c: int,
    *,
    base_seed: int = 0,
    enforce_commutative: bool = True,
) -> int:
    vals = [int(uid_a), int(uid_b), int(uid_c)]
    if enforce_commutative:
        vals.sort()
    a, b, c = vals
    s = np.uint32(base_seed)
    with np.errstate(over="ignore", invalid="ignore"):
        mixed = (
            s
            ^ (np.uint32(a) * np.uint32(_PAIR_MUL_A))
            ^ (np.uint32(b) * np.uint32(_PAIR_MUL_B))
            ^ (np.uint32(c) * np.uint32(_PAIR_MUL_C))
        )
    z = _splitmix_word_np(mixed)
    return int(z % np.uint32(_DOMAIN_SIZE))


def _triple_to_output_uid_global(
    uid_a: jax.Array,
    uid_b: jax.Array,
    uid_c: jax.Array,
    *,
    base_seed: int = 0,
    enforce_commutative: bool = True,
) -> jax.Array:
    vals = jnp.stack(
        [
            jnp.asarray(uid_a, dtype=jnp.int32),
            jnp.asarray(uid_b, dtype=jnp.int32),
            jnp.asarray(uid_c, dtype=jnp.int32),
        ]
    )
    if enforce_commutative:
        vals = jnp.sort(vals, axis=0)
    a, b, c = vals[0], vals[1], vals[2]
    s = jnp.uint32(base_seed)
    mixed = (
        s
        ^ (jnp.asarray(a, dtype=jnp.uint32) * jnp.uint32(_PAIR_MUL_A))
        ^ (jnp.asarray(b, dtype=jnp.uint32) * jnp.uint32(_PAIR_MUL_B))
        ^ (jnp.asarray(c, dtype=jnp.uint32) * jnp.uint32(_PAIR_MUL_C))
    )
    z = _splitmix_word(mixed)
    return (z % jnp.uint32(_DOMAIN_SIZE)).astype(jnp.int32)


def _unary_to_output_uid_global_np(uid_in: int, *, base_seed: int = 0) -> int:
    unary_salt_uid = _DOMAIN_SIZE + 97
    return _pair_to_output_uid_global_np(
        int(uid_in),
        int(unary_salt_uid),
        base_seed=base_seed,
        enforce_commutative=False,
    )


def _unary_to_output_uid_global(uid_in: jax.Array, *, base_seed: int = 0) -> jax.Array:
    unary_salt_uid = jnp.asarray(_DOMAIN_SIZE + 97, dtype=jnp.int32)
    return _pair_to_output_uid_global(
        uid_in,
        unary_salt_uid,
        base_seed=base_seed,
        enforce_commutative=False,
    )


def _uid_to_item_color_np(uid: int) -> Tuple[int, int]:
    item = uid // int(Colors.BLACK)
    col = uid % int(Colors.BLACK)
    return item, col


# Deterministic pairing search: no idempotence; unique outputs across the ruleset
def _gen_pairings_for_level(nodes: list[int], used_out_uids: set[int], base_seed: int):
    n = len(nodes)
    idxs = list(range(n))

    def rec(rem, pairs, outs):
        if not rem:
            yield pairs, outs
            return
        i = rem[0]
        a = nodes[i]
        for k in range(1, len(rem)):
            j = rem[k]
            b = nodes[j]
            out = _pair_to_output_uid_global_np(a, b, base_seed=base_seed)
            if out == a or out == b:
                continue
            if out in used_out_uids or out in outs:
                continue
            new_rem = rem[1:k] + rem[k + 1 :]
            yield from rec(new_rem, pairs + [(a, b, out)], outs + [out])

    yield from rec(idxs, [], [])


def _build_tree_global(
    leaves_uids: list[int],
    base_seed: int,
    required_first_level_pairs: Optional[List[Tuple[int, int]]] = None,
    required_pairs_by_level: Optional[List[Optional[List[Tuple[int, int]]]]] = None,
):
    """
    Build a binary tree of combine rules from leaves.

    Args:
        leaves_uids: List of unique item IDs for leaves
        base_seed: Seed for deterministic hash function
        required_first_level_pairs: If provided, use exactly these (uid_a, uid_b) pairs
            for the first level of combining. This ensures subtree-closure consistency
            by preserving the exact pairings practiced at lower depths.
            Collisions are NOT checked for these pairs (caller must ensure validity).
        required_pairs_by_level: Optional list of required pairs per combine level,
            starting at the leaf-combine level. If provided, it overrides
            required_first_level_pairs for level 0.

    Returns:
        List of (uid_a, uid_b, uid_out) triples, or None if no valid tree found.
    """
    # Track both outputs AND original leaves to prevent collisions
    leaf_set = set(leaves_uids)
    used_outs: set[int] = set()
    rules: list[Tuple[int, int, int]] = []

    pairs_by_level = required_pairs_by_level
    if pairs_by_level is None and required_first_level_pairs is not None:
        pairs_by_level = [required_first_level_pairs]

    def dfs(level_nodes: list[int], level: int = 0) -> bool:
        nonlocal used_outs, rules
        if len(level_nodes) == 1:
            return True

        # If required pairs are provided for this level, use them directly
        required_pairs = None
        if pairs_by_level is not None and level < len(pairs_by_level):
            required_pairs = pairs_by_level[level]
        if required_pairs is not None:
            # Compute outputs for required pairs with collision validation
            # (caller should pre-validate, but we double-check for safety)
            pairs_with_outs = []
            outs = []
            level_outs_set = set()

            for uid_a, uid_b in required_pairs:
                out = _pair_to_output_uid_global_np(uid_a, uid_b, base_seed=base_seed)

                # Validate: output != inputs (idempotence check)
                if out == uid_a or out == uid_b:
                    return False

                # Validate: output not in leaves (prevents DAG)
                if out in leaf_set:
                    return False

                # Validate: output not already used (collision with prior levels)
                if out in used_outs:
                    return False

                # Validate: no duplicate outputs within this level
                if out in level_outs_set:
                    return False

                pairs_with_outs.append((uid_a, uid_b, out))
                outs.append(out)
                level_outs_set.add(out)

            rules.extend(pairs_with_outs)
            used_outs.update(outs)
            return dfs(outs, level=level + 1)

        # Standard exploration for other levels
        for pairs, outs in _gen_pairings_for_level(
            level_nodes, used_outs | leaf_set, base_seed
        ):
            old_rules_len = len(rules)
            old_used = set(used_outs)
            rules.extend(pairs)
            used_outs.update(outs)
            if dfs(outs, level=level + 1):
                return True
            rules = rules[:old_rules_len]
            used_outs = old_used
        return False

    ok = dfs(leaves_uids, level=0)
    return rules if ok else None


# ---------------------------------------------------------------------
# Tree-builder
# ---------------------------------------------------------------------


def _leaf_count(depth: int) -> int:
    # depth=1 -> 1 leaf (collect only)
    # depth>=2 -> 2^(depth-1) leaves
    return 1 if depth == 1 else (1 << (depth - 1))


def _combine_count(depth: int) -> int:
    # For depth>=2, binary tree internal nodes = 2^(depth-1)-1; else 0
    return 0 if depth == 1 else ((1 << (depth - 1)) - 1)


def _rule_count(depth: int) -> int:
    """Number of rules for a ruleset of given depth."""
    return 1 if depth == 1 else ((1 << (depth - 1)) - 1)


def compute_depth_task_counts(
    max_depth: int, n_highest: int, pool_size: int
) -> dict[int, int]:
    """Compute number of tasks at each depth for dataset generation."""
    structure: dict[int, int] = {1: pool_size}
    if max_depth >= 2:
        structure[2] = pool_size * (pool_size - 1) // 2
    for depth in range(max_depth, 2, -1):
        if depth == max_depth:
            structure[depth] = n_highest
        else:
            structure[depth] = 2 * structure[depth + 1]
    return structure


def build_ruleset(
    key: chex.PRNGKey,
    *,
    depth: int = 3,
    base_seed: int = 0,
    required_adjacent: bool = True,
    leaf_uid_pool: Optional[Union[jnp.ndarray, np.ndarray, List[int]]] = None,
    fixed_leaves: Optional[Sequence[int]] = None,
    required_first_level_pairs: Optional[List[Tuple[int, int]]] = None,
    required_pairs_by_level: Optional[List[Optional[List[Tuple[int, int]]]]] = None,
) -> jnp.ndarray:
    """
    Build ONE ruleset (shape = (R, 6), int32) with global pair-consistency.
    - depth in [1, 6]
    - If `fixed_leaves` (len = #leaves) is provided, use exactly those leaves in that order.
      This makes it possible to embed known depth-2 / depth-3 subtrees by forcing pairings.
    - If `required_first_level_pairs` is provided (list of (uid_a, uid_b) tuples),
      these exact pairings are used for the first level of combining. This ensures
      subtree-closure consistency by reusing the exact intermediate combines practiced
      at lower depths. When provided, collision checks are skipped for these pairs
      (caller must ensure validity).
    - If `required_pairs_by_level` is provided, it overrides `required_first_level_pairs`
      and uses the given pairs at each combine level (level 0 = leaf pairs).
    - If `leaf_uid_pool` is provided, leaves are sampled only from this pool of
      unique item ids (uid = item * _NUM_REAL_COLORS + color, with color in [0..BLACK-1]).
      This enables cross-depth coupling of item vocabularies.
    - base_seed fixes the global mapping for the whole benchmark.
    - required_adjacent: whether combine rules require adjacency.
    Ordering:
      * For depth>=2, the FIRST combination rule is the FINAL rule (so your
        current AgentHasItemFromRulesetGoal, which reads the first combine,
        targets the true final).
      * No movement/collect rules are stored for depth>=2.
      * Depth=1 uses a single leaf descriptor (collect-encoding).
    """
    depth = int(depth)
    if depth < 1 or depth > 6:
        raise ValueError("depth must be in [1, 6].")

    L = _leaf_count(depth)
    R = _rule_count(depth)
    # 1) Exact leaves (highest precedence) -------------------------------------
    chosen_rules: Optional[List[Tuple[int, int, int]]] = None
    chosen_leaves: Optional[List[int]] = None

    if fixed_leaves is not None:
        fl = np.asarray(fixed_leaves, dtype=np.int32)
        if fl.shape[0] == L:
            # Ensure domain bounds
            ok = np.all((fl >= 0) & (fl < _DOMAIN_SIZE))
            if ok:
                leaves_uids = fl.tolist()
                if depth >= 2:
                    triples = _build_tree_global(
                        leaves_uids,
                        base_seed,
                        required_first_level_pairs=required_first_level_pairs,
                        required_pairs_by_level=required_pairs_by_level,
                    )
                    if triples is not None:
                        chosen_rules = triples
                        chosen_leaves = leaves_uids
                else:
                    chosen_rules = []
                    chosen_leaves = leaves_uids
        # If invalid or fails, we will fall back to pool/global below.
    # Choose leaves either from a provided pool (for coupling) or from the full domain.
    if chosen_leaves is None and leaf_uid_pool is not None:
        pool = np.asarray(leaf_uid_pool, dtype=np.int32)
        # ensure valid, unique uids within domain
        M = _DOMAIN_SIZE
        pool = np.unique(pool[(pool >= 0) & (pool < M)])
        if pool.shape[0] >= L:
            # Try a few permutations of the pool to avoid collisions in the global map.
            max_tries = 256
            k_try = key
            for _ in range(max_tries):
                k_try, k_sample = jax.random.split(k_try)
                idx = np.array(
                    jax.random.permutation(k_sample, pool.shape[0]), dtype=np.int32
                )
                leaves_uids = pool[idx[:L]].tolist()
                if depth >= 2:
                    triples = _build_tree_global(leaves_uids, base_seed)
                    if triples is not None:
                        chosen_rules = triples
                        chosen_leaves = leaves_uids
                        break
                else:
                    # depth==1 needs only a single leaf; always OK
                    chosen_rules = []
                    chosen_leaves = leaves_uids
                    break
        # If pool too small or failed, fall back to global domain sampling below.

    if chosen_leaves is None:
        # Deterministic permutation over all 80 leaf UIDs
        key, k_perm = jax.random.split(key)
        M = NUM_RULESET_ITEMS * _NUM_REAL_COLORS
        perm = np.array(jax.random.permutation(k_perm, M), dtype=np.int32)
        # Try sliding windows until we can build a valid tree under the global map
        for off in range(0, M - L + 1):
            leaves_uids = perm[off : off + L].tolist()
            triples = _build_tree_global(leaves_uids, base_seed) if depth >= 2 else []
            if (depth == 1) or (triples is not None):
                chosen_rules = triples  # [(uid_a, uid_b, uid_out)] bottom-up
                chosen_leaves = leaves_uids
                break
        if chosen_leaves is None:
            raise RuntimeError(
                "Could not build a collision-free tree with global mapping."
            )
    # Encode combine rules; FINAL combine goes first
    combine_rules: List[jnp.ndarray] = []
    if depth >= 2 and chosen_rules:
        comb_triples = [chosen_rules[-1]] + chosen_rules[:-1]
        for ua, ub, uo in comb_triples:
            ia, ca = _uid_to_item_color_np(ua)
            ib, cb = _uid_to_item_color_np(ub)
            io, co = _uid_to_item_color_np(uo)
            combine_rules.append(
                _enc_combine(ia, ib, io, ca, cb, co, required_adjacent)
            )
    encs: List[jnp.ndarray] = []
    if depth >= 2:
        encs.extend(combine_rules)
    else:
        # Depth-1: single leaf descriptor
        uid = chosen_leaves[0]
        item, color = _uid_to_item_color_np(uid)
        tile_tt = int(_ITEM_TO_TILE[item])
        encs.append(_enc_collect(tile_tt, item, color))
    ruleset = jnp.stack(encs, axis=0).astype(jnp.int32)
    assert ruleset.shape == (R, 6), f"ruleset shape {ruleset.shape} != {(R, 6)}"
    return ruleset


# ---------------------------------------------------------------------
# Benchmark builders
# ---------------------------------------------------------------------


def build_ruleset_batch(
    key: chex.PRNGKey,
    *,
    n: int,
    depth: int = 3,
    base_seed: int = 0,
    required_adjacent: bool = True,
    leaf_uid_pool: Optional[Union[jnp.ndarray, np.ndarray, List[int]]] = None,
    fixed_leaves_list: Optional[List[Optional[Sequence[int]]]] = None,
    required_first_level_pairs_list: Optional[
        List[Optional[List[Tuple[int, int]]]]
    ] = None,
    required_pairs_by_level_list: Optional[
        List[Optional[List[Optional[List[Tuple[int, int]]]]]]
    ] = None,
) -> jnp.ndarray:
    depth = int(depth)
    R = _rule_count(depth)
    keys = list(jax.random.split(key, n))
    out = []
    for i, k in enumerate(keys):
        fixed = None
        if fixed_leaves_list is not None:
            if i < len(fixed_leaves_list):
                fixed = fixed_leaves_list[i]
        required_pairs = None
        if required_first_level_pairs_list is not None:
            if i < len(required_first_level_pairs_list):
                required_pairs = required_first_level_pairs_list[i]
        required_pairs_by_level = None
        if required_pairs_by_level_list is not None:
            if i < len(required_pairs_by_level_list):
                required_pairs_by_level = required_pairs_by_level_list[i]
        rs = build_ruleset(
            k,
            depth=depth,
            base_seed=base_seed,
            required_adjacent=required_adjacent,
            leaf_uid_pool=leaf_uid_pool,
            fixed_leaves=fixed,
            required_first_level_pairs=required_pairs,
            required_pairs_by_level=required_pairs_by_level,
        )
        out.append(np.array(rs, dtype=np.int32))
    return jnp.asarray(out, dtype=jnp.int32).reshape(n, R, 6)


def iter_rulesets(
    seed: int,
    *,
    n: int,
    depth: int = 3,
    base_seed: int = 0,
    required_adjacent: bool = True,
    leaf_uid_pool: Optional[Union[jnp.ndarray, np.ndarray, List[int]]] = None,
    fixed_leaves_list: Optional[List[Optional[Sequence[int]]]] = None,
    required_first_level_pairs_list: Optional[
        List[Optional[List[Tuple[int, int]]]]
    ] = None,
    required_pairs_by_level_list: Optional[
        List[Optional[List[Optional[List[Tuple[int, int]]]]]]
    ] = None,
) -> Iterator[jnp.ndarray]:
    """
    Streaming generator: yields one (R, 6) ruleset at a time; use this when n is huge.
    """
    depth = int(depth)
    key = jax.random.PRNGKey(seed)
    for i in range(n):
        key, sub = jax.random.split(key)
        fixed = None
        if fixed_leaves_list is not None and i < len(fixed_leaves_list):
            fixed = fixed_leaves_list[i]
        required_pairs = None
        if required_first_level_pairs_list is not None and i < len(
            required_first_level_pairs_list
        ):
            required_pairs = required_first_level_pairs_list[i]
        required_pairs_by_level = None
        if required_pairs_by_level_list is not None and i < len(
            required_pairs_by_level_list
        ):
            required_pairs_by_level = required_pairs_by_level_list[i]
        yield build_ruleset(
            sub,
            depth=depth,
            base_seed=base_seed,
            required_adjacent=required_adjacent,
            leaf_uid_pool=leaf_uid_pool,
            fixed_leaves=fixed,
            required_first_level_pairs=required_pairs,
            required_pairs_by_level=required_pairs_by_level,
        )


# Optional helper if you ever need the "goal item" (type,color) for a ruleset:
def goal_unique_item_from_ruleset(ruleset: jnp.ndarray) -> Tuple[int, int]:
    """
    If there is at least one producer rule (combine/transform), return the sink
    producer output (goal item/color). If none (depth=1), return collect item.
    """
    rule_type = ruleset[:, 0]
    is_collect = rule_type == 1
    is_binary_combine = rule_type == RULE_TYPE_COMBINE
    is_ternary_combine = rule_type == RULE_TYPE_TERNARY_COMBINE
    is_comb = is_binary_combine | is_ternary_combine
    is_transform = rule_type == 4
    is_prod = is_comb | is_transform
    any_prod = jnp.any(is_prod)

    def on_producer(rs):
        packed = rs[:, 5]
        c1 = packed & 0xF
        c2 = (packed >> 4) & 0xF
        c3 = jnp.where(is_ternary_combine, (packed >> 8) & 0xF, -1)
        cout = jnp.where(
            is_ternary_combine,
            (packed >> 12) & 0xF,
            (packed >> 8) & 0xF,
        )
        out_item = jnp.where(is_prod, rs[:, 3], -1)
        out_color = jnp.where(is_prod, cout, -1)
        in1_item = jnp.where(is_prod, rs[:, 1], -2)
        in1_color = jnp.where(is_prod, c1, -2)
        in2_item = jnp.where(is_comb, rs[:, 2], -2)
        in2_color = jnp.where(is_comb, c2, -2)
        in3_item = jnp.where(is_ternary_combine, rs[:, 4] >> 1, -2)
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
        used_as_input = (
            (uses_out_as_in1 | uses_out_as_in2 | uses_out_as_in3)
            & is_prod[:, None]
            & is_prod[None, :]
            & (~jnp.eye(rs.shape[0], dtype=bool))
        )
        out_is_used = jnp.any(used_as_input, axis=0)
        sink_mask = is_prod & (~out_is_used)
        have_sink = jnp.any(sink_mask)
        idx = jax.lax.select(
            have_sink,
            jnp.argmax(sink_mask.astype(jnp.int32)),
            jnp.argmax(is_prod.astype(jnp.int32)),
        )
        return int(out_item[idx]), int(out_color[idx])

    def on_collect(rs):
        idx = jnp.argmax(is_collect.astype(jnp.int32))
        return int(rs[idx, 2]), int(rs[idx, 3])

    return jax.lax.cond(any_prod, on_producer, on_collect, operand=ruleset)


# Backwards-compatible alias for the pre-release function name.
compute_curriculum_structure = compute_depth_task_counts
