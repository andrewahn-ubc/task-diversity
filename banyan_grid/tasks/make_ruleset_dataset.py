# tools/make_ruleset_dataset.py
"""
RULESET DATASET GENERATION

Default mode enforces strict subtree closure: EVERY subtree at depth D appears
as a standalone task at depth D. This is ALWAYS enforced, not probabilistic.

Alternative mixed modes generate topology-diverse trees. `mixed_u1_b2` uses
unary transforms + binary combines. `mixed_u1_b2_t3` additionally allows
ternary combines, increasing topology diversity.

Structure:
- Depth 1: Pick up single items (one task per unique item in pool)
- Depth 2: Combine 2 depth-1 items (each depth-2 task uses items from depth-1)
- Depth 3: ALWAYS embeds 2 complete depth-2 subtrees (4 leaves from depth-2)
- Depth 4: ALWAYS embeds 2 complete depth-3 subtrees (8 leaves from depth-3)

Number of tasks per depth:
- User specifies N tasks for the HIGHEST depth
- Lower depths are computed automatically:
  - depth D: N tasks (user-specified)
  - depth D-1: 2*N tasks (to provide subtrees for depth D)
  - depth D-2: 4*N tasks (to provide subtrees for depth D-1)
  - depth 1: pool_size tasks (one per unique item)

Usage:
  python make_ruleset_dataset.py \\
    --n 1000 --max-depth 3 --pool-size 16 \\
    --out-dir ./rulesets
"""

import argparse
import bz2
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from math import comb

import jax
import jax.numpy as jnp
import numpy as np

from banyan_grid.environment.constants import (
    NUM_COLORS,
    NUM_ITEMS,
    NUM_RULESET_ITEMS,
    RULE_TYPE_COMBINE,
    RULE_TYPE_DISTRACTOR_COMBINE,
    RULE_TYPE_TERNARY_COMBINE,
    RULE_TYPE_TRANSFORM,
    Colors,
)
from banyan_grid.tasks.ruleset_codec import (
    pack_rules_uint32_np,
    unpack_rules_uint32,
)
from banyan_grid.tasks.ruleset_factory import (
    _pair_to_output_uid_global_np,
    _rule_count,
    _triple_to_output_uid_global_np,
    _unary_to_output_uid_global_np,
    build_ruleset_batch,
    compute_depth_task_counts,
)

_NUM_REAL_COLORS = int(Colors.BLACK)
_DOMAIN_SIZE = NUM_RULESET_ITEMS * _NUM_REAL_COLORS
MAX_TRIES = int(os.environ.get("RULESET_GEN_MAX_TRIES", "200"))
SEED_BLOCK_SIZE = int(os.environ.get("RULESET_GEN_SEED_BLOCK_SIZE", "8192"))
MIXED_DEPTH_WORKERS = int(os.environ.get("RULESET_GEN_MIXED_DEPTH_WORKERS", "1"))
# Cap strict unique-first probing to avoid heavy tail slowdowns.
STRICT_UNIQUE_TRIES = int(os.environ.get("RULESET_GEN_STRICT_UNIQUE_TRIES", "64"))
MIXED_UNARY_PROB = max(0.0, min(1.0, float(os.environ.get("RULESET_GEN_MIXED_UNARY_PROB", "0.42"))))
_mixed_unary_prob_depth5_raw = os.environ.get("RULESET_GEN_MIXED_UNARY_PROB_DEPTH5")
MIXED_UNARY_PROB_DEPTH5 = (
    None
    if _mixed_unary_prob_depth5_raw is None
    else max(0.0, min(1.0, float(_mixed_unary_prob_depth5_raw)))
)
MIXED_D5_HIGH_LEAF_MIN = max(1, int(os.environ.get("RULESET_GEN_MIXED_D5_HIGH_LEAF_MIN", "10")))
MIXED_D5_LOW_LEAF_REJECT_PROB = max(
    0.0,
    min(1.0, float(os.environ.get("RULESET_GEN_MIXED_D5_LOW_LEAF_REJECT_PROB", "0.0"))),
)
TREE_TOPOLOGY_BALANCED = "balanced"
TREE_TOPOLOGY_ASYM_D2_D3_6L = "asym_d2_d3_6l"
TREE_TOPOLOGY_MIXED_U1_B2 = "mixed_u1_b2"
TREE_TOPOLOGY_MIXED_U1_B2_T3 = "mixed_u1_b2_t3"
MIXED_TERNARY_PROB = max(
    0.0, min(1.0, float(os.environ.get("RULESET_GEN_MIXED_TERNARY_PROB", "0.18")))
)


def _is_mixed_topology_mode(tree_topology: str) -> bool:
    return tree_topology in {
        TREE_TOPOLOGY_MIXED_U1_B2,
        TREE_TOPOLOGY_MIXED_U1_B2_T3,
    }


def _mixed_mode_supports_ternary(tree_topology: str) -> bool:
    return tree_topology == TREE_TOPOLOGY_MIXED_U1_B2_T3


def _R(d):
    """Number of rules for depth d - delegates to canonical implementation"""
    return _rule_count(d)


def _rules_per_depth(depth: int, tree_topology: str) -> int:
    if tree_topology == TREE_TOPOLOGY_ASYM_D2_D3_6L and int(depth) == 3:
        # 6 leaves => 5 combine rules.
        return 5
    return _R(int(depth))


def _mixed_unary_prob_for_depth(depth: int) -> float:
    """Unary-root probability for mixed topology sampling."""
    if int(depth) == 5 and MIXED_UNARY_PROB_DEPTH5 is not None:
        return float(MIXED_UNARY_PROB_DEPTH5)
    return float(MIXED_UNARY_PROB)


def _fmt_density_tag(value: float) -> str:
    clamped = max(0.0, min(1.0, float(value)))
    text = f"{clamped:.6f}".rstrip("0").rstrip(".")
    if text == "":
        text = "0"
    return text.replace(".", "p")


def _distractor_suffix(value: float) -> str:
    if float(value) <= 0.0:
        return ""
    return f"_dd{_fmt_density_tag(value)}"


def _uid_to_item_color(uid: int) -> tuple[int, int]:
    return uid // _NUM_REAL_COLORS, uid % _NUM_REAL_COLORS


def _item_color_to_uid(item: int, color: int) -> int:
    return int(item) * _NUM_REAL_COLORS + int(color)


def _pack_colors(c1: int, c2: int, c_out: int) -> int:
    return ((c_out & 0xF) << 8) | ((c2 & 0xF) << 4) | (c1 & 0xF)


def _pack_ternary_colors(c1: int, c2: int, c3: int, c_out: int) -> int:
    return ((c_out & 0xF) << 12) | ((c3 & 0xF) << 8) | ((c2 & 0xF) << 4) | (c1 & 0xF)


def _canonical_task_signature(
    leaves: list[int], required_pairs_by_level: list[list[tuple[int, int]]]
) -> tuple[tuple[int, ...], tuple[tuple[tuple[int, int], ...], ...]]:
    """Canonical signature for subtree identity (order-invariant within levels)."""
    leaf_sig = tuple(sorted(int(x) for x in leaves))
    levels = []
    for level_pairs in required_pairs_by_level:
        canon_pairs = tuple(
            sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in level_pairs)
        )
        levels.append(canon_pairs)
    return leaf_sig, tuple(levels)


def _encode_combine_row(uid_a: int, uid_b: int, uid_out: int, rule_type: int) -> np.ndarray:
    in1_item, in1_color = _uid_to_item_color(int(uid_a))
    in2_item, in2_color = _uid_to_item_color(int(uid_b))
    out_item, out_color = _uid_to_item_color(int(uid_out))
    return np.array(
        [
            int(rule_type),
            int(in1_item),
            int(in2_item),
            int(out_item),
            1,  # required adjacent
            _pack_colors(in1_color, in2_color, out_color),
        ],
        dtype=np.int32,
    )


def _decode_combine_row_uids(row: np.ndarray) -> tuple[int, int, int]:
    packed = int(row[5])
    in1_uid = _item_color_to_uid(int(row[1]), packed & 0xF)
    in2_uid = _item_color_to_uid(int(row[2]), (packed >> 4) & 0xF)
    out_uid = _item_color_to_uid(int(row[3]), (packed >> 8) & 0xF)
    return in1_uid, in2_uid, out_uid


def _encode_ternary_combine_row(uid_a: int, uid_b: int, uid_c: int, uid_out: int) -> np.ndarray:
    ordered = sorted([int(uid_a), int(uid_b), int(uid_c)])
    in1_item, in1_color = _uid_to_item_color(ordered[0])
    in2_item, in2_color = _uid_to_item_color(ordered[1])
    in3_item, in3_color = _uid_to_item_color(ordered[2])
    out_item, out_color = _uid_to_item_color(int(uid_out))
    meta = (int(in3_item) << 1) | 1
    return np.array(
        [
            int(RULE_TYPE_TERNARY_COMBINE),
            int(in1_item),
            int(in2_item),
            int(out_item),
            int(meta),
            _pack_ternary_colors(in1_color, in2_color, in3_color, out_color),
        ],
        dtype=np.int32,
    )


def _choose_asym_d2_d3_pairs(
    d2_pairs_lib: list[tuple[int, int]],
    rng: np.random.Generator,
    pair_output_fn,
    max_tries: int = 256,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    """Pick 3 leaf pairs for a 6-leaf asymmetric depth-3 topology.

    Topology:
      p_shallow -> o_shallow
      p_deep_a  -> o_deep_a
      p_deep_b  -> o_deep_b
      o_deep_a + o_deep_b -> o_deep
      o_shallow + o_deep -> o_root
    """
    if len(d2_pairs_lib) < 3:
        return None

    for _ in range(max_tries):
        order = rng.permutation(len(d2_pairs_lib))
        selected: list[tuple[int, int]] = []
        used_inputs: set[int] = set()

        for idx in order:
            a, b = d2_pairs_lib[int(idx)]
            if a in used_inputs or b in used_inputs:
                continue
            selected.append((int(a), int(b)))
            used_inputs.add(int(a))
            used_inputs.add(int(b))
            if len(selected) == 3:
                break

        if len(selected) != 3:
            continue

        p_shallow, p_deep_a, p_deep_b = selected
        o_shallow = int(pair_output_fn(*p_shallow))
        o_deep_a = int(pair_output_fn(*p_deep_a))
        o_deep_b = int(pair_output_fn(*p_deep_b))

        leaves = {
            int(p_shallow[0]),
            int(p_shallow[1]),
            int(p_deep_a[0]),
            int(p_deep_a[1]),
            int(p_deep_b[0]),
            int(p_deep_b[1]),
        }
        level0_outs = {o_shallow, o_deep_a, o_deep_b}

        # Prevent DAG collisions at level 0.
        if len(level0_outs) != 3:
            continue
        if any(o in leaves for o in level0_outs):
            continue

        o_deep = int(pair_output_fn(o_deep_a, o_deep_b))
        if o_deep in leaves or o_deep in level0_outs:
            continue

        o_root = int(pair_output_fn(o_shallow, o_deep))
        if o_root in leaves or o_root in level0_outs or o_root == o_deep:
            continue

        return p_shallow, p_deep_a, p_deep_b

    return None


def _build_asym_d2_d3_ruleset_from_pairs(
    p_shallow: tuple[int, int],
    p_deep_a: tuple[int, int],
    p_deep_b: tuple[int, int],
    pair_output_fn,
) -> np.ndarray:
    """Build 5-row ruleset for the 6-leaf asymmetric depth-3 topology."""
    o_shallow = int(pair_output_fn(*p_shallow))
    o_deep_a = int(pair_output_fn(*p_deep_a))
    o_deep_b = int(pair_output_fn(*p_deep_b))
    o_deep = int(pair_output_fn(o_deep_a, o_deep_b))
    o_root = int(pair_output_fn(o_shallow, o_deep))

    rows = [
        _encode_combine_row(o_shallow, o_deep, o_root, RULE_TYPE_COMBINE),
        _encode_combine_row(p_shallow[0], p_shallow[1], o_shallow, RULE_TYPE_COMBINE),
        _encode_combine_row(o_deep_a, o_deep_b, o_deep, RULE_TYPE_COMBINE),
        _encode_combine_row(p_deep_a[0], p_deep_a[1], o_deep_a, RULE_TYPE_COMBINE),
        _encode_combine_row(p_deep_b[0], p_deep_b[1], o_deep_b, RULE_TYPE_COMBINE),
    ]
    return np.asarray(rows, dtype=np.int32)


def _verify_single_asym_d2_d3_ruleset(rows: np.ndarray) -> None:
    """Raise ValueError if rows are not a valid 6-leaf asym depth-3 tree."""
    combine_rows = rows[rows[:, 0] == RULE_TYPE_COMBINE]
    if combine_rows.shape[0] != 5:
        raise ValueError(f"Expected 5 combine rows, got {combine_rows.shape[0]}")

    children: dict[int, tuple[int, int]] = {}
    outputs = []
    input_uids = []
    for row in combine_rows:
        in1_uid, in2_uid, out_uid = _decode_combine_row_uids(row)
        if out_uid in children:
            raise ValueError("Duplicate output uid in asym ruleset")
        children[out_uid] = (in1_uid, in2_uid)
        outputs.append(out_uid)
        input_uids.extend([in1_uid, in2_uid])

    outputs_set = set(outputs)
    inputs_set = set(input_uids)
    leaves = inputs_set - outputs_set
    roots = outputs_set - inputs_set

    if len(leaves) != 6:
        raise ValueError(f"Expected 6 leaves, got {len(leaves)}")
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, got {len(roots)}")

    root_uid = next(iter(roots))
    first_root = _decode_combine_row_uids(combine_rows[0])[2]
    if first_root != root_uid:
        raise ValueError("Final combine is not first row (goal semantics broken)")

    leaf_depths = []
    stack: set[int] = set()

    def _walk(uid: int, dist: int) -> None:
        if uid in stack:
            raise ValueError("Cycle detected in asym ruleset")
        if uid not in children:
            leaf_depths.append(dist)
            return
        stack.add(uid)
        c1, c2 = children[uid]
        _walk(c1, dist + 1)
        _walk(c2, dist + 1)
        stack.remove(uid)

    _walk(root_uid, 0)
    if sorted(leaf_depths) != [2, 2, 3, 3, 3, 3]:
        raise ValueError(f"Unexpected leaf depth profile: {sorted(leaf_depths)}")


def _sample_linked_balanced_depth123(
    *,
    n_tasks: int,
    base_pool: np.ndarray,
    base_seed: int,
    key_root: jax.Array,
) -> tuple[
    list[int],
    list[tuple[int, int]],
    list[list[int]],
    list[list[list[tuple[int, int]]]],
]:
    """Sample linked depth-1/2/3 tasks for balanced depth-3 curricula.

    For each sampled depth-3 tree, choose one depth-2 subtree and one leaf token:
      depth-3 task i -> depth-2 subtree i -> depth-1 leaf i
    """
    if int(n_tasks) <= 0:
        return [], [], [], []
    if int(base_pool.shape[0]) < 4:
        raise ValueError("linked depth sampling requires at least 4 pool items for depth-3 trees")
    max_unique_leafsets = int(comb(int(base_pool.shape[0]), 4))
    if int(n_tasks) > max_unique_leafsets:
        raise ValueError(
            "linked depth sampling requires unique depth-3 leafsets, "
            f"but requested n_tasks={int(n_tasks)} exceeds C(pool_size,4)={max_unique_leafsets}."
        )

    depth1_leaves: list[int] = []
    depth2_pairs: list[tuple[int, int]] = []
    depth3_leaves: list[list[int]] = []
    depth3_pairs_by_level: list[list[list[tuple[int, int]]]] = []
    seen_leafsets: set[tuple[int, int, int, int]] = set()

    patterns = (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    )
    key_stream = jax.random.fold_in(key_root, 515151)

    def _pair_out(uid_a: int, uid_b: int) -> int:
        a, b = (int(uid_a), int(uid_b)) if int(uid_a) <= int(uid_b) else (int(uid_b), int(uid_a))
        return int(_pair_to_output_uid_global_np(a, b, base_seed=base_seed))

    for i in range(int(n_tasks)):
        k_i = jax.random.fold_in(key_stream, int(i))
        accepted = False

        for attempt in range(MAX_TRIES):
            k_try = jax.random.fold_in(k_i, int(attempt))
            seed_i = int(jax.random.randint(k_try, (), 0, 2**31 - 1))
            rng_i = np.random.default_rng(seed_i)

            leaves_arr = rng_i.choice(base_pool, size=4, replace=False)
            leaves = [int(u) for u in np.asarray(leaves_arr, dtype=np.int32).tolist()]
            rng_i.shuffle(leaves)
            leafset_key = tuple(sorted(int(u) for u in leaves))
            if leafset_key in seen_leafsets:
                continue

            pa, pb = patterns[int(rng_i.integers(0, len(patterns)))]
            pair_a = tuple(sorted((int(leaves[pa[0]]), int(leaves[pa[1]]))))  # type: ignore[arg-type]
            pair_b = tuple(sorted((int(leaves[pb[0]]), int(leaves[pb[1]]))))  # type: ignore[arg-type]

            # Optional order swap for level-0 pair list.
            if bool(rng_i.integers(0, 2)):
                pair_a, pair_b = pair_b, pair_a

            out_a = _pair_out(pair_a[0], pair_a[1])
            out_b = _pair_out(pair_b[0], pair_b[1])
            leaves_set = set(leaves)
            if out_a in leaves_set or out_b in leaves_set or out_a == out_b:
                continue

            root_out = _pair_out(out_a, out_b)
            if root_out in leaves_set or root_out in {out_a, out_b}:
                continue

            chosen_pair = pair_a if bool(rng_i.integers(0, 2)) else pair_b
            chosen_leaf = int(chosen_pair[int(rng_i.integers(0, 2))])

            seen_leafsets.add(leafset_key)
            depth1_leaves.append(chosen_leaf)
            depth2_pairs.append((int(chosen_pair[0]), int(chosen_pair[1])))
            depth3_leaves.append([int(u) for u in leaves])
            depth3_pairs_by_level.append([[pair_a, pair_b]])
            accepted = True
            break

        if not accepted:
            raise ValueError(
                f"Failed to sample linked depth-3 tree {i} after {MAX_TRIES} tries. "
                "Try increasing pool_size."
            )

    return depth1_leaves, depth2_pairs, depth3_leaves, depth3_pairs_by_level


def _encode_transform_row(uid_in: int, uid_out: int) -> np.ndarray:
    """Encode unary transform row: in -> out via TOGGLE."""
    in_item, in_color = _uid_to_item_color(int(uid_in))
    out_item, out_color = _uid_to_item_color(int(uid_out))
    return np.array(
        [
            int(RULE_TYPE_TRANSFORM),
            int(in_item),
            0,  # unary op has no second input
            int(out_item),
            1,  # required adjacent
            _pack_colors(in_color, 0, out_color),
        ],
        dtype=np.int32,
    )


def _unary_to_output_uid_global_np(uid_in: int, *, base_seed: int = 0) -> int:
    """Deterministic global mapping for unary transform signatures."""
    unary_salt_uid = _DOMAIN_SIZE + 97
    return _pair_to_output_uid_global_np(
        int(uid_in),
        int(unary_salt_uid),
        base_seed=base_seed,
        enforce_commutative=False,
    )


def _topology_key(node) -> tuple:
    """Canonical key for unordered unary/binary/ternary rooted trees."""
    kind = node[0]
    if kind == "L":
        return ("L",)
    if kind == "U":
        return ("U", _topology_key(node[1]))
    if kind == "B":
        left_key = _topology_key(node[1])
        right_key = _topology_key(node[2])
        if right_key < left_key:
            left_key, right_key = right_key, left_key
        return ("B", left_key, right_key)
    child_keys = sorted([_topology_key(node[1]), _topology_key(node[2]), _topology_key(node[3])])
    return ("T", child_keys[0], child_keys[1], child_keys[2])


def _topology_leaf_count(node) -> int:
    kind = node[0]
    if kind == "L":
        return 1
    if kind == "U":
        return _topology_leaf_count(node[1])
    if kind == "B":
        return _topology_leaf_count(node[1]) + _topology_leaf_count(node[2])
    return (
        _topology_leaf_count(node[1])
        + _topology_leaf_count(node[2])
        + _topology_leaf_count(node[3])
    )


def _topology_rule_count(node) -> int:
    kind = node[0]
    if kind == "L":
        return 0
    if kind == "U":
        return 1 + _topology_rule_count(node[1])
    if kind == "B":
        return 1 + _topology_rule_count(node[1]) + _topology_rule_count(node[2])
    return (
        1
        + _topology_rule_count(node[1])
        + _topology_rule_count(node[2])
        + _topology_rule_count(node[3])
    )


def _topology_unary_binary_ternary_counts(node) -> tuple[int, int, int]:
    """Return (#unary_nodes, #binary_nodes, #ternary_nodes) for a topology tree."""
    kind = node[0]
    if kind == "L":
        return 0, 0, 0
    if kind == "U":
        u_child, b_child, t_child = _topology_unary_binary_ternary_counts(node[1])
        return 1 + u_child, b_child, t_child
    if kind == "B":
        u_left, b_left, t_left = _topology_unary_binary_ternary_counts(node[1])
        u_right, b_right, t_right = _topology_unary_binary_ternary_counts(node[2])
        return u_left + u_right, 1 + b_left + b_right, t_left + t_right
    u1, b1, t1 = _topology_unary_binary_ternary_counts(node[1])
    u2, b2, t2 = _topology_unary_binary_ternary_counts(node[2])
    u3, b3, t3 = _topology_unary_binary_ternary_counts(node[3])
    return u1 + u2 + u3, b1 + b2 + b3, 1 + t1 + t2 + t3


def _topology_key_to_jsonable(node_key):
    if isinstance(node_key, tuple):
        return [_topology_key_to_jsonable(x) for x in node_key]
    return node_key


def _topology_signature_from_key(node_key) -> str:
    """Stable string signature for topology keys (order-invariant)."""
    return json.dumps(_topology_key_to_jsonable(node_key), separators=(",", ":"), ensure_ascii=True)


def _jsonable_to_topology_key(obj):
    if isinstance(obj, list):
        return tuple(_jsonable_to_topology_key(x) for x in obj)
    return obj


def _topology_key_from_signature(sig: str) -> tuple:
    try:
        parsed = json.loads(str(sig))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid topology signature JSON: {sig!r}") from exc
    key = _jsonable_to_topology_key(parsed)
    if not isinstance(key, tuple):
        raise ValueError(f"Invalid topology signature root (expected list/tuple): {sig!r}")
    return key


def _parse_depth_csv(text: str) -> list[int]:
    vals: list[int] = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        vals.append(int(part))
    return vals


def _load_topology_exclusions_from_meta(meta_path: str, depths: set[int]) -> dict[int, set[tuple]]:
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to read topology exclusion meta: {meta_path}") from exc

    if not _is_mixed_topology_mode(str(meta.get("tree_topology"))):
        raise ValueError(
            "Topology exclusions require a mixed-topology source meta file "
            f"(got tree_topology={meta.get('tree_topology')!r})"
        )

    signatures_per_depth = meta.get("topology_signatures_per_depth")
    if not isinstance(signatures_per_depth, dict):
        raise ValueError(
            "Source meta missing 'topology_signatures_per_depth'. "
            "Regenerate source dataset with updated generator."
        )

    out: dict[int, set[tuple]] = {}
    for d in depths:
        raw = signatures_per_depth.get(str(d), [])
        if not isinstance(raw, list):
            raise ValueError(f"Invalid topology_signatures_per_depth[{d}] in meta: expected list")
        converted: set[tuple] = set()
        for x in raw:
            converted.add(_topology_key_from_signature(str(x)))
        out[d] = converted
    return out


def _task_signature_from_rows(rows: np.ndarray) -> str:
    """Stable signature for one task ruleset row-array."""
    arr = np.asarray(rows, dtype=np.int32)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"Unexpected ruleset row shape for task signature: {arr.shape}")
    # Ignore no-op padded rows so signature is independent of outer padding width.
    core = arr[arr[:, 0] != 5]
    core_c = np.ascontiguousarray(core, dtype=np.int32)
    return hashlib.sha1(core_c.tobytes()).hexdigest()


def _load_task_exclusions_from_meta(meta_path: str, depths: set[int]) -> dict[int, set[str]]:
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to read task exclusion meta: {meta_path}") from exc

    if not _is_mixed_topology_mode(str(meta.get("tree_topology"))):
        raise ValueError(
            "Task exclusions require a mixed-topology source meta file "
            f"(got tree_topology={meta.get('tree_topology')!r})"
        )

    signatures_per_depth = meta.get("task_signatures_per_depth")
    if not isinstance(signatures_per_depth, dict):
        # Fallback for older metas: derive signatures directly from packed dataset.
        structure_raw = meta.get("structure")
        if not isinstance(structure_raw, dict):
            raise ValueError(
                "Source meta missing 'task_signatures_per_depth' and invalid/missing "
                "'structure'; cannot derive task signatures."
            )
        try:
            structure = {int(k): int(v) for k, v in structure_raw.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid 'structure' in source meta") from exc

        data_path = str(meta_path).replace("_meta.json", ".npy.bz2")
        try:
            with bz2.BZ2File(data_path, "rb") as f:
                packed = np.load(f)
        except OSError as exc:
            raise ValueError(
                "Source meta missing 'task_signatures_per_depth' and failed to read "
                f"packed dataset for fallback derivation: {data_path}"
            ) from exc

        signatures_per_depth = {}
        offset = 0
        for d in sorted(structure):
            n_d = int(structure[d])
            rows_d = packed[offset : offset + n_d]
            # Convert packed task rows back to int32 [R,6], then hash in the same
            # no-op-insensitive format used during generation-time exclusion.
            rows_d_decoded = np.asarray(unpack_rules_uint32(rows_d), dtype=np.int32)
            signatures_per_depth[str(d)] = [
                _task_signature_from_rows(rows_d_decoded[i]) for i in range(n_d)
            ]
            offset += n_d

    out: dict[int, set[str]] = {}
    for d in depths:
        raw = signatures_per_depth.get(str(d), [])
        if not isinstance(raw, list):
            raise ValueError(f"Invalid task_signatures_per_depth[{d}] in meta: expected list")
        out[d] = set(str(x) for x in raw)
    return out


def _choose_mixed_root_kind(depth: int, rng: np.random.Generator, tree_topology: str) -> str:
    unary_prob = float(_mixed_unary_prob_for_depth(depth))
    ternary_prob = float(MIXED_TERNARY_PROB) if _mixed_mode_supports_ternary(tree_topology) else 0.0
    if unary_prob + ternary_prob > 1.0:
        ternary_prob = max(0.0, 1.0 - unary_prob)
    draw = float(rng.random())
    if draw < unary_prob:
        return "U"
    if ternary_prob > 0.0 and draw < unary_prob + ternary_prob:
        return "T"
    return "B"


def _sample_mixed_topology_exact(depth: int, rng: np.random.Generator, tree_topology: str):
    """Sample a random mixed topology with exact max depth=depth."""
    if depth <= 1:
        return ("L",)

    root_kind = _choose_mixed_root_kind(depth, rng, tree_topology)
    if root_kind == "U":
        return ("U", _sample_mixed_topology_exact(depth - 1, rng, tree_topology))

    if root_kind == "B":
        while True:
            da = int(rng.integers(1, depth))
            db = int(rng.integers(1, depth))
            if max(da, db) == depth - 1:
                break
        left = _sample_mixed_topology_exact(da, rng, tree_topology)
        right = _sample_mixed_topology_exact(db, rng, tree_topology)
        if _topology_key(right) < _topology_key(left):
            left, right = right, left
        return ("B", left, right)

    while True:
        da = int(rng.integers(1, depth))
        db = int(rng.integers(1, depth))
        dc = int(rng.integers(1, depth))
        if max(da, db, dc) == depth - 1:
            break
    children = [
        _sample_mixed_topology_exact(da, rng, tree_topology),
        _sample_mixed_topology_exact(db, rng, tree_topology),
        _sample_mixed_topology_exact(dc, rng, tree_topology),
    ]
    children.sort(key=_topology_key)
    return ("T", children[0], children[1], children[2])


def _sample_mixed_topology_exact_with_key_counts(
    depth: int, rng: np.random.Generator, tree_topology: str
) -> tuple[tuple, tuple, int, int]:
    """Sample topology + canonical key + (leaf_count, rule_count) in one pass."""
    if depth <= 1:
        leaf = ("L",)
        return leaf, ("L",), 1, 0

    root_kind = _choose_mixed_root_kind(depth, rng, tree_topology)
    if root_kind == "U":
        child_topo, child_key, child_leaf_count, child_rule_count = (
            _sample_mixed_topology_exact_with_key_counts(depth - 1, rng, tree_topology)
        )
        topo = ("U", child_topo)
        key = ("U", child_key)
        return topo, key, child_leaf_count, 1 + child_rule_count

    if root_kind == "B":
        while True:
            da = int(rng.integers(1, depth))
            db = int(rng.integers(1, depth))
            if max(da, db) == depth - 1:
                break

        left_topo, left_key, left_leaf_count, left_rule_count = (
            _sample_mixed_topology_exact_with_key_counts(da, rng, tree_topology)
        )
        right_topo, right_key, right_leaf_count, right_rule_count = (
            _sample_mixed_topology_exact_with_key_counts(db, rng, tree_topology)
        )
        if right_key < left_key:
            left_topo, right_topo = right_topo, left_topo
            left_key, right_key = right_key, left_key
            left_leaf_count, right_leaf_count = right_leaf_count, left_leaf_count
            left_rule_count, right_rule_count = right_rule_count, left_rule_count

        topo = ("B", left_topo, right_topo)
        key = ("B", left_key, right_key)
        leaf_count = left_leaf_count + right_leaf_count
        rule_count = 1 + left_rule_count + right_rule_count
        return topo, key, leaf_count, rule_count

    while True:
        da = int(rng.integers(1, depth))
        db = int(rng.integers(1, depth))
        dc = int(rng.integers(1, depth))
        if max(da, db, dc) == depth - 1:
            break

    children = [
        _sample_mixed_topology_exact_with_key_counts(da, rng, tree_topology),
        _sample_mixed_topology_exact_with_key_counts(db, rng, tree_topology),
        _sample_mixed_topology_exact_with_key_counts(dc, rng, tree_topology),
    ]
    children.sort(key=lambda x: x[1])
    topo = ("T", children[0][0], children[1][0], children[2][0])
    key = ("T", children[0][1], children[1][1], children[2][1])
    leaf_count = int(children[0][2] + children[1][2] + children[2][2])
    rule_count = 1 + int(children[0][3] + children[1][3] + children[2][3])
    return topo, key, leaf_count, rule_count


@lru_cache(maxsize=128)
def _mixed_topology_exact_capacity_for_depth(
    depth: int, leaf_cap: int, rule_cap: int, tree_topology: str
) -> int:
    """Exact count of sampler-reachable mixed topologies for one exact depth."""
    d_max = int(depth)
    leaf_cap = int(max(1, leaf_cap))
    rule_cap = int(max(0, rule_cap))
    if d_max < 1:
        return 0

    supports_ternary = _mixed_mode_supports_ternary(tree_topology)
    leq_by_depth: dict[int, dict[tuple[int, int], int]] = {
        1: {(1, 0): 1} if leaf_cap >= 1 and rule_cap >= 0 else {}
    }
    total_prev = 1

    for d in range(2, d_max + 1):
        prev = leq_by_depth[d - 1]
        prev_items = sorted(prev.items())
        curr: defaultdict[tuple[int, int], int] = defaultdict(int)

        # Reuse prior topologies.
        for key, count in prev_items:
            curr[key] += int(count)

        # Unary root over any prior topology.
        for (leaves, rules), count in prev_items:
            leaves_u = int(leaves)
            rules_u = int(rules) + 1
            if leaves_u <= leaf_cap and rules_u <= rule_cap:
                curr[(leaves_u, rules_u)] += int(count)

        # Binary root over unordered pairs with replacement.
        for i, ((la, ra), ca) in enumerate(prev_items):
            for j in range(i, len(prev_items)):
                (lb, rb), cb = prev_items[j]
                leaves_b = int(la) + int(lb)
                rules_b = 1 + int(ra) + int(rb)
                if leaves_b > leaf_cap or rules_b > rule_cap:
                    continue
                mult = int(ca) * int(cb)
                if i == j:
                    mult = int(ca) * (int(ca) + 1) // 2
                curr[(leaves_b, rules_b)] += mult

        if supports_ternary:
            for i, ((la, ra), ca) in enumerate(prev_items):
                for j in range(i, len(prev_items)):
                    (lb, rb), cb = prev_items[j]
                    for k in range(j, len(prev_items)):
                        (lc, rc), cc = prev_items[k]
                        leaves_t = int(la) + int(lb) + int(lc)
                        rules_t = 1 + int(ra) + int(rb) + int(rc)
                        if leaves_t > leaf_cap or rules_t > rule_cap:
                            continue
                        if i == j == k:
                            mult = int(ca) * (int(ca) + 1) * (int(ca) + 2) // 6
                        elif i == j:
                            mult = (int(ca) * (int(ca) + 1) // 2) * int(cc)
                        elif j == k:
                            mult = int(ca) * (int(cb) * (int(cb) + 1) // 2)
                        else:
                            mult = int(ca) * int(cb) * int(cc)
                        curr[(leaves_t, rules_t)] += mult

        total_curr = int(sum(int(v) for v in curr.values()))
        leq_by_depth[d] = dict(curr)
        if d == d_max:
            return max(0, total_curr - total_prev)
        total_prev = total_curr

    return 0


@jax.jit
def _legacy_seed_block(
    key_depth: jax.Array, sample_ids: jax.Array, attempt_ids: jax.Array
) -> jax.Array:
    """Vectorized legacy seed derivation for (sample_idx, attempt_idx)."""
    keys_i = jax.vmap(lambda idx: jax.random.fold_in(key_depth, idx))(sample_ids)
    keys_ia = jax.vmap(lambda key_i: jax.vmap(lambda a: jax.random.fold_in(key_i, a))(attempt_ids))(
        keys_i
    )
    return jax.vmap(
        lambda row: jax.vmap(lambda key_try: jax.random.randint(key_try, (), 0, 2**31 - 1))(row)
    )(keys_ia)


def _instantiate_mixed_topology_ruleset(
    topology,
    *,
    base_pool: np.ndarray,
    rng: np.random.Generator,
    base_seed: int,
    tree_topology: str,
    leaf_count: int | None = None,
) -> np.ndarray | None:
    """Instantiate a sampled topology into ruleset rows."""
    if leaf_count is None:
        leaf_count = _topology_leaf_count(topology)
    if leaf_count < 1 or leaf_count > len(base_pool):
        return None

    sampled = rng.choice(base_pool, size=leaf_count, replace=False)
    leaves = [int(u) for u in np.asarray(sampled, dtype=np.int32).tolist()]
    leaf_set = set(leaves)
    produced: set[int] = set()
    rows_post: list[np.ndarray] = []
    leaf_idx = 0

    def walk(node) -> tuple[int, bool]:
        nonlocal leaf_idx
        kind = node[0]
        if kind == "L":
            uid = int(leaves[leaf_idx])
            leaf_idx += 1
            return uid, True

        if kind == "U":
            in_uid, ok = walk(node[1])
            if not ok:
                return -1, False
            out_uid = int(_unary_to_output_uid_global_np(in_uid, base_seed=base_seed))
            if out_uid == in_uid or out_uid in leaf_set or out_uid in produced:
                return -1, False
            produced.add(out_uid)
            rows_post.append(_encode_transform_row(in_uid, out_uid))
            return out_uid, True

        if kind == "B":
            left_uid, ok_l = walk(node[1])
            if not ok_l:
                return -1, False
            right_uid, ok_r = walk(node[2])
            if not ok_r:
                return -1, False
            a, b = (left_uid, right_uid) if left_uid <= right_uid else (right_uid, left_uid)
            out_uid = int(_pair_to_output_uid_global_np(a, b, base_seed=base_seed))
            if out_uid == a or out_uid == b or out_uid in leaf_set or out_uid in produced:
                return -1, False
            produced.add(out_uid)
            rows_post.append(_encode_combine_row(a, b, out_uid, RULE_TYPE_COMBINE))
            return out_uid, True

        uid_a, ok_a = walk(node[1])
        if not ok_a:
            return -1, False
        uid_b, ok_b = walk(node[2])
        if not ok_b:
            return -1, False
        uid_c, ok_c = walk(node[3])
        if not ok_c:
            return -1, False
        ordered = sorted([uid_a, uid_b, uid_c])
        out_uid = int(
            _triple_to_output_uid_global_np(
                ordered[0],
                ordered[1],
                ordered[2],
                base_seed=base_seed,
            )
        )
        if out_uid in ordered or out_uid in leaf_set or out_uid in produced:
            return -1, False
        produced.add(out_uid)
        rows_post.append(_encode_ternary_combine_row(ordered[0], ordered[1], ordered[2], out_uid))
        return out_uid, True

    _root_uid, ok = walk(topology)
    if not ok or leaf_idx != leaf_count:
        return None

    # Reverse postorder so root producer appears first.
    rows = rows_post[::-1]
    return np.asarray(rows, dtype=np.int32)


def _generate_mixed_rulesets_for_depth(
    *,
    depth: int,
    n_tasks: int,
    base_pool: np.ndarray,
    base_seed: int,
    tree_topology: str,
    key_depth: jax.Array,
    max_rules_for_depth: int,
    excluded_topology_signatures: set[tuple] | None = None,
    excluded_task_signatures: set[str] | None = None,
    target_unique_topologies: int = 0,
    collect_rulesets: bool = True,
) -> tuple[np.ndarray | None, dict]:
    """Generate topology-diverse mixed rulesets for one exact depth."""
    if depth <= 1:
        raise ValueError("Mixed topology generator expects depth >= 2.")

    rows_all: list[np.ndarray] | None = [] if collect_rulesets else None
    unique_topology_keys: set[tuple] = set()
    unique_topology_key_list: list[tuple] = []
    topology_by_key: dict[tuple, tuple] = {}
    leaf_rule_by_key: dict[tuple, tuple[int, int]] = {}
    unique_task_signatures: set[str] = set()
    unary_nodes_total = 0
    binary_nodes_total = 0
    ternary_nodes_total = 0
    target_unique_relaxations = 0
    leaf_count_dist: dict[int, int] = {}
    rule_count_dist: dict[int, int] = {}
    excluded = excluded_topology_signatures or set()
    excluded_tasks = excluded_task_signatures or set()
    mapping_base_seed = 0 if _mixed_mode_supports_ternary(tree_topology) else int(base_seed)
    depth_capacity = _mixed_topology_exact_capacity_for_depth(
        int(depth),
        int(len(base_pool)),
        int(max_rules_for_depth),
        tree_topology,
    )
    available_unique_capacity = max(0, int(depth_capacity) - int(len(excluded)))
    requested_unique_target = max(0, int(target_unique_topologies))
    forced_unique_target = min(
        int(n_tasks), int(available_unique_capacity), int(requested_unique_target)
    )
    force_reuse_after_target = forced_unique_target > 0
    # Legacy adaptive unique preference at deeper mixed depths.
    adaptive_unique_target = 0
    if int(depth) >= 5:
        adaptive_unique_target = min(int(n_tasks), int(available_unique_capacity))
    unique_target = (
        int(forced_unique_target) if force_reuse_after_target else int(adaptive_unique_target)
    )
    # Keep duplicate-acceptance delay bounded even when MAX_TRIES is very large.
    relaxed_duplicate_guard = min(
        max(1, int(MAX_TRIES) // 3),
        max(1, int(STRICT_UNIQUE_TRIES)),
    )
    attempt_ids = jnp.arange(MAX_TRIES, dtype=jnp.int32)
    block_size = max(1, int(SEED_BLOCK_SIZE))
    seed_block_start = -1
    seed_block_np: np.ndarray | None = None

    for i in range(n_tasks):
        if (
            seed_block_np is None
            or i < seed_block_start
            or i >= seed_block_start + seed_block_np.shape[0]
        ):
            seed_block_start = (i // block_size) * block_size
            # Keep jit input shapes fixed for better cache reuse; slice tail locally.
            block_count = min(block_size, n_tasks - seed_block_start)
            sample_ids_full = jnp.arange(
                seed_block_start, seed_block_start + block_size, dtype=jnp.int32
            )
            seed_block_full = _legacy_seed_block(key_depth, sample_ids_full, attempt_ids)
            seed_block_np = np.asarray(seed_block_full[:block_count], dtype=np.int64)

        assert seed_block_np is not None
        seeds_i = seed_block_np[i - seed_block_start]
        rows_i = None
        topo_key_i = None
        topo_i = None
        task_sig_i = None
        leaf_count_i = None
        rule_count_i = None
        reuse_only = force_reuse_after_target and len(unique_topology_keys) >= unique_target

        if reuse_only:
            if not unique_topology_key_list:
                raise ValueError("Internal error: fixed topology catalog is empty")
            for attempt in range(MAX_TRIES):
                rng_i = np.random.default_rng(int(seeds_i[attempt]))
                pick_idx = int(rng_i.integers(0, len(unique_topology_key_list)))
                topo_key = unique_topology_key_list[pick_idx]
                topo = topology_by_key[topo_key]
                leaf_count, rule_count = leaf_rule_by_key[topo_key]
                rows_candidate = _instantiate_mixed_topology_ruleset(
                    topo,
                    base_pool=base_pool,
                    rng=rng_i,
                    base_seed=mapping_base_seed,
                    tree_topology=tree_topology,
                    leaf_count=leaf_count,
                )
                if rows_candidate is None:
                    continue
                task_sig_candidate = _task_signature_from_rows(rows_candidate)
                if task_sig_candidate in excluded_tasks:
                    continue
                rows_i = rows_candidate
                topo_key_i = topo_key
                topo_i = topo
                task_sig_i = task_sig_candidate
                leaf_count_i = int(leaf_count)
                rule_count_i = int(rule_count)
                break
        else:
            force_unique_now = (
                force_reuse_after_target and len(unique_topology_keys) < unique_target
            )
            need_unique = force_unique_now or (
                int(depth) >= 5 and bool(excluded) and len(unique_topology_keys) < unique_target
            )

            # Phase 1 (depth>=5): strict unique preference.
            # Phase 2 fallback: duplicates allowed after initial retries.
            phase_count = 1 if force_unique_now else (2 if need_unique else 1)
            for phase in range(phase_count):
                strict_unique_phase = bool(need_unique and phase == 0)
                attempt_cap = (
                    int(MAX_TRIES)
                    if force_unique_now
                    else (
                        min(int(MAX_TRIES), max(1, int(STRICT_UNIQUE_TRIES)))
                        if strict_unique_phase
                        else int(MAX_TRIES)
                    )
                )

                for attempt in range(attempt_cap):
                    rng_i = np.random.default_rng(int(seeds_i[attempt]))

                    topo, topo_key, leaf_count, rule_count = (
                        _sample_mixed_topology_exact_with_key_counts(depth, rng_i, tree_topology)
                    )

                    # Enforce cross-dataset topology disjointness (if requested).
                    if topo_key in excluded:
                        continue

                    # Enforce pool feasibility and max-rule budget.
                    if leaf_count > len(base_pool):
                        continue
                    if rule_count > max_rules_for_depth:
                        continue
                    # Optional soft bias for depth-5: reject some low-leaf candidates
                    # to increase representation of high-leaf topologies.
                    if (
                        int(depth) == 5
                        and MIXED_D5_LOW_LEAF_REJECT_PROB > 0.0
                        and int(leaf_count) < int(MIXED_D5_HIGH_LEAF_MIN)
                        and float(rng_i.random()) < MIXED_D5_LOW_LEAF_REJECT_PROB
                    ):
                        continue

                    if topo_key in unique_topology_keys:
                        if strict_unique_phase:
                            continue
                        if attempt < relaxed_duplicate_guard:
                            continue

                    rows_candidate = _instantiate_mixed_topology_ruleset(
                        topo,
                        base_pool=base_pool,
                        rng=rng_i,
                        base_seed=mapping_base_seed,
                        tree_topology=tree_topology,
                        leaf_count=leaf_count,
                    )
                    if rows_candidate is None:
                        continue
                    task_sig_candidate = _task_signature_from_rows(rows_candidate)
                    if task_sig_candidate in excluded_tasks:
                        continue

                    rows_i = rows_candidate
                    topo_key_i = topo_key
                    topo_i = topo
                    task_sig_i = task_sig_candidate
                    leaf_count_i = int(leaf_count)
                    rule_count_i = int(rule_count)
                    break

                if rows_i is not None:
                    break

        if (
            (
                rows_i is None
                or topo_key_i is None
                or topo_i is None
                or leaf_count_i is None
                or rule_count_i is None
            )
            and force_reuse_after_target
            and unique_topology_key_list
        ):
            # Soft target mode: if unique-first sampling stalls, reuse already discovered
            # topologies so dataset generation can complete with a reported shortfall.
            for attempt in range(MAX_TRIES):
                rng_i = np.random.default_rng(int(seeds_i[attempt]))
                pick_idx = int(rng_i.integers(0, len(unique_topology_key_list)))
                topo_key = unique_topology_key_list[pick_idx]
                topo = topology_by_key[topo_key]
                leaf_count, rule_count = leaf_rule_by_key[topo_key]
                rows_candidate = _instantiate_mixed_topology_ruleset(
                    topo,
                    base_pool=base_pool,
                    rng=rng_i,
                    base_seed=mapping_base_seed,
                    tree_topology=tree_topology,
                    leaf_count=leaf_count,
                )
                if rows_candidate is None:
                    continue
                task_sig_candidate = _task_signature_from_rows(rows_candidate)
                if task_sig_candidate in excluded_tasks:
                    continue
                rows_i = rows_candidate
                topo_key_i = topo_key
                topo_i = topo
                task_sig_i = task_sig_candidate
                leaf_count_i = int(leaf_count)
                rule_count_i = int(rule_count)
                target_unique_relaxations += 1
                break

        if (
            rows_i is None
            or topo_key_i is None
            or topo_i is None
            or task_sig_i is None
            or leaf_count_i is None
            or rule_count_i is None
        ):
            raise ValueError(
                f"Failed to construct mixed topology sample {i} at depth {depth} "
                f"after {MAX_TRIES} tries; consider larger pool_size or smaller depth. "
                f"(excluded_topologies={len(excluded)}, "
                f"unique_so_far={len(unique_topology_keys)}, "
                f"unique_target={unique_target}, depth_capacity={depth_capacity}, "
                f"target_unique_topologies={requested_unique_target})"
            )

        if rows_all is not None:
            padded = np.zeros((max_rules_for_depth, 6), dtype=np.int32)
            padded[..., 0] = 5  # no-op rule
            padded[: rows_i.shape[0], :] = rows_i
            rows_all.append(padded)

        if topo_key_i not in unique_topology_keys:
            unique_topology_keys.add(topo_key_i)
            unique_topology_key_list.append(topo_key_i)
            topology_by_key[topo_key_i] = topo_i
            leaf_rule_by_key[topo_key_i] = (int(leaf_count_i), int(rule_count_i))
        unique_task_signatures.add(str(task_sig_i))
        unary_i, binary_i, ternary_i = _topology_unary_binary_ternary_counts(topo_i)
        unary_nodes_total += unary_i
        binary_nodes_total += binary_i
        ternary_nodes_total += ternary_i
        leaf_count_dist[leaf_count_i] = leaf_count_dist.get(leaf_count_i, 0) + 1
        rule_count_dist[rule_count_i] = rule_count_dist.get(rule_count_i, 0) + 1

    total_internal_nodes = unary_nodes_total + binary_nodes_total + ternary_nodes_total
    unary_fraction = (
        float(unary_nodes_total) / float(total_internal_nodes) if total_internal_nodes > 0 else 0.0
    )
    binary_fraction = (
        float(binary_nodes_total) / float(total_internal_nodes) if total_internal_nodes > 0 else 0.0
    )
    ternary_fraction = (
        float(ternary_nodes_total) / float(total_internal_nodes)
        if total_internal_nodes > 0
        else 0.0
    )
    unique_count = int(len(unique_topology_keys))
    unique_shortfall = max(0, int(unique_target) - int(unique_count))

    stats = {
        "unique_topologies": int(unique_count),
        "num_tasks": int(n_tasks),
        "unique_topology_fraction": (float(unique_count) / float(n_tasks) if n_tasks > 0 else 0.0),
        "unary_nodes_total": int(unary_nodes_total),
        "binary_nodes_total": int(binary_nodes_total),
        "ternary_nodes_total": int(ternary_nodes_total),
        "unary_node_fraction": float(unary_fraction),
        "binary_node_fraction": float(binary_fraction),
        "ternary_node_fraction": float(ternary_fraction),
        "leaf_count_distribution": {
            str(k): int(leaf_count_dist[k]) for k in sorted(leaf_count_dist)
        },
        "rule_count_distribution": {
            str(k): int(rule_count_dist[k]) for k in sorted(rule_count_dist)
        },
        "excluded_topologies": int(len(excluded)),
        "excluded_task_signatures": int(len(excluded_tasks)),
        "depth_capacity": int(depth_capacity),
        "available_unique_capacity": int(available_unique_capacity),
        "unique_target": int(unique_target),
        "target_unique_topologies_requested": int(requested_unique_target),
        "target_unique_topologies_effective": int(forced_unique_target),
        "target_unique_topologies_achieved": int(unique_count),
        "target_unique_topologies_shortfall": int(unique_shortfall),
        "target_unique_topologies_met": bool(unique_shortfall == 0),
        "target_unique_relaxations": int(target_unique_relaxations),
        "topology_signatures": sorted(
            _topology_signature_from_key(k) for k in unique_topology_keys
        ),
        "task_signatures": sorted(unique_task_signatures),
    }
    if rows_all is None:
        return None, stats
    return np.asarray(rows_all, dtype=np.int32), stats


def _generate_mixed_rulesets_for_depth_worker(
    payload: tuple[int, int, np.ndarray, int, int, int, int, str, set[tuple], set[str]],
) -> tuple[int, np.ndarray, dict]:
    (
        depth,
        n_tasks,
        base_pool,
        base_seed,
        bench_seed,
        max_rules_for_depth,
        target_unique_topologies,
        tree_topology,
        excluded_topology_signatures,
        excluded_task_signatures,
    ) = payload
    key_root = jax.random.PRNGKey(int(bench_seed))
    key_depth = jax.random.fold_in(key_root, int(depth))
    rs_np, stats = _generate_mixed_rulesets_for_depth(
        depth=int(depth),
        n_tasks=int(n_tasks),
        base_pool=np.asarray(base_pool, dtype=np.int32),
        base_seed=int(base_seed),
        tree_topology=str(tree_topology),
        key_depth=key_depth,
        max_rules_for_depth=int(max_rules_for_depth),
        target_unique_topologies=int(target_unique_topologies),
        excluded_topology_signatures=excluded_topology_signatures,
        excluded_task_signatures=excluded_task_signatures,
        collect_rulesets=True,
    )
    assert rs_np is not None
    return int(depth), rs_np, stats


def _mixed_topology_capacity_per_depth(
    *, max_depth: int, pool_size: int, tree_topology: str
) -> dict[int, int]:
    """Count exact mixed topology capacity per depth with leaf-count cap.

    Counts topology shapes (not item assignments), for exact max-depth buckets.
    """
    if max_depth < 1:
        return {}
    leaf_cap = max(1, int(pool_size))
    supports_ternary = _mixed_mode_supports_ternary(tree_topology)

    # depth -> {leaf_count -> num_topologies_with_max_depth<=depth}
    leq_by_depth: dict[int, dict[int, int]] = {1: {1: 1}}
    total_leq_prev = 1
    exact_counts: dict[int, int] = {1: 1}

    for depth in range(2, int(max_depth) + 1):
        prev = leq_by_depth[depth - 1]
        curr: dict[int, int] = {}

        # Reuse previous trees + unary root over previous trees.
        for leaves, count in prev.items():
            if leaves <= leaf_cap:
                curr[leaves] = curr.get(leaves, 0) + 2 * int(count)

        # Binary root over unordered pairs with replacement.
        leaf_vals = sorted(prev.keys())
        for i, la in enumerate(leaf_vals):
            ca = int(prev[la])
            diag_leaves = la + la
            if diag_leaves <= leaf_cap:
                curr[diag_leaves] = curr.get(diag_leaves, 0) + (ca * (ca + 1) // 2)

            for lb in leaf_vals[i + 1 :]:
                total_leaves = la + lb
                if total_leaves > leaf_cap:
                    break
                cb = int(prev[lb])
                curr[total_leaves] = curr.get(total_leaves, 0) + (ca * cb)

        if supports_ternary:
            for i, la in enumerate(leaf_vals):
                ca = int(prev[la])
                for j in range(i, len(leaf_vals)):
                    lb = leaf_vals[j]
                    cb = int(prev[lb])
                    for k in range(j, len(leaf_vals)):
                        lc = leaf_vals[k]
                        cc = int(prev[lc])
                        total_leaves = la + lb + lc
                        if total_leaves > leaf_cap:
                            continue
                        if i == j == k:
                            mult = ca * (ca + 1) * (ca + 2) // 6
                        elif i == j:
                            mult = (ca * (ca + 1) // 2) * cc
                        elif j == k:
                            mult = ca * (cb * (cb + 1) // 2)
                        else:
                            mult = ca * cb * cc
                        curr[total_leaves] = curr.get(total_leaves, 0) + mult

        total_curr = sum(int(v) for v in curr.values())
        exact_counts[depth] = int(total_curr - total_leq_prev)
        leq_by_depth[depth] = curr
        total_leq_prev = total_curr

    return exact_counts


def _estimate_mixed_unique_total_for_n(
    *,
    n_highest: int,
    max_depth: int,
    pool_size: int,
    base_pool: np.ndarray,
    base_seed: int,
    tree_topology: str,
    key_root: jax.Array,
    excluded_topologies_by_depth: dict[int, set[tuple]],
) -> tuple[int, dict[int, int], dict[int, int]]:
    structure = compute_depth_task_counts(int(max_depth), int(n_highest), int(pool_size))
    depths = sorted(structure.keys())
    r_per_depth = {d: _rules_per_depth(d, tree_topology) for d in depths}

    unique_per_depth: dict[int, int] = {}
    unique_total = 0
    for depth in depths:
        if depth < 2:
            continue
        n_depth = int(structure[depth])
        key_depth = jax.random.fold_in(key_root, int(depth))
        excluded = excluded_topologies_by_depth.get(int(depth), set())
        _, stats = _generate_mixed_rulesets_for_depth(
            depth=int(depth),
            n_tasks=n_depth,
            base_pool=base_pool,
            base_seed=int(base_seed),
            tree_topology=str(tree_topology),
            key_depth=key_depth,
            max_rules_for_depth=int(r_per_depth[depth]),
            excluded_topology_signatures=excluded,
            collect_rulesets=False,
        )
        uniq = int(stats.get("unique_topologies", 0))
        unique_per_depth[int(depth)] = uniq
        unique_total += uniq
    return int(unique_total), unique_per_depth, structure


def _solve_n_for_target_unique_total(
    *,
    target_unique_total: int,
    max_depth: int,
    pool_size: int,
    base_pool: np.ndarray,
    base_seed: int,
    tree_topology: str,
    key_root: jax.Array,
    excluded_topologies_by_depth: dict[int, set[tuple]],
) -> tuple[int, int, dict[int, int], dict[int, int]]:
    if target_unique_total <= 0:
        raise ValueError("target_unique_total must be > 0")

    cache: dict[int, tuple[int, dict[int, int], dict[int, int]]] = {}

    def eval_n(n_val: int) -> tuple[int, dict[int, int], dict[int, int]]:
        n_val = max(1, int(n_val))
        if n_val not in cache:
            cache[n_val] = _estimate_mixed_unique_total_for_n(
                n_highest=n_val,
                max_depth=int(max_depth),
                pool_size=int(pool_size),
                base_pool=base_pool,
                base_seed=int(base_seed),
                tree_topology=str(tree_topology),
                key_root=key_root,
                excluded_topologies_by_depth=excluded_topologies_by_depth,
            )
        return cache[n_val]

    low = 1
    low_total, _, _ = eval_n(low)
    if low_total >= target_unique_total:
        total, per_depth, structure = eval_n(low)
        return low, total, per_depth, structure

    high = max(2, int(target_unique_total))
    high_total, _, _ = eval_n(high)
    while high_total < target_unique_total:
        low = high
        high = int(high * 2)
        if high > 10_000_000:
            raise ValueError(
                "Could not satisfy target unique topology count; "
                "try increasing --pool-size or lowering the target."
            )
        high_total, _, _ = eval_n(high)

    while low + 1 < high:
        mid = (low + high) // 2
        mid_total, _, _ = eval_n(mid)
        if mid_total >= target_unique_total:
            high = mid
        else:
            low = mid

    total, per_depth, structure = eval_n(high)
    return high, total, per_depth, structure


def _perm_all_from_bench_seed(bench_seed: int) -> np.ndarray:
    key_root = jax.random.PRNGKey(int(bench_seed))
    k_pool = jax.random.fold_in(key_root, 9991)
    return np.array(jax.random.permutation(k_pool, _DOMAIN_SIZE), dtype=np.int32)


def _parse_pool_indices(text: str):
    if not text:
        return None
    parts = [p for p in text.replace(",", " ").split() if p]
    if not parts:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid --pool-indices entry: {exc}") from exc


def _validate_pool_uids(uids, label: str):
    if any((u < 0 or u >= _DOMAIN_SIZE) for u in uids):
        raise ValueError(f"{label} contains values outside [0, {_DOMAIN_SIZE - 1}]")
    if len(set(uids)) != len(uids):
        raise ValueError(f"{label} must be unique (no duplicates)")


def _load_pool_uids_from_meta(meta_path: str):
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to read meta file: {meta_path}") from exc

    pool_uids = meta.get("pool_uids", None)
    pool_size_meta = meta.get("pool_size", None)
    if pool_uids is None:
        bench_seed = meta.get("bench_seed", None)
        if pool_size_meta is None or bench_seed is None:
            raise ValueError("Meta file missing pool_uids and cannot be reconstructed")
        perm_all = _perm_all_from_bench_seed(int(bench_seed))
        pool_uids = perm_all[: int(pool_size_meta)].tolist()

    pool_uids = [int(u) for u in pool_uids]
    if pool_size_meta is not None and len(pool_uids) != int(pool_size_meta):
        raise ValueError("Meta pool_uids length does not match pool_size")
    _validate_pool_uids(pool_uids, "pool_uids in meta")
    return pool_uids


def main():
    ap = argparse.ArgumentParser(description="Generate rulesets with strict subtree reuse")
    ap.add_argument(
        "--n",
        type=int,
        default=None,
        help=(
            "Number of tasks at the HIGHEST depth. "
            "Optional when --target-unique-topologies-total is set."
        ),
    )
    ap.add_argument("--max-depth", type=int, required=True, help="Maximum depth to generate")
    ap.add_argument(
        "--pool-size",
        type=int,
        required=True,
        help="Number of unique (item,color) pairs for depth-1 tasks",
    )
    ap.add_argument(
        "--distractor-density",
        type=float,
        default=0.0,
        help=(
            "Dataset-global distractor density over all pool pairs in [0,1]. "
            "0.0 disables distractors; 1.0 enables all pool-pair distractors."
        ),
    )
    ap.add_argument(
        "--pool-indices",
        type=str,
        default="",
        help=f"Comma/space-separated list of pool uids in [0,{_DOMAIN_SIZE - 1}] (uid=item*{_NUM_REAL_COLORS}+color)",
    )
    ap.add_argument(
        "--pool-meta",
        type=str,
        default="",
        help="Path to a previous dataset meta.json to reuse or exclude its pool items",
    )
    ap.add_argument(
        "--pool-meta-mode",
        type=str,
        default="superset",
        choices=["superset", "disjoint"],
        help="How to use pool items from --pool-meta (superset includes them, disjoint excludes them)",
    )
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--bench-seed", type=int, default=12345)
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--name", type=str, default="rulesets", help="Prefix for output files")
    ap.add_argument("--preview", type=int, default=0)
    ap.add_argument("--preview-seed", type=int, default=0)
    ap.add_argument(
        "--pool-partition",
        type=int,
        default=None,
        help="Partition index for disjoint pools (0, 1, 2, ...). "
        "Deterministically partitions item TYPES (disjoint tasks) instead of random sampling.",
    )
    ap.add_argument(
        "--num-partitions",
        type=int,
        default=3,
        help="Total number of partitions when using --pool-partition (default 3).",
    )
    ap.add_argument(
        "--tree-topology",
        type=str,
        default=TREE_TOPOLOGY_BALANCED,
        choices=[
            TREE_TOPOLOGY_BALANCED,
            TREE_TOPOLOGY_ASYM_D2_D3_6L,
            TREE_TOPOLOGY_MIXED_U1_B2,
            TREE_TOPOLOGY_MIXED_U1_B2_T3,
        ],
        help=(
            "Core task tree topology. "
            "'balanced' keeps the existing strict binary trees. "
            "'asym_d2_d3_6l' uses a 6-leaf asymmetric depth-3 tree "
            "(one branch depth-2, one branch depth-3) for depth-3 tasks. "
            "'mixed_u1_b2' samples topology-diverse trees with unary transforms "
            "(1-input) and binary combines (2-input), unordered binary children. "
            "'mixed_u1_b2_t3' additionally samples ternary combines (3-input)."
        ),
    )
    ap.add_argument(
        "--enforce-max-depth-unique",
        action="store_true",
        help=(
            "Reject duplicate max-depth task structures during generation. "
            "For depth-3 this enforces global uniqueness of sampled task trees."
        ),
    )
    ap.add_argument(
        "--linked-depth-samples",
        action="store_true",
        help=(
            "Special-case balanced depth-3 mode where depths 1/2/3 each have N tasks. "
            "Each depth-3 task contributes exactly one depth-2 subtree task and one depth-1 "
            "leaf task. Intended for controlled tiny-N sweeps only."
        ),
    )
    ap.add_argument(
        "--target-unique-topologies-total",
        type=int,
        default=0,
        help=(
            "Auto-solve for --n (mixed mode) to hit at least this many distinct "
            "topologies summed across depths >=2."
        ),
    )
    ap.add_argument(
        "--target-topologies-per-depth",
        type=int,
        default=0,
        help=(
            "Mixed mode only. If >0, cap each mixed depth bucket to this many "
            "unique topologies (or feasible maximum), then reuse those topologies "
            "to fill the requested task count for that depth."
        ),
    )
    ap.add_argument(
        "--target-topologies-depths",
        type=str,
        default="",
        help=(
            "Optional comma-separated depths where --target-topologies-per-depth applies "
            "(e.g. '6'). Default: all mixed depths present in the run."
        ),
    )
    ap.add_argument(
        "--mixed-tasks-per-depth-ge3",
        type=int,
        default=0,
        help=(
            "Mixed mode only. If >0, set tasks for each depth >=3 to this value. "
            "Depth 1 and 2 remain at their maximal standard counts "
            "(pool_size and C(pool_size,2))."
        ),
    )
    ap.add_argument(
        "--exclude-topologies-from-meta",
        type=str,
        default="",
        help=(
            "Path to mixed dataset meta.json whose topology signatures should be "
            "excluded while sampling (for cross-dataset disjointness)."
        ),
    )
    ap.add_argument(
        "--exclude-topology-depths",
        type=str,
        default="",
        help=(
            "Comma-separated depths where topology exclusions apply "
            "(e.g. '4,5,6'). Default: all mixed depths present in structure."
        ),
    )
    ap.add_argument(
        "--exclude-task-signatures-from-meta",
        type=str,
        default="",
        help=(
            "Path to mixed dataset meta.json whose task signatures should be "
            "excluded while sampling (for cross-dataset task-tree disjointness)."
        ),
    )
    ap.add_argument(
        "--exclude-task-depths",
        type=str,
        default="",
        help=(
            "Comma-separated depths where task-signature exclusions apply "
            "(e.g. '4,5,6'). Default: all mixed depths present in structure."
        ),
    )

    args = ap.parse_args()

    if (
        args.n is None
        and args.target_unique_topologies_total <= 0
        and args.mixed_tasks_per_depth_ge3 <= 0
    ):
        ap.error("Provide --n, --target-unique-topologies-total, or --mixed-tasks-per-depth-ge3.")
    if args.n is not None and int(args.n) <= 0:
        ap.error("--n must be > 0")
    if args.target_unique_topologies_total < 0:
        ap.error("--target-unique-topologies-total must be >= 0")
    if args.target_topologies_per_depth < 0:
        ap.error("--target-topologies-per-depth must be >= 0")
    if args.mixed_tasks_per_depth_ge3 < 0:
        ap.error("--mixed-tasks-per-depth-ge3 must be >= 0")
    if args.target_topologies_depths and args.target_topologies_per_depth <= 0:
        ap.error("--target-topologies-depths requires --target-topologies-per-depth > 0")
    if args.exclude_task_depths and not args.exclude_task_signatures_from_meta:
        ap.error("--exclude-task-depths requires --exclude-task-signatures-from-meta")

    if args.pool_size < 1 or args.pool_size > _DOMAIN_SIZE:
        ap.error(f"--pool-size must be in [1,{_DOMAIN_SIZE}]")
    if args.max_depth < 1 or args.max_depth > 6:
        ap.error("--max-depth must be in [1, 6]")
    if args.distractor_density < 0.0 or args.distractor_density > 1.0:
        ap.error("--distractor-density must be in [0.0, 1.0]")
    if args.tree_topology == TREE_TOPOLOGY_ASYM_D2_D3_6L and args.max_depth > 3:
        ap.error("--tree-topology asym_d2_d3_6l currently supports --max-depth <= 3")
    if args.tree_topology == TREE_TOPOLOGY_ASYM_D2_D3_6L and args.max_depth < 3:
        ap.error("--tree-topology asym_d2_d3_6l requires --max-depth >= 3")
    if args.exclude_topologies_from_meta and not _is_mixed_topology_mode(args.tree_topology):
        ap.error("--exclude-topologies-from-meta is only supported for mixed topology modes")
    if args.exclude_task_signatures_from_meta and not _is_mixed_topology_mode(args.tree_topology):
        ap.error("--exclude-task-signatures-from-meta is only supported for mixed topology modes")
    if args.linked_depth_samples:
        if args.tree_topology != TREE_TOPOLOGY_BALANCED:
            ap.error("--linked-depth-samples requires --tree-topology balanced")
        if args.max_depth != 3:
            ap.error("--linked-depth-samples requires --max-depth 3")
    if args.target_unique_topologies_total > 0:
        if not _is_mixed_topology_mode(args.tree_topology):
            ap.error("--target-unique-topologies-total requires a mixed topology mode")
        if args.max_depth < 2:
            ap.error("--target-unique-topologies-total requires --max-depth >= 2")
        if args.linked_depth_samples:
            ap.error(
                "--target-unique-topologies-total cannot be combined with --linked-depth-samples"
            )
    if args.target_topologies_per_depth > 0 and not _is_mixed_topology_mode(args.tree_topology):
        ap.error("--target-topologies-per-depth requires a mixed topology mode")
    if args.target_topologies_per_depth > 0 and args.target_unique_topologies_total > 0:
        ap.error(
            "--target-topologies-per-depth cannot be combined with --target-unique-topologies-total"
        )
    if args.mixed_tasks_per_depth_ge3 > 0:
        if not _is_mixed_topology_mode(args.tree_topology):
            ap.error("--mixed-tasks-per-depth-ge3 requires a mixed topology mode")
        if args.linked_depth_samples:
            ap.error("--mixed-tasks-per-depth-ge3 cannot be combined with --linked-depth-samples")
        if args.target_unique_topologies_total > 0:
            ap.error(
                "--mixed-tasks-per-depth-ge3 cannot be combined with "
                "--target-unique-topologies-total"
            )

    if args.pool_indices and args.pool_meta:
        ap.error("Use only one of --pool-indices or --pool-meta")

    try:
        pool_indices = _parse_pool_indices(args.pool_indices)
    except ValueError as exc:
        ap.error(str(exc))

    if pool_indices is not None:
        try:
            _validate_pool_uids(pool_indices, "--pool-indices")
        except ValueError as exc:
            ap.error(str(exc))
        if len(pool_indices) != args.pool_size:
            ap.error("--pool-indices length must match --pool-size")

    os.makedirs(args.out_dir, exist_ok=True)

    n_highest = int(args.n) if args.n is not None else 1

    # Compute subtree-closed structure (may be overridden later by target-unique mode).
    if args.linked_depth_samples:
        structure = {1: int(n_highest), 2: int(n_highest), 3: int(n_highest)}
    elif _is_mixed_topology_mode(args.tree_topology) and args.mixed_tasks_per_depth_ge3 > 0:
        structure = {1: int(args.pool_size)}
        if int(args.max_depth) >= 2:
            structure[2] = int(args.pool_size) * (int(args.pool_size) - 1) // 2
        for d in range(3, int(args.max_depth) + 1):
            structure[int(d)] = int(args.mixed_tasks_per_depth_ge3)
        n_highest = int(structure.get(int(args.max_depth), int(args.pool_size)))
    else:
        structure = compute_depth_task_counts(args.max_depth, n_highest, args.pool_size)
    depths = sorted(structure.keys())
    mixed_depths = [d for d in depths if d >= 2]
    target_topology_depths: set[int] = set()
    if args.target_topologies_depths:
        try:
            target_topology_depths = set(
                int(d) for d in _parse_depth_csv(args.target_topologies_depths)
            )
        except ValueError:
            ap.error("--target-topologies-depths must be a comma-separated list of ints")
        invalid_target_depths = sorted(d for d in target_topology_depths if d not in mixed_depths)
        if invalid_target_depths:
            ap.error(
                "--target-topologies-depths contains depths not present in this run: "
                f"{invalid_target_depths}"
            )
    mixed_topology_target_per_depth: dict[int, int] = {}
    if _is_mixed_topology_mode(args.tree_topology) and args.target_topologies_per_depth > 0:
        target_depths = (
            target_topology_depths if target_topology_depths else set(int(d) for d in mixed_depths)
        )
        mixed_topology_target_per_depth = {
            int(d): int(args.target_topologies_per_depth)
            for d in mixed_depths
            if int(d) in target_depths
        }

    exclude_topology_depths: set[int] = set()
    excluded_topologies_by_depth: dict[int, set[tuple]] = {}
    if args.exclude_topologies_from_meta:
        if args.exclude_topology_depths:
            try:
                exclude_topology_depths = set(_parse_depth_csv(args.exclude_topology_depths))
            except ValueError:
                ap.error("--exclude-topology-depths must be a comma-separated list of ints")
        else:
            exclude_topology_depths = set(mixed_depths)
        invalid_excl = sorted(d for d in exclude_topology_depths if d not in mixed_depths)
        if invalid_excl:
            ap.error(
                f"--exclude-topology-depths contains depths not present in this run: {invalid_excl}"
            )
        try:
            excluded_topologies_by_depth = _load_topology_exclusions_from_meta(
                args.exclude_topologies_from_meta, exclude_topology_depths
            )
        except ValueError as exc:
            ap.error(str(exc))

    exclude_task_depths: set[int] = set()
    excluded_tasks_by_depth: dict[int, set[str]] = {}
    if args.exclude_task_signatures_from_meta:
        if args.exclude_task_depths:
            try:
                exclude_task_depths = set(_parse_depth_csv(args.exclude_task_depths))
            except ValueError:
                ap.error("--exclude-task-depths must be a comma-separated list of ints")
        else:
            exclude_task_depths = set(mixed_depths)
        invalid_task_excl = sorted(d for d in exclude_task_depths if d not in mixed_depths)
        if invalid_task_excl:
            ap.error(
                "--exclude-task-depths contains depths not present in this run: "
                f"{invalid_task_excl}"
            )
        try:
            excluded_tasks_by_depth = _load_task_exclusions_from_meta(
                args.exclude_task_signatures_from_meta, exclude_task_depths
            )
        except ValueError as exc:
            ap.error(str(exc))

    # Initialize RNGs: base_seed fixes the depth-1 pool, bench_seed drives composition.
    key_root = jax.random.PRNGKey(args.bench_seed)
    base_key = jax.random.PRNGKey(args.base_seed)

    # Create base pool for depth-1 items
    # Priority: pool_indices > pool_meta > pool_partition > default
    k_pool = jax.random.fold_in(base_key, 9991)
    perm_all = np.array(jax.random.permutation(k_pool, _DOMAIN_SIZE), dtype=np.int32)

    if pool_indices is not None:
        # Explicit pool uids provided
        base_pool = np.array(pool_indices, dtype=np.int32)
    elif args.pool_meta:
        # Use/exclude items from another dataset's pool
        try:
            meta_pool = _load_pool_uids_from_meta(args.pool_meta)
        except ValueError as exc:
            ap.error(str(exc))
        meta_pool_set = set(meta_pool)
        if args.pool_meta_mode == "superset":
            if args.pool_size < len(meta_pool):
                ap.error("superset mode requires --pool-size >= pool size in meta")
            remaining = [int(u) for u in perm_all.tolist() if int(u) not in meta_pool_set]
            extra_needed = args.pool_size - len(meta_pool)
            base_pool = np.array(meta_pool + remaining[:extra_needed], dtype=np.int32)
        else:
            available = [int(u) for u in perm_all.tolist() if int(u) not in meta_pool_set]
            if args.pool_size > len(available):
                ap.error("disjoint mode requires --pool-size <= remaining pool size")
            base_pool = np.array(available[: args.pool_size], dtype=np.int32)
    elif args.pool_partition is not None:
        if args.pool_partition >= args.num_partitions:
            raise ValueError(
                f"pool_partition ({args.pool_partition}) must be < num_partitions ({args.num_partitions})"
            )
        items_per_part = NUM_RULESET_ITEMS // args.num_partitions
        if items_per_part == 0:
            raise ValueError("pool_partition requires num_partitions <= num_items")
        usable_items = items_per_part * args.num_partitions
        item_start = args.pool_partition * items_per_part
        item_end = item_start + items_per_part
        # Disjoint item types with full color palette
        pool_uids = [
            item * _NUM_REAL_COLORS + color
            for item in range(item_start, item_end)
            for color in range(_NUM_REAL_COLORS)
        ]
        if args.pool_size > len(pool_uids):
            raise ValueError(
                f"pool_size ({args.pool_size}) exceeds partition size "
                f"({len(pool_uids)}) for items[{item_start}:{item_end})"
            )
        pool_set = set(pool_uids)
        candidate_uids = [int(u) for u in perm_all.tolist() if int(u) in pool_set]
        base_pool = np.array(candidate_uids[: args.pool_size], dtype=np.int32)
        print(
            f"Using partition {args.pool_partition}/{args.num_partitions} "
            f"items[{item_start}:{item_end}) "
            f"(usable items: {usable_items}/{NUM_RULESET_ITEMS}, "
            f"colors: 0..{_NUM_REAL_COLORS - 1})"
        )
    else:
        # Default: random pool from base_seed
        base_pool = perm_all[: args.pool_size]

    if len(base_pool) != args.pool_size:
        ap.error("Failed to construct pool with requested --pool-size")

    if args.target_unique_topologies_total > 0:
        depth_caps = _mixed_topology_capacity_per_depth(
            max_depth=int(args.max_depth),
            pool_size=int(args.pool_size),
            tree_topology=str(args.tree_topology),
        )
        max_possible_unique_total = int(
            sum(depth_caps.get(int(d), 0) for d in depths if int(d) >= 2)
        )
        if int(args.target_unique_topologies_total) > max_possible_unique_total:
            ap.error(
                "--target-unique-topologies-total exceeds feasible topology capacity "
                f"for max_depth={args.max_depth}, pool_size={args.pool_size}: "
                f"{args.target_unique_topologies_total} > {max_possible_unique_total}"
            )

        solved_n, est_unique_total, est_unique_per_depth, solved_structure = (
            _solve_n_for_target_unique_total(
                target_unique_total=int(args.target_unique_topologies_total),
                max_depth=int(args.max_depth),
                pool_size=int(args.pool_size),
                base_pool=base_pool,
                base_seed=int(args.base_seed),
                tree_topology=str(args.tree_topology),
                key_root=key_root,
                excluded_topologies_by_depth=excluded_topologies_by_depth,
            )
        )
        args.n = int(solved_n)
        n_highest = int(solved_n)
        structure = solved_structure
        depths = sorted(structure.keys())
        mixed_depths = [d for d in depths if d >= 2]
        print(
            "Auto-selected n for topology target: "
            f"target_unique_total={int(args.target_unique_topologies_total)}, "
            f"n={n_highest}, estimated_unique_total={est_unique_total}"
        )
        print(
            "Estimated unique topologies by depth:",
            {f"d{d}": int(v) for d, v in sorted(est_unique_per_depth.items())},
        )
    else:
        args.n = int(n_highest)

    total_n = sum(structure.values())

    if _is_mixed_topology_mode(args.tree_topology):
        print("=== TOPOLOGY-DIVERSE TASK STRUCTURE ===")
    else:
        print("=== STRICT SUBTREE-CLOSED STRUCTURE ===")
    print(f"Max depth: {args.max_depth}")
    print(f"Pool size: {args.pool_size}")
    print(f"Tree topology: {args.tree_topology}")
    if _is_mixed_topology_mode(args.tree_topology):
        print(
            "Mixed sampler knobs: "
            f"unary_prob={MIXED_UNARY_PROB:.3f}, "
            f"ternary_prob={MIXED_TERNARY_PROB:.3f}, "
            f"unary_prob_depth5="
            f"{'default' if MIXED_UNARY_PROB_DEPTH5 is None else f'{MIXED_UNARY_PROB_DEPTH5:.3f}'}, "
            f"d5_high_leaf_min={MIXED_D5_HIGH_LEAF_MIN}, "
            f"d5_low_leaf_reject_prob={MIXED_D5_LOW_LEAF_REJECT_PROB:.3f}"
        )
    if args.linked_depth_samples:
        print("Generation mode: linked_depth_samples (N at each depth 1/2/3)")
    if _is_mixed_topology_mode(args.tree_topology) and args.mixed_tasks_per_depth_ge3 > 0:
        print(
            "Generation mode: fixed mixed tasks for depths>=3 "
            f"(tasks_per_depth_ge3={int(args.mixed_tasks_per_depth_ge3)})"
        )
    print(f"n (highest depth tasks): {int(args.n)}")
    print("Tasks per depth:")
    for d in depths:
        print(f"  Depth {d}: {structure[d]} tasks")
    if mixed_topology_target_per_depth:
        print(
            "Topology target per mixed depth:",
            {f"d{d}": int(v) for d, v in sorted(mixed_topology_target_per_depth.items())},
        )
    if args.exclude_topologies_from_meta:
        print(
            "Topology exclusions from meta:",
            args.exclude_topologies_from_meta,
        )
        for d in sorted(exclude_topology_depths):
            print(
                f"  Depth {d}: excluding {len(excluded_topologies_by_depth.get(d, set()))} topology signatures"
            )
    if args.exclude_task_signatures_from_meta:
        print(
            "Task-signature exclusions from meta:",
            args.exclude_task_signatures_from_meta,
        )
        for d in sorted(exclude_task_depths):
            print(
                f"  Depth {d}: excluding {len(excluded_tasks_by_depth.get(d, set()))} task signatures"
            )

    linked_depth1_leaves: list[int] | None = None
    linked_depth2_pairs: list[tuple[int, int]] | None = None
    linked_depth3_leaves: list[list[int]] | None = None
    linked_depth3_pairs_by_level: list[list[list[tuple[int, int]]]] | None = None
    if args.linked_depth_samples:
        (
            linked_depth1_leaves,
            linked_depth2_pairs,
            linked_depth3_leaves,
            linked_depth3_pairs_by_level,
        ) = _sample_linked_balanced_depth123(
            n_tasks=int(args.n),
            base_pool=base_pool,
            base_seed=int(args.base_seed),
            key_root=key_root,
        )
        print(
            "Linked sampling prepared: "
            f"d1={len(linked_depth1_leaves)} d2={len(linked_depth2_pairs)} "
            f"d3={len(linked_depth3_leaves)}"
        )

    # Dataset-global distractor rules sampled once from the pool pair universe.
    pool_pairs = [(int(a), int(b)) for a, b in combinations(base_pool.tolist(), 2)]
    distractor_seed = None
    distractor_pairs: list[tuple[int, int]] = []
    if args.distractor_density > 0.0 and pool_pairs:
        k_distr = jax.random.fold_in(key_root, 271828)
        distractor_seed = int(jax.random.randint(k_distr, (), 0, 2**31 - 1))
        rng_distr = np.random.default_rng(distractor_seed)
        keep_mask = rng_distr.random(len(pool_pairs)) < float(args.distractor_density)
        distractor_pairs = [pool_pairs[i] for i, keep in enumerate(keep_mask) if keep]

    distractor_rows = np.zeros((len(distractor_pairs), 6), dtype=np.int32)
    for idx, (uid_a, uid_b) in enumerate(distractor_pairs):
        uid_out = _pair_to_output_uid_global_np(uid_a, uid_b, base_seed=args.base_seed)
        distractor_rows[idx] = _encode_combine_row(
            uid_a, uid_b, uid_out, RULE_TYPE_DISTRACTOR_COMBINE
        )

    distractor_count = int(distractor_rows.shape[0])
    print(
        f"Distractors: density={args.distractor_density:.3f}, "
        f"pairs={distractor_count}/{len(pool_pairs)}"
    )

    # Build distractor lookup table: (NUM_ITEMS, NUM_COLORS, NUM_ITEMS, NUM_COLORS, 3)
    # table[i1, c1, i2, c2] = [out_item, out_color, valid]
    # Distractors are NOT stored as rule rows — they use this table at runtime instead.
    distractor_table = np.zeros(
        (NUM_ITEMS, NUM_COLORS, NUM_ITEMS, NUM_COLORS, 3),
        dtype=np.int32,
    )
    for row in distractor_rows:
        packed = int(row[5])
        i1, c1 = int(row[1]), packed & 0xF
        i2, c2 = int(row[2]), (packed >> 4) & 0xF
        out_i, out_c = int(row[3]), (packed >> 8) & 0xF
        # Populate both orderings (symmetric)
        distractor_table[i1, c1, i2, c2] = [out_i, out_c, 1]
        distractor_table[i2, c2, i1, c1] = [out_i, out_c, 1]

    # R_max = core rules only (distractors handled via lookup table, not rule rows)
    R_per_depth = {d: _rules_per_depth(d, args.tree_topology) for d in depths}
    R_max_core = max(R_per_depth.values())
    R_max = R_max_core

    # Generate filename
    depth_str = "-".join(map(str, depths))
    filename = (
        f"{args.name}_d{depth_str}_n{total_n}_ps{args.pool_size}"
        f"_bs{args.base_seed}_rs{args.bench_seed}"
        f"{_distractor_suffix(args.distractor_density)}"
        ".uint32.npy.bz2"
    )
    out_path = os.path.join(args.out_dir, filename)

    # Libraries to track subtrees
    d2_pairs_lib = []  # list of (uid_a, uid_b) sorted pairs
    d3_subtrees_lib = []  # list of (leaves[4], level0_pairs[2], level1_pair) for depth-3 tasks
    # depth -> list of (leaves, pairs_by_level) harvested from generated balanced rulesets;
    # pairs_by_level[k] holds the sorted (uid_a, uid_b) pairs at combine level k,
    # including the subtree's root pair as the last level
    balanced_subtree_lib: dict[int, list[tuple[list[int], list[list[tuple[int, int]]]]]] = {}

    all_rules = []
    mixed_topology_stats: dict[int, dict] = {}
    mixed_rules_by_depth: dict[int, np.ndarray] = {}

    if _is_mixed_topology_mode(args.tree_topology):
        mixed_depths_to_generate = [int(d) for d in depths if int(d) >= 2]
        if (
            mixed_depths_to_generate
            and MIXED_DEPTH_WORKERS > 1
            and len(mixed_depths_to_generate) > 1
        ):
            workers = min(int(MIXED_DEPTH_WORKERS), len(mixed_depths_to_generate))
            print(
                f"Precomputing mixed depths in parallel: workers={workers}, depths={mixed_depths_to_generate}"
            )
            payloads = []
            for d in mixed_depths_to_generate:
                payloads.append(
                    (
                        int(d),
                        int(structure[d]),
                        np.asarray(base_pool, dtype=np.int32),
                        int(args.base_seed),
                        int(args.bench_seed),
                        int(R_per_depth[d]),
                        int(mixed_topology_target_per_depth.get(int(d), 0)),
                        str(args.tree_topology),
                        set(excluded_topologies_by_depth.get(int(d), set())),
                        set(excluded_tasks_by_depth.get(int(d), set())),
                    )
                )
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("spawn"),
            ) as executor:
                for d, rs_np, stats_d in executor.map(
                    _generate_mixed_rulesets_for_depth_worker, payloads
                ):
                    mixed_rules_by_depth[int(d)] = np.asarray(rs_np, dtype=np.int32)
                    mixed_topology_stats[int(d)] = dict(stats_d)

    def _pair_output(a, b):
        """Compute output UID for a pair using the global hash."""
        return _pair_to_output_uid_global_np(a, b, base_seed=args.base_seed)

    def _choose_k_non_overlapping_pairs(pairs, k, rng, max_tries: int = 256):
        """Return up to k disjoint pairs from `pairs` using rng.

        FIXED: Also checks that the OUTPUT of one pair doesn't equal the INPUT
        of another pair, preventing DAG structures that are unsolvable.

        Rules:
        1. No two pairs share an input UID
        2. No pair's output equals any selected input or output
        3. The final combine's output (hash of all selected outputs)
           doesn't equal any input or intermediate output
        """
        if not pairs or k == 0:
            return []

        for _ in range(max_tries):
            order = rng.permutation(len(pairs))
            used_inputs = set()
            used_outputs = set()
            out = []

            for idx in order:
                a, b = pairs[idx]
                # Check inputs don't overlap with already-used inputs
                if (a in used_inputs) or (b in used_inputs):
                    continue

                # Compute this pair's output
                pair_out = _pair_output(a, b)

                # Check output doesn't collide with any used input or output
                if pair_out in used_inputs or pair_out in used_outputs:
                    continue

                # Check this pair's inputs don't collide with any used output
                # (i.e., we're not using an intermediate as a leaf)
                if a in used_outputs or b in used_outputs:
                    continue

                # This pair is valid - add it
                out.append((a, b))
                used_inputs.add(a)
                used_inputs.add(b)
                used_outputs.add(pair_out)

                if len(out) == k:
                    break

            if len(out) != k:
                continue

            # Final validation: check the level-1 combine (if k >= 2)
            if len(out) >= 2:
                # For depth-3, the two level-0 outputs combine at level-1
                # Check that the level-1 output doesn't collide
                out1 = _pair_output(out[0][0], out[0][1])
                out2 = _pair_output(out[1][0], out[1][1])
                final_out = _pair_output(out1, out2)

                # If final output collides with any input or intermediate, retry
                all_uids = used_inputs | used_outputs
                if final_out in all_uids:
                    continue

            return out

        return []

    def _choose_balanced_subtrees(lib, k_needed, rng):
        """Pick k_needed disjoint subtrees from a harvested balanced-subtree library.

        Each lib entry is (leaves, pairs_by_level). All intermediate outputs are
        recomputed via _pair_output and checked for collisions against already
        used leaves and outputs (same checks as _choose_d3_subtrees).
        """
        if not lib:
            return []
        order = rng.permutation(len(lib))
        chosen = []
        used_leaves = set()
        used_outputs = set()  # Track ALL intermediate outputs

        for idx in order:
            leaves_k, pairs_by_level_k = lib[int(idx)]

            # Check leaf collisions
            if any(u in used_leaves for u in leaves_k):
                continue

            # Compute this subtree's intermediate outputs (all levels, incl. root)
            all_subtree_outputs = {
                _pair_output(a, b) for level_pairs in pairs_by_level_k for a, b in level_pairs
            }

            # Check: outputs don't collide with already-used leaves
            if any(out in used_leaves for out in all_subtree_outputs):
                continue

            # Check: outputs don't collide with already-used outputs
            if any(out in used_outputs for out in all_subtree_outputs):
                continue

            # Check: this subtree's leaves don't collide with used outputs
            if any(leaf in used_outputs for leaf in leaves_k):
                continue

            # Valid subtree - add it
            chosen.append((leaves_k, pairs_by_level_k))
            used_leaves.update(leaves_k)
            used_outputs.update(all_subtree_outputs)

            if len(chosen) == k_needed:
                break

        # Final validation: check the new root combine doesn't collide
        if len(chosen) == 2:
            root1 = _pair_output(*chosen[0][1][-1][0])
            root2 = _pair_output(*chosen[1][1][-1][0])
            new_root = _pair_output(root1, root2)
            if new_root in used_leaves or new_root in used_outputs:
                return []

        return chosen

    def _subtree_uids(leaves, all_pairs):
        """All uids a candidate touches: leaves plus every intermediate output."""
        uids = set(leaves)
        uids.update(_pair_output(a, b) for a, b in all_pairs)
        return uids

    def _lib_entry_uids(entry):
        # Entry shapes: (a, b) d2 pair; (leaves, pairs_4, lvl1_pair) d3
        # subtree; (leaves, pairs_by_level) harvested balanced subtree.
        if len(entry) == 2 and isinstance(entry[0], (int, np.integer)):
            return _subtree_uids(entry, [entry])
        if len(entry) == 3:
            return _subtree_uids(entry[0], entry[1] + [entry[2]])
        return _subtree_uids(entry[0], [p for lvl in entry[1] for p in lvl])

    def _lib_entry_root(entry):
        if len(entry) == 2 and isinstance(entry[0], (int, np.integer)):
            return _pair_output(entry[0], entry[1])
        if len(entry) == 3:
            return _pair_output(entry[2][0], entry[2][1])
        return _pair_output(*entry[1][-1][0])

    def _match_disjoint_units(tagged, n_units, rng, forbidden=frozenset(), max_tries: int = 64):
        """Partition tagged lib entries into n_units uid-disjoint pairs.

        `tagged` is a list of (entry, uid_set). Each unit is 2 entries plus
        their invented root combine; every unit's full uid set must be
        disjoint from all other units' and from `forbidden` (uids already
        claimed by earlier tasks). Returns a list of entry pairs, or []
        if no complete matching is found.
        """
        for _ in range(max_tries):
            order = rng.permutation(len(tagged))
            units = []
            used = set()
            pending = None
            for idx in order:
                e, u = tagged[int(idx)]
                if u & used or (pending is not None and u & pending[1]):
                    continue
                if pending is None:
                    pending = (e, u)
                    continue
                root = _pair_output(_lib_entry_root(pending[0]), _lib_entry_root(e))
                if root in used or root in u or root in pending[1] or root in forbidden:
                    continue
                units.append((pending[0], e, pending[1] | u | {root}, root))
                used |= pending[1] | u | {root}
                pending = None
                if len(units) == n_units:
                    return units
        return []

    def _pair_items(items, n_units, forbidden):
        """Backtracking-pair `items` ((uid_set, root)) into n_units disjoint
        units; returns produced (uid_set, root) list, or None if impossible."""
        used = set()
        picked = [False] * len(items)
        produced = []

        def bt():
            if len(produced) == n_units:
                return True
            for i in range(len(items)):
                ui, ri = items[i]
                if picked[i] or ui & used:
                    continue
                for j in range(i + 1, len(items)):
                    uj, rj = items[j]
                    if picked[j] or uj & used or uj & ui:
                        continue
                    root = _pair_output(ri, rj)
                    if root in used or root in ui or root in uj or root in forbidden:
                        continue
                    unit_u = ui | uj | {root}
                    picked[i] = picked[j] = True
                    used.update(unit_u)
                    produced.append((unit_u, root))
                    if bt():
                        return True
                    produced.pop()
                    used.difference_update(unit_u)
                    picked[i] = picked[j] = False
            return False

        return produced if bt() else None

    def _chain_pairable(items, caps, forbidden):
        """True if `items` can be successively paired into `caps` units per
        level (each produced unit's root avoiding all uids in play)."""
        for n_units in caps:
            produced = _pair_items(items, n_units, forbidden)
            if produced is None:
                return False
            items = produced
        return True

    # Generate rulesets for each depth
    max_depth_signatures = set()
    for d in depths:
        print(f"\nGenerating depth {d}...")
        key_d = jax.random.fold_in(key_root, d)
        n_d = structure[d]
        R_d = R_per_depth[d]
        need_d3_disjoint = d == 3 and structure.get(4, 0) > 0
        disjoint_d3_leaf_sets = []
        # When a higher balanced level exists, the first 2**(max_depth - d)
        # tasks at this depth must be mutually disjoint over ALL uids (leaves
        # AND intermediate outputs) so the next level can embed enough
        # disjoint subtrees (each depth-(d+1) task consumes 2 of them, and
        # 2 disjoint depth-(d+1) tasks are needed one level higher, etc.).
        # Depth-3 keeps its historical leaf-only check for max_depth <= 4.
        balanced_disjoint_cap = 0
        if (
            args.tree_topology == TREE_TOPOLOGY_BALANCED
            and not args.linked_depth_samples
            and structure.get(d + 1, 0) > 0
            and (d >= 4 or (d == 3 and int(args.max_depth) >= 5))
        ):
            balanced_disjoint_cap = min(n_d, 1 << (int(args.max_depth) - d))
        disjoint_task_uid_sets = []
        cap_plan = None

        # Determine leaf pool and fixed leaves
        rs_np = None
        leaf_pool_for_depth = None
        fixed_leaves_list = None
        required_pairs_by_level_list = None

        if args.linked_depth_samples and args.tree_topology == TREE_TOPOLOGY_BALANCED:
            if d == 1:
                assert linked_depth1_leaves is not None
                fixed_leaves_list = [[int(uid)] for uid in linked_depth1_leaves]
                leaf_pool_for_depth = base_pool
            elif d == 2:
                assert linked_depth2_pairs is not None
                fixed_leaves_list = [[int(pair[0]), int(pair[1])] for pair in linked_depth2_pairs]
                leaf_pool_for_depth = base_pool
            elif d == 3:
                assert linked_depth3_leaves is not None
                assert linked_depth3_pairs_by_level is not None
                fixed_leaves_list = [
                    [int(uid) for uid in leaves] for leaves in linked_depth3_leaves
                ]
                required_pairs_by_level_list = linked_depth3_pairs_by_level
                leaf_pool_for_depth = base_pool
            else:
                raise ValueError("--linked-depth-samples currently supports depths 1/2/3 only")

        elif _is_mixed_topology_mode(args.tree_topology) and d >= 2:
            if int(d) in mixed_rules_by_depth:
                rs_np = mixed_rules_by_depth[int(d)]
                stats_d = mixed_topology_stats[int(d)]
            else:
                excluded_d = excluded_topologies_by_depth.get(int(d), set())
                excluded_tasks_d = excluded_tasks_by_depth.get(int(d), set())
                rs_np, stats_d = _generate_mixed_rulesets_for_depth(
                    depth=d,
                    n_tasks=n_d,
                    base_pool=base_pool,
                    base_seed=args.base_seed,
                    tree_topology=args.tree_topology,
                    key_depth=key_d,
                    max_rules_for_depth=R_d,
                    target_unique_topologies=int(mixed_topology_target_per_depth.get(int(d), 0)),
                    excluded_topology_signatures=excluded_d,
                    excluded_task_signatures=excluded_tasks_d,
                )
                assert rs_np is not None
                mixed_topology_stats[int(d)] = stats_d
            unique_d = int(stats_d.get("unique_topologies", 0))
            num_tasks_d = int(stats_d.get("num_tasks", 0))
            unique_frac_d = float(stats_d.get("unique_topology_fraction", 0.0))
            unary_frac_d = float(stats_d.get("unary_node_fraction", 0.0))
            binary_frac_d = float(stats_d.get("binary_node_fraction", 0.0))
            ternary_frac_d = float(stats_d.get("ternary_node_fraction", 0.0))
            excluded_count_d = int(stats_d.get("excluded_topologies", 0))
            excluded_tasks_d = int(stats_d.get("excluded_task_signatures", 0))
            target_eff_d = int(stats_d.get("target_unique_topologies_effective", 0))
            shortfall_d = int(stats_d.get("target_unique_topologies_shortfall", 0))
            relax_d = int(stats_d.get("target_unique_relaxations", 0))
            target_msg = ""
            if target_eff_d > 0:
                target_msg = f", target={unique_d}/{target_eff_d}"
                if shortfall_d > 0:
                    target_msg += f", shortfall={shortfall_d}"
                if relax_d > 0:
                    target_msg += f", relaxed_fills={relax_d}"
            print(
                "  Mixed topology stats: "
                f"unique={unique_d}/{num_tasks_d} ({unique_frac_d:.3f}), "
                f"unary_frac={unary_frac_d:.3f}, binary_frac={binary_frac_d:.3f}, "
                f"ternary_frac={ternary_frac_d:.3f}, "
                f"excluded={excluded_count_d}, excluded_tasks={excluded_tasks_d}{target_msg}"
            )
            if int(d) == 5:
                print(
                    "  Depth 5 leaf_count_distribution:",
                    stats_d.get("leaf_count_distribution", {}),
                )
                print(
                    "  Depth 5 rule_count_distribution:",
                    stats_d.get("rule_count_distribution", {}),
                )

        elif d == 1:
            # Depth 1: use base pool directly
            leaf_pool_for_depth = base_pool
            # For depth 1, we want exactly one task per item in pool
            # So we fix the leaves to be the pool items
            fixed_leaves_list = [[uid] for uid in base_pool[:n_d]]

        elif d == 2:
            # Depth 2: Generate ALL possible pairs from depth-1 (full coverage)
            # This ensures ANY pairing depth-3+ chooses will be valid
            # With 80 items: C(80,2) = 3,160 pairs
            print(f"  Total possible pairs from {len(base_pool)} items: {len(pool_pairs)}")

            # Use the requested n_d pairs (or all if n_d > total)
            if n_d > len(pool_pairs):
                print(
                    f"  WARNING: Requested {n_d} tasks but only {len(pool_pairs)} unique pairs exist"
                )
                print(f"  Using all {len(pool_pairs)} pairs and padding with repeats")
                # Use all pairs, then repeat some to reach n_d
                selected_pairs = pool_pairs * (n_d // len(pool_pairs) + 1)
                selected_pairs = selected_pairs[:n_d]
            else:
                # Randomly select n_d pairs
                k_d2 = jax.random.fold_in(key_d, 99)
                seed_d2 = int(jax.random.randint(k_d2, (), 0, 2**31 - 1))
                rng_d2 = np.random.default_rng(seed_d2)
                selected_indices = rng_d2.choice(len(pool_pairs), size=n_d, replace=False)
                selected_pairs = [pool_pairs[i] for i in selected_indices]

            # Convert pairs to fixed_leaves_list format
            fixed_leaves_list = [[a, b] for a, b in selected_pairs]
            leaf_pool_for_depth = base_pool

        elif d == 3 and args.tree_topology == TREE_TOPOLOGY_ASYM_D2_D3_6L:
            # Asymmetric depth-3 topology with 6 leaves:
            # one branch depth-2, one branch depth-3.
            if len(d2_pairs_lib) < 3:
                raise ValueError(
                    "Need at least 3 depth-2 pairs before generating asym depth-3 trees"
                )

            rs_rows = []
            for i in range(n_d):
                k_i = jax.random.fold_in(key_d, i)
                rows_i = None
                for attempt in range(MAX_TRIES):
                    k_try = jax.random.fold_in(k_i, attempt)
                    seed_i = int(jax.random.randint(k_try, (), 0, 2**31 - 1))
                    rng_i = np.random.default_rng(seed_i)

                    picked = _choose_asym_d2_d3_pairs(d2_pairs_lib, rng_i, _pair_output)
                    if picked is None:
                        continue

                    rows_i = _build_asym_d2_d3_ruleset_from_pairs(
                        picked[0], picked[1], picked[2], _pair_output
                    )
                    try:
                        _verify_single_asym_d2_d3_ruleset(rows_i)
                    except ValueError:
                        rows_i = None
                        continue
                    break

                if rows_i is None:
                    raise ValueError(
                        f"Failed to construct asym depth-3 sample {i} after {MAX_TRIES} tries"
                    )
                rs_rows.append(rows_i)

            rs_np = np.asarray(rs_rows, dtype=np.int32)
            if rs_np.shape != (n_d, R_d, 6):
                raise ValueError(f"Unexpected asym ruleset shape: {rs_np.shape}")

        elif d >= 3:
            # Depth 3+: ALWAYS use complete lower-depth subtrees with EXPLICIT PAIRINGS
            # This is critical: we pass the exact pairings from lower depths to ensure
            # the tree builder uses the same intermediate combines the agent practiced.
            L = 1 << (d - 1)  # number of leaves
            fixed_leaves_list = [None] * n_d
            required_pairs_by_level_list = [None] * n_d

            for i in range(n_d):
                k_i = jax.random.fold_in(key_d, i)
                leaves = None
                required_pairs_by_level = None
                for attempt in range(MAX_TRIES):
                    k_try = jax.random.fold_in(k_i, attempt)
                    seed_i = int(jax.random.randint(k_try, (), 0, 2**31 - 1))
                    rng_i = np.random.default_rng(seed_i)

                    leaves = None
                    required_pairs_by_level = None

                    # While the disjoint cap is unfilled, match the remaining
                    # cap tasks' subtree pairs jointly (each pair's invented
                    # root included) so an early greedy pick can't strand a
                    # self-colliding pair.
                    cap_left = (
                        balanced_disjoint_cap - len(disjoint_task_uid_sets)
                        if balanced_disjoint_cap
                        else 0
                    )
                    if cap_left > 0:
                        if cap_plan is None:
                            used_union = (
                                set().union(*disjoint_task_uid_sets)
                                if disjoint_task_uid_sets
                                else set()
                            )
                            lib = (
                                d2_pairs_lib
                                if d == 3
                                else d3_subtrees_lib
                                if d == 4
                                else balanced_subtree_lib.get(d - 1, [])
                            )
                            tagged = []
                            for e in lib:
                                u = _lib_entry_uids(e)
                                if not (u & used_union):
                                    tagged.append((e, u))
                            # The cap tasks' pairing must stay completable all
                            # the way up, so require the produced units to be
                            # recursively pairable for the remaining levels.
                            caps_ahead = []
                            for d_up in range(d + 1, int(args.max_depth) + 1):
                                if d_up == int(args.max_depth):
                                    caps_ahead.append(min(structure.get(d_up, 0), 1))
                                else:
                                    caps_ahead.append(
                                        min(
                                            structure.get(d_up, 0),
                                            1 << (int(args.max_depth) - d_up),
                                        )
                                    )
                            for _ in range(64):
                                units = _match_disjoint_units(
                                    tagged,
                                    balanced_disjoint_cap,
                                    rng_i,
                                    forbidden=used_union,
                                )
                                if not units:
                                    continue
                                if _chain_pairable(
                                    [(u, r) for _, _, u, r in units],
                                    caps_ahead,
                                    used_union,
                                ):
                                    cap_plan = [(a, b) for a, b, _, _ in units]
                                    break
                        if cap_plan is not None:
                            e1, e2 = cap_plan[len(disjoint_task_uid_sets)]
                            if d == 3:
                                leaves = [e1[0], e1[1], e2[0], e2[1]]
                                required_pairs_by_level = [[e1, e2]]
                            elif d == 4:
                                leaves = e1[0] + e2[0]
                                required_pairs_by_level = [
                                    e1[1] + e2[1],
                                    [e1[2], e2[2]],
                                ]
                            else:
                                leaves = e1[0] + e2[0]
                                required_pairs_by_level = [
                                    e1[1][k] + e2[1][k] for k in range(d - 2)
                                ]

                    elif d == 3:
                        # Depth 3: embed 2 disjoint depth-2 subtrees (pairs)
                        # Pass BOTH leaves AND the exact pairings to preserve subtree-closure consistency
                        if len(d2_pairs_lib) >= 2:
                            pairs = _choose_k_non_overlapping_pairs(d2_pairs_lib, 2, rng_i)
                            if len(pairs) == 2:
                                (a, b), (c, d_leaf) = pairs
                                leaves = [a, b, c, d_leaf]
                                # These are the exact pairings from depth-2 that we want to reuse
                                required_pairs_by_level = [[(a, b), (c, d_leaf)]]

                    elif d == 4:
                        # Depth 4: embed 2 disjoint depth-3 subtrees
                        # Each depth-3 subtree has 4 leaves, 2 first-level pairs, and 1 second-level pair
                        #
                        # BUG FIX: Must validate ALL intermediate outputs, not just leaves.
                        # Previously only checked leaf collisions, missing:
                        # - Level-0 output collisions between subtrees
                        # - Level-1 output collisions between subtrees
                        # - Output collisions with leaves from the other subtree
                        # - Level-2 (final) combine output collisions

                        def _compute_subtree_outputs(pairs_4, lvl1_pair):
                            """Compute all intermediate outputs for a depth-3 subtree."""
                            # Level-0 outputs: combine each leaf pair
                            level0_outs = [_pair_output(a, b) for a, b in pairs_4]
                            # Level-1 output: combine the two level-0 outputs (depth-3 final)
                            level1_out = _pair_output(lvl1_pair[0], lvl1_pair[1])
                            return level0_outs, level1_out

                        def _choose_d3_subtrees(lib, k_needed, rng):
                            if not lib:
                                return []
                            order = rng.permutation(len(lib))
                            chosen = []
                            used_leaves = set()
                            used_outputs = set()  # Track ALL intermediate outputs

                            for idx in order:
                                L4, pairs_4, lvl1_pair = lib[idx]

                                # Check leaf collisions
                                if any(u in used_leaves for u in L4):
                                    continue

                                # Compute this subtree's intermediate outputs
                                level0_outs, level1_out = _compute_subtree_outputs(
                                    pairs_4, lvl1_pair
                                )
                                all_subtree_outputs = set(level0_outs) | {level1_out}

                                # Check: outputs don't collide with already-used leaves
                                if any(out in used_leaves for out in all_subtree_outputs):
                                    continue

                                # Check: outputs don't collide with already-used outputs
                                if any(out in used_outputs for out in all_subtree_outputs):
                                    continue

                                # Check: this subtree's leaves don't collide with used outputs
                                # (prevents DAG structures where intermediate is reused as leaf)
                                if any(leaf in used_outputs for leaf in L4):
                                    continue

                                # Valid subtree - add it
                                chosen.append((L4, pairs_4, lvl1_pair))
                                used_leaves.update(L4)
                                used_outputs.update(all_subtree_outputs)

                                if len(chosen) == k_needed:
                                    break

                            # Final validation: check the level-2 combine doesn't collide
                            if len(chosen) == 2:
                                _, pairs_1, lvl1_1 = chosen[0]
                                _, pairs_2, lvl1_2 = chosen[1]

                                # Get the final outputs of each depth-3 subtree
                                final1 = _pair_output(lvl1_1[0], lvl1_1[1])
                                final2 = _pair_output(lvl1_2[0], lvl1_2[1])

                                # The level-2 combine output
                                level2_out = _pair_output(final1, final2)

                                # Check level-2 output doesn't collide with anything
                                all_used = used_leaves | used_outputs
                                if level2_out in all_used:
                                    # This pairing produces a collision at level-2
                                    # Try to find an alternative (in practice, regenerate)
                                    return []

                            return chosen

                        chosen_d3 = _choose_d3_subtrees(d3_subtrees_lib, 2, rng_i)
                        if len(chosen_d3) == 2:
                            (L4_1, pairs_1, lvl1_1), (L4_2, pairs_2, lvl1_2) = chosen_d3
                            leaves = L4_1 + L4_2  # 8 leaves
                            # Combine the first-level pairs from both depth-3 subtrees
                            level0_pairs = pairs_1 + pairs_2  # 4 leaf-pairs
                            level1_pairs = [
                                lvl1_1,
                                lvl1_2,
                            ]  # 2 pairs of intermediate outputs
                            required_pairs_by_level = [level0_pairs, level1_pairs]

                    elif d >= 5:
                        # Depths 5+: embed 2 disjoint harvested depth-(d-1) subtrees,
                        # concatenating their pairings level-wise so every subtree of
                        # the task is itself a standalone task (strict closure).
                        chosen_subtrees = _choose_balanced_subtrees(
                            balanced_subtree_lib.get(d - 1, []), 2, rng_i
                        )
                        if len(chosen_subtrees) == 2:
                            (L1, lvl_1), (L2, lvl_2) = chosen_subtrees
                            leaves = L1 + L2
                            required_pairs_by_level = [lvl_1[k] + lvl_2[k] for k in range(d - 2)]

                    if (
                        leaves is not None
                        and len(leaves) == L
                        and required_pairs_by_level is not None
                    ):
                        if need_d3_disjoint and len(disjoint_d3_leaf_sets) < 2:
                            leaf_set = set(leaves)
                            if any(leaf_set & used for used in disjoint_d3_leaf_sets):
                                continue
                            disjoint_d3_leaf_sets.append(leaf_set)
                        if (
                            balanced_disjoint_cap
                            and len(disjoint_task_uid_sets) < balanced_disjoint_cap
                        ):
                            uid_set = set(leaves)
                            level_outs = []
                            for lvl in required_pairs_by_level:
                                outs_k = [_pair_output(a, b) for a, b in lvl]
                                uid_set.update(outs_k)
                                level_outs = outs_k
                            # Root combines invented by the tree builder
                            while len(level_outs) > 1:
                                level_outs = [
                                    _pair_output(level_outs[i], level_outs[i + 1])
                                    for i in range(0, len(level_outs) - 1, 2)
                                ]
                                uid_set.update(level_outs)
                            if any(uid_set & used for used in disjoint_task_uid_sets):
                                continue
                            disjoint_task_uid_sets.append(uid_set)
                        if args.enforce_max_depth_unique and d == args.max_depth:
                            sig = _canonical_task_signature(leaves, required_pairs_by_level)
                            if sig in max_depth_signatures:
                                continue
                            max_depth_signatures.add(sig)
                        fixed_leaves_list[i] = leaves
                        required_pairs_by_level_list[i] = required_pairs_by_level
                        break
                else:
                    raise ValueError(
                        f"Failed to construct depth-{d} subtree for sample {i} after {MAX_TRIES} tries; "
                        "check pool_size and depth counts."
                    )

        # Build rulesets (manual branch may have already populated rs_np)
        if rs_np is None:
            rs = build_ruleset_batch(
                key_d,
                n=n_d,
                depth=d,
                base_seed=args.base_seed,
                leaf_uid_pool=leaf_pool_for_depth,
                fixed_leaves_list=fixed_leaves_list,
                required_pairs_by_level_list=required_pairs_by_level_list,
            )
            rs_np = np.array(rs, dtype=np.int32)

        # Pad to R_max (distractors are in the lookup table, not in rule rows)
        if rs_np.shape[1] < R_max:
            pad = np.zeros((n_d, R_max - rs_np.shape[1], 6), dtype=np.int32)
            pad[..., 0] = 5  # no-op rule type
            rs_np = np.concatenate([rs_np, pad], axis=1)

        all_rules.append(rs_np)

        # Collect subtrees for next depth
        if not _is_mixed_topology_mode(args.tree_topology) and d == 2:
            comb_mask = rs_np[..., 0] == RULE_TYPE_COMBINE
            if comb_mask.any():
                comb_rows = rs_np[comb_mask]
                for row in comb_rows:
                    in1_item = int(row[1])
                    in2_item = int(row[2])
                    packed = int(row[5])
                    in1_c = packed & 0xF
                    in2_c = (packed >> 4) & 0xF
                    uid1 = in1_item * _NUM_REAL_COLORS + in1_c
                    uid2 = in2_item * _NUM_REAL_COLORS + in2_c
                    a, b = (uid1, uid2) if uid1 <= uid2 else (uid2, uid1)
                    d2_pairs_lib.append((a, b))

        if not _is_mixed_topology_mode(args.tree_topology) and d == 3:
            # Harvest depth-3 subtrees for depth-4
            # Store both leaves AND the first-level pairings for subtree-closure consistency
            for rs_one in rs_np:
                is_comb = rs_one[:, 0] == RULE_TYPE_COMBINE
                comb_idx = np.where(is_comb)[0]
                if comb_idx.size != 3:
                    continue

                rows = rs_one[comb_idx]
                outs = rows[:, 3]
                ins1 = rows[:, 1]
                ins2 = rows[:, 2]
                packed = rows[:, 5]
                out_c = (packed >> 8) & 0xF
                in1_c = packed & 0xF
                in2_c = (packed >> 4) & 0xF

                # Find final combine (output not used as input)
                used_as_input = np.zeros(outs.shape[0], dtype=bool)
                for i in range(outs.shape[0]):
                    oi, oc = int(outs[i]), int(out_c[i])
                    for j in range(outs.shape[0]):
                        if i == j:
                            continue
                        if (int(ins1[j]) == oi and int(in1_c[j]) == oc) or (
                            int(ins2[j]) == oi and int(in2_c[j]) == oc
                        ):
                            used_as_input[i] = True

                final_rel = int(np.where(~used_as_input)[0][0]) if np.any(~used_as_input) else 0
                child_rel = [k for k in range(3) if k != final_rel]

                # Extract leaves from child subtrees
                fin = rows[final_rel]
                fin_in1 = int(fin[1])
                fin_c1 = int(fin[5] & 0xF)
                child_rows = [rows[child_rel[0]], rows[child_rel[1]]]

                def _out_of(r):
                    return (int(r[3]), int((int(r[5]) >> 8) & 0xF))

                def _out_uid(r):
                    oi, oc = _out_of(r)
                    return oi * _NUM_REAL_COLORS + oc

                if _out_of(child_rows[1]) == (fin_in1, fin_c1):
                    child_rows = [child_rows[1], child_rows[0]]

                leaves_4 = []
                first_level_pairs = []
                child_out_uids = []
                for r in child_rows:
                    p = int(r[5])
                    li1 = int(r[1])
                    lc1 = p & 0xF
                    li2 = int(r[2])
                    lc2 = (p >> 4) & 0xF
                    uid1 = li1 * _NUM_REAL_COLORS + lc1
                    uid2 = li2 * _NUM_REAL_COLORS + lc2
                    leaves_4.extend([uid1, uid2])
                    # Store the pair in sorted order for consistency
                    first_level_pairs.append((min(uid1, uid2), max(uid1, uid2)))
                    child_out_uids.append(_out_uid(r))

                if len(leaves_4) == 4 and len(child_out_uids) == 2:
                    lvl1_pair = (
                        min(child_out_uids[0], child_out_uids[1]),
                        max(child_out_uids[0], child_out_uids[1]),
                    )
                    d3_subtrees_lib.append((leaves_4, first_level_pairs, lvl1_pair))

        if not _is_mixed_topology_mode(args.tree_topology) and d >= 4:
            # Harvest balanced depth-d subtrees for depth-(d+1).
            # Parse the generated rows (source of truth) rather than what we asked
            # for, since build_ruleset can fall back to pool sampling.
            lib_d = balanced_subtree_lib.setdefault(d, [])
            expected_combines = (1 << (d - 1)) - 1
            expected_leaves = 1 << (d - 1)
            for rs_one in rs_np:
                comb_rows = rs_one[rs_one[:, 0] == RULE_TYPE_COMBINE]
                if comb_rows.shape[0] != expected_combines:
                    continue
                children_map = {}  # out_uid -> sorted (in1_uid, in2_uid)
                outputs = set()
                inputs = set()
                valid = True
                for row in comb_rows:
                    in1_uid, in2_uid, out_uid = _decode_combine_row_uids(row)
                    if out_uid in children_map:
                        valid = False
                        break
                    children_map[out_uid] = (
                        min(in1_uid, in2_uid),
                        max(in1_uid, in2_uid),
                    )
                    outputs.add(out_uid)
                    inputs.update((in1_uid, in2_uid))
                if not valid:
                    continue
                leaves_set = inputs - outputs
                roots = outputs - inputs
                if len(leaves_set) != expected_leaves or len(roots) != 1:
                    continue
                # Bottom-up heights (leaf = 0); skip on cycles/dangling nodes
                height = {u: 0 for u in leaves_set}
                remaining = dict(children_map)
                progress = True
                while remaining and progress:
                    progress = False
                    for out_uid, (a, b) in list(remaining.items()):
                        if a in height and b in height:
                            height[out_uid] = 1 + max(height[a], height[b])
                            del remaining[out_uid]
                            progress = True
                if remaining:
                    continue
                # Perfectly balanced: both children of every node at equal height
                if any(height[a] != height[b] for a, b in children_map.values()):
                    continue
                pairs_by_level = [[] for _ in range(d - 1)]
                for out_uid, pair in children_map.items():
                    pairs_by_level[height[out_uid] - 1].append(pair)
                if any(len(pairs_by_level[k]) != (1 << (d - 2 - k)) for k in range(d - 1)):
                    continue
                lib_d.append((list(leaves_set), pairs_by_level))

        print(f"  Generated {n_d} rulesets for depth {d}")
        if not _is_mixed_topology_mode(args.tree_topology) and d == 2:
            print(f"  Collected {len(d2_pairs_lib)} depth-2 pairs for depth-3")
        if not _is_mixed_topology_mode(args.tree_topology) and d == 3:
            print(f"  Collected {len(d3_subtrees_lib)} depth-3 subtrees for depth-4")
        if not _is_mixed_topology_mode(args.tree_topology) and d >= 4:
            print(
                f"  Collected {len(balanced_subtree_lib[d])} depth-{d} subtrees for depth-{d + 1}"
            )

    # Concatenate all rules
    rules = np.concatenate(all_rules, axis=0)

    if args.tree_topology == TREE_TOPOLOGY_ASYM_D2_D3_6L:
        offset = 0
        checked = 0
        for d in depths:
            n_d = int(structure[d])
            next_offset = offset + n_d
            if d == 3:
                for i in range(offset, next_offset):
                    _verify_single_asym_d2_d3_ruleset(rules[i])
                    checked += 1
            offset = next_offset
        print(f"Verified {checked} asymmetric depth-3 rulesets")

    # Save metadata
    meta = {
        "n": total_n,
        "structure": structure,
        "max_depth": args.max_depth,
        "tree_topology": args.tree_topology,
        "rules_per_depth": {str(k): int(v) for k, v in R_per_depth.items()},
        "pool_size": args.pool_size,
        "pool_uids": [int(u) for u in base_pool.tolist()],
        "base_seed": args.base_seed,
        "bench_seed": args.bench_seed,
        "base_seed_in_pool": True,
        "rule_shape": [R_max, 6],
        "generation_mode": "strict",
        "description": "Every subtree at depth D appears as standalone task at depth D",
        "distractor_density": float(args.distractor_density),
        "distractor_table": True,
        "distractor_table_shape": list(distractor_table.shape),
        "distractor_pair_universe": len(pool_pairs),
        "distractor_pair_count": distractor_count,
        "distractor_seed": distractor_seed,
        "distractor_rule_type": int(RULE_TYPE_DISTRACTOR_COMBINE),
        "enforce_max_depth_unique": bool(args.enforce_max_depth_unique),
    }
    if args.linked_depth_samples:
        meta["generation_mode"] = "linked_depth_samples"
        meta["description"] = (
            "Depth-1/2/3 each use N tasks sampled from linked chains: each depth-3 "
            "task contributes one depth-2 subtree task and one depth-1 leaf task."
        )
        meta["linked_depth_samples"] = True
        meta["linked_depth_sample_count_per_depth"] = int(args.n)
    if _is_mixed_topology_mode(args.tree_topology):
        meta["generation_mode"] = "topology_diverse"
        if _mixed_mode_supports_ternary(args.tree_topology):
            meta["description"] = (
                "Topology-diverse unary/binary/ternary trees by depth "
                "(exact max depth per bucket), unordered children within each arity, "
                "no global inverse consistency."
            )
            meta["topology_mode"] = "mixed_unary_binary_ternary_unordered"
            meta["topology_node_types"] = {
                "leaf": 0,
                "transform": 1,
                "combine_binary": 2,
                "combine_ternary": 3,
            }
        else:
            meta["description"] = (
                "Topology-diverse unary/binary trees by depth (exact max depth per bucket), "
                "unordered binary children, no global inverse consistency."
            )
            meta["topology_mode"] = "mixed_unary_binary_unordered"
            meta["topology_node_types"] = {"leaf": 0, "transform": 1, "combine": 2}
        if args.mixed_tasks_per_depth_ge3 > 0:
            meta["mixed_tasks_per_depth_ge3"] = int(args.mixed_tasks_per_depth_ge3)
            meta["mixed_tasks_schedule"] = "depth1_pool_depth2_all_pairs_depth3plus_fixed"
        if args.target_topologies_per_depth > 0:
            meta["target_topologies_per_depth_requested"] = int(args.target_topologies_per_depth)
            meta["target_topologies_per_depth_scope"] = "per_mixed_depth"
            if target_topology_depths:
                meta["target_topologies_depths"] = sorted(int(d) for d in target_topology_depths)
        meta["mixed_sampler_knobs"] = {
            "unary_prob": float(MIXED_UNARY_PROB),
            "ternary_prob": float(MIXED_TERNARY_PROB),
            "unary_prob_depth5": (
                None if MIXED_UNARY_PROB_DEPTH5 is None else float(MIXED_UNARY_PROB_DEPTH5)
            ),
            "depth5_high_leaf_min": int(MIXED_D5_HIGH_LEAF_MIN),
            "depth5_low_leaf_reject_prob": float(MIXED_D5_LOW_LEAF_REJECT_PROB),
            "strict_unique_tries": int(STRICT_UNIQUE_TRIES),
            "max_tries": int(MAX_TRIES),
        }
        meta["topology_unique_per_depth"] = {
            str(d): int(stats.get("unique_topologies", 0))
            for d, stats in mixed_topology_stats.items()
        }
        meta["topology_unique_fraction_per_depth"] = {
            str(d): float(stats.get("unique_topology_fraction", 0.0))
            for d, stats in mixed_topology_stats.items()
        }
        meta["topology_unary_fraction_per_depth"] = {
            str(d): float(stats.get("unary_node_fraction", 0.0))
            for d, stats in mixed_topology_stats.items()
        }
        meta["topology_binary_fraction_per_depth"] = {
            str(d): float(stats.get("binary_node_fraction", 0.0))
            for d, stats in mixed_topology_stats.items()
        }
        meta["topology_ternary_fraction_per_depth"] = {
            str(d): float(stats.get("ternary_node_fraction", 0.0))
            for d, stats in mixed_topology_stats.items()
        }
        meta["topology_leaf_count_distribution_per_depth"] = {
            str(d): stats.get("leaf_count_distribution", {})
            for d, stats in mixed_topology_stats.items()
        }
        meta["topology_rule_count_distribution_per_depth"] = {
            str(d): stats.get("rule_count_distribution", {})
            for d, stats in mixed_topology_stats.items()
        }
        meta["topology_signatures_per_depth"] = {
            str(d): stats.get("topology_signatures", [])
            for d, stats in mixed_topology_stats.items()
        }
        meta["task_signatures_per_depth"] = {
            str(d): stats.get("task_signatures", []) for d, stats in mixed_topology_stats.items()
        }
        if args.target_topologies_per_depth > 0:
            meta["target_topologies_per_depth_effective"] = {
                str(d): int(stats.get("target_unique_topologies_effective", 0))
                for d, stats in mixed_topology_stats.items()
            }
            meta["target_topologies_per_depth_achieved"] = {
                str(d): int(stats.get("target_unique_topologies_achieved", 0))
                for d, stats in mixed_topology_stats.items()
            }
            meta["target_topologies_per_depth_shortfall"] = {
                str(d): int(stats.get("target_unique_topologies_shortfall", 0))
                for d, stats in mixed_topology_stats.items()
            }
            meta["target_topologies_per_depth_met"] = {
                str(d): bool(stats.get("target_unique_topologies_met", False))
                for d, stats in mixed_topology_stats.items()
            }
        if args.exclude_topologies_from_meta:
            meta["topology_exclusion_source_meta"] = args.exclude_topologies_from_meta
            meta["topology_exclusion_depths"] = sorted(int(d) for d in exclude_topology_depths)
            meta["topology_excluded_count_per_depth"] = {
                str(d): int(len(excluded_topologies_by_depth.get(d, set())))
                for d in sorted(exclude_topology_depths)
            }
        if args.exclude_task_signatures_from_meta:
            meta["task_exclusion_source_meta"] = args.exclude_task_signatures_from_meta
            meta["task_exclusion_depths"] = sorted(int(d) for d in exclude_task_depths)
            meta["task_excluded_count_per_depth"] = {
                str(d): int(len(excluded_tasks_by_depth.get(d, set())))
                for d in sorted(exclude_task_depths)
            }
        if 5 in mixed_topology_stats:
            d5_stats = mixed_topology_stats[5]
            meta["topology_depth5_leaf_count_distribution"] = d5_stats.get(
                "leaf_count_distribution", {}
            )
            meta["topology_depth5_rule_count_distribution"] = d5_stats.get(
                "rule_count_distribution", {}
            )
            meta["topology_depth5_unary_fraction"] = float(d5_stats.get("unary_node_fraction", 0.0))
            meta["topology_depth5_binary_fraction"] = float(
                d5_stats.get("binary_node_fraction", 0.0)
            )
            meta["topology_depth5_ternary_fraction"] = float(
                d5_stats.get("ternary_node_fraction", 0.0)
            )
            meta["topology_depth5_unique_topologies"] = int(d5_stats.get("unique_topologies", 0))
    if args.tree_topology == TREE_TOPOLOGY_ASYM_D2_D3_6L:
        meta["asym_depth3_leaf_depths"] = [2, 2, 3, 3, 3, 3]
    if args.pool_partition is not None:
        meta["pool_partition"] = args.pool_partition
        meta["num_partitions"] = args.num_partitions

    mixed_unique_total = int(
        sum(int(stats.get("unique_topologies", 0)) for stats in mixed_topology_stats.values())
    )
    if _is_mixed_topology_mode(args.tree_topology):
        if args.target_unique_topologies_total > 0:
            meta["target_unique_topologies_total"] = int(args.target_unique_topologies_total)
            meta["target_unique_topologies_scope"] = "sum_depth2_to_max"
            meta["target_unique_topologies_achieved"] = mixed_unique_total
            if mixed_unique_total < int(args.target_unique_topologies_total):
                raise ValueError(
                    "Generated dataset did not reach target unique topologies: "
                    f"achieved={mixed_unique_total}, "
                    f"target={int(args.target_unique_topologies_total)}"
                )

    if _is_mixed_topology_mode(args.tree_topology) and mixed_topology_stats:
        print("\n=== MIXED TOPOLOGY SUMMARY ===")
        for depth in sorted(mixed_topology_stats):
            stats_d = mixed_topology_stats[depth]
            unique_d = int(stats_d.get("unique_topologies", 0))
            num_tasks_d = int(stats_d.get("num_tasks", 0))
            unique_frac_d = float(stats_d.get("unique_topology_fraction", 0.0))
            unary_frac_d = float(stats_d.get("unary_node_fraction", 0.0))
            binary_frac_d = float(stats_d.get("binary_node_fraction", 0.0))
            ternary_frac_d = float(stats_d.get("ternary_node_fraction", 0.0))
            excluded_count_d = int(stats_d.get("excluded_topologies", 0))
            excluded_tasks_d = int(stats_d.get("excluded_task_signatures", 0))
            target_eff_d = int(stats_d.get("target_unique_topologies_effective", 0))
            shortfall_d = int(stats_d.get("target_unique_topologies_shortfall", 0))
            relax_d = int(stats_d.get("target_unique_relaxations", 0))
            target_msg = ""
            if target_eff_d > 0:
                target_msg = f", target={unique_d}/{target_eff_d}"
                if shortfall_d > 0:
                    target_msg += f", shortfall={shortfall_d}"
                if relax_d > 0:
                    target_msg += f", relaxed_fills={relax_d}"
            print(
                f"  depth {depth}: unique={unique_d}/{num_tasks_d} "
                f"({unique_frac_d:.3f}), unary_frac={unary_frac_d:.3f}, "
                f"binary_frac={binary_frac_d:.3f}, ternary_frac={ternary_frac_d:.3f}, "
                f"excluded={excluded_count_d}, "
                f"excluded_tasks={excluded_tasks_d}{target_msg}"
            )
            if int(depth) == 5:
                print(
                    "    depth5 leaf_count_distribution:",
                    stats_d.get("leaf_count_distribution", {}),
                )
                print(
                    "    depth5 rule_count_distribution:",
                    stats_d.get("rule_count_distribution", {}),
                )
        print(f"  total_unique_depth2_plus: {mixed_unique_total}")

    meta_name = os.path.splitext(os.path.splitext(filename)[0])[0]
    meta_path = os.path.join(args.out_dir, f"{meta_name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Pack and save rules
    packed = pack_rules_uint32_np(np.array(rules, dtype=np.int32)).astype(np.uint32)
    with bz2.BZ2File(out_path, "wb") as f:
        np.save(f, packed, allow_pickle=False)

    # Save distractor lookup table
    if distractor_count > 0:
        table_path = os.path.join(args.out_dir, f"{meta_name}_distractor_table.npy")
        np.save(table_path, distractor_table)
        print(f"  Distractor table: {table_path} (shape={distractor_table.shape})")
    else:
        table_path = None

    print(f"\nR_max: {R_max} (core rules only, distractors in lookup table)")
    print("\n=== COMPLETE ===")
    print(f"Total rulesets: {total_n}")
    print(f"  Output: {out_path}")
    print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    main()
