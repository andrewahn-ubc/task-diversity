#!/usr/bin/env python3
"""
Round-disjoint depth-6 dataset generator.

This generator is intentionally different from tools/make_ruleset_dataset.py:

- It creates EXACTLY n tasks at every depth 1..6.
- It samples n exact-depth-6 trees for each round.
- For each sampled depth-6 tree, it extracts one strict subtree of exact depth
  5, 4, 3, 2, and 1 along a deepest-path chain, so every lower-depth task is a
  strict subset of a depth-6 task from the same round.
- It generates multiple rounds (default: 10).
- Within one n-run, it guarantees across rounds:
    * no producer signature repeats across rounds;
    * no depth-6 topology repeats across rounds;
    * reuse within the same round is allowed.

Producer signature means:
- unary transform: ("U", in_uid)
- binary combine: ("B", min(uid_a, uid_b), max(uid_a, uid_b))
- ternary combine: ("T", sorted(uid_a, uid_b, uid_c))

Notes:
- The packed codec currently reconstructs compact outputs with a FIXED base seed
  of 0 when item ids 8/9 are present. With the full 120-uid domain, this file
  therefore defaults to --base-seed 0 and validates that choice.
- Unary transforms are supported, but the default configuration disables them,
  because the unary signature space is only 120 and becomes a bottleneck for
  large n.
- The default sampler now targets a more balanced unary/binary/ternary mix.
- Same-round reuse is allowed, but new unary/binary signatures introduced by a
  round are budgeted so later rounds still have disjoint signatures available.

Example:
  python tools/make_depth6_round_disjoint_dataset.py \
    --n 512 \
    --rounds 10 \
    --out-dir ./round_disjoint_depth6

  python tools/make_depth6_round_disjoint_dataset.py \
    --n-values 1,2,4,6,8,10,16,32,64,128,256,512 \
    --rounds 10 \
    --out-dir ./round_disjoint_depth6
"""

from __future__ import annotations

import argparse
import bz2
from collections import Counter
from dataclasses import dataclass
import json
from math import comb
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from banyan_grid.environment.constants import (
    NUM_RULESET_ITEMS,
    RULE_TYPE_COLLECT,
    RULE_TYPE_COMBINE,
    RULE_TYPE_TERNARY_COMBINE,
    RULE_TYPE_TRANSFORM,
    Colors,
)
from banyan_grid.tasks.ruleset_codec import (
    pack_rules_uint32_np,
)
from banyan_grid.tasks.ruleset_factory import (
    _ITEM_TO_TILE,
    _pair_to_output_uid_global_np,
    _triple_to_output_uid_global_np,
    _unary_to_output_uid_global_np,
)

NOOP_RULE_TYPE = 5
DEFAULT_N_VALUES = [1, 2, 4, 6, 8, 10, 16, 32, 64, 128, 256, 512]
_NUM_REAL_COLORS = int(Colors.BLACK)
_DOMAIN_SIZE = int(NUM_RULESET_ITEMS) * _NUM_REAL_COLORS
LEGACY_CODEC_DOMAIN_SIZE = 8 * _NUM_REAL_COLORS  # compact codec kicks in beyond this.


Topology = tuple
OperationSignature = tuple


@dataclass(frozen=True)
class InstantiatedNode:
    kind: str
    depth: int
    uid: int
    children: tuple["InstantiatedNode", ...]
    row: np.ndarray | None
    topology_key: tuple


@dataclass(frozen=True)
class InstantiatedTree:
    root: InstantiatedNode
    operation_signatures: tuple[OperationSignature, ...]
    unary_count: int
    binary_count: int
    ternary_count: int
    leaf_count: int
    rule_count: int
    topology_signature: str


class SamplingError(RuntimeError):
    """Raised when a tree or round could not be sampled under the constraints."""


def _parse_csv_ints(text: str) -> list[int]:
    values: list[int] = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        values.append(int(part))
    return values


def _uid_to_item_color(uid: int) -> tuple[int, int]:
    return int(uid) // _NUM_REAL_COLORS, int(uid) % _NUM_REAL_COLORS


def _item_color_to_uid(item: int, color: int) -> int:
    return int(item) * _NUM_REAL_COLORS + int(color)


def _pack_colors(c1: int, c2: int, c_out: int) -> int:
    return ((c_out & 0xF) << 8) | ((c2 & 0xF) << 4) | (c1 & 0xF)


def _pack_ternary_colors(c1: int, c2: int, c3: int, c_out: int) -> int:
    return (
        ((c_out & 0xF) << 12)
        | ((c3 & 0xF) << 8)
        | ((c2 & 0xF) << 4)
        | (c1 & 0xF)
    )


def _encode_collect_row(uid: int) -> np.ndarray:
    item, color = _uid_to_item_color(int(uid))
    tile_type = int(np.asarray(_ITEM_TO_TILE)[item])
    return np.array(
        [
            int(RULE_TYPE_COLLECT),
            tile_type,
            item,
            color,
            0,
            0,
        ],
        dtype=np.int32,
    )


def _encode_transform_row(uid_in: int, uid_out: int) -> np.ndarray:
    in_item, in_color = _uid_to_item_color(int(uid_in))
    out_item, out_color = _uid_to_item_color(int(uid_out))
    return np.array(
        [
            int(RULE_TYPE_TRANSFORM),
            int(in_item),
            0,
            int(out_item),
            1,
            _pack_colors(in_color, 0, out_color),
        ],
        dtype=np.int32,
    )


def _encode_combine_row(uid_a: int, uid_b: int, uid_out: int) -> np.ndarray:
    a, b = sorted((int(uid_a), int(uid_b)))
    in1_item, in1_color = _uid_to_item_color(a)
    in2_item, in2_color = _uid_to_item_color(b)
    out_item, out_color = _uid_to_item_color(int(uid_out))
    return np.array(
        [
            int(RULE_TYPE_COMBINE),
            int(in1_item),
            int(in2_item),
            int(out_item),
            1,
            _pack_colors(in1_color, in2_color, out_color),
        ],
        dtype=np.int32,
    )


def _encode_ternary_combine_row(
    uid_a: int, uid_b: int, uid_c: int, uid_out: int
) -> np.ndarray:
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


def _topology_key(node: Topology) -> tuple:
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
    if kind == "T":
        child_keys = sorted(
            [_topology_key(node[1]), _topology_key(node[2]), _topology_key(node[3])]
        )
        return ("T", child_keys[0], child_keys[1], child_keys[2])
    raise ValueError(f"Unsupported topology kind: {kind!r}")


def _topology_depth(node: Topology) -> int:
    kind = node[0]
    if kind == "L":
        return 1
    if kind == "U":
        return 1 + _topology_depth(node[1])
    if kind == "B":
        return 1 + max(_topology_depth(node[1]), _topology_depth(node[2]))
    if kind == "T":
        return 1 + max(
            _topology_depth(node[1]),
            _topology_depth(node[2]),
            _topology_depth(node[3]),
        )
    raise ValueError(f"Unsupported topology kind: {kind!r}")


def _topology_leaf_count(node: Topology) -> int:
    kind = node[0]
    if kind == "L":
        return 1
    if kind == "U":
        return _topology_leaf_count(node[1])
    if kind == "B":
        return _topology_leaf_count(node[1]) + _topology_leaf_count(node[2])
    if kind == "T":
        return (
            _topology_leaf_count(node[1])
            + _topology_leaf_count(node[2])
            + _topology_leaf_count(node[3])
        )
    raise ValueError(f"Unsupported topology kind: {kind!r}")


def _topology_rule_count(node: Topology) -> int:
    kind = node[0]
    if kind == "L":
        return 0
    if kind == "U":
        return 1 + _topology_rule_count(node[1])
    if kind == "B":
        return 1 + _topology_rule_count(node[1]) + _topology_rule_count(node[2])
    if kind == "T":
        return (
            1
            + _topology_rule_count(node[1])
            + _topology_rule_count(node[2])
            + _topology_rule_count(node[3])
        )
    raise ValueError(f"Unsupported topology kind: {kind!r}")


def _topology_internal_counts(node: Topology) -> tuple[int, int, int]:
    kind = node[0]
    if kind == "L":
        return 0, 0, 0
    if kind == "U":
        u_child, b_child, t_child = _topology_internal_counts(node[1])
        return 1 + u_child, b_child, t_child
    if kind == "B":
        u_left, b_left, t_left = _topology_internal_counts(node[1])
        u_right, b_right, t_right = _topology_internal_counts(node[2])
        return u_left + u_right, 1 + b_left + b_right, t_left + t_right
    if kind == "T":
        u1, b1, t1 = _topology_internal_counts(node[1])
        u2, b2, t2 = _topology_internal_counts(node[2])
        u3, b3, t3 = _topology_internal_counts(node[3])
        return u1 + u2 + u3, b1 + b2 + b3, 1 + t1 + t2 + t3
    raise ValueError(f"Unsupported topology kind: {kind!r}")


def _topology_key_to_jsonable(node_key: tuple | str) -> list | str:
    if isinstance(node_key, tuple):
        return [_topology_key_to_jsonable(x) for x in node_key]
    return str(node_key)


def _topology_signature(node: Topology | tuple) -> str:
    topo_key = _topology_key(node)  # type: ignore[arg-type]
    return json.dumps(
        _topology_key_to_jsonable(topo_key),
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _operation_signature_unary(uid_in: int) -> OperationSignature:
    return ("U", int(uid_in))


def _operation_signature_binary(uid_a: int, uid_b: int) -> OperationSignature:
    a, b = sorted((int(uid_a), int(uid_b)))
    return ("B", a, b)


def _operation_signature_ternary(uid_a: int, uid_b: int, uid_c: int) -> OperationSignature:
    ordered = tuple(sorted((int(uid_a), int(uid_b), int(uid_c))))
    return ("T",) + ordered


def _max_binary_nodes_possible(depth: int) -> int:
    if depth <= 1:
        return 0
    return (1 << (depth - 1)) - 1


def _max_unary_nodes_possible(depth: int) -> int:
    return max(0, int(depth) - 1)


def _split_budget(
    total: int,
    child_depths: Sequence[int],
    cap_fn,
    rng: np.random.Generator,
) -> list[int]:
    allocation = [0 for _ in child_depths]
    capacities = [int(cap_fn(int(d))) for d in child_depths]
    if total <= 0:
        return allocation
    if sum(capacities) < total:
        raise SamplingError(
            f"Cannot allocate budget {total} across child depths {list(child_depths)}"
        )
    for _ in range(int(total)):
        eligible = [
            i for i, cap in enumerate(capacities) if allocation[i] < int(cap)
        ]
        if not eligible:
            raise SamplingError(
                f"Budget allocation unexpectedly failed for child depths {list(child_depths)}"
            )
        chosen = int(rng.choice(np.asarray(eligible, dtype=np.int32)))
        allocation[chosen] += 1
    return allocation


def _sample_side_depth(
    parent_depth: int,
    side_depth_cap: int,
    deep_side_prob: float,
    rng: np.random.Generator,
) -> int:
    max_side = max(1, min(int(side_depth_cap), int(parent_depth) - 1))
    if int(parent_depth) > 3 and float(rng.random()) < float(deep_side_prob):
        return int(rng.integers(1, int(parent_depth)))
    return int(rng.integers(1, max_side + 1))


def _sample_budgeted_topology_exact(
    depth: int,
    *,
    remaining_unary: int,
    remaining_binary: int,
    unary_root_weight: float,
    binary_root_weight: float,
    side_depth_cap: int,
    deep_side_prob: float,
    rng: np.random.Generator,
) -> Topology:
    """Sample a mixed exact-depth topology under unary/binary budgets.

    The sampler is intentionally ternary-heavy by default. Budgets cap the total
    number of unary and binary nodes in each exact-depth-6 tree, which prevents
    the small unary/binary signature spaces from being exhausted.
    """
    depth = int(depth)
    if depth <= 1:
        return ("L",)

    kinds = ["T"]
    weights = [max(1e-9, 1.0 - float(unary_root_weight) - float(binary_root_weight))]
    if remaining_binary > 0:
        kinds.append("B")
        weights.append(max(0.0, float(binary_root_weight)))
    if remaining_unary > 0:
        kinds.append("U")
        weights.append(max(0.0, float(unary_root_weight)))

    weight_sum = float(sum(weights))
    probs = np.asarray([w / weight_sum for w in weights], dtype=np.float64)
    kind = str(rng.choice(np.asarray(kinds, dtype=object), p=probs))

    if kind == "U":
        child = _sample_budgeted_topology_exact(
            depth - 1,
            remaining_unary=remaining_unary - 1,
            remaining_binary=remaining_binary,
            unary_root_weight=unary_root_weight,
            binary_root_weight=binary_root_weight,
            side_depth_cap=side_depth_cap,
            deep_side_prob=deep_side_prob,
            rng=rng,
        )
        return ("U", child)

    if kind == "B":
        side_depth = _sample_side_depth(depth, side_depth_cap, deep_side_prob, rng)
        child_depths = [depth - 1, side_depth]
        rng.shuffle(child_depths)
        unary_alloc = _split_budget(
            max(0, int(remaining_unary)),
            child_depths,
            _max_unary_nodes_possible,
            rng,
        )
        binary_alloc = _split_budget(
            max(0, int(remaining_binary) - 1),
            child_depths,
            _max_binary_nodes_possible,
            rng,
        )
        children = []
        for idx, child_depth in enumerate(child_depths):
            children.append(
                _sample_budgeted_topology_exact(
                    int(child_depth),
                    remaining_unary=int(unary_alloc[idx]),
                    remaining_binary=int(binary_alloc[idx]),
                    unary_root_weight=unary_root_weight,
                    binary_root_weight=binary_root_weight,
                    side_depth_cap=side_depth_cap,
                    deep_side_prob=deep_side_prob,
                    rng=rng,
                )
            )
        children.sort(key=_topology_key)
        return ("B", children[0], children[1])

    child_depths = [
        depth - 1,
        _sample_side_depth(depth, side_depth_cap, deep_side_prob, rng),
        _sample_side_depth(depth, side_depth_cap, deep_side_prob, rng),
    ]
    rng.shuffle(child_depths)
    unary_alloc = _split_budget(
        max(0, int(remaining_unary)),
        child_depths,
        _max_unary_nodes_possible,
        rng,
    )
    binary_alloc = _split_budget(
        max(0, int(remaining_binary)),
        child_depths,
        _max_binary_nodes_possible,
        rng,
    )
    children = []
    for idx, child_depth in enumerate(child_depths):
        children.append(
            _sample_budgeted_topology_exact(
                int(child_depth),
                remaining_unary=int(unary_alloc[idx]),
                remaining_binary=int(binary_alloc[idx]),
                unary_root_weight=unary_root_weight,
                binary_root_weight=binary_root_weight,
                side_depth_cap=side_depth_cap,
                deep_side_prob=deep_side_prob,
                rng=rng,
            )
        )
    children.sort(key=_topology_key)
    return ("T", children[0], children[1], children[2])


def _instantiate_topology(
    topology: Topology,
    *,
    base_pool: np.ndarray,
    base_seed: int,
    blocked_operation_signatures: set[OperationSignature],
    round_operation_signatures: set[OperationSignature],
    remaining_new_signatures_by_kind: dict[str, int],
    rng: np.random.Generator,
) -> InstantiatedTree | None:
    leaf_count = _topology_leaf_count(topology)
    if leaf_count < 1 or leaf_count > int(base_pool.shape[0]):
        return None

    sampled = rng.choice(base_pool, size=leaf_count, replace=False)
    leaves = [int(u) for u in np.asarray(sampled, dtype=np.int32).tolist()]
    leaf_iter = iter(leaves)
    leaf_set = set(leaves)
    produced_uids: set[int] = set()
    tree_operation_signatures: list[OperationSignature] = []
    tree_operation_signature_set: set[OperationSignature] = set()
    new_signature_budget = {
        str(kind): int(remaining_new_signatures_by_kind.get(str(kind), 0))
        for kind in ("unary", "binary", "ternary")
    }

    def can_use_operation_signature(op_sig: OperationSignature) -> bool:
        if op_sig in blocked_operation_signatures:
            return False
        if op_sig in tree_operation_signature_set:
            return False
        if op_sig in round_operation_signatures:
            return True

        kind = str(op_sig[0])
        budget_key = {"U": "unary", "B": "binary", "T": "ternary"}.get(kind)
        if budget_key is None:
            raise ValueError(f"Unsupported operation signature kind: {kind!r}")
        if int(new_signature_budget[budget_key]) <= 0:
            return False
        new_signature_budget[budget_key] -= 1
        return True

    def walk(node: Topology) -> InstantiatedNode | None:
        kind = str(node[0])
        if kind == "L":
            uid = int(next(leaf_iter))
            return InstantiatedNode(
                kind="L",
                depth=1,
                uid=uid,
                children=(),
                row=None,
                topology_key=("L",),
            )

        if kind == "U":
            child = walk(node[1])
            if child is None:
                return None
            op_sig = _operation_signature_unary(int(child.uid))
            if not can_use_operation_signature(op_sig):
                return None
            out_uid = int(_unary_to_output_uid_global_np(int(child.uid), base_seed=base_seed))
            if out_uid == int(child.uid) or out_uid in leaf_set or out_uid in produced_uids:
                return None
            produced_uids.add(out_uid)
            tree_operation_signature_set.add(op_sig)
            tree_operation_signatures.append(op_sig)
            return InstantiatedNode(
                kind="U",
                depth=1 + int(child.depth),
                uid=out_uid,
                children=(child,),
                row=_encode_transform_row(int(child.uid), out_uid),
                topology_key=_topology_key(node),
            )

        if kind == "B":
            left = walk(node[1])
            if left is None:
                return None
            right = walk(node[2])
            if right is None:
                return None
            a, b = sorted((int(left.uid), int(right.uid)))
            if a == b:
                return None
            op_sig = _operation_signature_binary(a, b)
            if not can_use_operation_signature(op_sig):
                return None
            out_uid = int(_pair_to_output_uid_global_np(a, b, base_seed=base_seed))
            if out_uid in {a, b} or out_uid in leaf_set or out_uid in produced_uids:
                return None
            produced_uids.add(out_uid)
            tree_operation_signature_set.add(op_sig)
            tree_operation_signatures.append(op_sig)
            return InstantiatedNode(
                kind="B",
                depth=1 + max(int(left.depth), int(right.depth)),
                uid=out_uid,
                children=(left, right),
                row=_encode_combine_row(a, b, out_uid),
                topology_key=_topology_key(node),
            )

        if kind == "T":
            c1 = walk(node[1])
            if c1 is None:
                return None
            c2 = walk(node[2])
            if c2 is None:
                return None
            c3 = walk(node[3])
            if c3 is None:
                return None
            ordered = tuple(sorted((int(c1.uid), int(c2.uid), int(c3.uid))))
            if len(set(ordered)) != 3:
                return None
            op_sig = _operation_signature_ternary(*ordered)
            if not can_use_operation_signature(op_sig):
                return None
            out_uid = int(
                _triple_to_output_uid_global_np(
                    int(ordered[0]),
                    int(ordered[1]),
                    int(ordered[2]),
                    base_seed=base_seed,
                )
            )
            if out_uid in set(ordered) or out_uid in leaf_set or out_uid in produced_uids:
                return None
            produced_uids.add(out_uid)
            tree_operation_signature_set.add(op_sig)
            tree_operation_signatures.append(op_sig)
            return InstantiatedNode(
                kind="T",
                depth=1 + max(int(c1.depth), int(c2.depth), int(c3.depth)),
                uid=out_uid,
                children=(c1, c2, c3),
                row=_encode_ternary_combine_row(
                    int(ordered[0]), int(ordered[1]), int(ordered[2]), out_uid
                ),
                topology_key=_topology_key(node),
            )

        raise ValueError(f"Unsupported topology kind: {kind!r}")

    root = walk(topology)
    if root is None:
        return None
    if root.depth != _topology_depth(topology):
        return None
    if root.depth < 1:
        return None

    unary_count, binary_count, ternary_count = _topology_internal_counts(topology)
    return InstantiatedTree(
        root=root,
        operation_signatures=tuple(tree_operation_signatures),
        unary_count=int(unary_count),
        binary_count=int(binary_count),
        ternary_count=int(ternary_count),
        leaf_count=int(_topology_leaf_count(topology)),
        rule_count=int(_topology_rule_count(topology)),
        topology_signature=_topology_signature(_topology_key(topology)),
    )


def _choose_deepest_child(node: InstantiatedNode) -> InstantiatedNode:
    candidates = [child for child in node.children if int(child.depth) == int(node.depth) - 1]
    if not candidates:
        raise SamplingError(
            f"Node of depth {node.depth} has no child at depth {node.depth - 1}."
        )
    candidates.sort(key=lambda child: _topology_signature(child.topology_key))
    return candidates[0]


def _extract_subtree_task_nodes(root: InstantiatedNode) -> dict[int, InstantiatedNode]:
    nodes: dict[int, InstantiatedNode] = {int(root.depth): root}
    current = root
    while int(current.depth) > 1:
        current = _choose_deepest_child(current)
        nodes[int(current.depth)] = current
    return nodes


def _collect_producer_rows_preorder(node: InstantiatedNode, out_rows: list[np.ndarray]) -> None:
    if node.kind == "L":
        return
    if node.row is None:
        raise SamplingError(f"Non-leaf node {node.kind!r} is missing an encoded row.")
    out_rows.append(np.asarray(node.row, dtype=np.int32))
    for child in node.children:
        _collect_producer_rows_preorder(child, out_rows)


def _rows_for_subtree(node: InstantiatedNode) -> np.ndarray:
    if node.kind == "L":
        return np.asarray([_encode_collect_row(int(node.uid))], dtype=np.int32)
    rows: list[np.ndarray] = []
    _collect_producer_rows_preorder(node, rows)
    return np.asarray(rows, dtype=np.int32)


def _pad_task_rows(task_rows: list[np.ndarray]) -> np.ndarray:
    if not task_rows:
        return np.zeros((0, 1, 6), dtype=np.int32)
    max_rules = max(int(rows.shape[0]) for rows in task_rows)
    padded = np.zeros((len(task_rows), max_rules, 6), dtype=np.int32)
    padded[..., 0] = int(NOOP_RULE_TYPE)
    for idx, rows in enumerate(task_rows):
        padded[idx, : rows.shape[0], :] = rows
    return padded


def _histogram(values: Iterable[int]) -> dict[str, int]:
    counter = Counter(int(v) for v in values)
    return {str(k): int(counter[k]) for k in sorted(counter)}


def _op_kind_counts(signatures: Iterable[OperationSignature]) -> dict[str, int]:
    counter = Counter(str(sig[0]) for sig in signatures)
    return {
        "unary": int(counter.get("U", 0)),
        "binary": int(counter.get("B", 0)),
        "ternary": int(counter.get("T", 0)),
    }


def _sample_unique_depth6_tree(
    *,
    depth: int,
    base_pool: np.ndarray,
    base_seed: int,
    blocked_topology_signatures: set[str],
    blocked_operation_signatures: set[OperationSignature],
    round_operation_signatures: set[OperationSignature],
    remaining_new_signatures_by_kind: dict[str, int],
    unary_budget: int,
    binary_budget: int,
    unary_root_weight: float,
    binary_root_weight: float,
    side_depth_cap: int,
    deep_side_prob: float,
    max_rule_count: int,
    max_topology_attempts: int,
    max_instantiation_attempts: int,
    rng: np.random.Generator,
) -> InstantiatedTree:
    last_reject_reason = 'no attempt made'
    for _ in range(int(max_topology_attempts)):
        try:
            topo = _sample_budgeted_topology_exact(
                int(depth),
                remaining_unary=int(unary_budget),
                remaining_binary=int(binary_budget),
                unary_root_weight=float(unary_root_weight),
                binary_root_weight=float(binary_root_weight),
                side_depth_cap=int(side_depth_cap),
                deep_side_prob=float(deep_side_prob),
                rng=rng,
            )
        except SamplingError as exc:
            last_reject_reason = f'topology-sampling:{exc}'
            continue
        topo_sig = _topology_signature(_topology_key(topo))
        if topo_sig in blocked_topology_signatures:
            last_reject_reason = 'topology-used-in-previous-round'
            continue
        leaf_count = _topology_leaf_count(topo)
        rule_count = _topology_rule_count(topo)
        if int(leaf_count) > int(base_pool.shape[0]):
            last_reject_reason = f'leaf-count>{int(base_pool.shape[0])}'
            continue
        if int(rule_count) > int(max_rule_count):
            last_reject_reason = f'rule-count>{int(max_rule_count)}'
            continue
        unary_count, binary_count, _ternary_count = _topology_internal_counts(topo)
        if int(unary_count) > int(unary_budget):
            last_reject_reason = 'unary-budget-exceeded'
            continue
        if int(binary_count) > int(binary_budget):
            last_reject_reason = 'binary-budget-exceeded'
            continue
        for _ in range(int(max_instantiation_attempts)):
            instantiated = _instantiate_topology(
                topo,
                base_pool=base_pool,
                base_seed=int(base_seed),
                blocked_operation_signatures=blocked_operation_signatures,
                round_operation_signatures=round_operation_signatures,
                remaining_new_signatures_by_kind=remaining_new_signatures_by_kind,
                rng=rng,
            )
            if instantiated is not None:
                return instantiated
        last_reject_reason = 'instantiation-failed'
    raise SamplingError(
        'Failed to sample a depth-6 tree under the current cross-round disjointness constraints. '
        f'Last rejection: {last_reject_reason}. '
        f'blocked_topologies={len(blocked_topology_signatures)}, '
        f'blocked_ops={len(blocked_operation_signatures)}, '
        f'round_ops={len(round_operation_signatures)}'
    )



def _validate_base_seed(base_pool: np.ndarray, base_seed: int) -> None:
    del base_pool  # The mixed producer outputs can hit the full compact-coded domain.
    if int(base_seed) != 0:
        raise ValueError(
            "This generator requires base_seed=0 because the packed codec "
            "reconstructs compact producer outputs with a fixed base seed of 0."
        )


def _build_base_pool(pool_size: int, pool_indices: str) -> np.ndarray:
    if pool_size < 1 or pool_size > _DOMAIN_SIZE:
        raise ValueError(f"pool_size must be in [1, {_DOMAIN_SIZE}].")
    if pool_indices:
        parsed = _parse_csv_ints(pool_indices)
        if len(parsed) != int(pool_size):
            raise ValueError(
                f"pool-indices length ({len(parsed)}) must match pool-size ({pool_size})."
            )
        if len(set(parsed)) != len(parsed):
            raise ValueError("pool-indices must be unique.")
        if any(int(uid) < 0 or int(uid) >= _DOMAIN_SIZE for uid in parsed):
            raise ValueError(f"pool-indices entries must be in [0, {_DOMAIN_SIZE - 1}].")
        return np.asarray(parsed, dtype=np.int32)
    return np.arange(int(pool_size), dtype=np.int32)


def _producer_signature_capacity() -> dict[str, int]:
    return {
        "unary": int(_DOMAIN_SIZE),
        "binary": int(comb(_DOMAIN_SIZE, 2)),
        "ternary": int(comb(_DOMAIN_SIZE, 3)),
        "total": int(_DOMAIN_SIZE + comb(_DOMAIN_SIZE, 2) + comb(_DOMAIN_SIZE, 3)),
    }


def _ceil_div(a: int, b: int) -> int:
    if int(b) <= 0:
        return int(a)
    return (int(a) + int(b) - 1) // int(b)


def _round_new_signature_caps(
    *,
    rounds_remaining: int,
    blocked_operation_signatures: set[OperationSignature],
    override_unary: int,
    override_binary: int,
    override_ternary: int = 0,
) -> dict[str, int]:
    capacities = _producer_signature_capacity()
    used_counts = _op_kind_counts(blocked_operation_signatures)

    def _cap(kind: str, override: int) -> int:
        remaining = max(0, int(capacities[kind]) - int(used_counts.get(kind, 0)))
        if int(override) > 0:
            return min(int(override), int(remaining))
        if kind == 'ternary':
            return int(remaining)
        return _ceil_div(int(remaining), max(1, int(rounds_remaining)))

    return {
        'unary': _cap('unary', int(override_unary)),
        'binary': _cap('binary', int(override_binary)),
        'ternary': _cap('ternary', int(override_ternary)),
    }

def _default_issue_report(
    *,
    n: int,
    rounds: int,
    max_depth: int,
    unary_budget: int,
    binary_budget: int,
    max_rule_count: int,
) -> list[str]:
    issues: list[str] = []
    depth6_tree_count = int(n) * int(rounds)
    min_ops_per_depth6_tree = max(0, int(max_depth) - 1)
    capacities = _producer_signature_capacity()
    avg_unary_new_per_round = _ceil_div(capacities['unary'], max(1, int(rounds)))
    avg_binary_new_per_round = _ceil_div(capacities['binary'], max(1, int(rounds)))

    if int(unary_budget) > 0:
        issues.append(
            'Within-round reuse is now allowed, so unary-heavy trees are feasible, but the '            f'unique unary vocabulary still averages only about {avg_unary_new_per_round} new signatures per round '
            f'with a {_DOMAIN_SIZE}-uid pool and {int(rounds)} rounds.'
        )
    if int(binary_budget) > 0 and int(n) * int(binary_budget) > avg_binary_new_per_round:
        issues.append(
            'Binary nodes can be balanced by reusing same-round pairs, but the generator may need '            f'to recycle roughly within ~{avg_binary_new_per_round} new binary signatures per round.'
        )
    worst_case_total = int(depth6_tree_count) * int(max_rule_count)
    if worst_case_total > capacities['total']:
        issues.append(
            'The total producer-node count across all rounds exceeds the raw producer signature space; '            'this is okay only because same-round reuse is allowed.'
        )
    best_case_total = int(depth6_tree_count) * int(min_ops_per_depth6_tree)
    if best_case_total > capacities['total']:
        issues.append(
            'Even the theoretical minimum total producer count exceeds the raw signature space; '            'future rounds rely on reusing signatures inside a round but never across rounds.'
        )
    return issues



def _generate_round_dataset(
    *,
    n: int,
    max_depth: int,
    round_index: int,
    rounds_remaining: int,
    base_pool: np.ndarray,
    base_seed: int,
    unary_budget: int,
    binary_budget: int,
    unary_root_weight: float,
    binary_root_weight: float,
    side_depth_cap: int,
    deep_side_prob: float,
    max_depth6_rules: int,
    max_topology_attempts: int,
    max_instantiation_attempts: int,
    blocked_topology_signatures: set[str],
    blocked_operation_signatures: set[OperationSignature],
    max_new_unary_signatures_per_round: int,
    max_new_binary_signatures_per_round: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict, set[str], set[OperationSignature]]:
    tasks_by_depth: dict[int, list[np.ndarray]] = {
        int(d): [] for d in range(1, int(max_depth) + 1)
    }
    round_topology_signatures: list[str] = []
    round_topology_signature_set: set[str] = set()
    round_operation_signatures: list[OperationSignature] = []
    round_operation_signature_set: set[OperationSignature] = set()
    round_new_sig_caps = _round_new_signature_caps(
        rounds_remaining=int(rounds_remaining),
        blocked_operation_signatures=blocked_operation_signatures,
        override_unary=int(max_new_unary_signatures_per_round),
        override_binary=int(max_new_binary_signatures_per_round),
    )
    depth6_rule_counts: list[int] = []
    depth6_leaf_counts: list[int] = []
    depth6_binary_counts: list[int] = []
    depth6_unary_counts: list[int] = []
    depth6_ternary_counts: list[int] = []

    for _chain_idx in range(int(n)):
        remaining_new = {
            'unary': max(0, int(round_new_sig_caps['unary']) - _op_kind_counts(round_operation_signature_set)['unary']),
            'binary': max(0, int(round_new_sig_caps['binary']) - _op_kind_counts(round_operation_signature_set)['binary']),
            'ternary': max(0, int(round_new_sig_caps['ternary']) - _op_kind_counts(round_operation_signature_set)['ternary']),
        }
        tree = _sample_unique_depth6_tree(
            depth=int(max_depth),
            base_pool=base_pool,
            base_seed=int(base_seed),
            blocked_topology_signatures=blocked_topology_signatures,
            blocked_operation_signatures=blocked_operation_signatures,
            round_operation_signatures=round_operation_signature_set,
            remaining_new_signatures_by_kind=remaining_new,
            unary_budget=int(unary_budget),
            binary_budget=int(binary_budget),
            unary_root_weight=float(unary_root_weight),
            binary_root_weight=float(binary_root_weight),
            side_depth_cap=int(side_depth_cap),
            deep_side_prob=float(deep_side_prob),
            max_rule_count=int(max_depth6_rules),
            max_topology_attempts=int(max_topology_attempts),
            max_instantiation_attempts=int(max_instantiation_attempts),
            rng=rng,
        )

        subtree_task_nodes = _extract_subtree_task_nodes(tree.root)
        missing_depths = [
            d for d in range(1, int(max_depth) + 1) if int(d) not in subtree_task_nodes
        ]
        if missing_depths:
            raise SamplingError(
                f'Depth-6 tree failed to expose a full subtree-task chain. Missing depths: {missing_depths}'
            )

        for depth in range(1, int(max_depth) + 1):
            rows = _rows_for_subtree(subtree_task_nodes[int(depth)])
            tasks_by_depth[int(depth)].append(rows)

        round_topology_signature_set.add(str(tree.topology_signature))
        round_operation_signature_set.update(tree.operation_signatures)
        round_topology_signatures.append(str(tree.topology_signature))
        round_operation_signatures.extend(tree.operation_signatures)
        depth6_rule_counts.append(int(tree.rule_count))
        depth6_leaf_counts.append(int(tree.leaf_count))
        depth6_unary_counts.append(int(tree.unary_count))
        depth6_binary_counts.append(int(tree.binary_count))
        depth6_ternary_counts.append(int(tree.ternary_count))

    ordered_task_rows: list[np.ndarray] = []
    structure = {int(d): int(n) for d in range(1, int(max_depth) + 1)}
    per_depth_max_rules: dict[str, int] = {}
    per_depth_rule_count_hist: dict[str, dict[str, int]] = {}
    for depth in range(1, int(max_depth) + 1):
        ordered_task_rows.extend(tasks_by_depth[int(depth)])
        per_depth_max_rules[str(depth)] = max(
            int(rows.shape[0]) for rows in tasks_by_depth[int(depth)]
        )
        per_depth_rule_count_hist[str(depth)] = _histogram(
            int(rows.shape[0]) for rows in tasks_by_depth[int(depth)]
        )

    padded = _pad_task_rows(ordered_task_rows)
    packed = pack_rules_uint32_np(np.asarray(padded, dtype=np.int32)).astype(np.uint32)

    round_op_counts = _op_kind_counts(round_operation_signatures)
    round_unique_op_counts = _op_kind_counts(round_operation_signature_set)
    meta = {
        'structure': structure,
        'n_per_depth': int(n),
        'max_depth': int(max_depth),
        'round_index': int(round_index),
        'pool_size': int(base_pool.shape[0]),
        'pool_uids': [int(uid) for uid in base_pool.tolist()],
        'base_seed': int(base_seed),
        'generation_mode': 'deepest_path_strict_subtrees',
        'round_depth6_topology_count': len(round_topology_signatures),
        'round_depth6_topology_signatures': round_topology_signatures,
        'round_unique_depth6_topology_count': len(round_topology_signature_set),
        'round_producer_signature_count': len(round_operation_signatures),
        'round_producer_signature_counts_by_kind': round_op_counts,
        'round_unique_producer_signature_count': len(round_operation_signature_set),
        'round_unique_producer_signature_counts_by_kind': round_unique_op_counts,
        'round_new_signature_caps': {k: int(v) for k, v in round_new_sig_caps.items()},
        'depth6_rule_count_distribution': _histogram(depth6_rule_counts),
        'depth6_leaf_count_distribution': _histogram(depth6_leaf_counts),
        'depth6_unary_count_distribution': _histogram(depth6_unary_counts),
        'depth6_binary_count_distribution': _histogram(depth6_binary_counts),
        'depth6_ternary_count_distribution': _histogram(depth6_ternary_counts),
        'task_rule_count_distribution_per_depth': per_depth_rule_count_hist,
        'task_max_rule_count_per_depth': per_depth_max_rules,
        'rule_shape': [int(padded.shape[1]), 6],
        'packed_shape': list(packed.shape),
        'no_repeat_guarantees': {
            'producer_signatures_across_rounds_only': True,
            'depth6_topologies_across_rounds_only': True,
            'same_round_signature_reuse_allowed': True,
            'same_round_topology_reuse_allowed': True,
            'items_may_repeat': True,
        },
        'sampling_budgets': {
            'max_unary_nodes_per_depth6_tree': int(unary_budget),
            'max_binary_nodes_per_depth6_tree': int(binary_budget),
            'unary_root_weight': float(unary_root_weight),
            'binary_root_weight': float(binary_root_weight),
            'side_depth_cap': int(side_depth_cap),
            'deep_side_prob': float(deep_side_prob),
            'max_depth6_rules': int(max_depth6_rules),
        },
    }
    return packed, meta, round_topology_signature_set, round_operation_signature_set



def _write_round_outputs(
    *,
    out_dir: Path,
    stem: str,
    packed: np.ndarray,
    meta: dict,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"{stem}.uint32.npy.bz2"
    meta_path = out_dir / f"{stem}_meta.json"

    with bz2.BZ2File(data_path, "wb") as f:
        np.save(f, packed, allow_pickle=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return data_path, meta_path


def _run_for_single_n(args: argparse.Namespace, n: int, out_root: Path) -> dict:
    if int(n) <= 0:
        raise ValueError(f'n must be > 0, got {n}')

    base_pool = _build_base_pool(int(args.pool_size), str(args.pool_indices))
    _validate_base_seed(base_pool, int(args.base_seed))

    issues = _default_issue_report(
        n=int(n),
        rounds=int(args.rounds),
        max_depth=int(args.max_depth),
        unary_budget=int(args.max_unary_nodes_per_depth6),
        binary_budget=int(args.max_binary_nodes_per_depth6),
        max_rule_count=int(args.max_depth6_rules),
    )
    if issues and bool(args.fail_on_issue):
        joined = '\n - '.join(issues)
        raise ValueError(f'Feasibility issue(s) detected for n={n}:\n - {joined}')

    run_dir = out_root / f'n{int(n):04d}'
    run_dir.mkdir(parents=True, exist_ok=True)

    master_seed = int(args.master_seed) ^ (int(n) * 1_000_003)
    rng = np.random.default_rng(master_seed)
    blocked_topology_signatures: set[str] = set()
    blocked_operation_signatures: set[OperationSignature] = set()
    round_paths: list[dict[str, str]] = []
    round_meta_summaries: list[dict] = []

    for round_index in range(int(args.rounds)):
        round_seed = int(rng.integers(0, 2**31 - 1))
        round_rng = np.random.default_rng(round_seed)
        rounds_remaining = int(args.rounds) - int(round_index)
        packed, meta, round_topology_signature_set, round_operation_signature_set = _generate_round_dataset(
            n=int(n),
            max_depth=int(args.max_depth),
            round_index=int(round_index),
            rounds_remaining=int(rounds_remaining),
            base_pool=base_pool,
            base_seed=int(args.base_seed),
            unary_budget=int(args.max_unary_nodes_per_depth6),
            binary_budget=int(args.max_binary_nodes_per_depth6),
            unary_root_weight=float(args.unary_root_weight),
            binary_root_weight=float(args.binary_root_weight),
            side_depth_cap=int(args.side_depth_cap),
            deep_side_prob=float(args.deep_side_prob),
            max_depth6_rules=int(args.max_depth6_rules),
            max_topology_attempts=int(args.max_topology_attempts),
            max_instantiation_attempts=int(args.max_instantiation_attempts),
            blocked_topology_signatures=blocked_topology_signatures,
            blocked_operation_signatures=blocked_operation_signatures,
            max_new_unary_signatures_per_round=int(args.max_new_unary_signatures_per_round),
            max_new_binary_signatures_per_round=int(args.max_new_binary_signatures_per_round),
            rng=round_rng,
        )
        blocked_topology_signatures.update(round_topology_signature_set)
        blocked_operation_signatures.update(round_operation_signature_set)
        meta['round_seed'] = int(round_seed)
        meta['master_seed'] = int(master_seed)
        meta['global_unique_depth6_topologies_after_round'] = len(blocked_topology_signatures)
        meta['global_unique_producer_signatures_after_round'] = len(blocked_operation_signatures)
        stem = f'{args.name}_n{int(n)}_r{int(round_index):02d}'
        data_path, meta_path = _write_round_outputs(
            out_dir=run_dir,
            stem=stem,
            packed=packed,
            meta=meta,
        )
        round_paths.append(
            {
                'round': int(round_index),
                'data_path': str(data_path),
                'meta_path': str(meta_path),
            }
        )
        round_meta_summaries.append(
            {
                'round': int(round_index),
                'depth6_topologies': int(meta['round_depth6_topology_count']),
                'unique_depth6_topologies': int(meta['round_unique_depth6_topology_count']),
                'producer_signatures': int(meta['round_producer_signature_count']),
                'unique_producer_signatures': int(meta['round_unique_producer_signature_count']),
                'producer_signature_counts_by_kind': dict(meta['round_producer_signature_counts_by_kind']),
                'unique_producer_signature_counts_by_kind': dict(meta['round_unique_producer_signature_counts_by_kind']),
                'depth6_rule_count_distribution': dict(meta['depth6_rule_count_distribution']),
                'round_new_signature_caps': dict(meta['round_new_signature_caps']),
            }
        )

    summary = {
        'n': int(n),
        'rounds': int(args.rounds),
        'max_depth': int(args.max_depth),
        'pool_size': int(args.pool_size),
        'pool_indices': str(args.pool_indices),
        'base_seed': int(args.base_seed),
        'master_seed': int(master_seed),
        'issues': issues,
        'sampling_budgets': {
            'max_unary_nodes_per_depth6_tree': int(args.max_unary_nodes_per_depth6),
            'max_binary_nodes_per_depth6_tree': int(args.max_binary_nodes_per_depth6),
            'unary_root_weight': float(args.unary_root_weight),
            'binary_root_weight': float(args.binary_root_weight),
            'side_depth_cap': int(args.side_depth_cap),
            'deep_side_prob': float(args.deep_side_prob),
            'max_depth6_rules': int(args.max_depth6_rules),
            'max_new_unary_signatures_per_round': int(args.max_new_unary_signatures_per_round),
            'max_new_binary_signatures_per_round': int(args.max_new_binary_signatures_per_round),
        },
        'cross_round_guarantees': {
            'depth6_topologies_disjoint_across_rounds': True,
            'producer_signatures_disjoint_across_rounds': True,
            'same_round_signature_reuse_allowed': True,
            'same_round_topology_reuse_allowed': True,
        },
        'global_unique_depth6_topologies_after_all_rounds': len(blocked_topology_signatures),
        'global_unique_producer_signatures_after_all_rounds': len(blocked_operation_signatures),
        'rounds_written': round_paths,
        'round_meta_summaries': round_meta_summaries,
    }
    summary_path = run_dir / f'{args.name}_n{int(n)}_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    summary['summary_path'] = str(summary_path)
    return summary



def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Generate round-disjoint depth-6 ruleset datasets."
    )
    ap.add_argument(
        "--n",
        type=int,
        default=None,
        help="Generate one n-run with exactly n tasks at every depth 1..6.",
    )
    ap.add_argument(
        "--n-values",
        type=str,
        default="",
        help=(
            "Optional comma-separated sweep, e.g. "
            "'1,2,4,6,8,10,16,32,64,128,256,512'."
        ),
    )
    ap.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of datasets / rounds to generate for each n.",
    )
    ap.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum tree depth. This file is written for depth 1..6.",
    )
    ap.add_argument(
        "--pool-size",
        type=int,
        default=_DOMAIN_SIZE,
        help="Number of pool uids. Default uses the full 120-uid domain.",
    )
    ap.add_argument(
        "--pool-indices",
        type=str,
        default="",
        help=(
            f"Optional explicit pool uids in [0, {_DOMAIN_SIZE - 1}] as a CSV list."
        ),
    )
    ap.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Output mapping seed. For the 120-uid pool this should remain 0.",
    )
    ap.add_argument(
        "--master-seed",
        type=int,
        default=0,
        help="Master RNG seed for topology / leaf sampling.",
    )
    ap.add_argument(
        "--max-unary-nodes-per-depth6",
        type=int,
        default=2,
        help=(
            "Cap unary nodes in each exact-depth-6 tree. Default 2 now that "
            "signatures only need to be disjoint across rounds, not within a round."
        ),
    )
    ap.add_argument(
        "--max-binary-nodes-per-depth6",
        type=int,
        default=2,
        help=(
            "Cap binary nodes in each exact-depth-6 tree. Default 2 for a more "
            "balanced unary/binary/ternary mix."
        ),
    )
    ap.add_argument(
        "--unary-root-weight",
        type=float,
        default=0.34,
        help="Root-selection weight for unary nodes when unary budget > 0.",
    )
    ap.add_argument(
        "--binary-root-weight",
        type=float,
        default=0.33,
        help="Root-selection weight for binary nodes when binary budget > 0.",
    )
    ap.add_argument(
        "--side-depth-cap",
        type=int,
        default=3,
        help="Max random side-branch depth before occasional deep-side override.",
    )
    ap.add_argument(
        "--deep-side-prob",
        type=float,
        default=0.18,
        help="Probability of allowing a deeper side branch during topology sampling.",
    )
    ap.add_argument(
        "--max-depth6-rules",
        type=int,
        default=32,
        help=(
            "Reject exact-depth-6 trees whose producer-node count exceeds this cap. "
            "The default keeps the largest requested n comfortably inside the total "
            "producer-signature space."
        ),
    )
    ap.add_argument(
        "--max-topology-attempts",
        type=int,
        default=20000,
        help="How many topology attempts to allow when sampling one depth-6 tree.",
    )
    ap.add_argument(
        "--max-instantiation-attempts",
        type=int,
        default=128,
        help="How many leaf-instantiation retries to allow for one sampled topology.",
    )
    ap.add_argument(
        "--max-new-unary-signatures-per-round",
        type=int,
        default=0,
        help=(
            "Optional hard cap on how many NEW unary signatures a round may introduce. "
            "0 means auto-budget from remaining cross-round capacity."
        ),
    )
    ap.add_argument(
        "--max-new-binary-signatures-per-round",
        type=int,
        default=0,
        help=(
            "Optional hard cap on how many NEW binary signatures a round may introduce. "
            "0 means auto-budget from remaining cross-round capacity."
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Root output directory.",
    )
    ap.add_argument(
        "--name",
        type=str,
        default="round_disjoint_depth6",
        help="Filename prefix.",
    )
    ap.add_argument(
        "--fail-on-issue",
        action="store_true",
        help="Raise immediately if the preflight issue report is non-empty.",
    )
    return ap


def main() -> None:
    ap = _build_arg_parser()
    args = ap.parse_args()

    if args.max_depth != 6:
        ap.error("This generator currently supports only --max-depth 6.")
    if args.rounds <= 0:
        ap.error("--rounds must be > 0")
    if args.max_unary_nodes_per_depth6 < 0:
        ap.error("--max-unary-nodes-per-depth6 must be >= 0")
    if args.max_binary_nodes_per_depth6 < 0:
        ap.error("--max-binary-nodes-per-depth6 must be >= 0")
    if args.max_depth6_rules < max(0, args.max_depth - 1):
        ap.error(
            f"--max-depth6-rules must be at least {args.max_depth - 1} for exact depth {args.max_depth}."
        )
    if not (0.0 <= float(args.deep_side_prob) <= 1.0):
        ap.error("--deep-side-prob must be in [0, 1]")
    if float(args.unary_root_weight) < 0.0:
        ap.error("--unary-root-weight must be >= 0")
    if float(args.binary_root_weight) < 0.0:
        ap.error("--binary-root-weight must be >= 0")

    n_values: list[int] = []
    if args.n is not None:
        n_values.append(int(args.n))
    if args.n_values:
        n_values.extend(_parse_csv_ints(str(args.n_values)))
    if not n_values:
        n_values = list(DEFAULT_N_VALUES)
    n_values = list(dict.fromkeys(int(v) for v in n_values))
    if any(int(v) <= 0 for v in n_values):
        ap.error("All n values must be > 0")

    out_root = Path(str(args.out_dir))
    out_root.mkdir(parents=True, exist_ok=True)

    sweep_summary = {
        "generator": "make_depth6_round_disjoint_dataset.py",
        "n_values": [int(v) for v in n_values],
        "rounds": int(args.rounds),
        "max_depth": int(args.max_depth),
        "results": [],
    }

    for n in n_values:
        summary = _run_for_single_n(args, int(n), out_root)
        sweep_summary["results"].append(summary)
        print(
            f"[n={n}] rounds={args.rounds} "
            f"depth6_topologies={summary['global_unique_depth6_topologies_after_all_rounds']} "
            f"producer_signatures={summary['global_unique_producer_signatures_after_all_rounds']}"
        )
        if summary["issues"]:
            print(f"[n={n}] issues:")
            for issue in summary["issues"]:
                print(f"  - {issue}")

    sweep_path = out_root / f"{args.name}_sweep_summary.json"
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(sweep_summary, f, indent=2)
    print(f"Wrote sweep summary: {sweep_path}")


if __name__ == "__main__":
    main()
