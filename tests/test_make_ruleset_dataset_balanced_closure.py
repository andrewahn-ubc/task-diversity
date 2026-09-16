"""End-to-end closure test for --tree-topology balanced at depths 5 and 6.

Generates small datasets with the real CLI and verifies:
(a) meta["structure"] matches the expected per-depth task counts and tasks are
    grouped in ascending depth with actual depth equal to the bucket;
(b) every task with depth >= 2 is a full balanced binary tree;
(c) strict subtree closure: every subtree of every task (including depth-1
    leaves) is itself a standalone task in the dataset;
(d) within each task all producer outputs are distinct and disjoint from leaves.
"""

import os
import subprocess
import sys

import pytest

from banyan_grid.environment.constants import (
    RULE_TYPE_COLLECT,
    RULE_TYPE_COMBINE,
    RULE_TYPE_TERNARY_COMBINE,
    RULE_TYPE_TRANSFORM,
)
from banyan_grid.tasks.ruleset_codec import unpack_rules_uint32_np
from banyan_grid.tasks.ruleset_dataset_compact import (
    load_meta,
    load_packed_u32_bz2,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PRODUCER_TYPES = (RULE_TYPE_COMBINE, RULE_TYPE_TRANSFORM, RULE_TYPE_TERNARY_COMBINE)


def _task_tree(rows):
    """Build the canonical tree for one task's decoded (R, 6) rows.

    Returns (canonical_tree, children_map) where canonical_tree is
    (token, tuple(sorted(children, key=repr))) and token is an (item, color) tuple.
    """
    children_map = {}
    outputs = []
    inputs = []
    collect_root = None
    for row in rows:
        rt = int(row[0])
        c = int(row[5])
        if rt == RULE_TYPE_COLLECT:
            collect_root = (int(row[2]), int(row[3]))
        elif rt in _PRODUCER_TYPES:
            ins = [(int(row[1]), c & 15), (int(row[2]), (c >> 4) & 15)]
            if rt == RULE_TYPE_TERNARY_COMBINE:
                ins.append((int(row[4]) >> 1, (c >> 8) & 15))
                out = (int(row[3]), (c >> 12) & 15)
            else:
                out = (int(row[3]), (c >> 8) & 15)
            children_map[out] = ins
            outputs.append(out)
            inputs.extend(ins)
    if collect_root is not None:
        return (collect_root, ()), children_map
    assert outputs, "task has no producer rows"
    # The final combine is stored first
    root = outputs[0]

    def _canon(tok):
        ins = children_map.get(tok)
        if ins is None:
            return (tok, ())
        return (tok, tuple(sorted((_canon(t) for t in ins), key=repr)))

    return _canon(root), children_map


def _subtrees(tree):
    """All canonical subtrees of a canonical tree (including the leaf itself)."""
    tok, kids = tree
    out = {tree}
    for k in kids:
        out |= _subtrees(k)
    return out


def _leaf_depths(tree, depth=0):
    tok, kids = tree
    if not kids:
        return [depth]
    ds = []
    for k in kids:
        ds.extend(_leaf_depths(k, depth + 1))
    return ds


def _internal_arity_ok(tree):
    tok, kids = tree
    if not kids:
        return True
    return len(kids) == 2 and all(_internal_arity_ok(k) for k in kids)


@pytest.mark.parametrize(
    "max_depth,pool_size",
    [(5, 24), (6, 40)],
)
def test_balanced_deep_closure(tmp_path, max_depth, pool_size):
    env = {**os.environ, "JAX_PLATFORM_NAME": "cpu"}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "banyan_grid.tasks.make_ruleset_dataset",
            "--n",
            "1",
            "--max-depth",
            str(max_depth),
            "--pool-size",
            str(pool_size),
            "--tree-topology",
            "balanced",
            "--base-seed",
            "0",
            "--bench-seed",
            "0",
            "--name",
            "t",
            "--out-dir",
            str(tmp_path),
        ],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )

    files = [f for f in os.listdir(tmp_path) if f.endswith(".uint32.npy.bz2")]
    assert len(files) == 1
    packed = load_packed_u32_bz2(str(tmp_path), files[0])
    meta = load_meta(str(tmp_path), files[0])
    rules = unpack_rules_uint32_np(packed)  # (N, R, 6) int32

    # (a) structure and depth grouping
    expected_structure = {"1": pool_size, "2": pool_size * (pool_size - 1) // 2}
    for d in range(3, max_depth + 1):
        expected_structure[str(d)] = 2 ** (max_depth - d)
    assert meta["structure"] == expected_structure

    trees = []
    children_maps = []
    for task_rows in rules:
        tree, cmap = _task_tree(task_rows)
        trees.append(tree)
        children_maps.append(cmap)

    # actual depth of each task = tree height + 1
    actual_depths = [max(_leaf_depths(t)) + 1 for t in trees]
    offset = 0
    for d in range(1, max_depth + 1):
        n_d = expected_structure[str(d)]
        assert actual_depths[offset : offset + n_d] == [d] * n_d
        offset += n_d
    assert offset == len(trees)

    # (b) full balanced binary trees for depth >= 2
    for tree, depth in zip(trees, actual_depths):
        if depth >= 2:
            assert _internal_arity_ok(tree)
            assert len(set(_leaf_depths(tree))) == 1

    # (c) strict closure: every subtree of every task is itself a task
    all_trees = set(trees)
    for tree in trees:
        assert _subtrees(tree) <= all_trees

    # (d) outputs distinct within task and disjoint from leaf tokens
    for tree, cmap in zip(trees, children_maps):
        outs = list(cmap.keys())
        assert len(outs) == len(set(outs))
        leaf_tokens = set()

        def _collect_leaves(t):
            tok, kids = t
            if not kids:
                leaf_tokens.add(tok)
            else:
                for k in kids:
                    _collect_leaves(k)

        _collect_leaves(tree)
        assert leaf_tokens.isdisjoint(outs)
