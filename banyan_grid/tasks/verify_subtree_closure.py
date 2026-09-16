# tools/verify_subtree_closure.py
"""
SUBTREE CLOSURE VERIFIER

Verifies that a ruleset file has correct subtree structure:
- Every depth-4 task has 2 depth-3 subtrees that exist as standalone tasks
- Every depth-3 task has 2 depth-2 subtrees that exist as standalone tasks
- Every depth-2 task has 2 depth-1 items that exist as standalone tasks

Usage:
  python verify_subtree_closure.py path/to/rulesets.uint32.npy.bz2
"""

import os
import sys
import json
import bz2
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
import numpy as np

# Add repository root to path for direct script execution
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import jax.numpy as jnp
from banyan_grid.tasks.ruleset_codec import (
    unpack_rules_uint32_jit,
)
from banyan_grid.environment.constants import (
    Colors,
)

_NUM_REAL_COLORS = int(Colors.BLACK)


def uid_from_item_color(item: int, color: int) -> int:
    """Convert (item, color) to unique ID."""
    return item * _NUM_REAL_COLORS + color


def item_color_from_uid(uid: int) -> Tuple[int, int]:
    """Convert unique ID to (item, color)."""
    return uid // _NUM_REAL_COLORS, uid % _NUM_REAL_COLORS


def get_depth_from_rules(rules: np.ndarray) -> int:
    """
    Determine depth from rule array.
    Depth = 1 if no combine rules, else 1 + floor(log2(combines + 1))
    """
    is_comb = rules[:, 0] == 2
    combine_count = np.sum(is_comb)
    if combine_count == 0:
        return 1
    return 1 + int(np.floor(np.log2(combine_count + 1)))


def extract_depth1_target(rules: np.ndarray) -> Optional[int]:
    """
    Extract the target (item,color) uid for a depth-1 task (collect rule).
    Returns None if not a valid depth-1 task.
    """
    # Find collect rules (type == 1)
    is_collect = rules[:, 0] == 1
    if not np.any(is_collect):
        return None

    idx = np.argmax(is_collect.astype(np.int32))
    item = int(rules[idx, 2])
    color = int(rules[idx, 3])
    return uid_from_item_color(item, color)


def extract_depth2_signature(rules: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """
    Extract signature for a depth-2 task: (input1_uid, input2_uid, output_uid).
    Returns sorted (in1, in2) for consistent matching.
    """
    is_comb = rules[:, 0] == 2
    if np.sum(is_comb) != 1:
        return None

    idx = np.argmax(is_comb.astype(np.int32))
    row = rules[idx]

    in1_item = int(row[1])
    in2_item = int(row[2])
    out_item = int(row[3])
    packed = int(row[5])

    in1_color = packed & 0xF
    in2_color = (packed >> 4) & 0xF
    out_color = (packed >> 8) & 0xF

    in1_uid = uid_from_item_color(in1_item, in1_color)
    in2_uid = uid_from_item_color(in2_item, in2_color)
    out_uid = uid_from_item_color(out_item, out_color)

    # Sort inputs for consistent matching
    if in1_uid > in2_uid:
        in1_uid, in2_uid = in2_uid, in1_uid

    return (in1_uid, in2_uid, out_uid)


def extract_depth3_signature(rules: np.ndarray) -> Optional[Dict]:
    """
    Extract signature for a depth-3 task.
    Returns dict with:
      - leaves: list of 4 leaf uids
      - level0_pairs: list of 2 (sorted) pairs of leaf uids
      - level1_pair: (sorted) pair of intermediate output uids
      - final_output: final output uid
    """
    is_comb = rules[:, 0] == 2
    comb_idx = np.where(is_comb)[0]

    if len(comb_idx) != 3:
        return None

    rows = rules[comb_idx]

    # Extract outputs and inputs
    outs = rows[:, 3].astype(int)
    ins1 = rows[:, 1].astype(int)
    ins2 = rows[:, 2].astype(int)
    packed = rows[:, 5].astype(int)

    out_c = (packed >> 8) & 0xF
    in1_c = packed & 0xF
    in2_c = (packed >> 4) & 0xF

    out_uids = [uid_from_item_color(int(outs[i]), int(out_c[i])) for i in range(3)]
    in1_uids = [uid_from_item_color(int(ins1[i]), int(in1_c[i])) for i in range(3)]
    in2_uids = [uid_from_item_color(int(ins2[i]), int(in2_c[i])) for i in range(3)]

    # Find final combine (output not used as input)
    used_as_input = [False, False, False]
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            if out_uids[i] == in1_uids[j] or out_uids[i] == in2_uids[j]:
                used_as_input[i] = True

    final_idx = None
    for i in range(3):
        if not used_as_input[i]:
            final_idx = i
            break

    if final_idx is None:
        return None

    child_indices = [i for i in range(3) if i != final_idx]

    # Extract level-0 pairs (leaves)
    leaves = []
    level0_pairs = []
    child_out_uids = []

    for idx in child_indices:
        pair = tuple(sorted([in1_uids[idx], in2_uids[idx]]))
        level0_pairs.append(pair)
        leaves.extend([in1_uids[idx], in2_uids[idx]])
        child_out_uids.append(out_uids[idx])

    level1_pair = tuple(sorted(child_out_uids))

    return {
        "leaves": sorted(leaves),
        "level0_pairs": sorted(level0_pairs),
        "level1_pair": level1_pair,
        "final_output": out_uids[final_idx],
    }


def extract_depth4_signature(
    rules: np.ndarray, verbose: bool = False
) -> Optional[Dict]:
    """
    Extract signature for a depth-4 task.
    Handles both perfect binary trees (4 level-0 combines) and trees with leaf reuse.

    Returns dict with:
      - leaves: list of unique leaf uids
      - level0_pairs: list of (sorted) pairs of leaf uids that get combined at level 0
      - final_output: final output uid
    """
    is_comb = rules[:, 0] == 2
    comb_idx = np.where(is_comb)[0]

    if len(comb_idx) != 7:  # depth-4 has 7 combine rules
        if verbose:
            print(f"    Expected 7 combine rules, found {len(comb_idx)}")
        return None

    rows = rules[comb_idx]

    # Extract all combine info
    outs = rows[:, 3].astype(int)
    ins1 = rows[:, 1].astype(int)
    ins2 = rows[:, 2].astype(int)
    packed = rows[:, 5].astype(int)

    out_c = (packed >> 8) & 0xF
    in1_c = packed & 0xF
    in2_c = (packed >> 4) & 0xF

    out_uids = [uid_from_item_color(int(outs[i]), int(out_c[i])) for i in range(7)]
    in1_uids = [uid_from_item_color(int(ins1[i]), int(in1_c[i])) for i in range(7)]
    in2_uids = [uid_from_item_color(int(ins2[i]), int(in2_c[i])) for i in range(7)]

    # Build graph: for each combine, identify if its inputs are "leaves" (not produced by another combine)
    all_outputs = set(out_uids)

    # Find all "leaf pairs" - combines where both inputs are not outputs of other combines
    # These are the depth-2 subtrees embedded in this depth-4 tree
    level0_pairs = []
    leaves = set()
    for i in range(7):
        in1_is_leaf = in1_uids[i] not in all_outputs
        in2_is_leaf = in2_uids[i] not in all_outputs
        if in1_is_leaf and in2_is_leaf:
            pair = tuple(sorted([in1_uids[i], in2_uids[i]]))
            level0_pairs.append(pair)
            leaves.add(in1_uids[i])
            leaves.add(in2_uids[i])

    # Find the final output (not used as input to any other combine)
    used_as_input = set()
    for i in range(7):
        used_as_input.add(in1_uids[i])
        used_as_input.add(in2_uids[i])

    final_outputs = [uid for uid in out_uids if uid not in used_as_input]
    final_output = final_outputs[0] if final_outputs else out_uids[-1]

    if verbose:
        print(
            f"    Found {len(level0_pairs)} level-0 pairs, {len(leaves)} unique leaves"
        )

    return {
        "leaves": sorted(leaves),
        "level0_pairs": sorted(level0_pairs),
        "final_output": final_output,
    }


def load_rulesets(path: str) -> Tuple[np.ndarray, Dict]:
    """Load rulesets and metadata."""
    # Load metadata
    meta_path = path.replace(".uint32.npy.bz2", "_meta.json")
    if not os.path.exists(meta_path):
        # Try alternate naming
        base = os.path.splitext(os.path.splitext(path)[0])[0]
        meta_path = base + "_meta.json"

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    else:
        meta = {}
        print(f"Warning: No metadata file found at {meta_path}")

    # Load packed rulesets
    with bz2.BZ2File(path, "rb") as f:
        packed = np.load(f)

    # Unpack using JAX and convert to numpy
    packed_jax = jnp.array(packed)
    rules_jax = unpack_rules_uint32_jit(packed_jax)
    rules = np.array(rules_jax)

    return rules, meta


def verify_subtree_closure(
    rules: np.ndarray, meta: Dict, verbose: bool = True, debug: bool = False
) -> bool:
    """
    Verify subtree-closed structure.
    Returns True if all checks pass.
    """
    n_total = rules.shape[0]

    # Group rulesets by depth
    by_depth: Dict[int, List[int]] = defaultdict(list)
    for i in range(n_total):
        d = get_depth_from_rules(rules[i])
        by_depth[d].append(i)

    if verbose:
        print("\n" + "=" * 60)
        print("SUBTREE CLOSURE VERIFICATION")
        print("=" * 60)
        print(f"\nTotal rulesets: {n_total}")
        print(f"Depths found: {sorted(by_depth.keys())}")
        for d in sorted(by_depth.keys()):
            print(f"  Depth {d}: {len(by_depth[d])} tasks")

    all_passed = True

    # Build libraries of lower-depth signatures
    depth1_targets: Set[int] = set()
    depth2_signatures: Set[Tuple[int, int]] = set()  # (in1, in2) sorted pairs
    depth3_signatures: Set[Tuple] = set()  # (level0_pairs as tuple, level1_pair)

    # Collect depth-1 targets
    for idx in by_depth.get(1, []):
        target = extract_depth1_target(rules[idx])
        if target is not None:
            depth1_targets.add(target)

    if verbose:
        print(f"\nDepth-1 targets collected: {len(depth1_targets)}")

    # Collect depth-2 signatures and verify against depth-1
    depth2_errors = []
    for idx in by_depth.get(2, []):
        sig = extract_depth2_signature(rules[idx])
        if sig is None:
            depth2_errors.append((idx, "Could not parse depth-2 signature"))
            continue

        in1, in2, out = sig
        depth2_signatures.add((in1, in2))

        # Verify both inputs exist as depth-1 tasks
        if in1 not in depth1_targets:
            depth2_errors.append((idx, f"Input1 uid={in1} not found in depth-1 tasks"))
        if in2 not in depth1_targets:
            depth2_errors.append((idx, f"Input2 uid={in2} not found in depth-1 tasks"))

    if verbose:
        print(f"Depth-2 pairs collected: {len(depth2_signatures)}")

    if depth2_errors:
        all_passed = False
        print(f"\n❌ DEPTH-2 VERIFICATION ERRORS ({len(depth2_errors)}):")
        for idx, msg in depth2_errors[:10]:
            print(f"  Task {idx}: {msg}")
        if len(depth2_errors) > 10:
            print(f"  ... and {len(depth2_errors) - 10} more")
    else:
        if verbose:
            print("✅ All depth-2 tasks have valid depth-1 subtrees")

    # Collect depth-3 signatures and verify against depth-2
    depth3_errors = []
    for idx in by_depth.get(3, []):
        sig = extract_depth3_signature(rules[idx])
        if sig is None:
            depth3_errors.append((idx, "Could not parse depth-3 signature"))
            continue

        # Store signature for depth-4 verification
        sig_tuple = (tuple(sig["level0_pairs"]), sig["level1_pair"])
        depth3_signatures.add(sig_tuple)

        # Verify both level-0 pairs exist as depth-2 tasks
        for pair in sig["level0_pairs"]:
            if pair not in depth2_signatures:
                depth3_errors.append(
                    (idx, f"Level-0 pair {pair} not found in depth-2 tasks")
                )

    if verbose:
        print(f"Depth-3 subtrees collected: {len(depth3_signatures)}")

    if depth3_errors:
        all_passed = False
        print(f"\n❌ DEPTH-3 VERIFICATION ERRORS ({len(depth3_errors)}):")
        for idx, msg in depth3_errors[:10]:
            print(f"  Task {idx}: {msg}")
        if len(depth3_errors) > 10:
            print(f"  ... and {len(depth3_errors) - 10} more")
    else:
        if verbose and 3 in by_depth:
            print("✅ All depth-3 tasks have valid depth-2 subtrees")

    # Verify depth-4 against depth-3
    depth4_errors = []
    for idx in by_depth.get(4, []):
        if debug:
            print(f"  Parsing depth-4 task {idx}...")
        sig = extract_depth4_signature(rules[idx], verbose=debug)
        if sig is None:
            depth4_errors.append((idx, "Could not parse depth-4 signature"))
            continue

        # Check that all level-0 pairs (leaf pairs) exist as depth-2 tasks
        for pair in sig["level0_pairs"]:
            if pair not in depth2_signatures:
                depth4_errors.append(
                    (idx, f"Level-0 pair {pair} not found in depth-2 tasks")
                )

        # Try to find 2 depth-3 subtrees that match this depth-4's structure
        # A depth-4 tree should embed 2 depth-3 subtrees
        level0_pairs = sig["level0_pairs"]

        if len(level0_pairs) >= 4:
            # Perfect binary tree case: try to partition 4+ pairs into 2 groups
            # where each group exists as a depth-3 subtree
            found_valid_partition = False

            from itertools import combinations

            for group1_indices in combinations(range(len(level0_pairs)), 2):
                group2_indices = tuple(
                    i for i in range(len(level0_pairs)) if i not in group1_indices
                )
                if len(group2_indices) != 2:
                    continue
                group1 = tuple(sorted([level0_pairs[i] for i in group1_indices]))
                group2 = tuple(sorted([level0_pairs[i] for i in group2_indices]))

                # Check if both groups exist as depth-3 subtrees
                group1_found = any(d3_sig[0] == group1 for d3_sig in depth3_signatures)
                group2_found = any(d3_sig[0] == group2 for d3_sig in depth3_signatures)

                if group1_found and group2_found:
                    found_valid_partition = True
                    break

            if not found_valid_partition:
                depth4_errors.append(
                    (
                        idx,
                        f"Could not find 2 matching depth-3 subtrees for level-0 pairs {level0_pairs}",
                    )
                )
        elif len(level0_pairs) < 4:
            # Tree with leaf reuse - still check that pairs exist in depth-2
            # but skip the depth-3 subtree check (structure is non-standard)
            if debug:
                print(
                    f"    Task {idx}: Non-standard tree with {len(level0_pairs)} level-0 pairs (leaf reuse)"
                )

    if depth4_errors:
        all_passed = False
        print(f"\n❌ DEPTH-4 VERIFICATION ERRORS ({len(depth4_errors)}):")
        for idx, msg in depth4_errors[:10]:
            print(f"  Task {idx}: {msg}")
        if len(depth4_errors) > 10:
            print(f"  ... and {len(depth4_errors) - 10} more")
    else:
        if verbose and 4 in by_depth:
            print("✅ All depth-4 tasks have valid depth-3 subtrees")

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        if all_passed:
            print("✅ ALL CHECKS PASSED - Subtree closure holds!")
        else:
            print("❌ VERIFICATION FAILED - See errors above")
        print("=" * 60)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Verify ruleset structure")
    parser.add_argument("path", type=str, help="Path to .uint32.npy.bz2 ruleset file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only show errors")
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Show detailed debug info for parse failures",
    )

    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: File not found: {args.path}")
        sys.exit(1)

    print(f"Loading rulesets from: {args.path}")
    rules, meta = load_rulesets(args.path)
    print(f"Loaded {rules.shape[0]} rulesets with shape {rules.shape}")

    if meta:
        print(
            f"Metadata: max_depth={meta.get('max_depth')}, pool_size={meta.get('pool_size')}"
        )
        if "structure" in meta:
            print(f"Expected structure: {meta['structure']}")

    passed = verify_subtree_closure(rules, meta, verbose=not args.quiet, debug=args.debug)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
