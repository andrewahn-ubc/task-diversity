from __future__ import annotations

import dataclasses
import hashlib
import math
import random
from functools import lru_cache
from typing import Iterable

import numpy as np
import torch


LEAF = 0
UNARY = 1
BINARY = 2
OBJECT_COUNT = 512
PRIMITIVE_COUNT = 8


@dataclasses.dataclass(frozen=True)
class Node:
    op: int
    children: tuple["Node", ...] = ()
    leaf_slot: int = -1

    @property
    def depth(self) -> int:
        if self.op == LEAF:
            return 0
        return 1 + max(child.depth for child in self.children)

    @property
    def leaves(self) -> int:
        if self.op == LEAF:
            return 1
        return sum(child.leaves for child in self.children)

    def signature(self) -> str:
        if self.op == LEAF:
            return "L"
        if self.op == UNARY:
            return f"U({self.children[0].signature()})"
        children = sorted(child.signature() for child in self.children)
        return f"B({children[0]},{children[1]})"


@dataclasses.dataclass(frozen=True)
class Task:
    task_id: int
    phase: int
    topology_index: int
    depth: int
    topology_signature: str
    leaves: tuple[int, ...]
    unary_rules: tuple[tuple[int, int], ...]
    binary_rules: tuple[tuple[int, int, int], ...]
    root_object: int


def unary_output(value: int) -> int:
    bits = value & 0x1FF
    rotated = ((bits << 1) | (bits >> 8)) & 0x1FF
    return PRIMITIVE_COUNT + ((rotated ^ 0x12D) % (OBJECT_COUNT - PRIMITIVE_COUNT))


def binary_output(left: int, right: int) -> int:
    low, high = sorted((left & 0x1FF, right & 0x1FF))
    rot_low = ((low << 2) | (low >> 7)) & 0x1FF
    rot_high = ((high << 5) | (high >> 4)) & 0x1FF
    mixed = rot_low ^ rot_high ^ ((3 * (low + high)) & 0x1FF) ^ 0x0D7
    return PRIMITIVE_COUNT + (mixed % (OBJECT_COUNT - PRIMITIVE_COUNT))


def object_features(feature_dim: int) -> torch.Tensor:
    if feature_dim < 12:
        raise ValueError("object_feature_dim must be at least 12")
    values = np.arange(OBJECT_COUNT, dtype=np.float32)
    features = np.zeros((OBJECT_COUNT, feature_dim), dtype=np.float32)
    for bit in range(9):
        features[:, bit] = ((values.astype(np.int64) >> bit) & 1).astype(np.float32)
    features[:, 9] = values / float(OBJECT_COUNT - 1)
    features[:, 10] = np.sin(values * (2.0 * math.pi / OBJECT_COUNT))
    features[:, 11] = np.cos(values * (2.0 * math.pi / OBJECT_COUNT))
    for index in range(12, feature_dim):
        frequency = index - 10
        features[:, index] = np.sin(values * frequency * (2.0 * math.pi / OBJECT_COUNT))
    return torch.from_numpy(features)


def _random_topology(depth: int, max_leaves: int, rng: random.Random) -> Node:
    leaf_counter = [0]

    def leaf() -> Node:
        slot = leaf_counter[0]
        leaf_counter[0] += 1
        return Node(LEAF, leaf_slot=slot)

    def build(remaining_depth: int, leaves_left: int, force_depth: bool) -> Node:
        if remaining_depth == 0 or leaves_left <= 1 and not force_depth:
            return leaf()
        choose_binary = leaves_left >= 2 and rng.random() < 0.62
        if not choose_binary:
            child_depth = remaining_depth - 1 if force_depth else rng.randrange(remaining_depth)
            return Node(UNARY, (build(child_depth, leaves_left, force_depth),))
        deep_on_left = rng.random() < 0.5
        side_depth = rng.randrange(remaining_depth)
        side_budget = rng.randint(1, max(1, leaves_left - 1))
        deep_budget = max(1, leaves_left - side_budget)
        deep = build(remaining_depth - 1, deep_budget, force_depth)
        side = build(side_depth, side_budget, False)
        children = (deep, side) if deep_on_left else (side, deep)
        return Node(BINARY, children)

    return build(depth, max_leaves, True)


def _curriculum_topology(depth: int, max_leaves: int, variant: int) -> Node:
    """Build a nested anchor whose shallow task teaches object composition.

    The random generator can place several unary transformations below the
    first binary node.  With one topology, that means the depth-1 and depth-2
    curriculum contains no pickup/drop/merge task at all.  Banyan relies on
    nested shallow tasks to teach the operations required by deeper tasks, so
    every distribution receives one deterministic anchor with a binary node at
    depth 1.  Phase zero is a left-deep binary chain; later phases use distinct
    unary/binary patterns while preserving the same curriculum property.
    """
    if depth < 1 or max_leaves < 2:
        raise ValueError("A curriculum topology requires depth >= 1 and max_leaves >= 2")
    patterns = (
        (BINARY, BINARY, BINARY, BINARY, BINARY, BINARY),
        (BINARY, UNARY, BINARY, UNARY, BINARY, UNARY),
        (BINARY, BINARY, UNARY, BINARY, UNARY, BINARY),
        (BINARY, UNARY, UNARY, BINARY, BINARY, UNARY),
    )
    pattern = patterns[variant % len(patterns)]
    node = Node(LEAF, leaf_slot=0)
    leaf_count = 1
    for level in range(depth):
        operation = pattern[level % len(pattern)]
        if operation == BINARY and leaf_count < max_leaves:
            node = Node(BINARY, (node, Node(LEAF, leaf_slot=leaf_count)))
            leaf_count += 1
        else:
            node = Node(UNARY, (node,))
    return node


def _deepest_chain(root: Node) -> dict[int, Node]:
    chain: dict[int, Node] = {root.depth: root}
    node = root
    while node.depth > 0:
        child = max(node.children, key=lambda candidate: candidate.depth)
        node = child
        chain[node.depth] = node
    return chain


def _collect_leaf_slots(root: Node) -> list[int]:
    if root.op == LEAF:
        return [root.leaf_slot]
    slots: list[int] = []
    for child in root.children:
        slots.extend(_collect_leaf_slots(child))
    return slots


def _ground(
    root: Node, assignment: tuple[int, ...]
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int, int]], set[int]]:
    if root.op == LEAF:
        value = assignment[root.leaf_slot]
        return value, [], [], {value}
    child_values: list[int] = []
    unary: list[tuple[int, int]] = []
    binary: list[tuple[int, int, int]] = []
    leaves: set[int] = set()
    for child in root.children:
        value, child_unary, child_binary, child_leaves = _ground(child, assignment)
        child_values.append(value)
        unary.extend(child_unary)
        binary.extend(child_binary)
        leaves.update(child_leaves)
    if root.op == UNARY:
        output = unary_output(child_values[0])
        unary.append((child_values[0], output))
    else:
        left, right = sorted(child_values)
        output = binary_output(left, right)
        binary.append((left, right, output))
    return output, unary, binary, leaves


def _valid_grounding(root: Node, assignment: tuple[int, ...]) -> bool:
    output, unary, binary, leaves = _ground(root, assignment)
    produced = [rule[1] for rule in unary] + [rule[2] for rule in binary]
    inputs_to_outputs: dict[tuple[int, ...], int] = {}
    for source, target in unary:
        inputs_to_outputs[(UNARY, source)] = target
    for left, right, target in binary:
        key = (BINARY, left, right)
        if key in inputs_to_outputs and inputs_to_outputs[key] != target:
            return False
        inputs_to_outputs[key] = target
    return output not in leaves and len(produced) == len(set(produced)) and all(
        source != target for source, target in unary
    )


def _assignment_for(root: Node, seed: int) -> tuple[int, ...]:
    count = root.leaves
    for attempt in range(1000):
        rng = random.Random(seed + 104729 * attempt)
        values = list(range(PRIMITIVE_COUNT))
        rng.shuffle(values)
        assignment = tuple(values[:count])
        if _valid_grounding(root, assignment):
            return assignment
    raise RuntimeError("Could not find a collision-free deterministic object grounding")


class TaskCatalog:
    """Deterministic, phase-disjoint topology catalog shared by every seed."""

    def __init__(
        self,
        *,
        num_phases: int,
        max_diversity: int,
        max_depth: int,
        max_leaves: int,
        seed: int,
    ) -> None:
        self.num_phases = num_phases
        self.max_diversity = max_diversity
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.seed = seed
        total = num_phases * max_diversity
        rng = random.Random(seed)
        # Round-robin assignment below maps global indices 0..num_phases-1 to
        # the first topology in each phase.  Seed that prefix with controlled,
        # curriculum-friendly anchors before filling the remaining catalog
        # with random topologies.
        roots = [
            _curriculum_topology(max_depth, max_leaves, phase)
            for phase in range(num_phases)
        ]
        signatures = {root.signature() for root in roots}
        if len(signatures) != len(roots):
            raise AssertionError("Curriculum anchor topologies must be phase-distinct")
        attempts = 0
        while len(roots) < total:
            attempts += 1
            if attempts > total * 10000:
                raise RuntimeError("Topology generator exhausted before filling the catalog")
            candidate = _random_topology(max_depth, max_leaves, rng)
            signature = candidate.signature()
            if candidate.depth != max_depth or candidate.leaves > max_leaves or signature in signatures:
                continue
            signatures.add(signature)
            roots.append(candidate)
        self.tasks: list[Task] = []
        self.phase_topology_tasks: list[list[tuple[int, ...]]] = [
            [] for _ in range(num_phases)
        ]
        # Round-robin phase assignment balances generator order and tree statistics.
        for global_index, root in enumerate(roots):
            phase = global_index % num_phases
            topology_index = len(self.phase_topology_tasks[phase])
            digest = hashlib.sha256(
                f"{seed}:{global_index}:{root.signature()}".encode()
            ).digest()
            assignment_seed = int.from_bytes(digest[:8], "little")
            assignment = _assignment_for(root, assignment_seed)
            chain = _deepest_chain(root)
            task_ids: list[int] = []
            for depth in range(1, max_depth + 1):
                subtree = chain[depth]
                output, unary, binary, _ = _ground(subtree, assignment)
                slots = _collect_leaf_slots(subtree)
                task_id = len(self.tasks)
                task = Task(
                    task_id=task_id,
                    phase=phase,
                    topology_index=topology_index,
                    depth=depth,
                    topology_signature=subtree.signature(),
                    leaves=tuple(assignment[slot] for slot in slots),
                    unary_rules=tuple(unary),
                    binary_rules=tuple(binary),
                    root_object=output,
                )
                self.tasks.append(task)
                task_ids.append(task_id)
            self.phase_topology_tasks[phase].append(tuple(task_ids))
        if any(len(items) != max_diversity for items in self.phase_topology_tasks):
            raise AssertionError("Round-robin phase construction failed")
        self._build_arrays()

    def _build_arrays(self) -> None:
        max_unary = max((len(task.unary_rules) for task in self.tasks), default=0)
        max_binary = max((len(task.binary_rules) for task in self.tasks), default=0)
        task_count = len(self.tasks)
        self.leaves = torch.full((task_count, self.max_leaves), -1, dtype=torch.int64)
        self.unary = torch.full((task_count, max(1, max_unary), 2), -1, dtype=torch.int64)
        self.binary = torch.full((task_count, max(1, max_binary), 3), -1, dtype=torch.int64)
        self.roots = torch.empty(task_count, dtype=torch.int64)
        self.depths = torch.empty(task_count, dtype=torch.int64)
        for task in self.tasks:
            self.leaves[task.task_id, : len(task.leaves)] = torch.tensor(task.leaves)
            if task.unary_rules:
                self.unary[task.task_id, : len(task.unary_rules)] = torch.tensor(task.unary_rules)
            if task.binary_rules:
                self.binary[task.task_id, : len(task.binary_rules)] = torch.tensor(task.binary_rules)
            self.roots[task.task_id] = task.root_object
            self.depths[task.task_id] = task.depth
        # Dead ends are defined relative to the global dynamics, not merely
        # the current task recipe.  An operation absent from these tables is
        # physically invalid and therefore a no-op.  An operation present
        # globally but absent from the current task is a valid transformation
        # that makes that task unsolvable and should terminate with -1.
        self.global_unary = torch.full((OBJECT_COUNT,), -1, dtype=torch.int64)
        self.global_binary = torch.full(
            (OBJECT_COUNT, OBJECT_COUNT), -1, dtype=torch.int64
        )
        for task in self.tasks:
            for source, target in task.unary_rules:
                existing = int(self.global_unary[source])
                if existing not in (-1, target):
                    raise AssertionError("Unary dynamics are inconsistent across tasks")
                self.global_unary[source] = target
            for left, right, target in task.binary_rules:
                existing = int(self.global_binary[left, right])
                if existing not in (-1, target):
                    raise AssertionError("Binary dynamics are inconsistent across tasks")
                self.global_binary[left, right] = target

    def task_ids(self, phase: int, diversity: int) -> torch.Tensor:
        if not 0 <= phase < self.num_phases:
            raise ValueError(f"Invalid phase {phase}")
        if not 1 <= diversity <= self.max_diversity:
            raise ValueError(f"Invalid diversity {diversity}")
        selected = self.phase_topology_tasks[phase][:diversity]
        return torch.tensor([task for topology in selected for task in topology], dtype=torch.int64)

    def mixed_task_ids(self, phases: Iterable[int], diversity: int) -> torch.Tensor:
        phase_ids = [self.task_ids(phase, diversity) for phase in phases]
        return torch.cat(phase_ids)


@lru_cache(maxsize=8)
def build_catalog(
    num_phases: int, max_diversity: int, max_depth: int, max_leaves: int, seed: int
) -> TaskCatalog:
    return TaskCatalog(
        num_phases=num_phases,
        max_diversity=max_diversity,
        max_depth=max_depth,
        max_leaves=max_leaves,
        seed=seed,
    )
