from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from functools import lru_cache

import torch


@dataclass(frozen=True)
class LayoutCatalog:
    """Deterministic isometric layout families for controlled layout diversity."""

    walls: torch.Tensor
    agent_starts: torch.Tensor
    object_slots: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.walls.shape[0])


@lru_cache(maxsize=8)
def build_fixed_layout_catalog(grid_size: int, max_leaves: int) -> LayoutCatalog:
    """Return the original single layout for backward-compatible pilot runs."""
    candidates = (
        (1, grid_size - 2),
        (grid_size - 2, 1),
        (grid_size - 2, grid_size - 2),
        (1, grid_size // 2 - 1),
        (grid_size // 2, 1),
        (grid_size // 2, grid_size - 2),
        (grid_size - 2, grid_size // 2 + 1),
        (1, 2),
    )
    if max_leaves > len(candidates):
        raise ValueError("max_leaves exceeds available fixed object slots")
    walls = torch.zeros((grid_size, grid_size), dtype=torch.bool)
    walls[0, :] = walls[-1, :] = True
    walls[:, 0] = walls[:, -1] = True
    middle = grid_size // 2
    walls[2 : grid_size - 2, middle] = True
    walls[middle, middle] = False
    walls[2, middle] = False
    walls[grid_size - 3, middle] = False
    return LayoutCatalog(
        walls=walls[None],
        agent_starts=torch.tensor(((1, 1),), dtype=torch.int64),
        object_slots=torch.tensor((candidates[:max_leaves],), dtype=torch.int64),
    )


def _rotate_coordinate(row: int, col: int, size: int, turns: int) -> tuple[int, int]:
    for _ in range(turns % 4):
        row, col = col, size - 1 - row
    return row, col


def _connected(walls: set[tuple[int, int]], size: int, start: tuple[int, int]) -> bool:
    free = {
        (row, col)
        for row in range(1, size - 1)
        for col in range(1, size - 1)
        if (row, col) not in walls
    }
    if start not in free:
        return False
    reached = {start}
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        for candidate in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if candidate in free and candidate not in reached:
                reached.add(candidate)
                queue.append(candidate)
    return reached == free


def _distances(
    walls: set[tuple[int, int]], size: int, start: tuple[int, int]
) -> dict[tuple[int, int], int]:
    result = {start: 0}
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        for candidate in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if (
                0 <= candidate[0] < size
                and 0 <= candidate[1] < size
                and candidate not in walls
                and candidate not in result
            ):
                result[candidate] = result[(row, col)] + 1
                queue.append(candidate)
    return result


def _spread_slots(
    walls: set[tuple[int, int]],
    size: int,
    start: tuple[int, int],
    count: int,
) -> tuple[tuple[int, int], ...]:
    candidates = [
        cell
        for cell in _distances(walls, size, start)
        if cell != start
    ]
    selected: list[tuple[int, int]] = []
    while len(selected) < count:
        def score(cell: tuple[int, int]) -> tuple[int, int, int, int]:
            start_distance = _distances(walls, size, start)[cell]
            separation = min(
                (_distances(walls, size, other)[cell] for other in selected),
                default=start_distance,
            )
            return separation, start_distance, cell[0], cell[1]

        choice = max((cell for cell in candidates if cell not in selected), key=score)
        selected.append(choice)
    return tuple(selected)


def _base_layout(
    *, size: int, max_leaves: int, seed: int, family_index: int
) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[tuple[int, int], ...]]:
    border = (
        {(0, col) for col in range(size)}
        | {(size - 1, col) for col in range(size)}
        | {(row, 0) for row in range(size)}
        | {(row, size - 1) for row in range(size)}
    )
    start = (1, 1)
    interior = [
        (row, col)
        for row in range(1, size - 1)
        for col in range(1, size - 1)
        if (row, col) != start
    ]
    wall_count = max(4, size - 1)
    for attempt in range(10_000):
        rng = random.Random(seed + 1_000_003 * family_index + 104_729 * attempt)
        internal = set(rng.sample(interior, wall_count))
        walls = border | internal
        if not _connected(walls, size, start):
            continue
        rotations = {
            tuple(sorted(_rotate_coordinate(row, col, size, turns) for row, col in walls))
            for turns in range(4)
        }
        if len(rotations) != 4:
            continue
        slots = _spread_slots(walls, size, start, max_leaves)
        return walls, start, slots
    raise RuntimeError("Could not generate a connected asymmetric layout family")


@lru_cache(maxsize=16)
def build_layout_catalog(
    count: int, grid_size: int, max_leaves: int, seed: int
) -> LayoutCatalog:
    if count < 1:
        raise ValueError("Layout count must be positive")
    if grid_size < 7:
        raise ValueError("grid_size must be at least 7")
    if max_leaves < 1:
        raise ValueError("max_leaves must be positive")
    walls_out: list[torch.Tensor] = []
    starts_out: list[tuple[int, int]] = []
    slots_out: list[tuple[tuple[int, int], ...]] = []
    signatures: set[bytes] = set()
    family_index = 0
    while len(walls_out) < count:
        walls, start, slots = _base_layout(
            size=grid_size,
            max_leaves=max_leaves,
            seed=seed,
            family_index=family_index,
        )
        family_index += 1
        for turns in range(4):
            rotated = torch.zeros((grid_size, grid_size), dtype=torch.bool)
            for row, col in walls:
                rotated[_rotate_coordinate(row, col, grid_size, turns)] = True
            signature = rotated.numpy().tobytes()
            if signature in signatures:
                continue
            signatures.add(signature)
            walls_out.append(rotated)
            starts_out.append(_rotate_coordinate(*start, grid_size, turns))
            slots_out.append(
                tuple(
                    _rotate_coordinate(row, col, grid_size, turns)
                    for row, col in slots
                )
            )
            if len(walls_out) == count:
                break
    return LayoutCatalog(
        walls=torch.stack(walls_out),
        agent_starts=torch.tensor(starts_out, dtype=torch.int64),
        object_slots=torch.tensor(slots_out, dtype=torch.int64),
    )
