from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .layouts import LayoutCatalog, build_fixed_layout_catalog
from .taskgen import TaskCatalog


UP, DOWN, LEFT, RIGHT, STAY, PICKUP, DROP, TOGGLE = range(8)
ACTION_COUNT = 8


@dataclass
class CompactObs:
    objects: torch.Tensor
    agent: torch.Tensor
    goal: torch.Tensor
    held: torch.Tensor
    elapsed: torch.Tensor
    walls: torch.Tensor

    def to(self, device: torch.device | str) -> "CompactObs":
        return CompactObs(*(value.to(device) for value in self.as_tuple()))

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        return self.objects, self.agent, self.goal, self.held, self.elapsed, self.walls

    def clone(self) -> "CompactObs":
        return CompactObs(*(value.clone() for value in self.as_tuple()))


def stack_observations(observations: list[CompactObs]) -> CompactObs:
    return CompactObs(
        *(torch.stack([getattr(obs, name) for obs in observations]) for name in CompactObs.__dataclass_fields__)
    )


class VectorBanyan:
    def __init__(
        self,
        catalog: TaskCatalog,
        task_ids: torch.Tensor,
        *,
        num_envs: int,
        grid_size: int,
        max_episode_steps: int,
        device: torch.device,
        seed: int,
        layout_catalog: LayoutCatalog | None = None,
        layout_ids: torch.Tensor | None = None,
        dead_end_penalty: float = -1.0,
    ) -> None:
        if grid_size < 7:
            raise ValueError("grid_size must be at least 7")
        self.catalog = catalog
        self.device = device
        self.num_envs = num_envs
        self.grid_size = grid_size
        self.max_episode_steps = max_episode_steps
        self.dead_end_penalty = float(dead_end_penalty)
        self.task_pool = task_ids.to(device)
        self.layout_catalog = layout_catalog or build_fixed_layout_catalog(
            grid_size, catalog.max_leaves
        )
        if self.layout_catalog.walls.shape[1:] != (grid_size, grid_size):
            raise ValueError("Layout grid size does not match the environment")
        if self.layout_catalog.object_slots.shape[1] < catalog.max_leaves:
            raise ValueError("Layout catalog has too few object slots")
        if layout_ids is None:
            layout_ids = torch.arange(self.layout_catalog.count, dtype=torch.int64)
        self.layout_pool = layout_ids.to(device)
        if not len(self.layout_pool):
            raise ValueError("layout_ids cannot be empty")
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        self.catalog_walls = self.layout_catalog.walls.to(device)
        self.catalog_agent_starts = self.layout_catalog.agent_starts.to(device)
        self.catalog_object_slots = self.layout_catalog.object_slots.to(device)
        self.catalog_leaves = catalog.leaves.to(device)
        self.catalog_unary = catalog.unary.to(device)
        self.catalog_binary = catalog.binary.to(device)
        self.catalog_global_unary = catalog.global_unary.to(device)
        self.catalog_global_binary = catalog.global_binary.to(device)
        self.catalog_roots = catalog.roots.to(device)
        self.catalog_depths = catalog.depths.to(device)
        self.objects = torch.full(
            (num_envs, grid_size, grid_size), -1, dtype=torch.int64, device=device
        )
        self.agent = torch.zeros((num_envs, 2), dtype=torch.int64, device=device)
        self.held = torch.full((num_envs,), -1, dtype=torch.int64, device=device)
        self.elapsed = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.task_index = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.layout_index = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.root = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.depth = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.walls = torch.zeros(
            (num_envs, grid_size, grid_size), dtype=torch.bool, device=device
        )
        self.object_slots = torch.zeros(
            (num_envs, catalog.max_leaves, 2), dtype=torch.int64, device=device
        )
        self.reset()

    def set_task_pool(self, task_ids: torch.Tensor) -> CompactObs:
        self.task_pool = task_ids.to(self.device)
        return self.reset()

    def set_pools(self, task_ids: torch.Tensor, layout_ids: torch.Tensor) -> CompactObs:
        self.task_pool = task_ids.to(self.device)
        self.layout_pool = layout_ids.to(self.device)
        if not len(self.task_pool) or not len(self.layout_pool):
            raise ValueError("Task and layout pools cannot be empty")
        return self.reset()

    def _sample_tasks(self, count: int) -> torch.Tensor:
        choices = torch.randint(
            0, len(self.task_pool), (count,), generator=self.generator, device=self.device
        )
        return self.task_pool[choices]

    def _sample_layouts(self, count: int) -> torch.Tensor:
        choices = torch.randint(
            0, len(self.layout_pool), (count,), generator=self.generator, device=self.device
        )
        return self.layout_pool[choices]

    def reset(self, mask: torch.Tensor | None = None) -> CompactObs:
        if mask is None:
            indices = torch.arange(self.num_envs, device=self.device)
        else:
            indices = torch.nonzero(mask, as_tuple=False).flatten()
        count = len(indices)
        if count == 0:
            return self.observe()
        tasks = self._sample_tasks(count)
        layouts = self._sample_layouts(count)
        self.task_index[indices] = tasks
        self.layout_index[indices] = layouts
        self.root[indices] = self.catalog_roots[tasks]
        self.depth[indices] = self.catalog_depths[tasks]
        self.walls[indices] = self.catalog_walls[layouts]
        self.object_slots[indices] = self.catalog_object_slots[layouts]
        self.objects[indices] = -1
        self.agent[indices] = self.catalog_agent_starts[layouts]
        self.held[indices] = -1
        self.elapsed[indices] = 0
        leaves = self.catalog_leaves[tasks]
        for slot_index in range(self.catalog.max_leaves):
            values = leaves[:, slot_index]
            valid = values >= 0
            if valid.any():
                valid_indices = indices[valid]
                positions = self.object_slots[valid_indices, slot_index]
                self.objects[valid_indices, positions[:, 0], positions[:, 1]] = values[valid]
        return self.observe()

    def observe(self) -> CompactObs:
        return CompactObs(
            objects=self.objects,
            agent=self.agent,
            goal=self.root,
            held=self.held,
            elapsed=self.elapsed.float() / float(self.max_episode_steps),
            walls=self.walls,
        )

    def step(
        self, actions: torch.Tensor
    ) -> tuple[CompactObs, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        actions = actions.to(self.device).long()
        if actions.shape != (self.num_envs,):
            raise ValueError(f"Expected actions shape {(self.num_envs,)}, got {tuple(actions.shape)}")
        env_ids = torch.arange(self.num_envs, device=self.device)
        proposed = self.agent.clone()
        proposed[:, 0] += (actions == DOWN).long() - (actions == UP).long()
        proposed[:, 1] += (actions == RIGHT).long() - (actions == LEFT).long()
        blocked = self.walls[env_ids, proposed[:, 0], proposed[:, 1]]
        moving = actions <= RIGHT
        apply_move = moving & ~blocked
        self.agent[apply_move] = proposed[apply_move]
        row, col = self.agent[:, 0], self.agent[:, 1]
        cell = self.objects[env_ids, row, col]
        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        dead_end = torch.zeros_like(success)

        pickup = (actions == PICKUP) & (self.held < 0) & (cell >= 0)
        self.held[pickup] = cell[pickup]
        self.objects[env_ids[pickup], row[pickup], col[pickup]] = -1

        drop = (actions == DROP) & (self.held >= 0)
        empty_drop = drop & (cell < 0)
        self.objects[env_ids[empty_drop], row[empty_drop], col[empty_drop]] = self.held[empty_drop]
        self.held[empty_drop] = -1
        merge = drop & (cell >= 0)
        valid_merge = torch.zeros_like(success)
        invalid_merge = torch.zeros_like(success)
        if merge.any():
            merge_ids = env_ids[merge]
            left = torch.minimum(self.held[merge], cell[merge])
            right = torch.maximum(self.held[merge], cell[merge])
            rules = self.catalog_binary[self.task_index[merge]]
            matches = (rules[:, :, 0] == left[:, None]) & (rules[:, :, 1] == right[:, None])
            found = matches.any(dim=1)
            selected = matches.float().argmax(dim=1)
            output = rules[torch.arange(len(merge_ids), device=self.device), selected, 2]
            valid_ids = merge_ids[found]
            if found.any():
                self.objects[valid_ids, row[valid_ids], col[valid_ids]] = output[found]
                self.held[valid_ids] = -1
                success[valid_ids] = output[found] == self.root[valid_ids]
                valid_merge[valid_ids] = True
            missing_ids = merge_ids[~found]
            if len(missing_ids):
                global_output = self.catalog_global_binary[left[~found], right[~found]]
                globally_valid = global_output >= 0
                dead_end[missing_ids[globally_valid]] = True
                invalid_merge[missing_ids[~globally_valid]] = True

        toggle = (actions == TOGGLE) & (cell >= 0)
        valid_toggle = torch.zeros_like(success)
        invalid_toggle = torch.zeros_like(success)
        if toggle.any():
            toggle_ids = env_ids[toggle]
            rules = self.catalog_unary[self.task_index[toggle]]
            matches = rules[:, :, 0] == cell[toggle, None]
            found = matches.any(dim=1)
            selected = matches.float().argmax(dim=1)
            output = rules[torch.arange(len(toggle_ids), device=self.device), selected, 1]
            valid_ids = toggle_ids[found]
            if found.any():
                self.objects[valid_ids, row[valid_ids], col[valid_ids]] = output[found]
                success[valid_ids] = output[found] == self.root[valid_ids]
                valid_toggle[valid_ids] = True
            missing_ids = toggle_ids[~found]
            if len(missing_ids):
                globally_valid = self.catalog_global_unary[cell[toggle][~found]] >= 0
                dead_end[missing_ids[globally_valid]] = True
                invalid_toggle[missing_ids[~globally_valid]] = True

        self.elapsed += 1
        timeout = (self.elapsed >= self.max_episode_steps) & ~success & ~dead_end
        done = success | dead_end | timeout
        rewards[success] = 1.0
        rewards[dead_end] = self.dead_end_penalty
        info = {
            "success": success.clone(),
            "dead_end": dead_end.clone(),
            "timeout": timeout.clone(),
            "task_index": self.task_index.clone(),
            "depth": self.depth.clone(),
            "layout_index": self.layout_index.clone(),
            "movement_effective": apply_move.clone(),
            "pickup_effective": pickup.clone(),
            "drop_effective": (empty_drop | valid_merge).clone(),
            "merge_effective": valid_merge.clone(),
            "toggle_effective": valid_toggle.clone(),
            "merge_invalid": invalid_merge.clone(),
            "toggle_invalid": invalid_toggle.clone(),
        }
        self.reset(done)
        return self.observe(), rewards, done, info

    def state_dict(self) -> dict[str, Any]:
        return {
            "task_pool": self.task_pool.cpu(),
            "layout_pool": self.layout_pool.cpu(),
            "generator_state": self.generator.get_state().cpu(),
            "objects": self.objects.cpu(),
            "agent": self.agent.cpu(),
            "held": self.held.cpu(),
            "elapsed": self.elapsed.cpu(),
            "task_index": self.task_index.cpu(),
            "layout_index": self.layout_index.cpu(),
            "root": self.root.cpu(),
            "depth": self.depth.cpu(),
            "walls": self.walls.cpu(),
            "object_slots": self.object_slots.cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.task_pool = state["task_pool"].to(self.device)
        if "layout_pool" in state:
            self.layout_pool = state["layout_pool"].to(self.device)
        # Generator state is serialized as a CPU ByteTensor. PyTorch expects
        # that representation even when the generator produces CUDA tensors.
        self.generator.set_state(state["generator_state"].cpu())
        for name in (
            "objects",
            "agent",
            "held",
            "elapsed",
            "task_index",
            "layout_index",
            "root",
            "depth",
            "walls",
            "object_slots",
        ):
            if name not in state:
                continue
            setattr(self, name, state[name].to(self.device))
