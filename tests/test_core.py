from __future__ import annotations

import unittest

import torch

from banyan_pilot.config import load_config
from banyan_pilot.env import DROP, TOGGLE, VectorBanyan
from banyan_pilot.model import RecurrentActorCritic
from banyan_pilot.ppo import collect_rollout, gradient_cosines, update_ppo
from banyan_pilot.taskgen import build_catalog


class PilotCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config("configs/smoke.toml")
        cls.catalog = build_catalog(4, 256, 6, 7, 260600880)
        cls.device = torch.device("cpu")

    def make_env(self, task_ids: torch.Tensor, num_envs: int = 4) -> VectorBanyan:
        return VectorBanyan(
            self.catalog,
            task_ids,
            num_envs=num_envs,
            grid_size=9,
            max_episode_steps=32,
            device=self.device,
            seed=123,
        )

    def test_catalog_is_phase_disjoint_and_nested(self) -> None:
        signatures = []
        for phase in range(4):
            phase_signatures = {
                self.catalog.tasks[index].topology_signature
                for index in self.catalog.task_ids(phase, 256).tolist()
                if self.catalog.tasks[index].depth == 6
            }
            self.assertEqual(len(phase_signatures), 256)
            signatures.append(phase_signatures)
        for left in range(4):
            for right in range(left + 1, 4):
                self.assertFalse(signatures[left] & signatures[right])
        topology_tasks = self.catalog.phase_topology_tasks[0][0]
        self.assertEqual([self.catalog.tasks[index].depth for index in topology_tasks], list(range(1, 7)))

    def test_unary_goal_success(self) -> None:
        task = next(
            task for task in self.catalog.tasks
            if task.phase == 0 and task.depth == 1 and len(task.unary_rules) == 1
        )
        env = self.make_env(torch.tensor([task.task_id]), num_envs=1)
        slot = env.object_slots[0]
        env.agent[0] = slot
        _, reward, done, info = env.step(torch.tensor([TOGGLE]))
        self.assertTrue(done.item())
        self.assertTrue(info["success"].item())
        self.assertEqual(reward.item(), 1.0)

    def test_binary_goal_and_invalid_merge(self) -> None:
        task = next(
            task for task in self.catalog.tasks
            if task.phase == 0 and task.depth == 1 and len(task.binary_rules) == 1
        )
        env = self.make_env(torch.tensor([task.task_id]), num_envs=2)
        left, right, _ = task.binary_rules[0]
        env.objects[:] = -1
        env.held[:] = torch.tensor([left, 6])
        env.objects[:, 1, 1] = torch.tensor([right, 7])
        _, reward, done, info = env.step(torch.tensor([DROP, DROP]))
        self.assertTrue(info["success"][0].item())
        self.assertEqual(reward[0].item(), 1.0)
        self.assertTrue(info["dead_end"][1].item())
        self.assertEqual(reward[1].item(), -1.0)
        self.assertTrue(done.all().item())

    def test_recurrent_ppo_update_and_gradient_cosine(self) -> None:
        env = self.make_env(self.catalog.task_ids(0, 1), num_envs=4)
        model = RecurrentActorCritic(
            grid_size=9,
            object_feature_dim=16,
            hidden_size=32,
            walls=env.walls,
        )
        hidden = model.initial_hidden(4, self.device)
        starts = torch.ones(4, dtype=torch.bool)
        rollout = collect_rollout(
            model,
            env,
            env.observe(),
            hidden,
            starts,
            rollout_steps=4,
            gamma=0.99,
            gae_lambda=0.95,
        )
        ppo = self.config.ppo
        ppo = type(ppo)(
            **{
                **ppo.__dict__,
                "update_epochs": 1,
                "minibatch_envs": 2,
                "hidden_size": 32,
            }
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=ppo.learning_rate)
        generator = torch.Generator().manual_seed(7)
        losses = update_ppo(model, optimizer, rollout, ppo, generator)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in losses.values()))
        cosines = gradient_cosines(model, rollout, rollout, ppo)
        self.assertAlmostEqual(cosines["cosine_all"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
