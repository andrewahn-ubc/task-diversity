from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import PurePosixPath

import torch

from banyan_pilot.config import CBPConfig, load_config
from banyan_pilot.continual_adam import ContinualAdam
from banyan_pilot.continual_backprop import ContinualBackprop
from banyan_pilot.dependency_audit import audit_resolution
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
        self.assertEqual(self.config.experiment.diversities, (1, 4, 16, 64, 256))
        for diversity in self.config.experiment.diversities:
            self.assertEqual(self.catalog.task_ids(0, diversity).numel(), 6 * diversity)
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

    def test_offline_dependency_report_rejects_non_wheelhouse_source(self) -> None:
        root = PurePosixPath("/cvmfs/soft.computecanada.ca/custom/python/wheelhouse")

        def item(name: str, version: str, path: str, requested: bool = True) -> dict:
            return {
                "download_info": {"url": f"file://{path}"},
                "metadata": {"name": name, "version": version},
                "requested": requested,
            }

        report = {
            "version": "1",
            "environment": {
                "implementation_name": "cpython",
                "platform_machine": "x86_64",
                "platform_system": "Linux",
                "python_version": "3.11",
            },
            "install": [
                item("torch", "2.6.0+computecanada", f"{root}/torch.whl"),
                item("numpy", "1.26.4+computecanada", f"{root}/numpy.whl"),
                item("matplotlib", "3.9.2+computecanada", f"{root}/matplotlib.whl"),
                item("filelock", "3.16.1", f"{root}/filelock.whl", requested=False),
            ]
        }
        self.assertEqual(len(audit_resolution(report)), 4)
        report["install"][-1]["download_info"]["url"] = "file:///home/user/filelock.whl"
        with self.assertRaisesRegex(ValueError, "non-Alliance"):
            audit_resolution(report)

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
        self.assertAlmostEqual(cosines["cosine_all"], 1.0, delta=1e-4)

    def test_continual_adam_matches_torch_adam_before_resets(self) -> None:
        torch.manual_seed(91)
        reference = torch.nn.Parameter(torch.randn(4, 5))
        continual = torch.nn.Parameter(reference.detach().clone())
        reference_optimizer = torch.optim.Adam([reference], lr=2.5e-4, eps=1e-5)
        continual_optimizer = ContinualAdam([continual], lr=2.5e-4, eps=1e-5)
        for _ in range(5):
            gradient = torch.randn_like(reference)
            reference.grad = gradient.clone()
            continual.grad = gradient.clone()
            reference_optimizer.step()
            continual_optimizer.step()
        torch.testing.assert_close(continual, reference, rtol=1e-6, atol=1e-7)
        state = continual_optimizer.state[continual]
        continual_optimizer.reset_state(continual, (torch.tensor([1, 3]), slice(None)))
        self.assertTrue((state["step"][[1, 3]] == 0).all())
        self.assertTrue((state["exp_avg"][[1, 3]] == 0).all())
        self.assertTrue((state["exp_avg_sq"][[1, 3]] == 0).all())
        self.assertTrue((state["step"][[0, 2]] == 5).all())

    def test_cbp_resets_feedforward_recurrent_and_optimizer_state(self) -> None:
        torch.manual_seed(17)
        model = RecurrentActorCritic(
            grid_size=7,
            object_feature_dim=16,
            hidden_size=8,
            walls=torch.zeros(7, 7, dtype=torch.bool),
        )
        optimizer = ContinualAdam(model.parameters(), lr=2.5e-4, eps=1e-5)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        cbp = ContinualBackprop(
            model,
            optimizer,
            CBPConfig(
                enabled=True,
                replacement_rate=1.0,
                decay_rate=0.99,
                maturity_threshold=0,
                utility="contribution",
            ),
        )
        pixels = model.grid_size * model.grid_size
        statistics = {
            "conv1": (torch.ones(32), torch.ones(32)),
            "conv2": (torch.ones(32, pixels), torch.ones(32, pixels)),
            "pre_gru": (torch.ones(8), torch.ones(8)),
            "gru": (torch.ones(8), torch.ones(8)),
        }
        replacements = cbp.step(statistics)
        self.assertEqual(replacements, {"conv1": 32, "conv2": 32, "pre_gru": 8, "gru": 8})
        self.assertTrue((model.actor.weight == 0).all())
        self.assertTrue((model.critic.weight == 0).all())
        self.assertTrue((model.gru.weight_hh == 0).all())
        self.assertTrue((optimizer.state[model.actor.weight]["step"] == 0).all())
        self.assertTrue((optimizer.state[model.gru.weight_hh]["exp_avg"] == 0).all())
        self.assertTrue(torch.isfinite(model.encoder[0].weight).all())
        self.assertGreater(float(model.encoder[0].weight.detach().norm()), 0.0)

        restored = ContinualBackprop(model, optimizer, cbp.config)
        restored.load_state_dict(cbp.state_dict())
        self.assertEqual(restored.totals(), cbp.totals())
        for name in cbp.LAYER_NAMES:
            torch.testing.assert_close(restored.layers[name].ages, cbp.layers[name].ages)

    def test_ppo_update_runs_cbp_after_each_minibatch(self) -> None:
        env = self.make_env(self.catalog.task_ids(0, 1), num_envs=4)
        model = RecurrentActorCritic(
            grid_size=9,
            object_feature_dim=16,
            hidden_size=32,
            walls=env.walls,
        )
        rollout = collect_rollout(
            model,
            env,
            env.observe(),
            model.initial_hidden(4, self.device),
            torch.ones(4, dtype=torch.bool),
            rollout_steps=4,
            gamma=0.99,
            gae_lambda=0.95,
        )
        ppo = replace(
            self.config.ppo,
            update_epochs=1,
            minibatch_envs=2,
            hidden_size=32,
        )
        optimizer = ContinualAdam(model.parameters(), lr=ppo.learning_rate, eps=1e-5)
        cbp = ContinualBackprop(
            model,
            optimizer,
            CBPConfig(
                enabled=True,
                replacement_rate=1.0 / 32.0,
                decay_rate=0.99,
                maturity_threshold=0,
                utility="contribution",
            ),
        )
        losses = update_ppo(
            model,
            optimizer,
            rollout,
            ppo,
            torch.Generator().manual_seed(7),
            cbp,
        )
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in losses.values()))
        self.assertEqual(cbp.totals(), {"conv1": 2, "conv2": 2, "pre_gru": 2, "gru": 2})


if __name__ == "__main__":
    unittest.main()
