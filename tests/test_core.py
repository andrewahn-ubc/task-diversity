from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import PurePosixPath

import numpy as np
import torch

from banyan_pilot.config import CBPConfig, load_config
from banyan_pilot.continual_adam import ContinualAdam
from banyan_pilot.continual_backprop import ContinualBackprop
from banyan_pilot.dependency_audit import audit_resolution
from banyan_pilot.env import DROP, TOGGLE, VectorBanyan
from banyan_pilot.layouts import build_layout_catalog
from banyan_pilot.learnability import summarize
from banyan_pilot.model import RecurrentActorCritic
from banyan_pilot.ppo import collect_rollout, evaluate_policy, gradient_cosines, update_ppo
from banyan_pilot.reanalyze import _clustered_ols
from banyan_pilot.taskgen import build_catalog
from banyan_pilot.train import curriculum_state


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
        for phase in range(4):
            anchor = self.catalog.phase_topology_tasks[phase][0]
            first = self.catalog.tasks[anchor[0]]
            self.assertEqual(first.depth, 1)
            self.assertEqual(len(first.binary_rules), 1)

    def test_learnability_curriculum_leaves_original_reward_training(self) -> None:
        config = load_config("configs/learnability.toml")
        stage = config.curriculum.stage_steps
        self.assertEqual(curriculum_state(config, 0), (1, 0.0, True))
        self.assertEqual(curriculum_state(config, stage), (2, 0.0, True))
        self.assertEqual(curriculum_state(config, 5 * stage), (6, 0.0, True))
        self.assertEqual(curriculum_state(config, 6 * stage), (6, -1.0, False))
        self.assertGreater(config.experiment.steps_per_distribution, 6 * stage)

    def test_learnability_gate_requires_every_depth(self) -> None:
        rows = []
        for algorithm in ("ppo", "ppo_cbp"):
            for diversity in (1, 4):
                for run in range(4):
                    row = {
                        "algorithm": algorithm,
                        "diversity": diversity,
                        "phase_env_steps": 100,
                        "success_rate": 0.2,
                        "timeout_rate": 0.5,
                        "effective_manipulation_rate": 0.1,
                    }
                    row.update(
                        {f"success_depth_{depth}": 0.2 for depth in range(1, 7)}
                    )
                    rows.append(row)
        self.assertEqual(summarize(rows, (1, 4), 100, 0.1)["status"], "pass")
        for row in rows:
            if row["algorithm"] == "ppo_cbp" and row["diversity"] == 1:
                row["success_depth_3"] = 0.0
        self.assertEqual(summarize(rows, (1, 4), 100, 0.1)["status"], "fail")

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

    def test_clustered_ols_recovers_known_coefficient(self) -> None:
        predictor = np.linspace(-1.0, 1.0, 50)
        design = np.column_stack((np.ones(50), predictor))
        outcome = 2.0 + 3.0 * predictor
        clusters = np.repeat(np.arange(25), 2)
        result = _clustered_ols(
            design,
            outcome,
            clusters,
            ("intercept", "predictor"),
            intercept=True,
        )
        self.assertEqual(result["n_run_clusters"], 25)
        self.assertAlmostEqual(
            result["coefficients"]["predictor"]["estimate"], 3.0
        )

    def test_unary_goal_success(self) -> None:
        task = next(
            task for task in self.catalog.tasks
            if task.phase == 0 and task.depth == 1 and len(task.unary_rules) == 1
        )
        env = self.make_env(torch.tensor([task.task_id]), num_envs=1)
        slot = env.object_slots[0, 0]
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
        env = self.make_env(torch.tensor([task.task_id]), num_envs=3)
        left, right, _ = task.binary_rules[0]
        task_pairs = {(a, b) for a, b, _ in task.binary_rules}
        globally_valid_wrong = next(
            (a, b)
            for a, b in (self.catalog.global_binary >= 0).nonzero().tolist()
            if (a, b) not in task_pairs
        )
        globally_invalid = next(
            (a, b)
            for a in range(8)
            for b in range(a, 8)
            if self.catalog.global_binary[a, b] < 0
        )
        env.objects[:] = -1
        env.held[:] = torch.tensor(
            [left, globally_valid_wrong[0], globally_invalid[0]]
        )
        env.objects[:, 1, 1] = torch.tensor(
            [right, globally_valid_wrong[1], globally_invalid[1]]
        )
        _, reward, done, info = env.step(torch.tensor([DROP, DROP, DROP]))
        self.assertTrue(info["success"][0].item())
        self.assertEqual(reward[0].item(), 1.0)
        self.assertTrue(info["dead_end"][1].item())
        self.assertEqual(reward[1].item(), -1.0)
        self.assertTrue(info["merge_invalid"][2].item())
        self.assertFalse(done[2].item())
        self.assertEqual(reward[2].item(), 0.0)
        self.assertEqual(env.held[2].item(), globally_invalid[0])

    def test_layout_catalog_is_distinct_connected_and_checkpointed(self) -> None:
        layouts = build_layout_catalog(4, 9, 7, 260608799)
        self.assertEqual(layouts.count, 4)
        self.assertEqual(
            len({walls.numpy().tobytes() for walls in layouts.walls}), 4
        )
        for layout in range(4):
            start = layouts.agent_starts[layout]
            distances = (layouts.object_slots[layout] - start).abs().sum(dim=1)
            self.assertLessEqual(int(distances.max()), 3)
        env = VectorBanyan(
            self.catalog,
            self.catalog.task_ids(0, 4),
            num_envs=256,
            grid_size=9,
            max_episode_steps=32,
            device=self.device,
            seed=123,
            layout_catalog=layouts,
            layout_ids=torch.arange(4),
        )
        self.assertEqual(set(env.layout_index.tolist()), {0, 1, 2, 3})
        rows = torch.arange(env.num_envs)
        self.assertFalse(
            env.walls[rows, env.agent[:, 0], env.agent[:, 1]].any().item()
        )
        state = env.state_dict()
        env.reset()
        expected_layouts = env.layout_index.clone()
        expected_walls = env.walls.clone()
        env.load_state_dict(state)
        env.reset()
        torch.testing.assert_close(env.layout_index, expected_layouts)
        torch.testing.assert_close(env.walls, expected_walls)

    def test_environment_checkpoint_preserves_random_stream(self) -> None:
        env = self.make_env(self.catalog.task_ids(0, 16), num_envs=8)
        state = env.state_dict()
        env.reset()
        expected_task_index = env.task_index.clone()
        expected_objects = env.objects.clone()
        env.load_state_dict(state)
        env.reset()
        torch.testing.assert_close(env.task_index, expected_task_index)
        torch.testing.assert_close(env.objects, expected_objects)

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

    def test_evaluation_is_equal_weighted_across_depths(self) -> None:
        task_ids = self.catalog.task_ids(0, 1)

        def factory(ids: torch.Tensor, seed: int) -> VectorBanyan:
            return VectorBanyan(
                self.catalog,
                ids,
                num_envs=4,
                grid_size=9,
                max_episode_steps=8,
                device=self.device,
                seed=seed,
            )

        probe = factory(task_ids, 1)
        model = RecurrentActorCritic(
            grid_size=9,
            object_feature_dim=16,
            hidden_size=32,
            walls=probe.walls[0],
        )
        result = evaluate_policy(model, task_ids, factory, episodes=12, seed=17)
        self.assertEqual(result["episodes"], 12)
        self.assertEqual([result[f"episodes_depth_{depth}"] for depth in range(1, 7)], [2] * 6)
        expected = sum(result[f"success_depth_{depth}"] for depth in range(1, 7)) / 6
        self.assertAlmostEqual(result["success_rate"], expected)
        action_rate = sum(
            result[f"action_{name}_rate"]
            for name in ("up", "down", "left", "right", "stay", "pickup", "drop", "toggle")
        )
        self.assertAlmostEqual(action_rate, 1.0)

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
