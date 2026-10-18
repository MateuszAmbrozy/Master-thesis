import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from primal_mapf.config import validate_config
from primal_mapf.environment import MAPFEnv
from primal_mapf.evaluation.evaluate import evaluate, policy_episode
from primal_mapf.gym_env import JointMAPFEnv
from primal_mapf.runtime import ROOT, environment, masked_policy, new_model
from primal_mapf.solvers.api import valid_plan, find_path
from primal_mapf.training import train, LOG_FIELDS, imitation_episode


class CorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def env(self, starts, goals=None):
        env = MAPFEnv(num_agents=len(starts))
        env.set_custom_map(np.zeros((4, 4), dtype=int), starts, goals or starts)
        return env

    def test_masked_policy_gradient(self):
        logits = torch.tensor([0.2, 1.5, -0.8, 0.9, -0.1], requires_grad=True)
        distribution = masked_policy(logits.softmax(-1), [0, 2])
        self.assertEqual(float(distribution[[1, 3, 4]].detach().sum()), 0.)
        torch.testing.assert_close(distribution[[0, 2]], logits[[0, 2]].softmax(-1))
        loss = -distribution[2].log()
        actual, = torch.autograd.grad(loss, logits)
        expected = torch.zeros(5)
        expected[[0, 2]] = logits[[0, 2]].softmax(-1).detach() - torch.tensor([0., 1.])
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        with self.assertRaises(ValueError):
            masked_policy(logits.softmax(-1), [])

    def test_joint_rotation_and_following(self):
        starts = [(0, 0), (0, 1), (1, 1), (1, 0)]
        goals = starts[1:] + starts[:1]
        env = self.env(starts, goals)
        result = env.step_joint([1, 2, 3, 4])
        self.assertTrue(result.done)
        self.assertTrue(all(result.executed))
        self.assertTrue(valid_plan([starts, env.getAgents()], np.zeros((4,4)), starts, goals))
        env = self.env([(0, 0), (0, 1), (0, 2)])
        env.step_joint([1, 1, 1])
        self.assertEqual(env.getAgents(), [(0, 1), (0, 2), (0, 3)])

    def test_joint_conflicts_and_cancellation_chain(self):
        cases = [
            ([(0, 0), (0, 1)], [1, 3]),  # edge swap
            ([(0, 0), (0, 2)], [1, 3]),  # shared destination
            ([(0, 0), (0, 1), (0, 2)], [1, 1, 0]),  # blocked chain
        ]
        for starts, actions in cases:
            with self.subTest(starts=starts):
                env = self.env(starts)
                env.step_joint(actions)
                self.assertEqual(env.getAgents(), starts)
        env = self.env([(0,0)])
        self.assertFalse(env.step_joint([4]).executed[0])
        self.assertEqual(env.getAgents(), [(0,0)])

    def test_random_joint_trajectories_remain_collision_free(self):
        rng = np.random.default_rng(18)
        env = self.env([(0,0),(0,1),(1,1),(2,2)])
        starts = env.getAgents().copy()
        trace = [starts]
        for _ in range(100):
            env.step_joint(rng.integers(0, 5, size=4).tolist())
            trace.append(env.getAgents().copy())
        self.assertTrue(valid_plan(trace, np.zeros((4,4)), starts, trace[-1]))

    def test_malformed_plans_and_instances(self):
        grid = np.zeros((3,3), dtype=int)
        for path in ([], [[(0,0)], [(1.5,0)]], [[(0,0)], [(1,)]], [[(0,0)], [None]]):
            self.assertFalse(valid_plan(path, grid, [(0,0)], [(1,0)]))
        with self.assertRaises(ValueError):
            find_path('pibt', grid, [(0,0),(0,0)], [(1,0),(2,0)])
        with self.assertRaises(ValueError):
            find_path('pibt', grid, [(0,0)], [(5,0)])

    def test_solver_plans_replay_exactly(self):
        grid = np.zeros((5, 5), dtype=int)
        starts, goals = [(0,0), (4,4), (2,1)], [(4,4), (0,0), (2,3)]
        path = find_path("pibt", grid, starts, goals, max_steps=64)
        self.assertIsNotNone(path)
        env = MAPFEnv(num_agents=3)
        env.set_custom_map(grid, starts, goals)
        for previous, next_step in zip(path, path[1:]):
            actions = [env.world.getAction((b[0]-a[0], b[1]-a[1])) for a,b in zip(previous, next_step)]
            self.assertTrue(all(env.step_joint(actions).executed))
            self.assertEqual(env.getAgents(), list(map(tuple, next_step)))

    def test_imitation_accepts_joint_rotation(self):
        starts = [(0,0),(0,1),(1,1),(1,0)]
        goals = starts[1:]+starts[:1]
        config = json.loads((ROOT/'configs/smoke_classic.json').read_text())
        for variant in ('classic', 'improved'):
            env = self.env(starts, goals)
            model = new_model(variant)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            with patch('primal_mapf.training.find_path', return_value=[starts, goals]):
                result = imitation_episode(model, optimizer, variant, env, config, 0)
            self.assertTrue(result.success)
            self.assertTrue(np.isfinite(result.loss))

    def test_common_profile_observations_and_rewards(self):
        grid = np.zeros((4,4), dtype=int)
        envs = [environment(v, grid, [(0,0)], [(3,3)], "common") for v in ('classic', 'improved')]
        np.testing.assert_array_equal(envs[0].get_obs(1)[0], envs[1].get_obs(1)[0])
        self.assertEqual(envs[0].get_obs(1)[1], envs[1].get_obs(1)[1])
        self.assertEqual(envs[0].step_joint([0]).rewards, envs[1].step_joint([0]).rewards)
        self.assertEqual(environment('classic', grid, [(0,0)], [(3,3)]).settings.distance_cap, 75.)

    def test_gymnasium_contract_and_seed(self):
        from gymnasium.utils.env_checker import check_env
        env = JointMAPFEnv(max_steps=2)
        check_env(env, skip_render_check=True)
        first, _ = env.reset(seed=123)
        second, _ = env.reset(seed=123)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        env.step(np.array([0,0]))
        _, _, terminated, truncated, _ = env.step(np.array([0,0]))
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_config_errors_are_clear(self):
        config = json.loads((ROOT/'configs/smoke_classic.json').read_text())
        for key, value in [('episodes', 0), ('seed', -1), ('gamma', 1.5), ('learning_rate', float('nan')),
                           ('variant', 'other'), ('size', [2]), ('execution', 'other')]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_config(dict(config, **{key:value}), 'train')
        with self.assertRaisesRegex(ValueError, 'learning_rate'):
            validate_config({k:v for k,v in config.items() if k != 'learning_rate'}, 'train')

    def test_resume_reconciles_log_after_interruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = json.loads((ROOT/'configs/smoke_classic.json').read_text())
            config.update(output=tmp, episodes=4, save_every=2, max_steps=2)
            from primal_mapf.training import rl_episode
            calls = 0
            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise KeyboardInterrupt
                return rl_episode(*args, **kwargs)
            with patch('primal_mapf.training.rl_episode', side_effect=interrupted):
                train(config)
            checkpoint = Path(tmp)/'checkpoint.pt'
            self.assertEqual(torch.load(checkpoint, weights_only=True)['episode'], 2)
            with (Path(tmp)/'training.csv').open(newline='') as stream:
                self.assertEqual([int(r['episode']) for r in csv.DictReader(stream)], [1,2,3])
            train(config, resume=checkpoint)
            with (Path(tmp)/'training.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(r['episode']) for r in rows], [1,2,3,4])
            self.assertNotEqual(float(rows[-1]['reward']), 999)

    def test_evaluation_without_historical_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = json.loads((ROOT/'configs/smoke_classic.json').read_text())
            config.update(output=str(Path(tmp)/'train'), episodes=1, max_steps=2)
            checkpoint = train(config)
            evaluation = dict(seed=12, agents=2, size=[6,6], density=0, episodes=2, max_steps=3,
                              models={'classic':str(checkpoint)}, solvers=['pibt'], output=str(Path(tmp)/'eval'))
            summary = evaluate(evaluation)
            self.assertEqual(summary['classic']['episodes'], 2)
            self.assertEqual(summary['pibt']['execution'], 'joint')
            self.assertEqual(len(summary['classic']['success_rate_ci95']), 2)
            metadata = json.loads((Path(tmp)/'eval/metadata.json').read_text())
            self.assertIn('classic', metadata['model_sha256'])
            self.assertIn('source_sha256', metadata)
            with self.assertRaises(FileExistsError):
                evaluate(evaluation)


if __name__ == '__main__':
    unittest.main()
