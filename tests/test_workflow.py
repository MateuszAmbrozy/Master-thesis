import json
import os
import time
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from primal_mapf.runtime import ROOT, environment, load_model, scenario
from primal_mapf.training import train, imitation_episode
from primal_mapf.solvers.api import find_path, valid_plan
from primal_mapf.evaluation.evaluate import evaluate, policy_episode


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, variant, output):
        config = json.loads((ROOT/f"configs/smoke_{variant}.json").read_text())
        return dict(config, output=str(output), episodes=1, max_steps=3, update_steps=2)

    def test_shared_scenario(self):
        args = dict(agents=2, size=[8, 8], density=.1)
        first, second = scenario(args, 15), scenario(args, 15)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1:], second[1:])
        classic = environment("classic", *first)
        improved = environment("improved", *first)
        np.testing.assert_array_equal(classic.world.state, improved.world.state)
        np.testing.assert_array_equal(classic.world.goals, improved.world.goals)

    def test_invalid_scenario(self):
        with self.assertRaises(ValueError):
            scenario(dict(agents=5, size=[2, 2], density=0), 0)

    def test_invalid_plans(self):
        obstacles = np.zeros((2, 2), dtype=int)
        self.assertFalse(valid_plan([[(0,0),(0,1)],[(0,1),(0,0)]], obstacles,
                                    [(0,0),(0,1)], [(0,1),(0,0)]))
        self.assertFalse(valid_plan([[(0,0)],[(1,1)]], obstacles, [(0,0)], [(1,1)]))
        self.assertIsNone(find_path("pibt", obstacles, [(0,0)], [(1,1)], max_steps=0))

    def check_native(self, name):
        import importlib.util
        if importlib.util.find_spec(f"primal_mapf.solvers.{name}_core") is None:
            if os.environ.get("PRIMAL_REQUIRE_NATIVE") == "1":
                self.fail(f"Required native solver {name} is missing")
            self.skipTest(f"Native solver {name} has not been built")
        obstacles = np.zeros((5,7), dtype=int)
        obstacles[2, 2:5] = 1
        starts, goals = [(0,0),(4,6)], [(0,4),(4,1)]
        path = find_path(name, obstacles, starts, goals, timeout=15)
        self.assertTrue(valid_plan(path, obstacles, starts, goals))

    def test_native_ca(self):
        self.check_native("ca")

    def test_native_cbs(self):
        self.check_native("cbs")

    def test_cbs_conflicts_and_agent_already_at_goal(self):
        import importlib.util
        if importlib.util.find_spec("primal_mapf.solvers.cbs_core") is None:
            if os.environ.get("PRIMAL_REQUIRE_NATIVE") == "1":
                self.fail("CBS extension is required")
            self.skipTest("CBS extension has not been built")
        cases = [
            ([(0,0),(0,2)], [(0,2),(0,0)]),  # vertex conflict
            ([(0,0),(0,1)], [(0,1),(0,0)]),  # edge swap
            ([(0,1),(0,0)], [(0,1),(0,2)]),  # stationary goal must be vacated
        ]
        obstacles = np.zeros((2,3), dtype=int)
        for starts, goals in cases:
            with self.subTest(starts=starts):
                path = find_path("cbs", obstacles, starts, goals, max_steps=32, timeout=10)
                self.assertTrue(valid_plan(path, obstacles, starts, goals))

    def test_native_timeout_terminates_worker(self):
        import importlib.util
        if importlib.util.find_spec("primal_mapf.solvers.cbs_core") is None:
            self.skipTest("CBS extension has not been built")
        started = time.perf_counter()
        path = find_path("cbs", np.zeros((4,4), dtype=int), [(0,0)], [(3,3)], timeout=0.0001)
        self.assertIsNone(path)
        self.assertLess(time.perf_counter()-started, 3.0)

    def test_imitation_updates_both_variants(self):
        for variant in ["classic", "improved"]:
            with self.subTest(variant=variant):
                from primal_mapf.runtime import new_model
                env = environment(variant, np.zeros((4,4), dtype=int), [(0,0),(3,3)], [(0,2),(3,1)])
                model = new_model(variant)
                before = model.policy_layer.weight.detach().clone()
                config = self.config(variant, "unused")
                config.update(max_steps=16)
                optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
                reward, steps, done, loss = imitation_episode(model, optimizer, variant, env, config, 42)
                self.assertTrue(done)
                self.assertGreater(steps, 0)
                self.assertTrue(np.isfinite(loss))
                self.assertFalse(torch.equal(before, model.policy_layer.weight))

    def test_rl_resume_matches_uninterrupted(self):
        for variant in ["classic", "improved"]:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                first = self.config(variant, Path(tmp)/"split")
                train(first)
                checkpoint = Path(first["output"])/"checkpoint.pt"
                initial = torch.load(checkpoint, weights_only=True)["model"]
                continued = dict(first, episodes=2)
                train(continued, resume=checkpoint)
                split = torch.load(checkpoint, weights_only=True)
                full_config = dict(continued, output=str(Path(tmp)/"full"))
                full = torch.load(train(full_config), weights_only=True)
                self.assertEqual(split["episode"], 2)
                for key, value in full["model"].items():
                    self.assertTrue(torch.equal(value, split["model"][key]), key)
                self.assertFalse(torch.equal(initial["policy_layer.weight"], split["model"]["policy_layer.weight"]))
                with self.assertRaises(FileExistsError):
                    train(first)

    def test_historical_models_and_evaluation(self):
        if not all((ROOT/f"checkpoints/base/{variant}.pt").exists() for variant in ["classic", "improved"]):
            self.skipTest("Download historical checkpoints to run compatibility checks")
        for variant in ["classic", "improved"]:
            model = load_model(variant, f"checkpoints/base/{variant}.pt")
            obstacles, starts, goals = scenario(dict(agents=2,size=[8,8],density=0), 12)
            success, steps = policy_episode(model, variant, obstacles, starts, goals, 2, 12)
            self.assertIsInstance(success, bool)
        with tempfile.TemporaryDirectory() as tmp:
            config = json.loads((ROOT/"configs/experiments/smoke.json").read_text())
            config["output"] = tmp
            summary = evaluate(config)
            self.assertEqual(set(summary), {"classic", "improved", "pibt"})
            records = json.loads((Path(tmp)/"scenarios.json").read_text())
            self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
