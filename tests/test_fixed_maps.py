import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from primal_mapf.config import validate_config
from primal_mapf.fixed_maps import MAPS, evaluation_config, initial_weights, training_config
from primal_mapf.runtime import ROOT, load_map, scenario
from primal_mapf.training import train
from primal_mapf.evaluation.evaluate import evaluate


class FixedMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_configs_use_requested_map_and_separate_outputs(self):
        outputs = set()
        for name, path in MAPS.items():
            expected = load_map(path)
            for variant in ('classic', 'improved'):
                for init in (None, 'base'):
                    config = training_config(name, variant, init)
                    validate_config(config, 'train')
                    actual, starts, goals = scenario(config, 19)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(len(starts), config['agents'])
                    self.assertNotIn(config['output'], outputs)
                    outputs.add(config['output'])
            for weights in ('base', 'fine-tuned', 'trained', 'scratch'):
                config = evaluation_config(name, weights)
                validate_config(config, 'evaluate')
                self.assertEqual(config['map'], path)
                self.assertNotIn(config['output'], outputs)
                outputs.add(config['output'])

    def test_cli_selects_map_specific_weights_and_variant(self):
        result = subprocess.run([sys.executable, str(ROOT/'scripts/fixed_map.py'), 'evaluate',
                                 '--map', 'warehouse', '--variant', 'improved', '--weights', 'fine-tuned',
                                 '--agents', '2', '--episodes', '1', '--dry-run'],
                                text=True, capture_output=True, check=True)
        config = json.loads(result.stdout)
        self.assertEqual(config['models'], {'improved':'checkpoints/fine_tuned/improved/warehouse.pt'})
        self.assertEqual(config['agents'], 2)
        self.assertNotIn('scenarios', config)
        with self.assertRaises(ValueError):
            evaluation_config('maze', 'base', 'both', 'custom.pt')
        self.assertEqual(initial_weights('room', 'classic', 'fine-tuned'), 'checkpoints/fine_tuned/classic/room.pt')

    def test_train_initialize_resume_and_evaluate_each_map_and_variant(self):
        for name in MAPS:
            for variant in ('classic', 'improved'):
                with self.subTest(map=name, variant=variant), tempfile.TemporaryDirectory() as tmp:
                    config = training_config(name, variant)
                    config.update(episodes=1, max_steps=2, update_steps=2, agents=2,
                                  demonstration_prob=0, save_every=1, output=str(Path(tmp)/'scratch'))
                    scratch = train(config)
                    finetune = dict(config, output=str(Path(tmp)/'finetune'))
                    initialized = train(finetune, init=scratch)
                    # Resume through the CLI, recovering its exact stored configuration.
                    command = [sys.executable, str(ROOT/'scripts/fixed_map.py'), 'train', '--map', name,
                               '--variant', variant, '--resume', str(initialized), '--episodes', '2']
                    subprocess.run(command, check=True, capture_output=True, text=True)
                    saved = torch.load(initialized, weights_only=True)
                    self.assertEqual(saved['episode'], 2)
                    np.testing.assert_array_equal(scenario(saved['config'], 42)[0], load_map(MAPS[name]))
                    evaluation = evaluation_config(name, 'scratch', variant, str(initialized))
                    evaluation.pop('scenarios')
                    evaluation.update(episodes=1, max_steps=2, agents=2, output=str(Path(tmp)/'evaluation'))
                    result = evaluate(evaluation)
                    self.assertEqual(set(result), {variant, 'pibt'})
                    records = json.loads((Path(tmp)/'evaluation/scenarios.json').read_text())
                    np.testing.assert_array_equal(records[0]['obstacles'], load_map(MAPS[name]))
                    with self.assertRaises(FileExistsError):
                        train(finetune, init=scratch)

    def test_historical_fine_tuned_weights_when_available(self):
        from primal_mapf.runtime import load_model
        records = ROOT/'checkpoints/fine_tuned/manifest.json'
        if not records.exists():
            self.skipTest('Historical fine-tuned weights have not been imported')
        for name, record in json.loads(records.read_text()).items():
            if not (ROOT/record['file']).exists():
                self.skipTest('Historical fine-tuned weights are local files')
        for name, record in json.loads(records.read_text()).items():
            variant, map_name = name.split('/')
            model = load_model(variant, record['file'])
            config = training_config(map_name, variant)
            grid, starts, goals = scenario(dict(config, agents=2), 123)
            from primal_mapf.evaluation.evaluate import policy_episode
            success, _ = policy_episode(model, variant, grid, starts, goals, 2, 123)
            self.assertIsInstance(success, bool)


if __name__ == '__main__':
    unittest.main()
