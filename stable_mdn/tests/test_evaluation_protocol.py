"""Synthetic protocol checks; no model training or dataset loading."""
import json
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vis import empirical_confidence_cdf
from utils.data_loader import DataLoader
from utils.experiment import isolated_evaluation_rng, set_global_seed
from eval import MDN_Forecaster


class ProtocolTests(unittest.TestCase):
    def test_calibration_thresholds_and_ties(self):
        values = np.array([0., .005, .01, .01, .015, .5, .995, 1.])
        thresholds = np.array([0., .01, .5, .99, 1.])
        expected = np.array([np.mean(values <= q) for q in thresholds])
        np.testing.assert_array_equal(empirical_confidence_cdf(values, thresholds), expected)
        self.assertEqual(expected[0], 1 / 8)
        self.assertEqual(expected[1], 4 / 8)
        self.assertEqual(expected[-1], 1.)

    def test_full_validation_repeated_calls(self):
        loader = DataLoader.__new__(DataLoader)
        loader.eval_data_reduction = 1.
        loader.eval_data = [np.arange(12).reshape(6, 2) + i for i in range(5)]
        loader.sample_ids = {'eval': ['s' + str(i) for i in range(6)]}
        first = loader.get_eval_data()
        random.random()
        second = loader.get_eval_data()
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(loader.current_eval_ids, loader.sample_ids['eval'])
        cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/imptc/stable_peds_imptc.json').read_text())
        self.assertEqual(cfg['train_params']['eval_data_reduction'], 1.)

    def test_evaluation_rng_restored_even_on_failure(self):
        class Evaluation:
            cfg = SimpleNamespace(experiment_params={'seed': 7, 'evaluation_seed': 11})

            @isolated_evaluation_rng
            def run(self, fail=False):
                values = random.random(), np.random.random(), torch.rand(3)
                if fail:
                    raise RuntimeError('intentional')
                return values

        evaluation = Evaluation()
        first, second = evaluation.run(), evaluation.run()
        self.assertEqual(first[:2], second[:2])
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
        for fail in (False, True):
            set_global_seed(23)
            expected = random.random(), np.random.random(), torch.rand(3)
            expected_cuda = torch.rand(3, device='cuda') if torch.cuda.is_available() else None
            set_global_seed(23)
            if fail:
                with self.assertRaises(RuntimeError):
                    evaluation.run(True)
            else:
                evaluation.run()
            actual = random.random(), np.random.random(), torch.rand(3)
            self.assertEqual(expected[:2], actual[:2])
            torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
            if expected_cuda is not None:
                torch.testing.assert_close(expected_cuda, torch.rand(3, device='cuda'), rtol=0, atol=0)
        self.assertTrue(hasattr(MDN_Forecaster.evaluate, '__wrapped__'))
        self.assertTrue(hasattr(MDN_Forecaster.save_examples, '__wrapped__'))


if __name__ == '__main__':
    unittest.main()
