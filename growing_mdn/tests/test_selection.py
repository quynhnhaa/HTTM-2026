import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from growing_mdn.model import build_model
from growing_mdn.selection import component_scores, pick_split, select_k

PARAMS = {'lstm_input_shape': 4, 'lstm_hidden_size': 8, 'lstm_num_layers': 1,
          'num_gaussians': 2, 'output_factor': 6, 'forecast_horizon': 48,
          'mdn_parameterization': {'mode': 'legacy'}}


def constant_model(sigma1_log=2.0, mu1=0.0):
    """K=2 model whose output ignores the input: comp0 N(0,1), comp1 broader."""
    model = build_model(PARAMS, 2)
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.zero_()
        bias = model.fc.bias.view(48, 6, 2)
        bias[:, 2, 1] = sigma1_log
        bias[:, 3, 1] = sigma1_log
        bias[:, 0, 1] = mu1
        bias[:, 1, 1] = mu1
    return model


class ScoreTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2024)

    def test_worse_fitting_component_scores_higher(self):
        model = constant_model()
        X = np.zeros((512, 32, 4), dtype=np.float32)
        y = torch.randn(512, 48, 2).numpy()
        scores = component_scores(model, X, y, 'cpu', 128)
        self.assertEqual(scores.shape, (2,))
        self.assertGreater(scores[1], scores[0])
        self.assertEqual(pick_split(scores), 1)

    def test_dead_component_is_minus_inf(self):
        model = constant_model(sigma1_log=0.0, mu1=100.0)
        X = np.zeros((256, 32, 4), dtype=np.float32)
        y = torch.randn(256, 48, 2).numpy()
        scores = component_scores(model, X, y, 'cpu', 128)
        self.assertTrue(np.isneginf(scores[1]))
        self.assertTrue(np.isfinite(scores[0]))
        self.assertEqual(pick_split(scores), 0)

    def test_pick_split_requires_finite_score(self):
        with self.assertRaises(ValueError):
            pick_split(np.array([-np.inf, -np.inf]))

    def test_pick_split_tie_takes_first(self):
        self.assertEqual(pick_split(np.array([1.0, 3.0, 3.0])), 1)


def record(k, nll, ravg, rmin, s68, s95):
    return {'k': k, 'validation_nll': nll, 'ravg_percent': ravg, 'rmin_percent': rmin,
            's68_m2_per_s': s68, 's95_m2_per_s': s95}


# Values from docs/BASE_MDN_K_COMPARISON.md (test table), used only as shape-of-data fixtures.
RECORDS = [record(1, -0.3446, 86.4988, 76.5309, 3.8193, 8.1405),
           record(2, -0.9265, 95.7992, 90.2453, 5.9493, 11.8354),
           record(3, -1.0921, 97.4709, 94.0646, 1.3575, 6.0610),
           record(5, -1.1317, 97.9062, 95.9466, 2.2009, 7.6275),
           record(8, -1.2066, 98.4943, 96.5872, 0.9022, 5.3482)]


class SelectTest(unittest.TestCase):
    def test_sharpness_guard_keeps_best_nll_k(self):
        out = select_k(RECORDS, 0.1, 1.0, 0.10)
        self.assertEqual(out['best_nll_k'], 8)
        self.assertEqual(out['selected_k'], 8)

    def test_loose_sharpness_selects_smaller_k(self):
        out = select_k(RECORDS, 0.1, 1.0, 10.0)
        self.assertEqual(out['selected_k'], 5)
        self.assertEqual(out['candidate_ks'], [5, 8])

    def test_zero_tolerances_return_best_nll_k(self):
        self.assertEqual(select_k(RECORDS, 0.0, 0.0, 0.0)['selected_k'], 8)

    def test_best_is_smallest_k(self):
        records = [record(1, -2.0, 99, 98, 1.0, 2.0), record(2, -1.0, 90, 80, 2.0, 4.0)]
        self.assertEqual(select_k(records, 0.5, 1.0, 0.1)['selected_k'], 1)

    def test_non_finite_metric_rejected(self):
        bad = RECORDS + [record(10, -1.3, float('nan'), 97, 1.0, 5.0)]
        with self.assertRaises(ValueError):
            select_k(bad, 0.1, 1.0, 0.1)

    def test_negative_tolerance_rejected(self):
        with self.assertRaises(ValueError):
            select_k(RECORDS, -0.1, 1.0, 0.1)


if __name__ == '__main__':
    unittest.main()
