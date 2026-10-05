import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / 'base_mdn')):
    if p not in sys.path:
        sys.path.insert(0, p)
from sparsemax_mdn.artifacts import support_statistics  # noqa: E402
from sparsemax_mdn.tests.test_model_loss import make_model, K  # noqa: E402
from sparsemax_mdn.model import SparsemaxMDN  # noqa: E402
from utils.mdn_distribution import decode_mdn_output  # noqa: E402


class SupportStatsTest(unittest.TestCase):
    def test_matches_direct_count(self):
        m = make_model(scale=2.0)
        X = np.random.default_rng(0).normal(size=(40, 32, 4)).astype(np.float32)
        stats = support_statistics(m, X, K, 'cpu', batch_size=16)
        with torch.no_grad():
            pi = decode_mdn_output(m(torch.as_tensor(X)), K)['pi']
        sizes = (pi > 0).sum(-1).float()
        self.assertAlmostEqual(stats['mean_support_size'], sizes.mean().item(), places=6)
        self.assertAlmostEqual(stats['fraction_full_support'], (sizes == K).float().mean().item(), places=6)
        self.assertEqual(stats['dead_components'], int((~(pi > 0).flatten(0, 1).any(0)).sum()))

    def test_dead_component_detected(self):
        m = make_model(scale=0.0)
        with torch.no_grad():
            m.fc.bias.view(48, 6 * K)[:, 5 * K + 3] = -100.0  # component 3 never gets weight
        X = np.random.default_rng(1).normal(size=(10, 32, 4)).astype(np.float32)
        stats = support_statistics(m, X, K, 'cpu')
        self.assertGreaterEqual(stats['dead_components'], 1)


if __name__ == '__main__':
    unittest.main()


class AnalyzeKTest(unittest.TestCase):
    def test_spearman_and_stats(self):
        from sparsemax_mdn.analyze_k import spearman, rankdata, analyze
        self.assertEqual(rankdata([10, 20, 20, 5]).tolist(), [2.0, 3.5, 3.5, 1.0])
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [1, 4, 9, 16]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))
        pi = np.zeros((3, 2, 4)); pi[..., 0] = 1.0; pi[1, :, 1] = 0.5; pi[1, :, 0] = 0.5
        mu = np.zeros((3, 2, 4, 2)); cov = np.tile(np.eye(2), (3, 2, 4, 1, 1))
        res = analyze(pi, mu, cov, np.ones((3, 2, 2)), 4)
        self.assertEqual(res['k_distribution']['histogram']['2'], 2)
        self.assertEqual(res['k_distribution']['histogram']['1'], 4)
        self.assertEqual(res['component_activity']['dead_components'], 2)
