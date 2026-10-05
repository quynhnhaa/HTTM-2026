import json
import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / 'base_mdn')):
    if p not in sys.path:
        sys.path.insert(0, p)
from base_lstm import LSTM_Trajectory_Forecast, NLL_MDN_loss  # noqa: E402
from utils.mdn_distribution import decode_mdn_output  # noqa: E402
from crps_mdn import base_hashes  # noqa: E402
from crps_mdn.loss import LossStats, crps_xy, make_loss_fn, mixture_crps_1d  # noqa: E402

K, T = 3, 48
CFG = {'lstm_input_shape': 4, 'lstm_hidden_size': 8, 'num_gaussians': K, 'output_factor': 6,
       'forecast_horizon': T, 'lstm_num_layers': 1}
CONF = ROOT / 'crps_mdn/configs/imptc'
BASE = ROOT / 'base_mdn/configs/imptc'


def mc_crps(pi, mu, sigma, y, n=400000, seed=0):
    """Monte-Carlo CRPS = E|X - y| - 1/2 E|X - X'| for a 1-D mixture (single mixture, tensors [K])."""
    g = torch.Generator().manual_seed(seed)
    comp = torch.multinomial(pi, n, replacement=True, generator=g)
    x = mu[comp] + sigma[comp] * torch.randn(n, generator=g)
    comp2 = torch.multinomial(pi, n, replacement=True, generator=g)
    x2 = mu[comp2] + sigma[comp2] * torch.randn(n, generator=g)
    return float((x - y).abs().mean() - 0.5 * (x - x2).abs().mean())


class CrpsFormulaTest(unittest.TestCase):
    def test_single_gaussian_matches_known_formula(self):
        for mu, sigma, y in ((0.0, 1.0, 0.0), (0.5, 2.0, -1.0), (-1.0, 0.3, 2.0)):
            z = (y - mu) / sigma
            phi = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
            cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
            expected = sigma * (z * (2 * cdf - 1) + 2 * phi - 1 / math.sqrt(math.pi))
            got = float(mixture_crps_1d(torch.tensor([1.0], dtype=torch.float64), torch.tensor([mu], dtype=torch.float64),
                                        torch.tensor([sigma], dtype=torch.float64), torch.tensor(y, dtype=torch.float64)))
            self.assertAlmostEqual(got, expected, places=10)

    def test_mixture_matches_monte_carlo(self):
        pi = torch.tensor([0.5, 0.3, 0.2]); mu = torch.tensor([-1.0, 0.5, 3.0]); sigma = torch.tensor([0.4, 1.2, 0.8])
        for y in (-0.7, 0.0, 2.5, 6.0):
            exact = float(mixture_crps_1d(pi, mu, sigma, torch.tensor(y)))
            self.assertAlmostEqual(exact, mc_crps(pi, mu, sigma, y), delta=0.01)

    def test_nonnegative_and_zero_for_point_mass_at_observation(self):
        pi = torch.softmax(torch.randn(10, K), -1); mu = torch.randn(10, K); sigma = torch.rand(10, K) + 0.1
        self.assertTrue(bool((mixture_crps_1d(pi, mu, sigma, torch.randn(10)) >= 0).all()))
        tiny = mixture_crps_1d(torch.tensor([1.0]), torch.tensor([1.5]), torch.tensor([1e-6]), torch.tensor(1.5))
        self.assertLess(abs(float(tiny)), 1e-5)

    def test_gradient_is_finite(self):
        pi = torch.softmax(torch.randn(5, K), -1).requires_grad_(); mu = torch.randn(5, K, requires_grad=True)
        sigma = (torch.rand(5, K) + 0.1).requires_grad_()
        mixture_crps_1d(pi, mu, sigma, torch.randn(5)).sum().backward()
        for t in (pi, mu, sigma):
            self.assertTrue(torch.isfinite(t.grad).all())

    def test_crps_xy_uses_marginals_and_ignores_rho(self):
        torch.manual_seed(0)
        out = torch.randn(4, T, 6 * K)
        target = torch.randn(4, T, 2)
        base = crps_xy(out, target, K)
        changed = out.clone()
        changed[..., 4 * K:5 * K] += 3.0   # rho pre-activation
        torch.testing.assert_close(crps_xy(changed, target, K), base)
        d = decode_mdn_output(out, K)
        manual = (mixture_crps_1d(d['pi'], d['mu'][..., 0], d['sigma'][..., 0], target[..., 0])
                  + mixture_crps_1d(d['pi'], d['mu'][..., 1], d['sigma'][..., 1], target[..., 1])).mean()
        torch.testing.assert_close(base, manual)


class LossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.model = LSTM_Trajectory_Forecast(CFG)
        self.x, self.y = torch.randn(6, 32, 4), torch.randn(6, T, 2)

    def test_lambda_zero_equals_baseline_loss(self):
        stats = LossStats()
        fn = make_loss_fn(self.model.train(), K, 0.0, stats)
        raw = self.model(self.x)
        ours, _ = fn(raw, self.y)
        base, _ = NLL_MDN_loss(raw, self.y, K, None)
        self.assertEqual(float(ours), float(base))

    def test_training_objective_is_nll_plus_lambda_crps_and_stats_recorded(self):
        stats = LossStats()
        fn = make_loss_fn(self.model.train(), K, 1.0, stats)
        raw = self.model(self.x)
        obj, _ = fn(raw, self.y)
        nll, _ = NLL_MDN_loss(raw, self.y, K, None)
        crps = crps_xy(raw, self.y, K)
        self.assertAlmostEqual(float(obj), float(nll + crps), places=5)
        got = stats.consume()
        self.assertAlmostEqual(got['train_nll'], float(nll), places=5)
        self.assertAlmostEqual(got['train_crps'], float(crps), places=5)
        self.assertIsNone(stats.consume())

    def test_eval_mode_returns_pure_nll_and_records_nothing(self):
        stats = LossStats()
        fn = make_loss_fn(self.model.eval(), K, 1.0, stats)
        with torch.no_grad():
            raw = self.model(self.x)
            val, _ = fn(raw, self.y)
            base, _ = NLL_MDN_loss(raw, self.y, K, None)
        self.assertEqual(float(val), float(base))
        self.assertIsNone(stats.consume())

    def test_all_parameters_get_finite_gradient(self):
        fn = make_loss_fn(self.model.train(), K, 1.0, LossStats())
        loss, diverged = fn(self.model(self.x), self.y)
        self.assertFalse(diverged)
        loss.backward()
        for name, p in self.model.named_parameters():
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_non_finite_is_reported_as_divergence(self):
        fn = make_loss_fn(self.model.train(), K, 1.0, LossStats())
        loss, diverged = fn(torch.full((1, T, 6 * K), float('nan')), torch.zeros(1, T, 2))
        self.assertTrue(diverged)
        self.assertIsNone(loss)


class ConfigTest(unittest.TestCase):
    def test_full_config_differs_from_baseline_only_in_declared_keys(self):
        full = json.loads((CONF / 'crps_peds_imptc.json').read_text())
        base = json.loads((BASE / 'default_peds_imptc.json').read_text())
        self.assertEqual(full['experiment_params'].pop('crps'), {'lambda': 1.0})
        for key in ('experiment_name', 'method'):
            full['experiment_params'].pop(key)
            base['experiment_params'].pop(key, None)
        self.assertEqual(full, base)

    def test_smoke_config_is_small(self):
        smoke = json.loads((CONF / 'smoke_crps_peds_imptc.json').read_text())
        self.assertLessEqual(smoke['train_params']['train_epochs'], 3)


class HashGuardTest(unittest.TestCase):
    def test_base_unchanged(self):
        base_hashes.verify()


if __name__ == '__main__':
    unittest.main()
