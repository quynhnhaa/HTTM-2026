import json
import math
import sys
import types
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / 'base_mdn')):
    if p not in sys.path:
        sys.path.insert(0, p)
from sparsemax_mdn import evaluate as ev  # noqa: E402
from utils.mdn_distribution import build_mdn_distribution  # noqa: E402


def raw_output(k, mu, log_sigma, rho_pre, log_pi):
    """[1, 1, 6K] in the repository layout: mu_x, mu_y, log sx, log sy, rho_pre, log pi."""
    mu = np.asarray(mu, dtype=np.float32)
    row = np.concatenate([mu[:, 0], mu[:, 1], [log_sigma] * k, [log_sigma] * k, [rho_pre] * k,
                          np.asarray(log_pi, dtype=np.float32)]).astype(np.float32)
    return torch.tensor(row).reshape(1, 1, -1)


class GuardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        (self.run / 'evaluation').mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, split='validation', confirm=False, limit=None):
        return ev.check_split_request(split, confirm, limit, self.run / 'evaluation')

    def test_validation_needs_no_confirmation(self):
        self.check('validation', False, None)
        self.check('validation', False, 100)

    def test_test_requires_confirmation(self):
        with self.assertRaisesRegex(ValueError, 'confirm-test-once'):
            self.check('test', False, None)
        self.check('test', True, None)

    def test_test_rejects_limit_even_if_confirmed(self):
        with self.assertRaisesRegex(ValueError, 'limit'):
            self.check('test', True, 10)
        with self.assertRaisesRegex(ValueError, 'limit'):
            self.check('test', True, 0)

    def test_confirm_flag_on_validation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'confirm-test-once'):
            self.check('validation', True, None)

    def test_unknown_or_miscased_split_rejected(self):
        for bad in ('TEST', 'Test', ' test', 'test ', 'testing', 'val', '', None, 'Validation'):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.check(bad, True, None)

    def test_limit_must_be_positive(self):
        with self.assertRaises(ValueError):
            self.check('validation', False, 0)
        with self.assertRaises(ValueError):
            self.check('validation', False, -3)

    def test_existing_test_result_blocks_any_combination(self):
        (self.run / 'evaluation' / 'test_best_clampedpi.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.check('test', True, None)
        # a different checkpoint / flag combination is still the same run
        self.assertTrue(ev.existing_test_results(self.run / 'evaluation'))

    def test_validation_results_do_not_block_test(self):
        for name in ('validation_best_exactpi.json', 'validation_best_clampedpi_limited.json'):
            (self.run / 'evaluation' / name).write_text('{}')
        self.check('test', True, None)

    def test_claim_lock_blocks_test(self):
        (self.run / 'evaluation' / ev.TEST_CLAIM).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'already'):
            self.check('test', True, None)


class NamingTest(unittest.TestCase):
    def test_names(self):
        self.assertEqual(ev.output_name('validation', 'best', True, False), 'validation_best_exactpi.json')
        self.assertEqual(ev.output_name('validation', 'epoch_0100', False, True),
                         'validation_epoch_0100_clampedpi_limited.json')
        self.assertEqual(ev.output_name('test', 'final', False, False), 'test_final_clampedpi.json')

    def test_checkpoint_spelling(self):
        for ok in ('best', 'final', 'last', 'epoch_0100'):
            self.assertEqual(ev.validate_checkpoint(ok), ok)
        for bad in ('../best', 'epoch_1', 'BEST', 'best.pt', 'epoch_00100', ''):
            with self.assertRaises(ValueError, msg=bad):
                ev.validate_checkpoint(bad)

    def test_argparse_rejects_miscased_split(self):
        with self.assertRaises(SystemExit):
            with mock.patch('sys.stderr'):
                ev.build_parser().parse_args(['--run-id', 'x', '--split', 'TEST'])


class ExactMixtureTest(unittest.TestCase):
    K = 2

    def reference_log_prob(self, pi, mu, sigma, point):
        """Independent numpy log sum_k pi_k N(point; mu_k, sigma^2 I), dead components skipped."""
        terms = []
        for w, m in zip(pi, mu):
            if w > 0:
                d2 = float(np.sum((np.asarray(point) - np.asarray(m)) ** 2))
                terms.append(math.log(w) - math.log(2 * math.pi * sigma ** 2) - d2 / (2 * sigma ** 2))
        top = max(terms)
        return top + math.log(sum(math.exp(t - top) for t in terms))

    def test_matches_reference_with_zero_weight_components(self):
        mu = [[1.0, -1.0], [0.3, 0.2], [5.0, 5.0], [0.0, 0.0]]
        log_pi = [math.log(0.25), -1e9, -1e9, math.log(0.75)]
        raw = raw_output(4, mu, 0.0, 0.0, log_pi)
        point = torch.tensor([[[0.4, 0.1]]])
        exact = ev.build_exact_distribution(raw, 4, {'mode': 'legacy'}).log_prob(point).item()
        ref = self.reference_log_prob([0.25, 0, 0, 0.75], mu, 1.0, [0.4, 0.1])
        self.assertAlmostEqual(exact, ref, places=5)

    def test_reviewer_example_exact_vs_clamped(self):
        # live component far from the point, dead component sitting exactly at the point
        mu = [[10.0, 0.0], [0.0, 0.0]]
        raw = raw_output(2, mu, 0.0, 0.0, [0.0, -1e9])
        point = torch.zeros(1, 1, 2)
        exact = ev.build_exact_distribution(raw, 2, {'mode': 'legacy'}).log_prob(point).item()
        clamped = build_mdn_distribution(raw, 2, {'mode': 'legacy'}).log_prob(point).item()
        self.assertAlmostEqual(exact, -math.log(2 * math.pi) - 50.0, places=4)  # about -51.84
        eps = float(torch.finfo(torch.float32).eps)
        self.assertAlmostEqual(clamped, math.log(eps) - math.log(2 * math.pi), places=3)  # about -17.78
        self.assertGreater(clamped - exact, 30.0)

    def test_identical_when_all_weights_positive(self):
        raw = raw_output(2, [[1.0, 0.0], [-1.0, 0.5]], 0.0, 0.3, [math.log(0.4), math.log(0.6)])
        point = torch.tensor([[[0.2, 0.1]]])
        a = ev.build_exact_distribution(raw, 2, {'mode': 'legacy'}).log_prob(point)
        b = build_mdn_distribution(raw, 2, {'mode': 'legacy'}).log_prob(point)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_sampling_never_uses_dead_component_and_is_finite(self):
        raw = raw_output(2, [[10.0, 0.0], [-10.0, 0.0]], -3.0, 0.0, [0.0, -1e9])
        gmm = ev.build_exact_distribution(raw, 2, {'mode': 'legacy'})
        s = gmm.sample(torch.Size([2000]))
        self.assertTrue(torch.isfinite(s).all())
        self.assertGreater(s[..., 0].min().item(), 5.0)
        self.assertTrue(torch.isfinite(gmm.log_prob(s)).all())

    def test_gradient_free_logits_have_exact_zero_weight(self):
        raw = raw_output(2, [[0.0, 0.0], [1.0, 1.0]], 0.0, 0.0, [0.0, -1e9])
        gmm = ev.build_exact_distribution(raw, 2, {'mode': 'legacy'})
        self.assertEqual(gmm.mixture_distribution.probs[0, 0, 1].item(), 0.0)


class ForecasterOverrideTest(unittest.TestCase):
    def test_instance_override_changes_only_distribution(self):
        raw = raw_output(2, [[10.0, 0.0], [0.0, 0.0]], 0.0, 0.0, [0.0, -1e9])
        holder = mock.Mock()
        holder.model_params = {'mdn_parameterization': {'mode': 'legacy'}}
        ev.ExactPiMixin.build_distribution(holder, output=raw, num_gaussians=2)  # must run on any host object
        lp = ev.ExactPiMixin.build_distribution(holder, output=raw, num_gaussians=2).log_prob(torch.zeros(1, 1, 2))
        self.assertAlmostEqual(lp.item(), -math.log(2 * math.pi) - 50.0, places=4)


class SupportSummaryTest(unittest.TestCase):
    def test_summary(self):
        a = torch.tensor([[[0.5, 0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])  # supports 2, 1
        b = torch.tensor([[[0.25, 0.25, 0.25, 0.25], [0.5, 0.5, 0.0, 0.0]]])  # supports 4, 2
        s = ev.support_summary([a, b], 4)
        self.assertAlmostEqual(s['mean_support_size'], (2 + 1 + 4 + 2) / 4)
        self.assertAlmostEqual(s['fraction_full_support'], 0.25)
        self.assertEqual(s['dead_components'], 0)
        self.assertEqual(s['num_sample_steps'], 4)
        c = torch.tensor([[[1.0, 0.0, 0.0]]])
        self.assertEqual(ev.support_summary([c], 3)['dead_components'], 2)


class NoTestDataBeforeGuardTest(unittest.TestCase):
    """Refusals happen before DataLoader / torch.load are ever touched."""

    def run_main(self, argv):
        boom = AssertionError('loaded something before the guard refused')
        with mock.patch.object(ev, 'DataLoader', side_effect=boom), \
                mock.patch.object(ev.torch, 'load', side_effect=boom), \
                mock.patch.object(ev, 'ConfigLoader', side_effect=boom), \
                mock.patch.object(ev.base_hashes, 'verify', side_effect=boom):
            return ev.main(argv)

    def test_test_without_confirm(self):
        with self.assertRaisesRegex(ValueError, 'confirm-test-once'):
            self.run_main(['--run-id', 'r', '--split', 'test'])

    def test_test_with_limit(self):
        with self.assertRaisesRegex(ValueError, 'limit'):
            self.run_main(['--run-id', 'r', '--split', 'test', '--limit', '10', '--confirm-test-once'])

    def test_miscased_split_exits_at_argparse(self):
        with mock.patch('sys.stderr'), self.assertRaises(SystemExit):
            self.run_main(['--run-id', 'r', '--split', 'Test', '--confirm-test-once'])


class PerSampleTest(unittest.TestCase):
    H, DT, FH = [4, 9, 19], 0.5, 20

    def test_naming(self):
        self.assertEqual(ev.persample_name('validation', 'best', False, True),
                         'validation_best_clampedpi_limited_persample.npz')
        self.assertEqual(ev.persample_name('test', 'final', True, False), 'test_final_exactpi_persample.npz')

    def test_wrapper_restores_and_returns(self):
        mod = types.SimpleNamespace(plot_sharpness_over_time=lambda data, k=1: ('ret', k),
                                    plot_aee_over_time=lambda data: 1,
                                    plot_reliability_calibration=lambda confidence_sets: 2)
        orig = mod.plot_sharpness_over_time
        arr, store = np.ones((2, 3)), {}
        with ev.capture_plot_inputs(mod, store):
            self.assertEqual(mod.plot_sharpness_over_time(data=arr, k=5), ('ret', 5))
            arr[:] = 7  # captured copy must be independent
        self.assertIs(mod.plot_sharpness_over_time, orig)
        self.assertTrue((store['sharpness_area_m2'] == 1).all())

    def test_wrapper_restores_on_error(self):
        def boom(data):
            raise RuntimeError('x')
        mod = types.SimpleNamespace(plot_sharpness_over_time=boom, plot_aee_over_time=boom,
                                    plot_reliability_calibration=lambda confidence_sets: 0)
        with self.assertRaises(RuntimeError):
            with ev.capture_plot_inputs(mod, {}):
                mod.plot_sharpness_over_time(data=np.zeros(1))
        self.assertIs(mod.plot_sharpness_over_time, boom)
        self.assertIs(mod.plot_aee_over_time, boom)

    def test_mean_matches_official_formula(self):
        rng = np.random.default_rng(0)
        areas = rng.gamma(2.0, 5.0, size=(500, 2, 3))
        scores = ev.sharpness_scores(areas, self.H, self.DT, self.FH)
        # official formula with the mean taken over samples
        ref = [sum(areas[:, lv, i].mean() / ((h + 1) * self.DT) for i, h in enumerate(self.H))
               / (self.FH * self.DT) for lv in range(2)]
        np.testing.assert_allclose(scores.mean(0), ref, rtol=1e-12)

    def test_summary_with_outlier(self):
        areas = np.ones((100, 2, 3))
        areas[0] *= 1e4
        s = ev.sharpness_per_sample_summary(areas, self.H, self.DT, self.FH, [0.95, 0.68])
        a = s['s95']
        self.assertEqual(a['num_samples'], 100)
        self.assertEqual(a['num_above_20'], 1)
        self.assertEqual(a['max'], ev.sharpness_scores(areas, self.H, self.DT, self.FH)[0, 0])
        self.assertAlmostEqual(a['median'], ev.sharpness_scores(areas, self.H, self.DT, self.FH)[1, 0])
        self.assertGreater(a['top5pct_share_of_total'], 0.9)
        self.assertEqual(set(s), {'s95', 's68'})

    def test_official_percentile_equals_mean_for_constant_data(self):
        areas = np.full((10, 2, 3), 3.0)
        np.testing.assert_allclose(ev.official_percentile_scores(areas, self.H, self.DT, self.FH),
                                   ev.sharpness_scores(areas, self.H, self.DT, self.FH).mean(0))


if __name__ == '__main__':
    unittest.main()
