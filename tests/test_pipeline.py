import sys
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'base_mdn'))

from base_lstm import LSTM_Trajectory_Forecast, NLL_MDN_loss
from utils.data_loader import DataLoader
from utils.experiment import capture_rng_state, restore_rng_state, set_global_seed
from utils.mdn_distribution import build_mdn_distribution, decode_mdn_output
from testing import resolve_test_checkpoint
from create_demo2_horizon_uncertainty import time_indices, region_area
from create_demo1_m1_vs_m3 import (
    choose_candidate, component_sigma, multimodality_score, require_matching_runs,
)


class TestMDNPipeline(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            'lstm_input_shape': 4,
            'lstm_hidden_size': 8,
            'num_gaussians': 3,
            'output_factor': 6,
            'forecast_horizon': 48,
            'lstm_num_layers': 1,
        }

    def test_model_and_parameter_shapes(self):
        model = LSTM_Trajectory_Forecast(self.cfg)
        raw = model(torch.randn(2, 32, 4))
        decoded = decode_mdn_output(raw, 3)
        self.assertEqual(tuple(raw.shape), (2, 48, 18))
        self.assertEqual(tuple(decoded['pi'].shape), (2, 48, 3))
        self.assertEqual(tuple(decoded['mu'].shape), (2, 48, 3, 2))
        self.assertEqual(tuple(decoded['covariance'].shape), (2, 48, 3, 2, 2))

    def test_mixture_weights_normalize_over_components(self):
        raw = torch.randn(2, 6, 18)
        pi = decode_mdn_output(raw, 3)['pi']
        self.assertTrue(torch.allclose(pi.sum(dim=-1), torch.ones(2, 6), atol=1e-6))

    def test_single_gaussian_shapes_and_unit_weight(self):
        cfg = dict(self.cfg, num_gaussians=1)
        model = LSTM_Trajectory_Forecast(cfg)
        raw = model(torch.randn(2, 32, 4))
        decoded = decode_mdn_output(raw, 1)
        self.assertEqual(tuple(raw.shape), (2, 48, 6))
        self.assertEqual(tuple(decoded['pi'].shape), (2, 48, 1))
        self.assertEqual(tuple(decoded['mu'].shape), (2, 48, 1, 2))
        self.assertEqual(tuple(decoded['covariance'].shape), (2, 48, 1, 2, 2))
        self.assertTrue(torch.allclose(decoded['pi'], torch.ones_like(decoded['pi'])))
        loss, diverged = NLL_MDN_loss(raw, torch.randn(2, 48, 2), 1)
        self.assertFalse(diverged)
        self.assertTrue(torch.isfinite(loss))

    def test_training_and_evaluation_share_distribution(self):
        raw = torch.randn(2, 48, 18)
        target = torch.randn(2, 48, 2)
        distribution = build_mdn_distribution(raw, 3)
        expected = -distribution.log_prob(target).mean()
        actual, diverged = NLL_MDN_loss(raw, target, 3)
        self.assertFalse(diverged)
        self.assertTrue(torch.allclose(actual, expected))

    def test_covariance_is_positive_definite(self):
        raw = torch.randn(2, 48, 18)
        covariance = decode_mdn_output(raw, 3)['covariance']
        eigenvalues = torch.linalg.eigvalsh(covariance)
        self.assertTrue(torch.all(eigenvalues > 0))

    def test_rng_state_restoration(self):
        set_global_seed(2024)
        state = capture_rng_state()
        expected_numpy = np.random.random(4)
        expected_torch = torch.rand(4)
        restore_rng_state(state)
        self.assertTrue(np.array_equal(np.random.random(4), expected_numpy))
        self.assertTrue(torch.equal(torch.rand(4), expected_torch))

    def test_corrected_mesh_area(self):
        mesh_range_x = 18
        mesh_range_y = 18
        corrected_area = (2 * mesh_range_x) * (2 * mesh_range_y)
        official_old_area = mesh_range_x * mesh_range_y
        self.assertEqual(corrected_area, 1296)
        self.assertEqual(corrected_area, 4 * official_old_area)

    def test_data_loader_order_restores_exact_next_epoch(self):
        def loader_with_canonical_data():
            loader = DataLoader.__new__(DataLoader)
            loader.randomize = True
            loader.train_data_reduction = 0.3
            loader.train_data = [
                np.arange(80, dtype=np.float32).reshape(20, 4),
                np.arange(40, dtype=np.float32).reshape(20, 2),
            ]
            loader.train_order = np.arange(20, dtype=np.int64)
            return loader

        set_global_seed(2024)
        uninterrupted = loader_with_canonical_data()
        uninterrupted.get_train_data()
        saved_loader_state = uninterrupted.state_dict()
        saved_rng_state = capture_rng_state()
        expected_X, expected_y = uninterrupted.get_train_data()

        resumed = loader_with_canonical_data()
        self.assertTrue(resumed.load_state_dict(saved_loader_state))
        restore_rng_state(saved_rng_state)
        actual_X, actual_y = resumed.get_train_data()
        self.assertTrue(np.array_equal(actual_X, expected_X))
        self.assertTrue(np.array_equal(actual_y, expected_y))
        self.assertTrue(np.array_equal(resumed.train_order, uninterrupted.train_order))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for the resume device test')
    def test_resumed_adam_state_is_on_gpu(self):
        cpu_model = LSTM_Trajectory_Forecast(self.cfg)
        cpu_optimizer = torch.optim.Adam(cpu_model.parameters(), lr=1e-3)
        output = cpu_model(torch.randn(2, 32, 4))
        loss, _ = NLL_MDN_loss(output, torch.randn(2, 48, 2), 3)
        loss.backward()
        cpu_optimizer.step()

        gpu_model = LSTM_Trajectory_Forecast(self.cfg).to('cuda')
        gpu_optimizer = torch.optim.Adam(gpu_model.parameters(), lr=1e-3)
        gpu_optimizer.load_state_dict(cpu_optimizer.state_dict())
        moment_devices = {
            value.device.type
            for state in gpu_optimizer.state.values()
            for key, value in state.items()
            if key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq') and torch.is_tensor(value)
        }
        self.assertEqual(moment_devices, {'cuda'})

        gpu_optimizer.zero_grad()
        output = gpu_model(torch.randn(2, 32, 4, device='cuda'))
        loss, _ = NLL_MDN_loss(output, torch.randn(2, 48, 2, device='cuda'), 3)
        loss.backward()
        gpu_optimizer.step()


class TestCheckpointSelection(unittest.TestCase):
    def _cfg(self, root):
        return SimpleNamespace(result_path=str(Path(root) / 'model_results'))

    @staticmethod
    def _touch_best(cfg, run_id):
        path = Path(cfg.result_path) / 'runs' / run_id / 'checkpoints' / 'best.pt'
        path.parent.mkdir(parents=True)
        path.touch()
        return path.resolve()

    def test_defaults_to_only_run_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            cfg = self._cfg(tmp)
            expected = self._touch_best(cfg, 'baseline')
            self.assertEqual(resolve_test_checkpoint(cfg), expected)

    def test_run_id_selects_its_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            cfg = self._cfg(tmp)
            self._touch_best(cfg, 'run_a')
            expected = self._touch_best(cfg, 'run_b')
            self.assertEqual(resolve_test_checkpoint(cfg, run_id='run_b'), expected)

    def test_explicit_checkpoint_overrides_run(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            cfg = self._cfg(tmp)
            self._touch_best(cfg, 'run_a')
            explicit = Path(tmp) / 'final.pt'
            explicit.touch()
            self.assertEqual(
                resolve_test_checkpoint(cfg, checkpoint=str(explicit), run_id='run_a'),
                explicit.resolve(),
            )

    def test_multiple_runs_require_explicit_run_id(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            cfg = self._cfg(tmp)
            self._touch_best(cfg, 'run_a')
            self._touch_best(cfg, 'run_b')
            with self.assertRaisesRegex(RuntimeError, 'Multiple runs'):
                resolve_test_checkpoint(cfg)

    def test_does_not_silently_fall_back_to_legacy_final(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            cfg = self._cfg(tmp)
            legacy = Path(cfg.result_path) / 'checkpoints' / 'model_final.pt'
            legacy.parent.mkdir(parents=True)
            legacy.touch()
            with self.assertRaisesRegex(FileNotFoundError, 'No run-level best.pt'):
                resolve_test_checkpoint(cfg)


class TestDemo2Helpers(unittest.TestCase):
    def test_time_indices_respect_dataset_sampling(self):
        self.assertEqual(time_indices([1.0, 2.0, 3.0, 4.8], 0.1, 48), [9, 19, 29, 47])

    def test_time_indices_reject_misaligned_time(self):
        with self.assertRaisesRegex(ValueError, 'not aligned'):
            time_indices([1.05, 2.0, 3.0, 4.8], 0.1, 48)

    def test_region_area_uses_full_mesh_area(self):
        confidence = np.asarray([[0.2, 0.7], [0.9, 1.0]])
        self.assertEqual(region_area(confidence, 0.68, 16.0), 4.0)
        self.assertEqual(region_area(confidence, 0.95, 16.0), 12.0)

class TestDemo1Selection(unittest.TestCase):
    """Sample selection must follow EXPERIMENT_PROTOCOL.md 15.2, not the eye."""

    @staticmethod
    def _covariance(sigma):
        return np.asarray([[sigma ** 2, 0.0], [0.0, sigma ** 2]], dtype=np.float64)

    def test_component_sigma_averages_the_diagonal(self):
        covariance = np.asarray([[4.0, 0.0], [0.0, 16.0]])
        self.assertAlmostEqual(component_sigma(covariance), np.sqrt(10.0))

    def test_separated_balanced_mixture_qualifies(self):
        pi = np.asarray([0.45, 0.40, 0.15])
        mu = np.asarray([[-3.0, 0.0], [3.0, 0.0], [0.0, 0.2]])
        covariance = np.stack([self._covariance(0.5)] * 3)
        stats = multimodality_score(pi, mu, covariance, min_weight=0.15)
        self.assertTrue(stats['qualified'])
        self.assertEqual(stats['significant_components'], 3)
        self.assertGreater(stats['max_separation_sigma'], 2.0)
        self.assertGreater(stats['score'], 0.0)

    def test_overlapping_components_do_not_qualify(self):
        """Three components stacked on one spot are not multimodal."""
        pi = np.asarray([0.34, 0.33, 0.33])
        mu = np.asarray([[0.0, 0.0], [0.05, 0.0], [0.0, 0.05]])
        covariance = np.stack([self._covariance(1.0)] * 3)
        stats = multimodality_score(pi, mu, covariance, min_weight=0.15)
        self.assertFalse(stats['qualified'])
        self.assertEqual(stats['score'], 0.0)

    def test_single_dominant_component_does_not_qualify(self):
        pi = np.asarray([0.96, 0.02, 0.02])
        mu = np.asarray([[0.0, 0.0], [5.0, 0.0], [-5.0, 0.0]])
        covariance = np.stack([self._covariance(0.5)] * 3)
        stats = multimodality_score(pi, mu, covariance, min_weight=0.15)
        self.assertFalse(stats['qualified'])
        self.assertEqual(stats['significant_components'], 1)

    def test_choose_candidate_refuses_when_nothing_qualifies(self):
        rows = [{'sample_id': 'a', 'score': 0.0}, {'sample_id': 'b', 'score': 0.0}]
        with self.assertRaises(SystemExit):
            choose_candidate(rows, None, None)

    def test_choose_candidate_honours_explicit_sample_id(self):
        rows = [{'sample_id': 'a', 'score': 9.0}, {'sample_id': 'b', 'score': 1.0}]
        selected, reason = choose_candidate(rows, 'b', None)
        self.assertEqual(selected['sample_id'], 'b')
        self.assertIn('explicit', reason)

    def test_runs_differing_beyond_num_gaussians_are_rejected(self):
        inputs = {
            'sample_ids': np.asarray(['s0']),
            'X': np.zeros((1, 32, 4)),
            'y': np.zeros((1, 48, 2)),
        }
        m1 = (Path('m1'), {'model_params': {'num_gaussians': 1, 'lstm_hidden_size': 8}},
              inputs, None)
        m3 = (Path('m3'), {'model_params': {'num_gaussians': 3, 'lstm_hidden_size': 16}},
              inputs, None)
        with self.assertRaisesRegex(ValueError, 'beyond num_gaussians'):
            require_matching_runs(m1, m3)

    def test_runs_with_different_fixed_samples_are_rejected(self):
        params = {'model_params': {'num_gaussians': 1}}
        other = {'model_params': {'num_gaussians': 3}}
        left = {'sample_ids': np.asarray(['s0']), 'X': np.zeros((1, 32, 4)),
                'y': np.zeros((1, 48, 2))}
        right = {'sample_ids': np.asarray(['s9']), 'X': np.zeros((1, 32, 4)),
                 'y': np.zeros((1, 48, 2))}
        with self.assertRaisesRegex(ValueError, 'same fixed samples'):
            require_matching_runs((Path('a'), params, left, None),
                                  (Path('b'), other, right, None))


if __name__ == '__main__':
    unittest.main()
