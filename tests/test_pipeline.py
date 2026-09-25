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


if __name__ == '__main__':
    unittest.main()
