import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'base_mdn'))

from eth_ucy_m_ablation import build_config, prepare, validate_selection


class EthUcyAblationTests(unittest.TestCase):
    def test_only_num_gaussians_changes_within_fold(self):
        m1 = build_config('eth', 1, 2024)
        m8 = build_config('eth', 8, 2024)
        left = dict(m1.model_params)
        right = dict(m8.model_params)
        self.assertEqual(left.pop('num_gaussians'), 1)
        self.assertEqual(right.pop('num_gaussians'), 8)
        self.assertEqual(left, right)
        self.assertEqual(m1.train_params, m8.train_params)
        self.assertEqual(m1.test_params, m8.test_params)
        self.assertEqual(m1.paths, m8.paths)

    def test_fixed_sample_identity_is_independent_of_m(self):
        m1 = build_config('hotel', 1, 2024)
        m5 = build_config('hotel', 5, 2024)
        self.assertEqual(
            m1.experiment_params['fixed_sample_id_prefix'],
            m5.experiment_params['fixed_sample_id_prefix'],
        )
        self.assertEqual(m1.experiment_params['fixed_sample_count'], 8)

    def test_prepare_builds_requested_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = prepare(Path(directory), ('eth', 'univ'), (1, 3), 2024)
            self.assertEqual(len(protocol['jobs']), 4)
            self.assertEqual(
                {(job['fold'], job['num_gaussians']) for job in protocol['jobs']},
                {('eth', 1), ('eth', 3), ('univ', 1), ('univ', 3)},
            )
            self.assertTrue((Path(directory) / 'experiment_matrix.json').is_file())

    def test_rejects_values_outside_protocol(self):
        with self.assertRaises(ValueError):
            validate_selection(('eth', 'unknown'), (1, 3))
        with self.assertRaises(ValueError):
            validate_selection(('eth',), (1, 4))


if __name__ == '__main__':
    unittest.main()
