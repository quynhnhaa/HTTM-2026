import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'base_mdn'))

from create_demo3_calibration import calibration_curves


class Demo3CalibrationTests(unittest.TestCase):
    def test_matches_repository_binning_definition(self):
        bins = np.arange(0.0, 1.01, 0.01)
        confidence = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
        result = calibration_curves(confidence, bins)
        digitized = np.digitize(confidence, bins=bins)
        counts = np.bincount(digitized[:, 0], minlength=len(bins) + 1)
        expected = np.cumsum(counts[1:]) / len(confidence)
        np.testing.assert_allclose(result['observed'][0], expected)
        self.assertEqual(result['observed'].shape, (1, 101))

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            calibration_curves(np.zeros(10), np.arange(0.0, 1.01, 0.01))


if __name__ == '__main__':
    unittest.main()
