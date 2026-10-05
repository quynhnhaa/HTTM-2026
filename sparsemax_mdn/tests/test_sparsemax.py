import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sparsemax_mdn.sparsemax import sparsemax, support_size  # noqa: E402


def reference(z):
    """Slow per-vector reference: Euclidean projection onto the simplex by bisection-free scan."""
    z = np.asarray(z, dtype=np.float64)
    order = np.argsort(-z)
    zs = z[order]
    cssv = np.cumsum(zs)
    k = 0
    for j in range(1, len(z) + 1):
        if 1 + j * zs[j - 1] > cssv[j - 1]:
            k = j
    tau = (cssv[k - 1] - 1) / k
    return np.maximum(z - tau, 0)


class SparsemaxTest(unittest.TestCase):
    def test_known_examples(self):
        for z, p in (([1.0, 0.0], [1.0, 0.0]), ([0.5, 0.5], [0.5, 0.5]),
                     ([0.0, 0.0, 0.0, 0.0], [0.25] * 4), ([2.0, 1.2, -3.0], [0.9, 0.1, 0.0])):
            out = sparsemax(torch.tensor(z, dtype=torch.float64)).numpy()
            np.testing.assert_allclose(out, p, atol=1e-12)

    def test_simplex_properties(self):
        g = torch.Generator().manual_seed(0)
        for dtype in (torch.float32, torch.float64):
            p = sparsemax(torch.randn(5, 7, 8, generator=g, dtype=dtype) * 3)
            self.assertEqual(p.dtype, dtype)
            self.assertTrue((p >= 0).all())
            torch.testing.assert_close(p.sum(-1), torch.ones(5, 7, dtype=dtype), atol=1e-5, rtol=0)

    def test_exact_zeros_for_separated_logits(self):
        p = sparsemax(torch.tensor([5.0, 0.0, -4.0, 4.5]))
        self.assertEqual(p[1].item(), 0.0)
        self.assertEqual(p[2].item(), 0.0)
        self.assertEqual(support_size(p).item(), 2)

    def test_shift_invariance(self):
        z = torch.randn(4, 6, dtype=torch.float64)
        torch.testing.assert_close(sparsemax(z), sparsemax(z + 123.4), atol=1e-10, rtol=0)

    def test_matches_reference(self):
        rng = np.random.default_rng(1)
        for shape in ((8,), (3, 1), (4, 2), (6, 3, 5), (2, 48, 8), (2, 3, 48, 12)):
            z = rng.normal(size=shape) * rng.choice([0.3, 1, 5])
            out = sparsemax(torch.tensor(z)).numpy()
            flat = z.reshape(-1, shape[-1])
            ref = np.stack([reference(r) for r in flat]).reshape(shape)
            np.testing.assert_allclose(out, ref, atol=1e-10)

    def test_dim_argument(self):
        z = torch.randn(3, 5, 4, dtype=torch.float64)
        torch.testing.assert_close(sparsemax(z, dim=1), sparsemax(z.transpose(1, 2), dim=-1).transpose(1, 2))

    def test_gradient_finite_and_jvp(self):
        z = torch.randn(6, 8, dtype=torch.float64, requires_grad=True)
        w = torch.randn(6, 8, dtype=torch.float64)
        (sparsemax(z) * w).sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        # Jacobian-vector product against central finite differences (away from kinks w.p. 1).
        z0 = torch.randn(5, 8, dtype=torch.float64)
        v = torch.randn(5, 8, dtype=torch.float64)
        eps = 1e-6
        fd = (sparsemax(z0 + eps * v) - sparsemax(z0 - eps * v)) / (2 * eps)
        _, jvp = torch.autograd.functional.jvp(sparsemax, z0, v)
        torch.testing.assert_close(jvp, fd, atol=1e-6, rtol=1e-5)

    def test_gradcheck(self):
        z = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(sparsemax, (z,), eps=1e-6, atol=1e-5))

    def test_large_logits_stable(self):
        p = sparsemax(torch.tensor([1e6, 1e6 - 0.5, -1e6]))
        self.assertTrue(torch.isfinite(p).all())
        self.assertAlmostEqual(p.sum().item(), 1.0, places=5)

    def test_support_size(self):
        p = torch.tensor([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
        self.assertEqual(support_size(p).tolist(), [2, 1])


if __name__ == '__main__':
    unittest.main()
