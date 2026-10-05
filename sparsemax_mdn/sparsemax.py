"""Sparsemax (Martins & Astudillo, 2016): Euclidean projection onto the simplex."""
import torch


def sparsemax(z, dim=-1):
    """p = max(z - tau, 0) with tau chosen so that p sums to 1; exact zeros are possible.

    Sort/cumsum formula: k(z) = max{k : 1 + k z_(k) > sum_{j<=k} z_(j)},
    tau = (sum_{j<=k(z)} z_(j) - 1) / k(z). Autograd-differentiable.
    """
    z = z.transpose(dim, -1)
    z = z - z.max(dim=-1, keepdim=True).values.detach()  # invariant to shifts; improves precision
    z_sorted, _ = torch.sort(z, dim=-1, descending=True)
    cumsum = z_sorted.cumsum(dim=-1)
    k = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
    in_support = (1 + k * z_sorted) > cumsum
    k_z = in_support.sum(dim=-1, keepdim=True)  # >= 1 always (largest element qualifies)
    tau = (cumsum.gather(-1, k_z - 1) - 1) / k_z.to(z.dtype)
    p = torch.clamp(z - tau, min=0)
    return p.transpose(dim, -1)


def support_size(p):
    """Number of strictly positive entries along the last dimension."""
    return (p > 0).sum(-1)
