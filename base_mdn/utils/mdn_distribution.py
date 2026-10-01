"""Shared MDN parameter decoding used by training, evaluation and artifacts."""

import torch
import torch.distributions as dist


def decode_mdn_output(output, num_gaussians, rho_bound=None):
    """Decode repository MDN layout without changing its parameterization.

    ``rho_bound`` is opt-in and defaults to ``None``, which reproduces the
    repository's bare ``tanh`` exactly. Passing a value in ``(0, 1)`` applies the
    paper's activation ``rho(o) = tanh(o) * eps_rho`` (Sec. 3), keeping rho
    strictly inside ]-1, 1[ so the covariance cannot become singular.

    Without the bound, rho can reach +-1, the covariance degenerates,
    MultivariateNormal raises, and NLL_MDN_loss reports the run as diverged.
    That is what killed lstm_hidden_size=32 at epoch 713 and 4 of the 5
    ETH/UCY folds at num_gaussians=8.
    """
    mu_x = output[..., :num_gaussians]
    mu_y = output[..., num_gaussians:2 * num_gaussians]
    sigma_x = torch.exp(output[..., 2 * num_gaussians:3 * num_gaussians])
    sigma_y = torch.exp(output[..., 3 * num_gaussians:4 * num_gaussians])
    rho = torch.tanh(output[..., 4 * num_gaussians:5 * num_gaussians])
    if rho_bound:
        rho = rho * rho_bound
    pi = torch.softmax(output[..., 5 * num_gaussians:], dim=-1)

    mu = torch.stack([mu_x, mu_y], dim=-1)
    sigma = torch.stack([sigma_x, sigma_y], dim=-1)
    covariance = torch.zeros(*mu.shape[:-1], 2, 2, device=output.device, dtype=output.dtype)
    covariance[..., 0, 0] = sigma_x.square()
    covariance[..., 0, 1] = rho * sigma_x * sigma_y
    covariance[..., 1, 0] = covariance[..., 0, 1]
    covariance[..., 1, 1] = sigma_y.square()
    return {
        'pi': pi,
        'mu': mu,
        'sigma': sigma,
        'rho': rho,
        'covariance': covariance,
    }


def build_mdn_distribution(output, num_gaussians, rho_bound=None):
    """Build a bivariate Gaussian mixture for every leading output position."""
    params = decode_mdn_output(output, num_gaussians, rho_bound=rho_bound)
    components = dist.MultivariateNormal(params['mu'], params['covariance'])
    mixture = dist.Categorical(probs=params['pi'])
    return dist.MixtureSameFamily(mixture, components)
