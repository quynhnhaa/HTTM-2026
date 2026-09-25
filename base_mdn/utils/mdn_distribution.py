"""Shared MDN parameter decoding used by training, evaluation and artifacts."""

import torch
import torch.distributions as dist


def decode_mdn_output(output, num_gaussians):
    """Decode repository MDN layout without changing its parameterization."""
    mu_x = output[..., :num_gaussians]
    mu_y = output[..., num_gaussians:2 * num_gaussians]
    sigma_x = torch.exp(output[..., 2 * num_gaussians:3 * num_gaussians])
    sigma_y = torch.exp(output[..., 3 * num_gaussians:4 * num_gaussians])
    rho = torch.tanh(output[..., 4 * num_gaussians:5 * num_gaussians])
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


def build_mdn_distribution(output, num_gaussians):
    """Build a bivariate Gaussian mixture for every leading output position."""
    params = decode_mdn_output(output, num_gaussians)
    components = dist.MultivariateNormal(params['mu'], params['covariance'])
    mixture = dist.Categorical(probs=params['pi'])
    return dist.MixtureSameFamily(mixture, components)
