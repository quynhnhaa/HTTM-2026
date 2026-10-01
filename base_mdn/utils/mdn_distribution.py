"""Shared MDN parameter decoding used by training, evaluation and artifacts."""

import torch
import torch.distributions as dist


def resolve_parameterization(settings=None):
    """Absent metadata means legacy; epsilon values are implementation choices."""
    settings = dict(settings or {'mode': 'legacy'})
    mode = settings.get('mode', 'legacy')
    if mode == 'legacy':
        return {'mode': 'legacy'}
    raise ValueError(
        f'MDN parameterization {mode!r} is disabled in this legacy version. '
        'Paper checkpoints require the implementation saved in commit 29485b8; '
        'do not decode them with the legacy formula.'
    )


def checkpoint_parameterization(checkpoint):
    return resolve_parameterization(checkpoint.get('resolved_config', {}).get(
        'model_params', {}).get('mdn_parameterization'))


def apply_checkpoint_parameterization(cfg, checkpoint):
    """Decode according to training metadata, including old legacy checkpoints."""
    cfg.model_params['mdn_parameterization'] = checkpoint_parameterization(checkpoint)


def decode_mdn_output(output, num_gaussians, parameterization=None):
    """Decode the shared layout with an explicit, checkpointed policy."""
    resolve_parameterization(parameterization)
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


def build_mdn_distribution(output, num_gaussians, parameterization=None):
    """Build a bivariate Gaussian mixture for every leading output position."""
    params = decode_mdn_output(output, num_gaussians, parameterization)
    components = dist.MultivariateNormal(params['mu'], params['covariance'])
    mixture = dist.Categorical(probs=params['pi'])
    return dist.MixtureSameFamily(mixture, components)
