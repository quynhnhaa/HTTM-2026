"""Trajectory-level mixture with time-consistent mode probabilities.

The forward tensor deliberately follows the baseline MDN layout so the
repository's marginal evaluator and fixed-sample tracker can read it without
changes. Only the loss and trajectory sampling interpret one mode across time.
"""

import torch
from torch import nn
from torch.nn import init

from utils.mdn_distribution import build_mdn_distribution, decode_mdn_output


class ModeConsistentMDN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_gaussians = int(cfg['num_gaussians'])
        self.forecast_horizon = int(cfg['forecast_horizon'])
        if int(cfg['output_factor']) != 6:
            raise ValueError('The repository MDN layout requires output_factor=6')

        self.lstm = nn.LSTM(
            input_size=int(cfg['lstm_input_shape']),
            hidden_size=int(cfg['lstm_hidden_size']),
            num_layers=int(cfg['lstm_num_layers']),
            batch_first=True,
        )
        hidden = int(cfg['lstm_hidden_size'])
        # Five bivariate Gaussian parameters at each horizon and mode.
        self.trajectory_head = nn.Linear(
            hidden, self.forecast_horizon * self.num_gaussians * 5
        )
        # One categorical distribution for the complete future trajectory.
        self.mode_head = nn.Linear(hidden, self.num_gaussians)

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)
            elif 'bias' in name:
                init.zeros_(param)
        for head in (self.trajectory_head, self.mode_head):
            init.xavier_uniform_(head.weight)
            init.zeros_(head.bias)

    def forward(self, x):
        encoded, _ = self.lstm(x)
        last = encoded[:, -1, :]
        batch = x.shape[0]
        k = self.num_gaussians
        # Layout: all mu_x, all mu_y, all log sigma_x, all log sigma_y,
        # all raw rho, then the same K mode logits at every horizon.
        components = self.trajectory_head(last).reshape(
            batch, self.forecast_horizon, 5 * k
        )
        logits = self.mode_head(last)[:, None, :].expand(
            batch, self.forecast_horizon, k
        )
        return torch.cat((components, logits), dim=-1)


def joint_nll_per_step(output, target, num_gaussians, parameterization=None):
    """Exact trajectory-mixture NLL divided by the fixed forecast horizon.

    The division changes only the scale of the optimization objective. It does
    not turn this into the baseline's mean marginal NLL; those are different
    probability models and must be reported separately.
    """
    if output.ndim != 3 or target.shape != (*output.shape[:2], 2):
        raise ValueError('Expected output [B,H,6K] and target [B,H,2]')
    if output.shape[-1] != 6 * num_gaussians:
        raise ValueError('Output width does not match num_gaussians')
    logits = output[..., 5 * num_gaussians:]
    if not torch.equal(logits, logits[:, :1, :].expand_as(logits)):
        raise ValueError('Mode logits must be shared across every horizon')

    params = decode_mdn_output(output, num_gaussians, parameterization)
    components = torch.distributions.MultivariateNormal(
        params['mu'], covariance_matrix=params['covariance']
    )
    log_position_given_mode = components.log_prob(target.unsqueeze(-2))
    log_mode = torch.log_softmax(logits[:, 0, :], dim=-1)
    joint_log_prob = torch.logsumexp(
        log_mode + log_position_given_mode.sum(dim=1), dim=-1
    )
    return -joint_log_prob.mean() / target.shape[1]


def marginal_nll(output, target, num_gaussians, parameterization=None):
    """The baseline's per-position NLL, for a like-for-like diagnostic."""
    return -build_mdn_distribution(output, num_gaussians, parameterization).log_prob(target).mean()


@torch.no_grad()
def sample_coherent_trajectories(output, num_gaussians, count, parameterization=None):
    """Return [count,B,H,2] samples; each sample keeps one mode for all H."""
    if count <= 0:
        raise ValueError('count must be positive')
    params = decode_mdn_output(output, num_gaussians, parameterization)
    # One shared categorical weight vector is required for trajectory sampling.
    if not torch.equal(params['pi'], params['pi'][:, :1, :].expand_as(params['pi'])):
        raise ValueError('Mode probabilities must be shared across horizons')
    batch, horizon, k, _ = params['mu'].shape
    modes = torch.distributions.Categorical(probs=params['pi'][:, 0, :]).sample((count,))
    # Expand only views; gather then materializes the selected K-independent
    # means/covariances for each of the requested trajectories.
    mu = params['mu'].unsqueeze(0).expand(count, -1, -1, -1, -1)
    covariance = params['covariance'].unsqueeze(0).expand(count, -1, -1, -1, -1, -1)
    mode_index = modes[:, :, None, None, None].expand(-1, -1, horizon, 1, 2)
    selected_mu = mu.gather(3, mode_index).squeeze(3)
    covariance_index = modes[:, :, None, None, None, None].expand(
        -1, -1, horizon, 1, 2, 2
    )
    selected_covariance = covariance.gather(3, covariance_index).squeeze(3)
    return torch.distributions.MultivariateNormal(
        selected_mu, covariance_matrix=selected_covariance
    ).sample(), modes
