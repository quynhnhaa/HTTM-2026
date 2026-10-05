"""Baseline LSTM-MDN with a fixed observed-velocity prior on component means."""
import torch

from base_lstm import LSTM_Trajectory_Forecast


class ResidualMDN(LSTM_Trajectory_Forecast):
    def __init__(self, cfg):
        if int(cfg['lstm_input_shape']) != 4 or int(cfg['output_factor']) != 6:
            raise ValueError('ResidualMDN requires [x,y,vx,vy] and the 6K layout')
        super().__init__(cfg)
        self.num_gaussians = int(cfg['num_gaussians'])
        self.delta_t = float(cfg['delta_t'])
        self.velocity_steps = int(cfg['residual_velocity_steps'])
        if self.delta_t <= 0 or self.velocity_steps < 1:
            raise ValueError('Need positive delta_t and velocity_steps')

    def cv_trajectory(self, x):
        """Return [B,H,2] in the same ego frame as observed/target positions."""
        if x.ndim != 3 or x.shape[-1] != 4 or x.shape[1] < self.velocity_steps:
            raise ValueError('Expected [B,T,4] with enough observed velocity steps')
        velocity = x[:, -self.velocity_steps:, 2:4].mean(dim=1)
        seconds = torch.arange(1, self.forecast_horizon + 1,
                               dtype=x.dtype, device=x.device) * self.delta_t
        return x[:, -1:, :2] + velocity[:, None, :] * seconds[None, :, None]

    def residual_output(self, x):
        """Same trainable network as baseline; first 2K entries are residuals."""
        return super().forward(x)

    def forward(self, x):
        raw = self.residual_output(x)
        cv = self.cv_trajectory(x)
        k = self.num_gaussians
        # Final means reach all shared loss/evaluation/capture paths directly.
        # No extra learned weights, no target transformation, no added loss.
        return torch.cat((raw[..., :k] + cv[..., 0:1],
                          raw[..., k:2*k] + cv[..., 1:2],
                          raw[..., 2*k:]), dim=-1)
