"""LSTM encoder with horizon-specific temporal attention and the baseline MDN head."""

import torch
from torch import nn
from torch.nn import init


class AttentionMDN(nn.Module):
    """Read all observed LSTM states separately for each forecast horizon.

    Output shape and parameter layout are identical to the baseline:
    [batch, 48, 6 * K], with per-horizon bivariate GMM parameters.
    """

    def __init__(self, cfg):
        super().__init__()
        if int(cfg['output_factor']) != 6:
            raise ValueError('The repository MDN layout requires output_factor=6')
        hidden = int(cfg['lstm_hidden_size'])
        heads = int(cfg['attention_heads'])
        if hidden % heads:
            raise ValueError('lstm_hidden_size must be divisible by attention_heads')
        horizon = int(cfg['forecast_horizon'])
        k = int(cfg['num_gaussians'])
        width = int(cfg['decoder_width'])
        self.lstm = nn.LSTM(
            input_size=int(cfg['lstm_input_shape']), hidden_size=hidden,
            num_layers=int(cfg['lstm_num_layers']), batch_first=True,
        )
        self.future_queries = nn.Parameter(torch.empty(horizon, hidden))
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=heads, batch_first=True,
        )
        self.mdn_head = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, width), nn.GELU(),
            nn.Linear(width, 6 * k),
        )
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)
            elif 'bias' in name:
                init.zeros_(param)
        init.normal_(self.future_queries, mean=0.0, std=0.02)
        for module in self.mdn_head.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError('Expected observed trajectory [B,T,4]')
        observed_states, _ = self.lstm(x)
        last_state = observed_states[:, -1:, :]
        future_queries = last_state + self.future_queries.unsqueeze(0)
        context, _ = self.temporal_attention(
            future_queries, observed_states, observed_states,
            need_weights=False,
        )
        return self.mdn_head(torch.cat((future_queries, context), dim=-1))
