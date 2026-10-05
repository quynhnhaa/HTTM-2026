"""Validation-only check of the constant-velocity prior (writes reports/CV_PRIOR_CHECK.json).

Verifies the input/target conventions the model relies on and compares the plain CV line with the baseline K = 3
mixture mean. Never touches the test split.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'base_mdn')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from base_lstm import LSTM_Trajectory_Forecast  # noqa: E402
from utils.config_loader import ConfigLoader  # noqa: E402
from utils.data_loader import DataLoader  # noqa: E402
from utils.mdn_distribution import decode_mdn_output  # noqa: E402
from cvres_mdn.model import cv_extrapolation  # noqa: E402

BASE_CONFIG = ROOT / 'base_mdn/configs/imptc/default_peds_imptc.json'
BASE_RUN = ROOT / 'results/trained_models/base_mdn/imptc/default_peds_imptc/runs/imptc_baseline_seed2024'
OUT = ROOT / 'cvres_mdn/reports/CV_PRIOR_CHECK.json'
HORIZON_STEPS = (1, 8, 16, 24, 32, 40, 48)


def main():
    cfg = ConfigLoader(str(BASE_CONFIG), 'imptc', False, False, BASE_CONFIG.stem, 'base_mdn', 'eval')
    loader = DataLoader(cfg)
    loader.load_eval_data()
    X, y = loader.eval_data[0], loader.eval_data[1]
    dt = float(cfg.model_params['delta_t'])
    pos, vel = X[..., 0:2], X[..., 2:4]
    diff = np.diff(pos, axis=1)
    corr = lambda a, b: [float(np.corrcoef(a[..., i].ravel(), b[..., i].ravel())[0, 1]) for i in range(2)]
    facts = {
        'num_samples': int(len(X)), 'delta_t': dt,
        'last_obs_position_abs_max': float(np.abs(pos[:, -1]).max()),
        'corr_vel_vs_diff_pos': corr(diff, vel[:, 1:]),
        'magnitude_ratio_diff_over_vel': float(np.abs(diff).mean() / np.abs(vel[:, 1:]).mean()),
        'corr_y0_vs_last_displacement': [float(np.corrcoef(y[:, 0, i], diff[:, -1, i])[0, 1]) for i in range(2)],
        'mean_y_last_step': y[:, -1].mean(0).tolist()}
    cv = cv_extrapolation(torch.as_tensor(X, dtype=torch.float32), (2, 3), dt, y.shape[1]).numpy()
    err_cv = np.linalg.norm(cv - y, axis=-1)
    checkpoint = torch.load(BASE_RUN / 'checkpoints/best.pt', map_location='cpu', weights_only=False)
    params = checkpoint['resolved_config']['model_params']
    model = LSTM_Trajectory_Forecast(params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        raw = torch.cat([model(torch.as_tensor(X[s:s + 2048], dtype=torch.float32)) for s in range(0, len(X), 2048)])
    decoded = decode_mdn_output(raw, int(params['num_gaussians']))
    mean = (decoded['pi'][..., None] * decoded['mu']).sum(2).numpy()
    err_model = np.linalg.norm(mean - y, axis=-1)
    result = {'split': 'validation', 'facts': facts,
              'cv_vs_baseline_k3_mean_error_m': {
                  str(k): {'cv_line': float(err_cv[:, k - 1].mean()), 'baseline_mixture_mean': float(err_model[:, k - 1].mean())}
                  for k in HORIZON_STEPS},
              'ade_m': {'cv_line': float(err_cv.mean()), 'baseline_mixture_mean': float(err_model.mean())},
              'fraction_samples_cv_better_than_baseline_by_ade': float((err_cv.mean(1) < err_model.mean(1)).mean())}
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
