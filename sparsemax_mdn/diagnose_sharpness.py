"""Chẩn đoán (chỉ validation): vì sao S68/S95 của sparsemax dao động giữa các mốc epoch?

Tính diện tích tập tin cậy TỪNG MẪU bằng chính các hàm của repo (MDN_Forecaster.build_confidence_set_mdn / estimate_sharpness),
để xem giá trị điển hình (median, p90, p99) và đuôi (max, số mẫu vượt 20 m²/s) thay vì chỉ trung bình.
Chạy: .venv/bin/python sparsemax_mdn/diagnose_sharpness.py N   (N = số mẫu validation đầu tiên, mặc định 600).
Ghi kết quả vào results/trained_models/sparsemax_mdn/sharpness_probe.json.

Ghi chú gốc: per-sample sharpness areas with the repo's OWN functions.

Uses MDN_Forecaster.build_mesh_grid / build_confidence_set_mdn / estimate_sharpness unchanged, on the first N
validation samples, for baseline K=3, baseline K=8 and sparsemax K_max=16 (best checkpoints).
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "base_mdn"))
from base_lstm import LSTM_Trajectory_Forecast  # noqa: E402
from eval import MDN_Forecaster  # noqa: E402
from sparsemax_mdn.model import SparsemaxMDN  # noqa: E402
from utils.config_loader import ConfigLoader  # noqa: E402
from utils.data_loader import DataLoader  # noqa: E402
from utils.experiment import set_global_seed  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
R = ROOT / "results/trained_models"
SPR = R / "sparsemax_mdn/imptc/sparsemax_k16_peds_imptc/runs/sparsemax_k16_seed2024"
B3 = R / "base_mdn/imptc/default_peds_imptc/runs/imptc_baseline_seed2024"
MODELS = {
    "sparsemax ep1500": (SPR, "sparse", "epoch_1500"),
    "baseline K=3 ep1500": (B3, "base", "epoch_1500"),
}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
cfgp = ROOT / "base_mdn/configs/imptc/default_peds_imptc.json"
cfg = ConfigLoader(str(cfgp), "imptc", False, False, cfgp.stem, "base_mdn", "training")
loader = DataLoader(cfg)
loader.load_eval_data()
X = torch.as_tensor(loader.eval_data[0][:N], dtype=torch.float32, device=dev)
HOR = cfg.test_params["test_horizons"]
dt = cfg.model_params["delta_t"]
out = {}
for name, (run, kind, ck) in MODELS.items():
    set_global_seed(2024)
    saved = torch.load(run / "checkpoints" / f"{ck}.pt", map_location="cpu", weights_only=False)
    mp = saved["resolved_config"]["model_params"]
    k = mp["num_gaussians"]
    cfg.model_params["num_gaussians"] = k
    model = (SparsemaxMDN(mp) if kind == "sparse" else LSTM_Trajectory_Forecast(mp)).to(dev)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    f = MDN_Forecaster(cfg, model, loader, "eval", logging.getLogger("probe"), dev)
    grid = f.build_mesh_grid(mesh_range_x=cfg.test_params["mesh_range_x"], mesh_range_y=cfg.test_params["mesh_range_y"],
                             mesh_resolution=cfg.test_params["mesh_resolution"])
    mesh_area = (2 * cfg.test_params["mesh_range_x"]) * (2 * cfg.test_params["mesh_range_y"])
    areas = {0.68: [], 0.95: []}
    with torch.no_grad():
        for s in range(0, N, 50):
            outputs = model(X[s:s + 50])
            filt = torch.stack([outputs[:, h, :] for h in HOR], dim=1)
            for i in range(filt.shape[0]):
                conf = f.build_confidence_set_mdn(output=filt[i][None, ...], target=grid, num_gaussians=k, n_samples=f.n_samples)
                for kappa in (0.68, 0.95):
                    areas[kappa].append((f.estimate_sharpness(conf, kappa=kappa) * mesh_area).cpu().numpy())  # [6]
    res = {}
    for kappa in (0.68, 0.95):
        a = np.stack(areas[kappa])  # [N, 6] m^2
        norm = a / (np.array(HOR)[None, :] + 1) / dt  # m^2/s as in the official score
        per_sample = norm.sum(1) / (cfg.model_params["forecast_horizon"] * dt)
        srt = np.sort(per_sample)[::-1]
        res[str(kappa)] = {
            "official_style_score_mean": float(per_sample.mean()),
            "median_sample": float(np.median(per_sample)), "p90": float(np.percentile(per_sample, 90)),
            "p99": float(np.percentile(per_sample, 99)), "max": float(per_sample.max()),
            "top5pct_share_of_total": float(srt[: max(1, int(0.05 * len(srt)))].sum() / srt.sum()), "p999": float(np.percentile(per_sample, 99.9)), "n_over_20": int((per_sample > 20).sum()), "n": int(len(per_sample)),
            "mean_area_m2_by_horizon": a.mean(0).tolist(), "median_area_m2_by_horizon": np.median(a, 0).tolist()}
    out[name] = res
    print(name, json.dumps({kk: {a: round(b, 3) for a, b in v.items() if not isinstance(b, list)} for kk, v in res.items()}), flush=True)
dest = ROOT / "results/trained_models/sparsemax_mdn/sharpness_probe.json"
dest.write_text(json.dumps(out, indent=2))
print("wrote", dest)
