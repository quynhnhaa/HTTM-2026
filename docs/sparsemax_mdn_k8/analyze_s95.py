"""Vì sao S95 chính thức của sparsemax K_max = 8 cao hơn baseline K = 3? (ghi data/s95_explained.json)

Dùng: (1) số liệu từng mẫu đã lưu của hai mô hình trên tập test (không chạy lại đánh giá test);
(2) MỘT lần đọc chẩn đoán đầu vào của đúng một mẫu test (mẫu rộng nhất ở 2.4 s) để xem các thành phần của nó;
(3) validation: khối lượng xác suất nằm trên thành phần có sigma lớn. Không dùng kết quả để chỉnh mô hình.

Chạy từ thư mục gốc repo:  .venv/bin/python docs/sparsemax_mdn_k8/analyze_s95.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "base_mdn")):
    sys.path.insert(0, p)
from base_lstm import LSTM_Trajectory_Forecast  # noqa: E402
from sparsemax_mdn.model import SparsemaxMDN  # noqa: E402
from utils.config_loader import ConfigLoader  # noqa: E402
from utils.data_loader import DataLoader  # noqa: E402
from utils.mdn_distribution import decode_mdn_output  # noqa: E402

R = ROOT / "results/trained_models"
K8_RUN = R / "sparsemax_mdn/imptc/sparsemax_k8_peds_imptc/runs/sparsemax_k8_v2_seed2024"
K3_RUN = R / "base_mdn/imptc/default_peds_imptc/runs/imptc_baseline_seed2024"
NPZ = {"K3": R / "sparsemax_mdn/baseline_persample/K3/test_best_persample.npz",
       "K8": K8_RUN / "evaluation/test_best_clampedpi_persample.npz"}
OUT = Path(__file__).resolve().parent / "data/s95_explained.json"
STEPS = [7, 15, 23, 31, 39, 47]


def decompose(areas, hor, dt, fh):
    """Điểm chính thức = trung bình 101 phân vị (0..100) của diện tích theo từng mốc; tách thân (0..99) và phân vị 100."""
    rows = []
    for i, h in enumerate(hor):
        q = np.percentile(areas[:, i], np.arange(0, 101))
        w = 1.0 / ((h + 1) * dt) / (fh * dt)
        rows.append({"time_s": float((h + 1) * dt), "body": float(q[:100].sum() / 101 * w), "p100": float(q[100] / 101 * w),
                     "total": float(q.mean() * w), "median_area_m2": float(np.median(areas[:, i])), "max_area_m2": float(q[100])})
    return rows


def load_model(kind, run, k):
    ck = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    mp = ck["resolved_config"]["model_params"]
    m = SparsemaxMDN(mp) if kind == "sparsemax" else LSTM_Trajectory_Forecast(mp)
    m.load_state_dict(ck["model_state_dict"])
    return m.eval(), k


def main():
    z = {n: np.load(p) for n, p in NPZ.items()}
    hor, dt, fh = z["K3"]["horizons"], float(z["K3"]["delta_t"]), float(z["K3"]["forecast_horizon"])
    levels = list(z["K3"]["confidence_levels"])
    out = {"num_test_samples": int(len(z["K3"]["sample_index"])), "horizons_s": [float((h + 1) * dt) for h in hor]}
    for lvl, name in ((0.95, "s95"), (0.68, "s68")):
        out[name] = {n: decompose(z[n]["sharpness_area_m2"][:, levels.index(lvl), :], hor, dt, fh) for n in z}
        for n in z:
            rows = out[name][n]
            out[name][f"{n}_sum"] = {k: float(sum(r[k] for r in rows)) for k in ("body", "p100", "total")}
    # --- mẫu rộng nhất ở 2.4 s (S95) của K_max = 8
    hi = 2
    a8 = z["K8"]["sharpness_area_m2"][:, levels.index(0.95), :]
    a3 = z["K3"]["sharpness_area_m2"][:, levels.index(0.95), :]
    idx = int(np.argmax(a8[:, hi]))
    top = lambda a: [float(v) for v in np.sort(a[:, hi])[::-1][:6]]
    out["widest_sample"] = {"test_index": idx, "horizon_s": float((hor[hi] + 1) * dt),
                            "top6_area_m2": {"K3": top(a3), "K8": top(a8)},
                            "s95_area_m2_by_horizon": {"K3": a3[idx].tolist(), "K8": a8[idx].tolist()},
                            "aee_m_by_horizon": {"K3": z["K3"]["aee_m"][idx].tolist(), "K8": z["K8"]["aee_m"][idx].tolist()},
                            "coverage_by_horizon": {"K3": z["K3"]["reliability_confidence_sets"][idx].tolist(),
                                                    "K8": z["K8"]["reliability_confidence_sets"][idx].tolist()}}
    # --- chẩn đoán một đầu vào test: thành phần của mẫu đó
    cfg = ConfigLoader(str(ROOT / "base_mdn/configs/imptc/default_peds_imptc.json"), "imptc", False, False, "default_peds_imptc", "base_mdn", "testing")
    loader = DataLoader(cfg)
    loader.load_test_data()
    X, y = loader.get_test_data()[0], loader.get_test_data()[1]
    x = torch.as_tensor(X[idx:idx + 1], dtype=torch.float32)
    models = {"K3": load_model("base", K3_RUN, 3), "K8": load_model("sparsemax", K8_RUN, 8)}
    comp = {}
    for n, (m, k) in models.items():
        with torch.no_grad():
            d = decode_mdn_output(m(x), k)
        pi, mu, sg = d["pi"][0, 23].numpy(), d["mu"][0, 23].numpy(), d["sigma"][0, 23].numpy()
        comp[n] = [{"pi": float(pi[j]), "mu": mu[j].tolist(), "sigma": sg[j].tolist()} for j in np.argsort(-pi) if pi[j] > 1e-6]
    out["widest_sample"]["components_at_2p4s"] = comp
    out["widest_sample"]["last_speed_m_per_s"] = float(np.linalg.norm(X[idx, -1, 2:4]))
    out["widest_sample"]["ground_truth_at_2p4s"] = y[idx, 23].tolist()
    # --- validation: khối lượng xác suất trên thành phần rộng
    cfgv = ConfigLoader(str(ROOT / "base_mdn/configs/imptc/default_peds_imptc.json"), "imptc", False, False, "default_peds_imptc", "base_mdn", "eval")
    lv = DataLoader(cfgv)
    lv.load_eval_data()
    Xv = torch.as_tensor(lv.eval_data[0], dtype=torch.float32)
    prev = {}
    for n, (m, k) in models.items():
        mass = {s: [] for s in STEPS}
        with torch.no_grad():
            for b in range(0, len(Xv), 2048):
                d = decode_mdn_output(m(Xv[b:b + 2048]), k)
                sg = d["sigma"].amax(-1)
                for s in STEPS:
                    mass[s].append((d["pi"][:, s] * (sg[:, s] > 5)).sum(-1).numpy())
        prev[n] = {f"{(s + 1) * dt:.1f}s": {"mean_mass": float(np.concatenate(mass[s]).mean()),
                                            "fraction_samples_mass_gt_5pct": float((np.concatenate(mass[s]) > 0.05).mean())} for s in STEPS}
    out["validation_broad_component_sigma_gt_5m"] = {"num_samples": int(len(Xv)), "by_model_and_horizon": prev}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("s95",)}["s95"]["K3_sum"]), json.dumps(out["s95"]["K8_sum"]))
    print("mẫu rộng nhất:", idx, "-> đã ghi", OUT)


if __name__ == "__main__":
    main()
