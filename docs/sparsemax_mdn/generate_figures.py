"""Sinh hình minh họa cho SPARSEMAX_MDN.md.

Nhãn nguồn dữ liệu của từng hình:
  * [thật]    : đọc từ artifact trong results/ (baseline, run sparsemax đang chạy, phân tích K);
  * [tính]    : tính trực tiếp bằng code của repo (sparsemax_mdn.sparsemax) trên số minh họa;
  * [sơ đồ]   : hình giải thích, không có số đo.

Chạy từ thư mục gốc repo:  .venv/bin/python docs/sparsemax_mdn/generate_figures.py
(Các hình [thật] của run đang chạy sẽ thay đổi mỗi lần chạy lại script.)
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "base_mdn"))
from sparsemax_mdn.sparsemax import sparsemax  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED, MAGENTA, VIOLET = "#e34948", "#e87ba4", "#4a3aa7"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
TINT = {BLUE: "#dbe8f9", ORANGE: "#fbe0d5", AQUA: "#d3f0e5", YELLOW: "#fbecc4"}
SEQ = ["#e8f1fc", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
SEQMAP = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print("saved", name)


def box(ax, x, y, w, h, text, fc=SURFACE, ec=INK2, lw=1.4, size=9.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, weight=weight, zorder=3)


def arrow(ax, p, q, color=INK2, lw=1.6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color=color, lw=lw, zorder=1))


def blank(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.grid(False)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


SPARSE_RUN = ROOT / "results/trained_models/sparsemax_mdn/imptc/sparsemax_k16_peds_imptc/runs/sparsemax_k16_seed2024"
ABL = sorted(json.loads((ROOT / "results/ablations/imptc_num_gaussians/ablation.json").read_text())["results"],
             key=lambda r: r["num_gaussians"])


def sm(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def spm(z):
    return sparsemax(torch.tensor(z, dtype=torch.float64)).numpy()


# =========================================================================== HÌNH
def fig01_pipeline_change():
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    blank(ax, (0, 11.5), (0, 4.6))
    steps_x = [0.1, 2.5, 4.9, 7.3, 9.4]
    names = ["LSTM", "Lớp fc\n8 → 48 × 6K", "6 khối × K\nμx, μy, log σx,\nlog σy, ρ, π-logit", "chuẩn hóa π", "Hỗn hợp Gaussian 2D\nmỗi bước tương lai"]
    for y, tag, norm, fc in ((2.6, "Baseline", "softmax\nπ_k > 0 với mọi k", SURFACE), (0.3, "sparsemax_mdn", "sparsemax\nπ_k có thể = 0", TINT[ORANGE])):
        for i, (x, n) in enumerate(zip(steps_x, names)):
            if i == 3:
                box(ax, x, y, 1.9, 1.4, norm, fc=fc, ec=ORANGE if tag != "Baseline" else INK2, lw=2.4 if tag != "Baseline" else 1.4, size=9.5, weight="bold")
            else:
                box(ax, x, y, 1.9 if i != 4 else 1.95, 1.4, n, size=9)
        for a, b in zip(steps_x[:-1], steps_x[1:]):
            arrow(ax, (a + 1.92, y + 0.7), (b - 0.02, y + 0.7))
        ax.text(0.0, y + 1.55, tag, fontsize=11, weight="bold")
    ax.text(5.75, 4.5, "Chỉ khối chuẩn hóa π bị thay đổi; mọi thứ khác (LSTM, μ, σ, ρ, loss NLL, optimizer, lịch học) giữ nguyên",
            ha="center", fontsize=10, color=INK2)
    save(fig, "01_what_changes.png")


def fig02_simplex():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    A, B, C = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3) / 2])

    def to2d(p):
        return p[..., 0:1] * A + p[..., 1:2] * B + p[..., 2:3] * C

    g = np.arange(-4, 4.01, 0.2)
    pts = [(a, b, 0.0) for a in g for b in g]
    soft = np.array([sm(p) for p in pts])
    spar = np.array([spm(np.array(p)) for p in pts])
    sup = (spar > 1e-12).sum(1)
    for ax, P, title in ((axes[0], soft, "softmax: mọi điểm nằm bên trong tam giác"), (axes[1], spar, "sparsemax: nhiều điểm rơi lên cạnh và đỉnh")):
        ax.add_patch(Polygon([A, B, C], closed=True, fill=False, ec=INK2, lw=1.6, zorder=1))
        xy = to2d(P)
        if ax is axes[0]:
            ax.scatter(xy[:, 0], xy[:, 1], s=8, color=BLUE, alpha=0.6, zorder=2)
        else:
            cols = {3: BLUE, 2: ORANGE, 1: AQUA}
            for k_, c in cols.items():
                m = sup == k_
                ax.scatter(xy[m, 0], xy[m, 1], s=14 if k_ != 3 else 8, color=c, alpha=0.75, zorder=3 if k_ != 3 else 2)
            ax.text(0.5, -0.20, "xanh dương: 3 thành phần dương    cam: 2 (một π = 0)    xanh lục: 1 (hai π = 0)", ha="center", fontsize=9, color=INK2)
        for pt, lab in ((A, "π₁ = 1"), (B, "π₂ = 1"), (C, "π₃ = 1")):
            ax.text(pt[0] + (-0.02 if pt[0] < 0.4 else (0.02 if pt[0] > 0.6 else 0)), pt[1] + (-0.07 if pt[1] < 0.5 else 0.05), lab,
                    ha="center", fontsize=9.5, color=INK2)
        ax.set_aspect("equal")
        ax.set_xlim(-0.12, 1.12)
        ax.set_ylim(-0.25, 1.02)
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=10.5, weight="bold")
    fig.suptitle("Ánh xạ từ logit (a, b, 0) sang trọng số π trên tam giác đơn vị (3 thành phần), a và b chạy trong [−4, 4]", fontsize=11, weight="bold", y=1.02)
    save(fig, "02_softmax_vs_sparsemax_simplex.png")
    print("fraction of grid with an exact zero (sparsemax):", float((sup < 3).mean()))
    return float((sup < 3).mean())


def fig03_algorithm():
    z = np.array([2.0, 1.4, 1.1, 0.2, -0.3, -1.0])
    order = np.argsort(-z)
    zs = z[order]
    cs = np.cumsum(zs)
    k = np.arange(1, len(z) + 1)
    in_sup = (1 + k * zs) > cs
    kz = int(in_sup.sum())
    tau = (cs[kz - 1] - 1) / kz
    p = np.maximum(zs - tau, 0)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    x = np.arange(len(z))
    ax = axes[0]
    ax.bar(x, zs, color=BLUE, width=0.6)
    for i, v in enumerate(zs):
        ax.text(i, v + (0.08 if v >= 0 else -0.25), f"{v:+.1f}", ha="center", fontsize=9)
    ax.set_ylim(-1.6, 2.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"z({i + 1})" for i in x])
    ax.set_title("1. Sắp xếp logit giảm dần", loc="left", fontsize=10.5, weight="bold")
    ax = axes[1]
    ax.bar(x, zs, color=[BLUE if s else "#b9b8b1" for s in in_sup], width=0.6)
    ax.axhline(tau, color=ORANGE, lw=2.0, ls="--")
    ax.text(len(z) - 0.5, tau + 0.08, f"ngưỡng τ = {tau:.2f}", color=INK, ha="right", fontsize=9.5)
    ax.text(0.0, -1.45, f"chỉ k(z) = {kz} thành phần đầu được giữ (màu xanh)", fontsize=9.3, color=INK2, ha="left")
    ax.set_ylim(-1.7, 2.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"z({i + 1})" for i in x])
    ax.set_title("2. Tìm τ sao cho tổng phần dương = 1", loc="left", fontsize=10.5, weight="bold")
    ax = axes[2]
    ax.bar(x - 0.18, sm(zs), 0.34, color="#b9b8b1")
    ax.bar(x + 0.18, p, 0.34, color=ORANGE)
    for i, v in enumerate(p):
        ax.text(i + 0.18, v + 0.01, f"{v:.2f}" if v > 0 else "0", ha="center", fontsize=9)
    ax.text(3.2, 0.5, "xám: softmax\ncam: sparsemax", fontsize=9.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"π({i + 1})" for i in x])
    ax.set_title("3. π = max(z − τ, 0): có số 0 chính xác", loc="left", fontsize=10.5, weight="bold")
    fig.suptitle("Sparsemax từng bước trên 6 logit ví dụ (z = 2.0, 1.4, 1.1, 0.2, −0.3, −1.0)", fontsize=11.5, weight="bold", y=1.04)
    save(fig, "03_sparsemax_algorithm.png")
    print("example:", p, "tau", tau, "support", kz)
    return {"tau": float(tau), "support": kz, "pi": p.tolist(), "softmax": sm(zs).tolist()}


def fig04_model_trick():
    fig, ax = plt.subplots(figsize=(11.8, 3.6))
    blank(ax, (0, 11.8), (0.6, 4.4))
    box(ax, 0.1, 2.5, 2.2, 1.4, "fc cho 6K số\n(khối cuối: logit của π)", fc=TINT[BLUE])
    box(ax, 3.1, 2.5, 2.2, 1.4, "π = sparsemax(logit)\nví dụ [0.42, 0.58, 0, 0]", fc=TINT[ORANGE], ec=ORANGE, lw=2.2)
    box(ax, 6.1, 2.5, 2.4, 1.4, "thay khối π bằng\nlog π  (π > 0)\n−1e9  (π = 0)", fc=SURFACE)
    box(ax, 9.3, 2.5, 2.3, 1.4, "bộ giải mã baseline\nsoftmax(log π) = π", fc=TINT[BLUE])
    for a, b in ((2.3, 3.1), (5.3, 6.1), (8.5, 9.3)):
        arrow(ax, (a, 3.2), (b, 3.2))
    ax.text(5.9, 4.2, "Ý tưởng cài đặt: giữ nguyên định dạng đầu ra của baseline", ha="center", fontsize=12, weight="bold")
    ax.text(5.9, 1.8, "softmax của (log π, −1e9) cho đúng π; mục có −1e9 ra đúng 0 trong float32,\nnên toàn bộ code dùng lại của baseline (tracker, dự đoán cố định, metric chính thức) chạy không cần sửa.",
            ha="center", va="top", fontsize=9.8, color=INK2)
    save(fig, "04_implementation_trick.png")


def fig05_loss_masking():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    blank(ax, (0, 6), (0, 4.4))
    ax.text(0.0, 4.2, "Hỗn hợp tại một bước: 4 thành phần", fontsize=10.5, weight="bold")
    comps = [("k=1", 0.55, BLUE, True), ("k=2", 0.45, ORANGE, True), ("k=3", 0.0, "#b9b8b1", False), ("k=4", 0.0, "#b9b8b1", False)]
    for i, (n, pi, c, live) in enumerate(comps):
        y = 3.2 - i * 0.85
        ax.add_patch(Rectangle((0.2, y), 5.0 * max(pi, 0.02), 0.55, fc=TINT.get(c, "#eceae6"), ec=c, lw=1.6, zorder=2))
        ax.text(0.0, y + 0.27, n, ha="right", va="center", fontsize=9.5)
        ax.text(0.25 + 5.0 * max(pi, 0.02), y + 0.27, f"  π = {pi:.2f}" + ("" if live else "  (bị loại khỏi tổng)"), va="center", fontsize=9.5, color=INK2)
    ax.text(0.0, -0.1, "loss: −log Σ_k π_k N_k(y). Thành phần π = 0 đóng góp đúng 0\nvà không nhận gradient ở mẫu đó.", fontsize=9.2, color=INK2, va="top")
    ax = axes[1]
    # ví dụ số: một thành phần sống ở xa điểm thật, một thành phần chết ngay tại điểm thật
    log_live, log_dead = -16.6, 5.0
    exact = -log_live
    eps = 1.19e-7
    clamped = -np.log(np.exp(log_live) + eps * np.exp(log_dead))
    ax.bar([0, 1], [exact, clamped], color=[BLUE, ORANGE], width=0.5)
    ax.text(0, exact + 0.5, f"{exact:.2f}", ha="center", fontsize=10)
    ax.text(1, clamped + 0.5, f"{clamped:.2f}", ha="center", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["đúng (π = 0\nbị loại hẳn)", "làm tròn kiểu\nbaseline (π ≈ 1.2e-7)"])
    ax.set_ylabel("NLL tại một điểm (nat)")
    ax.set_title("Vì sao tự cài NLL chính xác", loc="left", fontsize=10.5, weight="bold")
    ax.text(0.5, 20.3, "ví dụ minh họa: 1 thành phần sống ở xa,\n1 thành phần π = 0 nằm ngay tại điểm thật", ha="center", fontsize=9, color=INK2, va="center")
    ax.set_ylim(0, 22.5)
    fig.suptitle("Cách tính NLL: loại hẳn thành phần có π = 0 (ví dụ số, không phải dữ liệu đo)", fontsize=11.5, weight="bold", y=1.03)
    save(fig, "05_exact_nll_masking.png")
    return {"exact": float(exact), "clamped": float(clamped)}


def fig06_baseline_gap():
    ks, gap_vt, gap_ts, ravg = [], [], [], []
    for r in ABL:
        rows = read_csv(Path(r["run_dir"]) / "history.csv")
        be = int(r["best_epoch"])
        row = [x for x in rows if int(x["epoch"]) == be][0]
        tr, va = float(row["train_nll"]), float(row["validation_nll"])
        ks.append(r["num_gaussians"])
        gap_vt.append(va - tr)
        gap_ts.append(r["test_nll"] - r["validation_nll"])
        ravg.append(r["ravg_percent"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    ax = axes[0]
    ax.plot(ks, gap_vt, "-o", color=ORANGE, lw=2, ms=6)
    ax.plot(ks, gap_ts, "-s", color=BLUE, lw=2, ms=6)
    ax.text(ks[-1] + 0.2, gap_vt[-1], "validation − train", va="center", fontsize=9.5, color=INK2)
    ax.text(ks[-1] + 0.2, gap_ts[-1], "test − validation", va="center", fontsize=9.5, color=INK2)
    ax.set_xticks(ks)
    ax.set_xlim(0.6, 11.5)
    ax.set_xlabel("K của baseline")
    ax.set_ylabel("chênh lệch NLL (nat)")
    ax.set_title("Khoảng cách NLL tăng theo K (validation − train)", loc="left", fontsize=10.5, weight="bold")
    ax = axes[1]
    ax.plot(ks, ravg, "-o", color=BLUE, lw=2, ms=6)
    ax.set_xticks(ks)
    ax.set_xlabel("K của baseline")
    ax.set_ylabel("Ravg trên test (%)")
    ax.set_title("nhưng độ tin cậy trên test vẫn tăng theo K", loc="left", fontsize=10.5, weight="bold")
    fig.suptitle("Baseline IMPTC [thật], các model độc lập, 1 seed: dấu hiệu quá khớp NLL, chưa thấy bằng chứng về độ tin cậy xấu đi", fontsize=10.5, weight="bold", y=1.03)
    save(fig, "06_baseline_overfit_signal.png")


def fig07_fixed_samples():
    z = np.load(SPARSE_RUN / "fixed_samples/predictions/best.npz")
    ep = int(z["epoch"])
    pi = z["pi"].astype(np.float64)
    sup = z["support_size"]
    fig = plt.figure(figsize=(14, 4.8))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.5, 0.05, 1, 1, 0.05], wspace=0.5)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(sup, aspect="auto", cmap=SEQMAP, origin="upper", extent=[0.5, 48.5, 8.5, 0.5], vmin=1, vmax=max(8, sup.max()))
    ax.set_xlabel("bước dự báo t (0.1 s mỗi bước)")
    ax.set_ylabel("mẫu validation cố định (1–8)")
    ax.set_yticks(range(1, 9))
    ax.grid(False)
    ax.set_title("Số thành phần K(x, t) = #{π > 0}", loc="left", fontsize=10.5, weight="bold")
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[1]))
    cb.ax.set_title("K", fontsize=9)
    means = sup.mean(1)
    pick = [int(np.argmin(means)), int(np.argmax(means))]
    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("p", ["#cde2fb", "#5598e7", "#104281"])
    cmap2.set_bad(SURFACE)
    im2 = None
    for j, s_ in enumerate(pick):
        ax = fig.add_subplot(gs[2 + j])
        m = np.ma.masked_less_equal(pi[s_].T, 0)
        im2 = ax.imshow(m, aspect="auto", cmap=cmap2, origin="upper", extent=[0.5, 48.5, 16.5, 0.5], vmin=0, vmax=1)
        ax.set_xlabel("bước t")
        ax.set_yticks([1, 4, 8, 12, 16])
        ax.set_ylabel("thành phần k" if j == 0 else "")
        ax.grid(False)
        ax.set_title(f"π theo (k, t): mẫu {s_ + 1}\n(K trung bình {means[s_]:.1f}; ô trắng = π = 0)", loc="left", fontsize=9.8, weight="bold")
    cb2 = fig.colorbar(im2, cax=fig.add_subplot(gs[4]))
    cb2.ax.set_title("π", fontsize=9)
    fig.suptitle(f"Dự đoán thật của run sparsemax K_max = 16 tại epoch {ep} (best), 8 mẫu validation cố định: K thay đổi theo mẫu và theo bước", fontsize=11, weight="bold", y=1.03)
    save(fig, "07_real_fixed_samples.png")
    return {"epoch": ep, "mean_k_per_sample": means.tolist()}


def fig08_k_usage():
    path = SPARSE_RUN / "analysis/k_usage.json"
    if not path.exists():
        print("chưa có k_usage.json, bỏ qua fig08")
        return None
    d = json.loads(path.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))
    ax = axes[0]
    hist = d["k_distribution"]["histogram"]
    ks = [int(k) for k in hist]
    vals = [hist[str(k)] / sum(hist.values()) for k in ks]
    ax.bar(ks, vals, color=BLUE, width=0.7)
    ax.set_xticks(range(0, d["k_max"] + 1, 2))
    ax.set_xlabel("K(x, t)")
    ax.set_ylabel("tỉ lệ (mẫu, bước)")
    ax.set_title(f"Phân bố K(x, t): trung bình {d['k_distribution']['mean']:.2f}", loc="left", fontsize=10.5, weight="bold")
    ax = axes[1]
    mk = d["mean_k_per_forecast_step"]
    ax.plot(np.arange(1, len(mk) + 1), mk, color=BLUE, lw=2.2)
    ax.set_xlabel("bước dự báo t")
    ax.set_ylabel("K trung bình")
    ax.set_ylim(0, d["k_max"])
    ax.set_title("K trung bình theo bước dự báo", loc="left", fontsize=10.5, weight="bold")
    ax = axes[2]
    act = d["component_activity"]["fraction_active_per_component"]
    ax.bar(range(1, len(act) + 1), act, color=BLUE, width=0.7)
    ax.set_xticks(range(1, len(act) + 1, 2))
    ax.set_xlabel("chỉ số thành phần")
    ax.set_ylabel("tỉ lệ (mẫu, bước) mà nó hoạt động")
    ax.set_title(f"Mức dùng từng thành phần (số thành phần chết: {d['component_activity']['dead_components']})", loc="left", fontsize=10.5, weight="bold")
    fig.suptitle(f"Phân tích thật trên {d['num_validation_samples']} mẫu validation, checkpoint best epoch {d['checkpoint_epoch']} (khám phá, chỉ validation)",
                 fontsize=11, weight="bold", y=1.04)
    save(fig, "08_real_k_usage.png")
    return {"epoch": d["checkpoint_epoch"], "mean_k": d["k_distribution"]["mean"],
            "spearman_ade": d["association_with_difficulty"]["spearman_k_vs_top_component_ade"],
            "spearman_spread": d["association_with_difficulty"]["spearman_k_vs_weighted_sqrt_trace"]}


def fig09_training():
    rows = read_csv(SPARSE_RUN / "history.csv")
    ep = np.array([int(r["epoch"]) for r in rows])
    val = np.array([float(r["validation_nll"]) for r in rows])
    kk = np.array([float(r["mean_support_size"]) for r in rows])
    full = np.array([float(r["fraction_full_support"]) for r in rows])
    dead = np.array([int(r["dead_components"]) for r in rows])
    ref = {}
    for r in ABL:
        if r["num_gaussians"] in (3, 8):
            rr = read_csv(Path(r["run_dir"]) / "history.csv")
            ref[r["num_gaussians"]] = (np.array([int(x["epoch"]) for x in rr]), np.array([float(x["validation_nll"]) for x in rr]))
    n = int(ep.max())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    ax = axes[0]
    m = (ep >= 15)
    ax.plot(ep[m], val[m], color=ORANGE, lw=2.2)
    for k_, c in ((3, BLUE), (8, AQUA)):
        e, v = ref[k_]
        mm = (e >= 15) & (e <= n)
        ax.plot(e[mm], v[mm], color=c, lw=1.6)
    ax.text(0.97, 0.97, "cam: sparsemax K_max = 16\nxanh dương: baseline K = 3\nxanh lục: baseline K = 8", transform=ax.transAxes, ha="right", va="top", fontsize=9.3, color=INK2)
    ax.set_ylim(-1.5, 1.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation NLL (tập con 50%)")
    ax.set_title("Validation NLL theo epoch, cùng giao thức", loc="left", fontsize=10.5, weight="bold")
    ax = axes[1]
    ax.plot(ep, kk, color=ORANGE, lw=2.2)
    ax.set_ylim(0, 16.5)
    ax.axhline(16, color=INK2, lw=1.1, ls="--")
    ax.text(n * 0.5, 15.0, "K_max = 16", color=INK2, fontsize=9)
    ax.set_xlabel("epoch")
    ax.set_ylabel("K trung bình (mẫu, bước)")
    ax.set_title("Số thành phần hiệu dụng tự thu hẹp", loc="left", fontsize=10.5, weight="bold")
    ax = axes[2]
    ax.plot(ep, full, color=ORANGE, lw=2.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("tỉ lệ dùng đủ K_max thành phần")
    ax.set_ylim(0, 1)
    ax.set_title(f"Kiểm tra trần: tỉ lệ dùng đủ 16 (số chết: {int(dead[-1])})", loc="left", fontsize=10.5, weight="bold")
    fig.suptitle(f"Run sparsemax K_max = 16 [thật, đã train xong {n} epoch], seed 2024; baseline cũng seed 2024 (một seed)", fontsize=11, weight="bold", y=1.04)
    save(fig, "09_training_dynamics.png")
    return {"epoch": n, "val": float(val[-1]), "k_eff": float(kk[-1]), "full": float(full[-1]), "dead": int(dead[-1])}


def fig13_metrics():
    runs = {"sparsemax K_max = 16": (SPARSE_RUN, ORANGE)}
    for r in ABL:
        if r["num_gaussians"] in (3, 8):
            runs[f"baseline K = {r['num_gaussians']}"] = (Path(r["run_dir"]), BLUE if r["num_gaussians"] == 3 else AQUA)
    spec = [("ravg_percent", "Ravg (%) ↑"), ("rmin_percent", "Rmin (%) ↑"), ("s68_m2_per_s", "S68 ↓"),
            ("s95_m2_per_s", "S95 ↓"), ("minade20_m", "minADE20 (m) ↓"), ("minfde20_m", "minFDE20 (m) ↓")]
    epochs = [250 * i for i in range(1, 11)]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.2))
    last = {}
    for ax, (key, title) in zip(axes.ravel(), spec):
        for name, (path, color) in runs.items():
            ys, es = [], []
            for e in epochs:
                f = path / f"metrics/epoch_{e:04d}.json"
                if f.exists():
                    es.append(e)
                    ys.append(json.loads(f.read_text())["metrics"][key])
            ax.plot(es, ys, "-o", color=color, lw=1.8, ms=4)
            last.setdefault(name, {})[key] = ys[-1] if ys else None
        ax.set_title(title, loc="left", fontsize=10.5, weight="bold")
        ax.set_xlabel("epoch")
    fig.text(0.5, 0.955, "cam: sparsemax K_max = 16     xanh dương: baseline K = 3     xanh lục: baseline K = 8", ha="center", fontsize=10, color=INK2)
    fig.suptitle("Metric chính thức của repo trên tập VALIDATION, mô hình tại mỗi mốc 250 epoch [thật], cùng giao thức và seed 2024 (một seed)", fontsize=11, weight="bold", y=1.02)
    save(fig, "13_official_metrics_vs_baseline.png")
    return last


TEST_JSON = SPARSE_RUN / "evaluation/test_best_clampedpi.json"


def fig14_test_metrics():
    if not TEST_JSON.exists():
        print("chưa có kết quả test, bỏ qua fig14")
        return None
    t = json.loads(TEST_JSON.read_text())
    om = t["official_metrics"]
    abl = {r["num_gaussians"]: r for r in ABL}
    rows = [("NLL ↓ (test)", "test_nll", t["exact_nll"]), ("Ravg (%) ↑", "ravg_percent", om["ravg_percent"]),
            ("Rmin (%) ↑", "rmin_percent", om["rmin_percent"]), ("minADE20 (m) ↓", "minade20_m", om["minade20_m"]),
            ("minFDE20 (m) ↓", "minfde20_m", om["minfde20_m"]), ("S68 ↓", "s68_m2_per_s", om["s68_m2_per_s"]),
            ("S95 ↓", "s95_m2_per_s", om["s95_m2_per_s"]), ("ASAEE ↓", "asaee_m_per_s", om["asaee_m_per_s"])]
    names = ["baseline K = 3", "baseline K = 8", "sparsemax K_max = 16"]
    cols = [BLUE, AQUA, ORANGE]
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.0))
    table = {}
    for ax, (title, key, sv) in zip(axes.ravel(), rows):
        vals = [abl[3][key], abl[8][key], sv]
        table[title] = vals
        ax.bar(range(3), vals, color=cols, width=0.62)
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.6 + 1e-9
        ax.set_ylim(lo - pad, 0 if hi < 0 else hi + pad * 0.5)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}" if abs(v) < 20 else f"{v:.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["K=3", "K=8", "sparsemax"], fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=10, weight="bold")
    fig.suptitle(f"Tập TEST, metric chính thức của repo, {t['num_samples']:,} mẫu, một lần chạy, một seed [thật]: xanh dương baseline K = 3, xanh lục baseline K = 8, cam sparsemax",
                 fontsize=10.5, weight="bold", y=1.0)
    save(fig, "14_test_metrics.png")
    return table


def fig15_test_sharpness():
    npz = SPARSE_RUN / "evaluation/test_best_clampedpi_persample.npz"
    if not (npz.exists() and TEST_JSON.exists()):
        return None
    z = np.load(npz)
    t = json.loads(TEST_JSON.read_text())
    areas = z["sharpness_area_m2"]  # [N, levels, horizons]
    hor, dt, fh = z["horizons"], float(z["delta_t"]), float(z["forecast_horizon"])
    lv = list(z["confidence_levels"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, level, color in ((axes[0], 0.95, ORANGE), (axes[1], 0.68, BLUE)):
        a = areas[:, lv.index(level), :]
        score = (a / ((hor[None, :] + 1) * dt)).sum(1) / (fh * dt)
        ax.hist(np.clip(score, 1e-2, None), bins=np.logspace(-1.5, 2.3, 70), color=color, alpha=0.85)
        ax.set_xscale("log")
        key = "s95" if level == 0.95 else "s68"
        sm = t["sharpness_per_sample_summary"][key]
        for v, lab, ls in ((sm["median"], "trung vị", ":"), (sm["mean"], "trung bình mẫu", "--"), (sm["official_percentile_grid_score"], "điểm chính thức", "-")):
            ax.axvline(v, color=INK, lw=1.4, ls=ls)
        ax.text(0.98, 0.97, f"chấm: trung vị {sm['median']:.2f}\nđứt: trung bình mẫu {sm['mean']:.2f}\nliền: điểm chính thức {sm['official_percentile_grid_score']:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK2)
        ax.set_xlabel("điểm sắc nét của từng mẫu (m²/s, thang log)")
        ax.set_ylabel("số mẫu")
        ax.set_title(f"S{int(level * 100)}: phân bố theo mẫu (sparsemax, test)", loc="left", fontsize=10, weight="bold")
    ax = axes[2]
    xs = np.arange(2)
    mean_v = [t["sharpness_per_sample_summary"][k]["mean"] for k in ("s68", "s95")]
    off_v = [t["sharpness_per_sample_summary"][k]["official_percentile_grid_score"] for k in ("s68", "s95")]
    ax.bar(xs - 0.18, mean_v, 0.34, color="#b9b8b1")
    ax.bar(xs + 0.18, off_v, 0.34, color=ORANGE)
    for i in range(2):
        ax.text(i - 0.18, mean_v[i], f"{mean_v[i]:.2f}", ha="center", va="bottom", fontsize=9.5)
        ax.text(i + 0.18, off_v[i], f"{off_v[i]:.2f}", ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(["S68", "S95"])
    ax.text(0.5, 0.9, "xám: trung bình theo mẫu\ncam: điểm chính thức (trung bình 101 phân vị,\ngồm cả mẫu lớn nhất)", transform=ax.transAxes, ha="center", va="top", fontsize=9, color=INK2)
    ax.set_ylim(0, max(off_v) * 1.3)
    ax.set_title("Điểm chính thức cao hơn trung bình mẫu", loc="left", fontsize=10, weight="bold")
    fig.suptitle("Độ sắc nét của sparsemax trên tập test [thật]: đa số mẫu rất sắc, điểm chính thức bị kéo lên bởi đuôi", fontsize=11, weight="bold", y=1.03)
    save(fig, "15_test_sharpness_distribution.png")
    return {"mean": mean_v, "official": off_v}


def fig10_protocol():
    fig, ax = plt.subplots(figsize=(11.8, 4.8))
    blank(ax, (0, 11.8), (0, 4.8))
    box(ax, 0.1, 2.9, 3.4, 1.4, "Công bố trước (spec)\nK_max = 16, sparsemax thuần,\ngiao thức baseline, seed 2024", fc=TINT[BLUE], size=9.3)
    box(ax, 4.2, 2.9, 3.4, 1.4, "MỘT lần huấn luyện\n2500 epoch, NLL như baseline\nkhông chọn giữa các biến thể", fc=TINT[ORANGE], size=9.3)
    box(ax, 8.2, 2.9, 3.4, 1.4, "Đánh giá\nmetric chính thức vs baseline K = 3, 8\n+ phân tích K(x, t) (validation)", fc=TINT[AQUA], size=9.3)
    arrow(ax, (3.5, 3.6), (4.2, 3.6))
    arrow(ax, (7.6, 3.6), (8.2, 3.6))
    ax.text(0.1, 2.3, "Tiêu chí thành công (công bố trước):", fontsize=10.5, weight="bold")
    ax.text(0.1, 1.85, "1. NLL, Ravg, Rmin không kém baseline; S68/S95 không xấu đi.", fontsize=9.8)
    ax.text(0.1, 1.4, "2. K trung bình nhỏ hơn K_max; K thay đổi theo mẫu và theo bước.", fontsize=9.8)
    ax.text(0.1, 0.95, "3. K(x, t) gắn với độ khó của mẫu (giả thuyết, cần kiểm chứng).", fontsize=9.8)
    ax.text(0.1, 0.4, "Không đạt tiêu chí nào cũng được báo cáo đúng như vậy. Một seed: khác biệt nhỏ chưa phải kết luận.", fontsize=9.3, color=INK2)
    ax.text(5.9, 4.65, "Giao thức đánh giá cho cải tiến sparsemax", ha="center", fontsize=12, weight="bold")
    save(fig, "10_protocol.png")


def fig11_two_philosophies():
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 3.9))
    ax = axes[0]
    blank(ax, (0, 6), (0, 4))
    ax.text(0.0, 3.8, "Chọn K bằng nhiều biến thể", fontsize=11, weight="bold")
    for i, k in enumerate((1, 2, 3, 5, 8)):
        box(ax, 0.1 + i * 1.15, 2.3, 1.0, 0.8, f"train K={k}", size=8.5)
    arrow(ax, (3.0, 2.25), (3.0, 1.6))
    box(ax, 1.5, 0.7, 3.0, 0.8, "so sánh rồi CHỌN K tốt nhất", fc=TINT[YELLOW], size=9.3)
    ax.text(0.0, 0.2, "đây là chọn mô hình (siêu tham số), chưa phải cải tiến", fontsize=9.3, color=INK2)
    ax = axes[1]
    blank(ax, (0, 6), (0, 4))
    ax.text(0.0, 3.8, "K thoát ra từ một lần huấn luyện", fontsize=11, weight="bold")
    box(ax, 0.2, 2.2, 5.4, 1.0, "một model, K_max lớn, softmax → sparsemax", fc=TINT[ORANGE], size=9.5)
    arrow(ax, (2.9, 2.15), (2.9, 1.6))
    box(ax, 0.2, 0.7, 5.4, 0.8, "mô hình tự đặt π = 0: K(x, t) theo từng mẫu", fc=TINT[AQUA], size=9.5)
    ax.text(0.0, 0.2, "đổi mô hình/ hàm chuẩn hóa, một cấu hình công bố trước", fontsize=9.3, color=INK2)
    fig.suptitle("Vì sao hướng sparsemax đáp ứng yêu cầu 'cải tiến không phải chọn biến thể'", fontsize=11.5, weight="bold", y=1.03)
    save(fig, "11_two_philosophies.png")


def fig12_difficulty_scatter():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from sparsemax_mdn.model import SparsemaxMDN
    from sparsemax_mdn.analyze_k import per_sample_statistics
    from utils.config_loader import ConfigLoader
    from utils.data_loader import DataLoader
    from utils.mdn_distribution import decode_mdn_output

    cfgp = ROOT / "sparsemax_mdn/configs/imptc/sparsemax_k16_peds_imptc.json"
    cfg = ConfigLoader(str(cfgp), "imptc", False, False, cfgp.stem, "sparsemax_mdn", "training")
    saved = torch.load(SPARSE_RUN / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    model = SparsemaxMDN(cfg.model_params)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    loader = DataLoader(cfg)
    loader.load_eval_data()
    X, y = loader.eval_data[0][:3000], loader.eval_data[1][:3000]
    parts = {k: [] for k in ("pi", "mu", "covariance")}
    with torch.no_grad():
        for s in range(0, len(X), 512):
            d = decode_mdn_output(model(torch.as_tensor(X[s:s + 512], dtype=torch.float32)), 16)
            for k in parts:
                parts[k].append(d[k].numpy())
    arr = {k: np.concatenate(v).astype(np.float64) for k, v in parts.items()}
    support, ade, spread = per_sample_statistics(arr["pi"], arr["mu"], arr["covariance"], np.asarray(y, dtype=np.float64))
    ks = support.mean(-1)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, v, name in ((axes[0], ade, "ADE của thành phần có π lớn nhất (m)"), (axes[1], spread, "độ phân tán dự báo (m)")):
        ax.scatter(ks, v, s=5, color=BLUE, alpha=0.35)
        bins = np.quantile(ks, np.linspace(0, 1, 9))
        idx = np.digitize(ks, bins[1:-1])
        cx = [ks[idx == b].mean() for b in range(8)]
        cy = [np.median(v[idx == b]) for b in range(8)]
        ax.plot(cx, cy, "-o", color=ORANGE, lw=2.2, ms=5)
        ax.set_xlabel("K trung bình của mẫu (trên 48 bước)")
        ax.set_ylabel(name)
        ax.set_yscale("log")
        ax.tick_params(axis="y", labelsize=8.5)
    axes[0].text(0.03, 0.95, "cam: trung vị theo 8 nhóm K", transform=axes[0].transAxes, fontsize=9, color=INK2, va="top")
    fig.suptitle(f"K trung bình của mẫu so với hai thước đo độ khó (khám phá, 3000 mẫu validation, epoch {int(saved['epoch'])}): quan hệ yếu hoặc chưa rõ", fontsize=10.5, weight="bold", y=1.03)
    save(fig, "12_k_vs_difficulty.png")
    return {"epoch": int(saved["epoch"])}


def main():
    info = {}
    fig01_pipeline_change()
    info["simplex_zero_fraction"] = fig02_simplex()
    info["algorithm"] = fig03_algorithm()
    fig04_model_trick()
    info["nll_example"] = fig05_loss_masking()
    fig06_baseline_gap()
    info["fixed"] = fig07_fixed_samples()
    info["usage"] = fig08_k_usage()
    info["training"] = fig09_training()
    info["metrics_last"] = fig13_metrics()
    info["test_metrics"] = fig14_test_metrics()
    info["test_sharp"] = fig15_test_sharpness()
    fig10_protocol()
    fig11_two_philosophies()
    info["scatter"] = fig12_difficulty_scatter()
    (OUT / "figure_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
