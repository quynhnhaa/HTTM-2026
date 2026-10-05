"""Sinh hình minh họa cho GROWING_MDN.md.

Nguồn dữ liệu của từng hình được ghi ngay trong hàm vẽ và trong chú thích của tài liệu:
  * "thật"  : đọc từ artifact trong results/ (baseline K, dự đoán baseline K=8, smoke run, run đang chạy);
  * "đồ chơi": mô phỏng 2D tự sinh để giải thích ý tưởng, KHÔNG phải dữ liệu IMPTC.

Chạy từ thư mục gốc repo:  .venv/bin/python docs/growing_mdn/generate_figures.py
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "base_mdn"))
from growing_mdn.model import fc_row_map  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# Bảng màu đã qua validate_palette (4 slot đầu, light) + mực chữ trung tính.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, VIOLET, RED, GREEN = "#e87ba4", "#4a3aa7", "#e34948", "#008300"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
TINT = {BLUE: "#dbe8f9", ORANGE: "#fbe0d5", AQUA: "#d3f0e5", YELLOW: "#fbecc4", MAGENTA: "#f9e0ea"}
SEQ = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

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


def box(ax, x, y, w, h, text, fc=SURFACE, ec=INK2, lw=1.4, size=9.5, weight="normal", style="round,pad=0.02,rounding_size=0.12"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, weight=weight, zorder=3)


def arrow(ax, p, q, color=INK2, lw=1.6, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def blank(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.grid(False)


# --------------------------------------------------------------------------- dữ liệu thật
ABL = json.loads((ROOT / "results/ablations/imptc_num_gaussians/ablation.json").read_text())["results"]
ABL = sorted(ABL, key=lambda r: r["num_gaussians"])
KS = [r["num_gaussians"] for r in ABL]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- đồ chơi 2D (EM + split)
def gauss_logpdf(X, mu, cov):
    d = X - mu
    inv = np.linalg.inv(cov)
    _, logdet = np.linalg.slogdet(cov)
    return -0.5 * np.einsum("ni,ij,nj->n", d, inv, d) - 0.5 * (logdet + 2 * np.log(2 * np.pi))


def logsumexp(a, axis=1):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def comp_logp(X, pis, mus, covs):
    return np.stack([np.log(pi) + gauss_logpdf(X, mu, c) for pi, mu, c in zip(pis, mus, covs)], axis=1)


def toy_nll(X, g):
    return float(-logsumexp(comp_logp(X, *g)).mean())


def em(X, g, iters=80):
    pis, mus, covs = [list(map(np.copy, part)) for part in g]
    for _ in range(iters):
        lp = comp_logp(X, pis, mus, covs)
        r = np.exp(lp - logsumexp(lp)[:, None])
        nk = r.sum(0) + 1e-9
        pis = list(nk / len(X))
        mus = [(r[:, k:k + 1] * X).sum(0) / nk[k] for k in range(len(nk))]
        covs = [((r[:, k:k + 1] * (X - mus[k])).T @ (X - mus[k])) / nk[k] + 1e-3 * np.eye(2) for k in range(len(nk))]
    return pis, mus, covs


def toy_scores(X, g):
    """Điểm giống code thật: trung bình có trọng số responsibility của -log N_k(y)."""
    lp = comp_logp(X, *g)
    r = np.exp(lp - logsumexp(lp)[:, None])
    neg_logn = -np.stack([gauss_logpdf(X, mu, c) for mu, c in zip(g[1], g[2])], axis=1)
    return (r * neg_logn).sum(0) / r.sum(0), r


def toy_split(g, j, shrink=0.4):
    pis, mus, covs = [list(map(np.copy, part)) for part in g]
    w, v = np.linalg.eigh(covs[j])
    axis = v[:, -1]
    off = 0.5 * np.sqrt(w[-1]) * axis
    small = covs[j] - (1 - shrink) * w[-1] * np.outer(axis, axis)
    pis[j] = pis[j] / 2
    pis.append(pis[j])
    mus.append(mus[j] - off)
    mus[j] = mus[j] + off
    covs[j] = small
    covs.append(small.copy())
    return pis, mus, covs


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def toy_data(n, seed):
    rng = np.random.default_rng(seed)
    lobes = [((-3.0, -0.5), (0.45, 1.0), 0.3, 0.2), ((3.2, -1.2), (1.0, 0.45), 0.25, -0.3),
             ((0.0, 2.8), (1.3, 0.4), 0.3, 0.0), ((1.6, -3.6), (0.45, 0.45), 0.15, 0.0)]
    counts = rng.multinomial(n, [l[2] for l in lobes])
    parts = []
    for (mu, sd, _, th), c in zip(lobes, counts):
        parts.append(rng.normal(size=(c, 2)) * sd @ rot(th).T + mu)
    X = np.vstack(parts)
    rng.shuffle(X)
    return X


def toy_run(k_max=7):
    Xtr, Xva = toy_data(2500, 1), toy_data(2500, 2)
    g = em(Xtr, ([1.0], [Xtr.mean(0)], [np.cov(Xtr.T)]), iters=40)
    history = []
    for k in range(1, k_max + 1):
        scores, _ = toy_scores(Xtr, g)
        history.append({"k": k, "g": g, "scores": scores, "train": toy_nll(Xtr, g), "val": toy_nll(Xva, g)})
        if k < k_max:
            g = em(Xtr, toy_split(g, int(np.argmax(scores))), iters=80)
    return Xtr, Xva, history


def draw_ellipses(ax, g, color=BLUE, lw=1.6, highlight=None, hcolor=ORANGE):
    for j, (pi, mu, c) in enumerate(zip(*g)):
        w, v = np.linalg.eigh(c)
        ang = np.degrees(np.arctan2(v[1, -1], v[0, -1]))
        hl = highlight == j
        ax.add_patch(Ellipse(mu, 4 * np.sqrt(w[-1]), 4 * np.sqrt(w[0]), angle=ang, fill=hl, alpha=0.25 if hl else 1,
                             fc=hcolor if hl else "none", ec=hcolor if hl else color, lw=lw + (0.6 if hl else 0), zorder=3))
        ax.add_patch(Ellipse(mu, 4 * np.sqrt(w[-1]), 4 * np.sqrt(w[0]), angle=ang, fill=False,
                             ec=hcolor if hl else color, lw=lw + (0.6 if hl else 0), zorder=3))


# =========================================================================== HÌNH
def fig01_pipeline():
    fig, ax = plt.subplots(figsize=(11, 3.7))
    blank(ax, (0, 11), (0, 3.7))
    xs = [0.1, 2.35, 4.6, 6.85, 9.0]
    labels = ["Quỹ đạo quan sát\n32 bước × 4 đặc trưng", "LSTM\n(bộ mã hóa chuỗi,\nhidden = 8)", "Lớp tuyến tính fc\n8 → 48 × 6K",
              "Giải mã MDN\nπ, μ, σ, ρ cho\nmỗi cặp (t, k)", "Hỗn hợp Gaussian 2D\ntại mỗi bước tương lai\n(48 phân phối)"]
    fills = [SURFACE, TINT[BLUE], TINT[ORANGE], TINT[BLUE], SURFACE]
    for x, t, f in zip(xs, labels, fills):
        box(ax, x, 1.5, 1.95, 1.4, t, fc=f)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 1.97, 2.2), (b - 0.03, 2.2))
    ax.text(5.5, 3.45, "Pipeline baseline (base_mdn)", ha="center", fontsize=12, weight="bold")
    ax.text(5.575, 1.2, "K = số thành phần Gaussian mỗi bước\nbaseline đặt cố định K = 3", ha="center", va="top", fontsize=9.5, color=INK)
    ax.add_patch(Rectangle((4.55, 1.38), 2.05, 1.64, fill=False, ec=ORANGE, lw=2.2, ls="--", zorder=4))
    ax.text(5.575, 0.15, "Hình này: K nằm ở kích thước đầu ra của fc (6K giá trị cho mỗi bước t). Đổi K = đổi kích thước lớp fc.",
            ha="center", fontsize=9, color=INK2)
    save(fig, "01_pipeline_baseline.png")


def fig02_layout():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    blocks = ["μx", "μy", "log σx", "log σy", "ρ (pre-tanh)", "π (logit)"]
    cols = [BLUE, ORANGE, AQUA, YELLOW]
    for ax, K, title in zip(axes, (3, 4), ("K = 3  (baseline)", "K = 4  (sau khi tách 1 thành phần)")):
        blank(ax, (-2.2, 4.8), (-1.4, 7.2))
        ax.text(1.2, 6.8, title, ha="center", fontsize=11, weight="bold")
        for b, name in enumerate(blocks):
            ax.text(-0.25, 5.5 - b, name, ha="right", va="center", fontsize=9.5, color=INK2)
            for k in range(K):
                new = (K == 4 and k == 3)
                ax.add_patch(Rectangle((k * 1.05, 5.1 - b), 1.0, 0.8, fc=TINT[cols[k]], ec=cols[k], lw=2.6 if new else 1.2,
                                       hatch="//" if new else None, zorder=2))
                ax.text(k * 1.05 + 0.5, 5.5 - b, str(b * K + k), ha="center", va="center", fontsize=9.5, zorder=3)
        for k in range(K):
            ax.text(k * 1.05 + 0.5, 6.15, f"k={k}" + (" (mới)" if K == 4 and k == 3 else ""), ha="center", fontsize=9,
                    color=INK2)
        ax.text(1.2, -0.45, f"chỉ số hàng trong một bước t  =  khối × K + k   (K = {K})", ha="center", fontsize=9.5)
        ax.text(1.2, -1.0, f"toàn bộ fc: chỉ số = t × 6K + khối × K + k   (6K = {6 * K})", ha="center", fontsize=9.5, color=INK2)
    fig.suptitle("Cách một bước dự báo được chứa trong đầu ra của fc: 6 khối × K thành phần", fontsize=12, weight="bold", y=1.02)
    save(fig, "02_fc_layout.png")


def fig03_baseline_nll():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    val = [r["validation_nll"] for r in ABL]
    test = [r["test_nll"] for r in ABL]
    ax = axes[0]
    ax.plot(KS, val, "-o", color=BLUE, lw=2, ms=6)
    ax.plot(KS, test, "-s", color=ORANGE, lw=2, ms=6)
    ax.text(KS[-1] + 0.25, val[-1] - 0.03, "validation", color=INK2, va="center", fontsize=9.5)
    ax.text(KS[-1] + 0.25, test[-1] + 0.06, "test", color=INK2, va="center", fontsize=9.5)
    ax.set_xticks(KS)
    ax.set_xlim(0.6, 10.4)
    ax.set_xlabel("K (số thành phần Gaussian)")
    ax.set_ylabel("NLL (càng thấp càng tốt)")
    ax.set_title("NLL giảm khi tăng K", loc="left", fontsize=11, weight="bold")
    ax = axes[1]
    gains = [(KS[i] + 0.0, (test[i - 1] - test[i]) / (KS[i] - KS[i - 1])) for i in range(1, len(KS))]
    labels = [f"{KS[i-1]}→{KS[i]}" for i in range(1, len(KS))]
    ax.bar(range(len(gains)), [g[1] for g in gains], color=BLUE, width=0.6)
    for i, (_, v) in enumerate(gains):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=9.5)
    ax.set_xticks(range(len(gains)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("bước tăng K")
    ax.set_ylabel("giảm test NLL trên MỖI thành phần thêm")
    ax.set_title("Lợi ích mỗi thành phần không giảm đều", loc="left", fontsize=11, weight="bold")
    fig.suptitle("Baseline IMPTC, các model train độc lập (1 seed), nguồn: results/ablations/imptc_num_gaussians", fontsize=9, color=INK2, y=-0.02)
    save(fig, "03_baseline_nll_vs_k.png")


def fig04_baseline_panels():
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    spec = [("Sai số vị trí (m)", [("minADE20", "minade20_m", BLUE), ("minFDE20", "minfde20_m", ORANGE)]),
            ("Độ tin cậy (%)", [("Ravg", "ravg_percent", BLUE), ("Rmin", "rmin_percent", ORANGE)]),
            ("Độ sắc nét (m²/s, thấp = sắc)", [("S68", "s68_m2_per_s", BLUE), ("S95", "s95_m2_per_s", ORANGE)]),
            ("ASAEE (m/s)", [("ASAEE", "asaee_m_per_s", BLUE)])]
    for ax, (title, series) in zip(axes, spec):
        for name, key, color in series:
            y = [r[key] for r in ABL]
            ax.plot(KS, y, "-o", color=color, lw=2, ms=5)
            ax.text(KS[-1] + 0.2, y[-1], name, color=INK2, va="center", fontsize=9)
        ax.set_xticks(KS)
        ax.set_xlim(0.6, 10.2)
        ax.set_xlabel("K")
        ax.set_title(title, loc="left", fontsize=10.5, weight="bold")
    fig.suptitle("Baseline IMPTC (tập test, best theo validation NLL): chất lượng thay đổi theo K, không phải metric nào cũng đơn điệu",
                 fontsize=11, weight="bold", y=1.04)
    save(fig, "04_baseline_metrics_vs_k.png")


def fig05_k8_usage():
    z = np.load(ROOT / "results/trained_models/base_mdn/imptc/m8_peds_imptc/runs/imptc_m8_seed2024/fixed_samples/predictions/best.npz")
    pi = z["pi"].astype(np.float64)
    neff = np.exp(-(pi * np.log(np.clip(pi, 1e-12, 1))).sum(-1))
    srt = -np.sort(-pi, axis=-1)
    cover = (np.cumsum(srt, -1) < 0.95).sum(-1) + 1
    print("K=8 baseline: n_eff min/mean/max", neff.min(), neff.mean(), neff.max(), "| cover95 min/mean/max", cover.min(), cover.mean(), cover.max())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    ax = axes[0]
    for s in range(pi.shape[0]):
        ax.plot(np.arange(1, 49), neff[s], color=INK2, lw=0.9, alpha=0.5)
    ax.plot(np.arange(1, 49), neff.mean(0), color=BLUE, lw=2.6)
    ax.text(49.2, neff.mean(0)[-1], "trung bình\n8 mẫu", color=INK2, va="center", fontsize=9)
    ax.set_xlim(1, 60)
    ax.set_ylim(0.8, 8.2)
    ax.axhline(8, color=ORANGE, lw=1.4, ls="--")
    ax.text(2, 7.7, "K = 8 (số thành phần có sẵn)", color=INK2, fontsize=9)
    ax.set_xlabel("bước dự báo t (0.1 s mỗi bước)")
    ax.set_ylabel("số thành phần hiệu dụng  exp(H(π))")
    ax.set_title("Mỗi đường là một mẫu validation cố định", loc="left", fontsize=10.5, weight="bold")
    ax = axes[1]
    im = ax.imshow(srt.mean(0).T, aspect="auto", cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", SEQ),
                   origin="upper", extent=[0.5, 48.5, 8.5, 0.5], vmin=0, vmax=1)
    ax.set_xlabel("bước dự báo t")
    ax.set_ylabel("hạng của thành phần\n(1 = trọng số π lớn nhất)")
    ax.set_yticks(range(1, 9))
    ax.set_title("Trọng số π trung bình theo hạng", loc="left", fontsize=10.5, weight="bold")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("π")
    fig.suptitle("Baseline K = 8: dự đoán thật trên 8 mẫu validation cố định (checkpoint best)", fontsize=11, weight="bold", y=1.03)
    save(fig, "05_baseline_k8_component_usage.png")
    return {"neff_min": float(neff.min()), "neff_mean": float(neff.mean()), "neff_max": float(neff.max()),
            "cover95_mean": float(cover.mean()), "cover95_max": int(cover.max())}


def fig06_params():
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ks = np.arange(1, 13)
    ax.plot(ks, 448 + 2592 * ks, "-", color=INK2, lw=1.4)
    ax.plot(KS, [r["parameter_count"] for r in ABL], "o", color=BLUE, ms=8)
    for k in (1, 3, 8, 12):
        ax.text(k, 448 + 2592 * k + 1800, f"{448 + 2592 * k:,}", ha="center", fontsize=9)
    ax.text(10.3, 9000, "mỗi thành phần thêm\n= 6 × 48 × (8 + 1) = 2 592 tham số", fontsize=9.5, color=INK2, ha="center")
    ax.set_xticks(ks)
    ax.set_xlabel("K")
    ax.set_ylabel("số tham số của model")
    ax.set_title("Chi phí tham số tăng tuyến tính theo K", loc="left", fontsize=11, weight="bold")
    ax.text(1.2, 29000, "chấm xanh: số tham số đo trên các model baseline (K = 1, 2, 3, 5, 8)", fontsize=8.8, color=INK2)
    save(fig, "06_parameters_vs_k.png")


def fig07_growth_toy(toy):
    Xtr, Xva, hist = toy
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.3))
    for ax, h in zip(axes, hist[:5]):
        ax.scatter(Xtr[:, 0], Xtr[:, 1], s=2.5, color="#b9b8b1", alpha=0.6, zorder=1)
        nxt = int(np.argmax(h["scores"])) if h["k"] < 5 else None
        draw_ellipses(ax, h["g"], highlight=nxt)
        ax.set_aspect("equal")
        ax.set_xlim(-7, 7)
        ax.set_ylim(-6, 6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f"K = {h['k']}" + (f"\nsẽ tách thành phần {nxt}" if nxt is not None else ""), fontsize=10.5, weight="bold", loc="left")
    fig.suptitle("Ý tưởng: bắt đầu K = 1, mỗi phase tách thành phần khớp kém nhất (màu cam), huấn luyện tiếp rồi lặp lại",
                 fontsize=11.5, weight="bold", y=1.0)
    fig.text(0.5, 0.03, "Mô phỏng 2D đồ chơi (EM), không phải dữ liệu IMPTC. Elip = 2σ của mỗi thành phần.",
             ha="center", fontsize=9, color=INK2)
    save(fig, "07_growth_idea_toy.png")


def fig08_split_mechanics():
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    grid = np.linspace(-4, 4, 301)
    xx, yy = np.meshgrid(grid, grid)
    pts = np.stack([xx.ravel(), yy.ravel()], 1)
    cov = np.array([[1.2, 0.5], [0.5, 0.8]])
    mu = np.array([0.0, 0.0])

    def density(delta):
        if delta == 0:
            return np.exp(gauss_logpdf(pts, mu, cov)).reshape(xx.shape)
        d = np.array([delta, delta])
        return (0.5 * np.exp(gauss_logpdf(pts, mu + d, cov)) + 0.5 * np.exp(gauss_logpdf(pts, mu - d, cov))).reshape(xx.shape)

    p0, p1, p2 = density(0), density(0.05), density(0.5)
    for ax, p, title in zip(axes[:3], (p0, p1, p2), ("1 thành phần (trước tách)", "sau tách, δ = 0.05 (dùng thật)", "sau tách, δ = 0.5 (phóng đại để thấy)")):
        ax.contourf(xx, yy, p, levels=8, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", SEQ))
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10, weight="bold", loc="left")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax, d in zip(axes[1:3], (0.05, 0.5)):
        ax.plot([d, -d], [d, -d], "o", color=ORANGE, ms=5, zorder=5)
    ax = axes[3]
    diff = np.abs(p1 - p0)
    im = ax.imshow(diff, extent=[-4, 4, -4, 4], origin="lower", cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", SEQ))
    ax.set_title("|p(δ=0.05) − p(trước)|", fontsize=10, weight="bold", loc="left")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=8)
    ratio = float(diff.max() / p0.max())
    print("split diff / peak density:", ratio)
    fig.text(0.5, 0.0, f"Sai khác lớn nhất bằng {100 * ratio:.2f}% mật độ đỉnh: tách xong, phân phối gần như không đổi nhưng hai bản đã khác nhau "
             "nên gradient sẽ kéo chúng tách dần.", ha="center", fontsize=9.5, color=INK2)
    fig.suptitle("Tách một thành phần: hai bản π/2 đặt lệch ±δ quanh tâm cũ (δ = 0.05)", fontsize=12, weight="bold", y=1.04)
    save(fig, "08_split_keeps_distribution.png")
    return ratio


def fig09_row_remap():
    horizon, k_old, src = 2, 3, 1
    rows = fc_row_map(k_old, src, horizon).numpy().reshape(horizon, 6, k_old + 1)
    cols = [BLUE, ORANGE, AQUA, YELLOW]
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    blank(ax, (-1.2, 12.2), (-1.0, 5.9))
    n_old, n_new = horizon * 6 * k_old, horizon * 6 * (k_old + 1)
    cw_old, cw_new = 12.0 / n_old, 12.0 / n_new
    for i in range(n_old):
        t, rem = divmod(i, 6 * k_old)
        b, k = divmod(rem, k_old)
        ax.add_patch(Rectangle((i * cw_old, 4.0), cw_old * 0.94, 0.75, fc=TINT[cols[k]], ec=cols[k], lw=1.0))
        ax.text(i * cw_old + cw_old * 0.47, 4.38, str(i), ha="center", va="center", fontsize=6.8)
    for j in range(n_new):
        t, rem = divmod(j, 6 * (k_old + 1))
        b, k = divmod(rem, k_old + 1)
        new = k == k_old
        ax.add_patch(Rectangle((j * cw_new, 0.2), cw_new * 0.94, 0.75, fc=TINT[cols[k]], ec=cols[k], lw=2.4 if new else 1.0,
                               hatch="//" if new else None))
        ax.text(j * cw_new + cw_new * 0.47, 0.58, str(j), ha="center", va="center", fontsize=6.8)
        old_row = int(rows[t, b, k])
        ax.plot([old_row * cw_old + cw_old * 0.47, j * cw_new + cw_new * 0.47], [4.0, 0.95], color=cols[src] if new else cols[k],
                lw=1.5 if new else 0.55, alpha=0.9 if new else 0.55, zorder=1)
    ax.text(-0.15, 4.38, "fc cũ\n(K = 3)", ha="right", va="center", fontsize=10, weight="bold")
    ax.text(-0.15, 0.58, "fc mới\n(K = 4)", ha="right", va="center", fontsize=10, weight="bold")
    ax.text(6, 5.45, "new_row[t, khối, k] = old_row[t, khối, k] nếu k < K;   = old_row[t, khối, nguồn] nếu k = K", ha="center", fontsize=10.5, bbox=dict(fc=SURFACE, ec=INK2, boxstyle="round,pad=0.35"), zorder=5)
    ax.text(6, 4.95, "hàng của bước t = 0 (18 hàng) và t = 1 (18 hàng) trong fc cũ", ha="center", fontsize=9.5, color=INK2)
    ax.text(6, -0.55, "fc mới: mỗi bước có 24 hàng, nên các hàng của t = 1 bị đẩy từ 18 → 24. Thành phần mới (k = 3, gạch chéo) sao chép hàng của k = 1.",
            ha="center", fontsize=9.5, color=INK2)
    save(fig, "09_fc_row_remap.png")


def fig10_pi_split():
    logits = np.array([1.2, 0.3, -0.5])
    p_old = np.exp(logits) / np.exp(logits).sum()
    src = 1
    new_logits = np.append(logits, logits[src])
    new_logits[src] -= np.log(2)
    new_logits[-1] -= np.log(2)
    p_new = np.exp(new_logits) / np.exp(new_logits).sum()
    print("pi old", p_old, "pi new", p_new, "src sum", p_new[src] + p_new[-1], "vs", p_old[src])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    cols = [BLUE, ORANGE, AQUA, YELLOW]
    for ax, p, lg, title in ((axes[0], p_old, logits, "Trước: K = 3"), (axes[1], p_new, new_logits, "Sau tách thành phần 1: K = 4")):
        ax.bar(range(len(p)), p, color=[cols[i] for i in range(len(p))], width=0.6)
        for i, v in enumerate(p):
            ax.text(i, v + 0.015, f"π = {v:.3f}", ha="center", fontsize=9.5)
            ax.text(i, -0.075, f"logit {lg[i]:+.2f}", ha="center", fontsize=8.5, color=INK2)
        ax.set_ylim(-0.1, 0.75)
        ax.set_xticks(range(len(p)))
        ax.set_xticklabels([f"k={i}" + (" (mới)" if i == 3 else "") for i in range(len(p))])
        ax.set_title(title, loc="left", fontsize=11, weight="bold")
        ax.set_ylabel("trọng số π")
    fig.suptitle("Chia π: hai bản đều lấy logit − ln 2, nên tổng trọng số của thành phần bị tách không đổi và tổng π vẫn bằng 1",
                 fontsize=10.5, weight="bold", y=1.02)
    save(fig, "10_pi_split.png")


def fig11_adam():
    rng = np.random.default_rng(3)
    horizon, k_old, src = 1, 3, 1
    m_old = rng.normal(size=(6 * k_old,)) * np.array([1, 0.6, 1.4, 0.8, 0.5, 1.1] * 1).repeat(k_old)
    rows = fc_row_map(k_old, src, horizon).numpy()
    m_new = m_old[rows]
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.4))
    fig.subplots_adjust(hspace=0.75)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("d", ["#2a78d6", "#f0efec", "#eb6834"])
    for ax, m, K, title in ((axes[0], m_old, 3, "Moment Adam exp_avg của fc cũ (K = 3)"), (axes[1], m_new, 4, "Sau remap: m_new = m_old[rows]  (K = 4)")):
        ax.imshow(m[None, :], cmap=cmap, vmin=-2, vmax=2, aspect="auto")
        for i, v in enumerate(m):
            ax.text(i, 0, f"{v:+.1f}", ha="center", va="center", fontsize=8)
        for b in range(1, 6):
            ax.axvline(b * K - 0.5, color=INK, lw=1.4)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(title, loc="left", fontsize=10.5, weight="bold")
        ax.grid(False)
        for b in range(6):
            ax.text(b * K + (K - 1) / 2, 0.68, ["μx", "μy", "log σx", "log σy", "ρ", "π"][b], ha="center", fontsize=8.5, color=INK2)
        if K == 4:
            for b in range(6):
                ax.add_patch(Rectangle((b * K + 2.5, -0.5), 1, 1, fill=False, ec=YELLOW, lw=2.6, hatch="//", zorder=4))
    fig.suptitle("Trạng thái Adam đi theo cùng bản đồ chỉ số: hàng mới thừa kế moment của hàng nguồn", fontsize=11.5, weight="bold", y=1.04)
    save(fig, "11_adam_state_remap.png")


def fig12_schedule():
    g = dict(p0=400, pe=200, k0=1, kmax=12)
    n_phase = g["kmax"] - g["k0"] + 1
    edges = [0, g["p0"]] + [g["p0"] + i * g["pe"] for i in range(1, n_phase)]
    total = edges[-1]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.25]})
    ax = axes[0]
    for i in range(n_phase):
        ax.hlines(g["k0"] + i, edges[i], edges[i + 1], color=BLUE, lw=3)
        if i:
            ax.vlines(edges[i], g["k0"] + i - 1, g["k0"] + i, color=ORANGE, lw=1.5, ls="--")
    ax.set_ylabel("K")
    ax.set_yticks([1, 3, 6, 9, 12])
    ax.set_title("K tăng 1 đơn vị sau mỗi phase (tách thành phần), 2600 epoch tổng cộng", loc="left", fontsize=11, weight="bold")
    ax = axes[1]
    e = np.arange(0, 2501)
    lr_base = 1e-3 * (1 + (1e-4 - 1) * e / 2500)
    ax.plot(e, lr_base, color=INK2, lw=1.8)
    ax.text(40, 3e-8, "đường xám: baseline, một lần giảm tuyến tính\nxuống 1e-7 trong 2500 epoch", fontsize=9, color=INK2)
    for i in range(n_phase):
        a, b = edges[i], edges[i + 1]
        x = np.arange(a, b + 1)
        peak = 1e-3 if i == 0 else 1e-4
        ax.plot(x, peak * (1 + (1e-4 - 1) * (x - a) / (b - a)), color=BLUE, lw=1.8)
    ax.text(30, 1.4e-3, "growing_mdn: phase 0 từ 1e-3; mỗi phase sau đặt lại LR ở 1e-4 rồi giảm", fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(5e-9, 4e-3)
    ax.set_xlim(0, 2900)
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate (log)")
    ax.set_title("LinearLR chạy lại trong mỗi phase, đỉnh 1e-4 từ phase thứ hai (khác baseline, đã ghi rõ)", loc="left", fontsize=11, weight="bold")
    save(fig, "12_schedule_k_and_lr.png")


def fig13_scores(toy):
    Xtr, Xva, hist = toy
    h = hist[2]
    g = h["g"]
    scores, r = toy_scores(Xtr, g)
    cols = [BLUE, ORANGE, AQUA]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    lab = r.argmax(1)
    for k in range(3):
        ax.scatter(Xtr[lab == k, 0], Xtr[lab == k, 1], s=4, color=cols[k], alpha=0.55)
    draw_ellipses(ax, g, color=INK)
    for k, mu in enumerate(g[1]):
        ax.text(mu[0], mu[1], str(k), fontsize=12, weight="bold", ha="center", va="center",
                bbox=dict(fc=SURFACE, ec=INK2, boxstyle="circle,pad=0.2"), zorder=6)
    ax.set_aspect("equal")
    ax.set_xlim(-7, 7)
    ax.set_ylim(-6, 6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("Mỗi điểm tô theo thành phần có responsibility lớn nhất", loc="left", fontsize=10.5, weight="bold")
    ax = axes[1]
    best = int(np.argmax(scores))
    ax.bar(range(3), scores, color=[ORANGE if i == best else BLUE for i in range(3)], width=0.55)
    for i, v in enumerate(scores):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"thành phần {i}" for i in range(3)])
    ax.set_ylabel("điểm = Σ r·(−log N_k) / Σ r")
    ax.set_ylim(0, scores.max() * 1.25)
    ax.set_title("Điểm cao nhất = khớp kém nhất → bị tách", loc="left", fontsize=10.5, weight="bold")
    ax.text(best, scores[best] * 0.5, "tách", ha="center", color=SURFACE, weight="bold", fontsize=11)
    fig.suptitle("Chọn thành phần để tách (mô phỏng 2D đồ chơi, cùng công thức với code thật)", fontsize=11.5, weight="bold", y=1.03)
    save(fig, "13_which_component_to_split.png")


def fig14_phase_loop():
    fig, ax = plt.subplots(figsize=(11.5, 4.7))
    blank(ax, (0, 11.5), (0, 4.7))
    steps = [("1. Huấn luyện phase\nK cố định, Adam\nLinearLR chạy lại", TINT[BLUE]),
             ("2. Nạp lại trạng thái\ntốt nhất của phase\n(best_k##.pt)", SURFACE),
             ("3. Metric chính thức\n+ lưu phase_k##.pt\n+ phase_summary.json", SURFACE),
             ("4. Chấm điểm thành\nphần trên tập train;\nchọn cái kém nhất", TINT[ORANGE]),
             ("5. Tách → K + 1,\nremap fc và Adam,\ntạo optimizer mới", TINT[ORANGE])]
    xs = [0.1, 2.35, 4.6, 6.85, 9.1]
    for x, (t, f) in zip(xs, steps):
        box(ax, x, 2.2, 2.1, 1.7, t, fc=f, size=8.8)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 2.12, 3.05), (b - 0.02, 3.05))
    ax.plot([10.15, 10.15, 1.15], [2.18, 1.95, 1.95], color=ORANGE, lw=1.8, zorder=1)
    arrow(ax, (1.15, 1.95), (1.15, 2.2), color=ORANGE)
    ax.text(5.6, 1.7, "lặp lại cho tới khi K = K_max (12)", ha="center", fontsize=10.5, color=INK, weight="bold")
    box(ax, 4.0, 0.15, 3.6, 0.95, "K_max: lưu final.pt, xong huấn luyện\n(K được chọn SAU, bằng select_k.py)", fc=SURFACE, size=9.5)
    arrow(ax, (5.8, 1.55), (5.8, 1.12), color=INK2)
    ax.text(5.75, 4.45, "Một vòng của growing_mdn", ha="center", fontsize=12, weight="bold")
    save(fig, "14_phase_loop.png")


def fig15_selection(toy):
    Xtr, Xva, hist = toy
    ks = [h["k"] for h in hist]
    val = np.array([h["val"] for h in hist])
    eps = 0.02
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    ax = axes[0]
    ax.plot(ks, val, "-o", color=BLUE, lw=2, ms=6)
    best = val.min()
    ax.axhspan(best, best + eps, color=ORANGE, alpha=0.18)
    ax.axhline(best, color=ORANGE, lw=1.2, ls="--")
    cand = [k for k, v in zip(ks, val) if v <= best + eps]
    sel = min(cand)
    ax.plot([sel], [val[sel - 1]], "o", ms=14, mfc="none", mec=ORANGE, mew=2.4)
    ax.text(sel + 0.2, val[sel - 1] + 0.12, f"K chọn = {sel}\n(K nhỏ nhất trong dải ε)", fontsize=9.5)
    ax.text(ks[-1] - 0.1, best + 0.07, "dải ε (cam)", color=INK2, fontsize=9, ha="right", va="bottom")
    ax.set_xticks(ks)
    ax.set_xlabel("K")
    ax.set_ylabel("validation NLL (đồ chơi)")
    ax.set_title("Mô phỏng đồ chơi: luật chọn K nhỏ nhất", loc="left", fontsize=10.5, weight="bold")
    ax = axes[1]
    v = np.array([r["validation_nll"] for r in ABL])
    ax.plot(KS, v, "-o", color=BLUE, lw=2, ms=6)
    ax.axhspan(v.min(), v.min() + eps, color=ORANGE, alpha=0.18)
    ax.axhline(v.min(), color=ORANGE, lw=1.2, ls="--")
    ax.set_xticks(KS)
    ax.set_xlabel("K")
    ax.set_ylabel("validation NLL (baseline)")
    ax.set_title("Baseline thật: với ε = 0.02 chỉ K = 8 nằm trong dải", loc="left", fontsize=10.5, weight="bold")
    fig.suptitle("Luật chọn K: K nhỏ nhất có validation NLL trong ε của tốt nhất, rồi kiểm thêm Ravg/Rmin/S68/S95", fontsize=11.5, weight="bold", y=1.04)
    fig.text(0.5, -0.04, "Bên phải chỉ minh họa luật trên các model baseline độc lập (1 seed), KHÔNG phải kết quả của growing_mdn. Tolerance ε = 0.02 là đề xuất.",
             ha="center", fontsize=9, color=INK2)
    save(fig, "15_selection_rule.png")


def fig16_protocol():
    fig, ax = plt.subplots(figsize=(11.8, 4.4))
    blank(ax, (0, 11.8), (0, 4.4))
    box(ax, 0.1, 2.3, 2.3, 1.6, "Huấn luyện growing_mdn\nK: 1 → 12\n(chỉ dùng train + validation)", fc=TINT[BLUE])
    box(ax, 3.3, 2.3, 2.7, 1.6, "select_k.py\n• đọc ngưỡng đã đóng băng\n• run phải completed, đủ K\n• không ghi đè", fc=TINT[ORANGE], size=9)
    box(ax, 6.9, 2.3, 1.9, 1.6, "selection.json\n(K đã chọn)", fc=SURFACE)
    box(ax, 9.4, 2.3, 2.3, 1.6, "evaluate --official\nTEST đúng 1 lần\ncho K đã chọn", fc=TINT[AQUA], size=9.5)
    for a, b in ((2.4, 3.3), (6.0, 6.9), (8.8, 9.4)):
        arrow(ax, (a, 3.1), (b, 3.1))
    box(ax, 3.3, 0.35, 5.5, 1.1, "evaluate --limit N  →  dùng N mẫu VALIDATION, không đọc test\n(chỉ kiểm tra đường ống)", fc=SURFACE, size=9)
    ax.text(5.9, 4.15, "Giao thức: dữ liệu test không bao giờ ảnh hưởng tới việc chọn K", ha="center", fontsize=12, weight="bold")
    ax.text(10.55, 1.85, "từ chối nếu: K ≠ selection,\nđã có kết quả official,\nselection của run khác", ha="center", fontsize=8.8, color=INK2, va="top")
    save(fig, "16_test_protocol.png")


def fig17_smoke():
    base = ROOT / "results/trained_models/growing_mdn/imptc/smoke_growing_peds_imptc/runs/growing_smoke_seed2024"
    rows = read_csv(base / "history.csv")
    growth = [json.loads(line) for line in (base / "growth_history.jsonl").read_text().splitlines()]
    ep = [int(r["epoch"]) for r in rows]
    val = [float(r["validation_nll"]) for r in rows]
    ks = [int(r["num_gaussians"]) for r in rows]
    cols = {1: BLUE, 2: ORANGE, 3: AQUA}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    ax.plot(ep, val, "-", color=INK2, lw=1.2, zorder=1)
    for e, v, k in zip(ep, val, ks):
        ax.plot(e, v, "o", color=cols[k], ms=9, zorder=3)
        ax.text(e, v - 0.012, f"K={k}", ha="center", va="top", fontsize=9)
    for gr in growth:
        ax.axvline(gr["epoch"] + 0.5, color=ORANGE, lw=1.3, ls="--")
        ax.text(gr["epoch"] + 0.55, max(val) - 0.01, f"tách\n{gr['k_before']}→{gr['k_after']}", fontsize=9, va="top")
    ax.set_xticks(ep)
    ax.set_xlim(0.6, 4.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation NLL")
    ax.set_title("Smoke run: K = 1, 1, 2, 3 qua 4 epoch", loc="left", fontsize=10.5, weight="bold")
    ax = axes[1]
    deltas = [gr["validation_nll_after"] - gr["validation_nll_before"] for gr in growth]
    ax.bar(range(len(deltas)), deltas, color=BLUE, width=0.5)
    for i, d in enumerate(deltas):
        ax.text(i, d - 0.0012, f"{d:+.4f}", ha="center", va="top", fontsize=10)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels([f"{g['k_before']}→{g['k_after']}" for g in growth])
    ax.set_ylim(-0.016, 0.004)
    ax.set_ylabel("ΔNLL validation ngay sau khi tách")
    ax.set_title("Thay đổi NLL do việc tách", loc="left", fontsize=10.5, weight="bold")
    fig.suptitle("Dữ liệu thật từ smoke run (4 epoch, tập train con rất nhỏ): chỉ kiểm tra đường ống, không nói gì về chất lượng", fontsize=10.5,
                 weight="bold", y=1.04)
    save(fig, "17_smoke_run.png")


def fig18_live():
    base = ROOT / "results/trained_models/growing_mdn/imptc/growing_peds_imptc/runs/growing_k12_seed2024"
    rows = read_csv(base / "history.csv")
    if len(rows) < 30:
        print("run chưa đủ dữ liệu, bỏ qua fig18")
        return None
    growth_path = base / "growth_history.jsonl"
    growth = [json.loads(line) for line in growth_path.read_text().splitlines()] if growth_path.exists() else []
    ep = np.array([int(r["epoch"]) for r in rows])
    val = np.array([float(r["validation_nll"]) for r in rows])
    tr = np.array([float(r["train_nll"]) for r in rows])
    ks = np.array([int(r["num_gaussians"]) for r in rows])
    palette = [BLUE, ORANGE, AQUA, YELLOW]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    m = ep >= 20
    for k in sorted(set(ks)):
        sel = m & (ks == k)
        if sel.any():
            ax.plot(ep[sel], val[sel], color=palette[(k - 1) % 4], lw=2)
            ax.text(ep[sel].mean(), 1.35, f"K={k}", ha="center", fontsize=9.5)
    ax.plot(ep[m], tr[m], color=INK2, lw=0.9, alpha=0.6)
    for gr in growth:
        ax.axvline(gr["epoch"] + 0.5, color=ORANGE, lw=1.1, ls="--")
    ax.set_ylim(-0.8, 1.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("NLL (đường màu: validation, xám: train)")
    ax.set_title(f"Toàn run v1 đến epoch {int(ep.max())} (trục y cắt ở 1.6)", loc="left", fontsize=10.5, weight="bold")
    ax = axes[1]
    z = (ep >= 385) & (ep <= 440)
    ax.plot(ep[z], val[z], "-o", color=INK, ms=3.5, lw=1.4)
    ax.plot(ep[z], tr[z], "-s", color=INK2, ms=3, lw=1, alpha=0.7)
    ax.set_yscale("symlog", linthresh=0.5)
    ax.axvline(400.5, color=ORANGE, lw=1.3, ls="--")
    ax.text(401.5, 40, "tách K: 1 → 2\n(LR đặt lại 1e-3)", fontsize=9, color=INK)
    if growth:
        ax.plot([400.5], [growth[0]["validation_nll_after"]], "v", color=RED, ms=9)
        ax.text(402, growth[0]["validation_nll_after"], f"validation NLL ngay sau tách = {growth[0]['validation_nll_after']:.1f}\n(trước tách: {growth[0]['validation_nll_before']:.3f})", fontsize=8.8, va="center")
    ax.text(417, 6, "chấm tròn đen: validation\nô vuông xám: train", fontsize=8.8, color=INK2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("NLL (thang symlog)")
    ax.set_title("Phóng to quanh lần tách đầu tiên", loc="left", fontsize=10.5, weight="bold")
    fig.suptitle("Run v1 growing_k12_seed2024 (đã dừng bởi người dùng, δ tuyệt đối 0.05): NLL KHÔNG liên tục qua lần tách", fontsize=11.5, weight="bold", y=1.03)
    save(fig, "18_live_run_snapshot.png")
    return {"epochs": int(ep.max()), "k_now": int(ks[-1]), "val_last": float(val[-1]),
            "split": growth[0] if growth else None}


def fig19_sigma_vs_delta():
    z = np.load(ROOT / "results/trained_models/growing_mdn/imptc/growing_peds_imptc/runs/growing_k12_seed2024/fixed_samples/predictions/phase_k01.npz")
    sg = z["sigma"][:, :, 0, :]
    t = np.arange(1, 49)
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for i in range(sg.shape[0]):
        ax.plot(t, sg[i, :, 0], color=INK2, lw=0.8, alpha=0.4)
        ax.plot(t, sg[i, :, 1], color=INK2, lw=0.8, alpha=0.4)
    ax.plot(t, np.median(sg[:, :, 0], 0), color=BLUE, lw=2.4)
    ax.plot(t, np.median(sg[:, :, 1], 0), color=ORANGE, lw=2.4)
    ax.axhline(0.05, color=RED, lw=1.8, ls="--")
    ax.set_yscale("log")
    ax.text(1.5, 0.062, "δ = 0.05 (tuyệt đối, đang dùng)", color=INK, fontsize=9.5)
    ax.text(40, np.median(sg[:, 40, 0]) * 0.55, "σx", color=INK2, fontsize=10)
    ax.text(40, np.median(sg[:, 40, 1]) * 1.25, "σy", color=INK2, fontsize=10)
    ax.set_xlabel("bước dự báo t (0.1 s mỗi bước)")
    ax.set_ylabel("σ của model K = 1 đã train (m, log)")
    ax.set_title("Ở các bước đầu σ chỉ cỡ 1e-3 m: lệch tâm 0.05 m bằng hàng chục σ", loc="left", fontsize=11, weight="bold")
    ax.text(1.5, 0.0004, "xám mảnh: 8 mẫu validation cố định; đường màu: trung vị (xanh dương = σx, cam = σy)", fontsize=8.8, color=INK2)
    save(fig, "19_sigma_vs_delta.png")


def fig20_split_check():
    data = json.loads((ROOT / "growing_mdn/reports/SPLIT_CHECK.json").read_text())
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    labels = [f"K = {d['k_before']} → {d['k_before'] + 1}" for d in data]
    x = np.arange(len(data))
    ab = [d["absolute_0.05"]["delta_nll"] for d in data]
    rel = [abs(d["relative"]["delta_nll"]) for d in data]
    ax.bar(x - 0.18, ab, 0.34, color=ORANGE)
    ax.bar(x + 0.18, rel, 0.34, color=BLUE)
    for i, (a_, r_) in enumerate(zip(ab, rel)):
        ax.text(i - 0.18, a_ * 1.25, f"{a_:.2f}", ha="center", fontsize=10)
        ax.text(i + 0.18, r_ * 1.5, f"{r_:.5f}", ha="center", fontsize=10)
    ax.set_yscale("log")
    ax.axhline(0.01, color=INK2, lw=1.2, ls="--")
    ax.text(len(data) - 0.5, 0.0115, "ngưỡng dừng run 0.01", ha="right", fontsize=9, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(5e-6, 600)
    ax.set_ylabel("|ΔNLL validation| ngay sau tách (log)")
    ax.set_title("Trên trọng số thật: tách tuyệt đối (cam) phá NLL, tách tương đối σ (xanh) gần như không đổi", loc="left", fontsize=10.5, weight="bold")
    ax.text(0.98, 0.62, "cam: δ = 0.05 m tuyệt đối (v1)\nxanh: δ = 0.05 × σ trung vị (v2)", transform=ax.transAxes, fontsize=9.5, va="top", ha="right", color=INK2)
    save(fig, "20_split_check_real_weights.png")


def fig21_v1_v2():
    base = ROOT / "results/trained_models/growing_mdn/imptc/growing_peds_imptc/runs"
    v1, v2 = base / "growing_k12_seed2024" / "history.csv", base / "growing_k12_v2_seed2024" / "history.csv"
    if not (v1.exists() and v2.exists()):
        return None
    r1, r2 = read_csv(v1), read_csv(v2)
    if len(r2) < 420:
        print("v2 chưa qua lần tách đầu, bỏ qua fig21")
        return None
    e1 = np.array([int(r["epoch"]) for r in r1]); n1 = np.array([float(r["validation_nll"]) for r in r1])
    e2 = np.array([int(r["epoch"]) for r in r2]); n2 = np.array([float(r["validation_nll"]) for r in r2])
    top = min(e1.max(), e2.max())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    z1, z2 = (e1 >= 385) & (e1 <= 440), (e2 >= 385) & (e2 <= 440)
    ax.plot(e1[z1], n1[z1], "-o", color=ORANGE, ms=3.5, lw=1.8)
    ax.plot(e2[z2], n2[z2], "-o", color=BLUE, ms=3.5, lw=1.8)
    ax.axvline(400.5, color=INK2, lw=1.1, ls="--")
    ax.set_yscale("symlog", linthresh=0.5)
    ax.set_ylim(-0.9, 3)
    ax.text(412, 0.9, "v1: δ tuyệt đối, LR đặt lại 1e-3", color=INK, fontsize=9.5)
    ax.text(412, -0.12, "v2: δ theo σ, LR đặt lại 1e-4", color=INK, fontsize=9.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation NLL (symlog)")
    ax.set_title("Quanh lần tách K = 1 → 2", loc="left", fontsize=10.5, weight="bold")
    ax = axes[1]
    m1, m2 = e1 <= top, e2 <= top
    ax.plot(e1[m1], n1[m1], color=ORANGE, lw=1.8)
    ax.plot(e2[m2], n2[m2], color=BLUE, lw=1.8)
    ax.set_ylim(-1.3, 0.8)
    ax.text(120, 0.62, "cam: v1 (δ tuyệt đối)", color=INK2, fontsize=9.5)
    ax.text(120, 0.46, "xanh: v2 (δ theo σ)", color=INK2, fontsize=9.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation NLL")
    ax.set_title(f"Toàn bộ đến epoch {int(top)} (trục y cắt)", loc="left", fontsize=10.5, weight="bold")
    fig.suptitle("Run v1 (δ tuyệt đối) so với run v2 (δ theo σ), cùng seed, dữ liệu thật", fontsize=11.5, weight="bold", y=1.03)
    save(fig, "21_v1_vs_v2.png")
    return {"v2_epochs": int(e2.max())}


def fig22_v2_curve():
    base = ROOT / "results/trained_models/growing_mdn/imptc/growing_peds_imptc/runs/growing_k12_v2_seed2024"
    sel_path, sm_path = base / "selection.json", base / "phase_summary.json"
    if not (sel_path.exists() and sm_path.exists()):
        print("v2 chưa có selection.json, bỏ qua fig22")
        return None
    sel = json.loads(sel_path.read_text())
    ph = json.loads(sm_path.read_text())["phases"]
    ks = sorted(int(k) for k in ph)
    nll = [ph[str(k)]["validation_nll"] for k in ks]
    get = lambda key: [ph[str(k)]["metrics"][key] for k in ks]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1))
    ax = axes[0]
    ax.plot(ks, nll, "-o", color=BLUE, lw=2, ms=5)
    best = min(nll)
    eps = sel["rule"]["epsilon_nll"]
    ax.axhspan(best, best + eps, color=ORANGE, alpha=0.18)
    ax.plot([sel["selected_k"]], [nll[ks.index(sel["selected_k"])]], "o", ms=14, mfc="none", mec=ORANGE, mew=2.4)
    ax.text(0.97, 0.95, f"K chọn = {sel['selected_k']}\n(K tốt nhất theo NLL = {sel['best_nll_k']})", transform=ax.transAxes, ha="right", va="top", fontsize=9.5)
    ax.set_xticks(ks)
    ax.set_xlabel("K")
    ax.set_ylabel("validation NLL (toàn bộ validation)")
    ax.set_title("NLL chưa bão hòa đến K = 12", loc="left", fontsize=10.5, weight="bold")
    ax = axes[1]
    ax.plot(ks, get("ravg_percent"), "-o", color=BLUE, lw=2, ms=4)
    ax.plot(ks, get("rmin_percent"), "-o", color=ORANGE, lw=2, ms=4)
    ax.text(ks[-1] + 0.2, get("ravg_percent")[-1], "Ravg", va="center", fontsize=9, color=INK2)
    ax.text(ks[-1] + 0.2, get("rmin_percent")[-1], "Rmin", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0.5, 13.3)
    ax.set_xticks(ks)
    ax.set_xlabel("K")
    ax.set_ylabel("độ tin cậy (%)")
    ax.set_title("Độ tin cậy", loc="left", fontsize=10.5, weight="bold")
    ax = axes[2]
    ax.plot(ks, get("s68_m2_per_s"), "-o", color=BLUE, lw=2, ms=4)
    ax.plot(ks, get("s95_m2_per_s"), "-o", color=ORANGE, lw=2, ms=4)
    ax.text(ks[-1] + 0.2, get("s68_m2_per_s")[-1], "S68", va="center", fontsize=9, color=INK2)
    ax.text(ks[-1] + 0.2, get("s95_m2_per_s")[-1], "S95", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0.5, 13.3)
    ax.set_xticks(ks)
    ax.set_xlabel("K")
    ax.set_ylabel("độ sắc nét (thấp = sắc)")
    ax.set_title("Độ sắc nét", loc="left", fontsize=10.5, weight="bold")
    fig.suptitle("Run growing_mdn v2 đã xong [thật], validation, một seed: mỗi K chỉ được train 200 epoch (K = 1 được 400), nên không so trực tiếp với baseline", fontsize=10.5, weight="bold", y=1.04)
    save(fig, "22_v2_curve_by_k.png")
    return {"selected_k": sel["selected_k"]}


def main():
    info = {}
    fig01_pipeline()
    fig02_layout()
    fig03_baseline_nll()
    fig04_baseline_panels()
    info["k8"] = fig05_k8_usage()
    fig06_params()
    toy = toy_run()
    print("toy val NLL by K:", [round(h["val"], 3) for h in toy[2]])
    fig07_growth_toy(toy)
    info["split_ratio"] = fig08_split_mechanics()
    fig09_row_remap()
    fig10_pi_split()
    fig11_adam()
    fig12_schedule()
    fig13_scores(toy)
    fig14_phase_loop()
    fig15_selection(toy)
    fig16_protocol()
    fig17_smoke()
    info["live"] = fig18_live()
    fig19_sigma_vs_delta()
    fig20_split_check()
    info["v1v2"] = fig21_v1_v2()
    info["v2curve"] = fig22_v2_curve()
    (OUT / "figure_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
