"""Generate synthetic teaching figures for MODE_CONSISTENT_MDN.md.

The coordinates and probabilities below are illustrative, not IMPTC outputs.
Run from the repository root with .venv/bin/python docs/mode_consistent_mdn/generate_figures.py.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": "white",
})

COLORS = {"straight": "#2563eb", "left": "#d97706", "right": "#16a34a",
          "history": "#334155", "bad": "#dc2626", "ink": "#1e293b"}


def save(fig, name):
    fig.savefig(OUT / name, dpi=190, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def box(ax, xy, wh, title, subtitle, color):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.025",
                           linewidth=1.7, edgecolor=color, facecolor="white")
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            weight="bold", fontsize=12, color=COLORS["ink"])
    ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center",
            fontsize=9, color="#475569")


def arrow(ax, x0, y0, x1, y1, color="#64748b"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=18, linewidth=1.8, color=color))


def pipelines():
    fig, ax = plt.subplots(figsize=(13.5, 5.0))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.02, 0.91, "BASELINE: hỗn hợp riêng ở mỗi thời điểm", weight="bold", fontsize=15, color=COLORS["ink"])
    ax.text(0.02, 0.43, "ĐỀ XUẤT: một mode cho toàn quỹ đạo", weight="bold", fontsize=15, color=COLORS["ink"])
    xs = [0.02, 0.27, 0.53, 0.78]
    widths = [0.18, 0.19, 0.19, 0.20]
    base = [("32 × 4 đầu vào", "vị trí + vận tốc"), ("LSTM", "mã hóa lịch sử"),
            ("MDN", "πₕₖ, μₕₖ, Σₕₖ"), ("48 hỗn hợp 2D", "NLL theo từng h")]
    new = [("32 × 4 đầu vào", "cùng dữ liệu"), ("LSTM", "mã hóa lịch sử"),
           ("Đầu mode + quỹ đạo", "πₖ và μₕₖ, Σₕₖ"), ("3 mode quỹ đạo", "NLL của cả chuỗi")]
    for j, row in enumerate([base, new]):
        y = 0.56 if j == 0 else 0.08
        for i, (title, sub) in enumerate(row):
            color = "#94a3b8" if i < 2 else (COLORS["straight"] if j == 0 else COLORS["left"])
            box(ax, (xs[i], y), (widths[i], 0.23), title, sub, color)
            if i < 3:
                arrow(ax, xs[i] + widths[i] + 0.012, y + 0.115, xs[i+1] - 0.012, y + 0.115)
    save(fig, "01_two_pipelines.png")


def trajectories():
    t = np.array([0, 1, 2, 3, 4], dtype=float)
    paths = {
        "Thẳng": (np.array([0, 1, 2, 3, 4]), np.array([0, 0, 0, 0, 0]), COLORS["straight"]),
        "Rẽ trái": (np.array([0, 1, 1.8, 2.3, 2.5]), np.array([0, .05, .45, 1.2, 2.2]), COLORS["left"]),
        "Rẽ phải": (np.array([0, 1, 1.8, 2.3, 2.5]), np.array([0, -.05, -.45, -1.2, -2.2]), COLORS["right"]),
    }
    return t, paths


def possible_futures():
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    _, paths = trajectories()
    ax.plot([-1.8, -1.2, -.6, 0], [0, 0, 0, 0], "o--", color=COLORS["history"], label="Quan sát")
    for label, (x, y, c) in paths.items():
        ax.plot(x, y, "o-", color=c, linewidth=2.6, label=f"Ví dụ mode: {label}")
    ax.scatter([0], [0], s=90, c="#111827", marker="s", zorder=5)
    ax.set(xlabel="x trong tọa độ ego (ví dụ)", ylabel="y trong tọa độ ego (ví dụ)",
           title="Một lịch sử quan sát có thể dẫn đến nhiều tương lai")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.2)
    ax.legend(loc="upper left")
    ax.text(.98, .04, "DỮ LIỆU GIẢ LẬP", transform=ax.transAxes, ha="right",
            va="bottom", color="#64748b", weight="bold")
    save(fig, "02_possible_futures.png")


def mode_matrix():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    names = ["Mode A", "Mode B", "Mode C"]
    palette = [COLORS["straight"], COLORS["left"], COLORS["right"]]
    for ax, title in zip(axes, ["Baseline: chọn mode ở từng h", "Đề xuất: chọn mode một lần"]):
        ax.set(xlim=(.5, 4.5), ylim=(-.5, 2.5), xticks=[1, 2, 3, 4],
               yticks=[0, 1, 2], yticklabels=names, xlabel="Bước dự báo h", title=title)
        ax.grid(axis="x", alpha=.2)
        for row in range(3):
            ax.scatter([1, 2, 3, 4], [row] * 4, s=290, facecolor="white",
                       edgecolor=palette[row], linewidth=2, zorder=2)
    baseline_rows = [0, 1, 2, 0]
    fixed_rows = [1, 1, 1, 1]
    for ax, rows, color in [(axes[0], baseline_rows, COLORS["bad"]),
                            (axes[1], fixed_rows, COLORS["left"])]:
        ax.plot([1, 2, 3, 4], rows, color=color, linewidth=2.5, zorder=3)
        ax.scatter([1, 2, 3, 4], rows, s=115, c=color, edgecolor="white", zorder=4)
    fig.text(.5, .01, "Ví dụ minh họa cách lấy mẫu; tên mode không được gán trước khi học.",
             ha="center", color="#64748b")
    save(fig, "03_mode_selection.png")


def sampled_paths():
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), sharex=True, sharey=True)
    _, paths = trajectories()
    ordered = list(paths.values())
    for ax in axes:
        ax.plot([-1.4, -.7, 0], [0, 0, 0], "o--", color=COLORS["history"], label="Quan sát")
        for x, y, c in ordered:
            ax.plot(x, y, "--", color=c, alpha=.45, linewidth=2)
        ax.set(xlabel="x (giả lập)", ylabel="y (giả lập)")
        ax.grid(alpha=.2)
        ax.set_aspect("equal", adjustable="box")
    chosen = [0, 1, 2, 0]
    xb = np.array([0] + [ordered[k][0][i] for i, k in enumerate(chosen, start=1)])
    yb = np.array([0] + [ordered[k][1][i] for i, k in enumerate(chosen, start=1)])
    axes[0].plot(xb, yb, "o-", color=COLORS["bad"], linewidth=3, label="Một mẫu có thể xảy ra")
    axes[0].set_title("Baseline: có thể đổi mode giữa các bước")
    axes[1].plot(ordered[1][0], ordered[1][1], "o-", color=COLORS["left"],
                 linewidth=3, label="Một mode xuyên suốt")
    axes[1].set_title("Đề xuất: giữ mode B xuyên suốt")
    for ax in axes:
        ax.legend(loc="lower left", fontsize=9)
    fig.text(.5, .01, "Đường đứt: tâm của ba khả năng giả lập. Đường đậm: một đường được chọn.",
             ha="center", color="#64748b")
    save(fig, "04_sampled_trajectories.png")


def marginal_versus_joint():
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 4.9), sharex=True, sharey=True)
    ts = np.array([1., 2., 3.])
    centers = [np.array([.2, .3, .4]), np.array([0., .6, 1.3]),
               np.array([-.2, -.3, -.4])]
    for ax, title in zip(axes, ["Các lát cắt thời gian nhìn giống nhau", "Cấu trúc nối qua thời gian khác nhau"]):
        ax.set(xlim=(.7, 3.3), ylim=(-.7, 1.7), xticks=ts, xticklabels=["h=1", "h=2", "h=3"],
               xlabel="Thời điểm tương lai", ylabel="Vị trí y (giả lập)", title=title)
        ax.grid(alpha=.16)
        for row, (y, c) in enumerate(zip(centers, [COLORS["straight"], COLORS["left"], COLORS["right"]])):
            for x0, y0 in zip(ts, y):
                ax.add_patch(Ellipse((x0, y0), .20, .24, edgecolor=c,
                                     facecolor=c, alpha=.24, linewidth=1.5))
                ax.scatter(x0, y0, color=c, s=25)
    for y, c in zip(centers, [COLORS["straight"], COLORS["left"], COLORS["right"]]):
        axes[1].plot(ts, y, color=c, linewidth=2.4)
    axes[0].text(2, -0.59, "Mỗi h: 3 Gaussian 2D", ha="center", color="#475569")
    axes[1].text(2, -0.59, "Mode k: một nhánh xuyên suốt", ha="center", color="#475569")
    fig.text(.5, .01, "Cùng phân phối từng bước vẫn có thể cho quy tắc lấy mẫu cả chuỗi khác nhau.",
             ha="center", color="#64748b")
    save(fig, "05_marginal_vs_joint.png")


def caveat():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharex=True, sharey=True)
    x = np.linspace(0, 4, 9)
    center = .13 * x**2
    noise = np.array([0, .14, -.12, .16, -.11, .11, -.13, .12, -.10])
    for ax in axes:
        ax.plot(x, center, "--", color=COLORS["left"], linewidth=2.5, label="Tâm của cùng một mode")
        ax.set(xlim=(-.1, 4.1), ylim=(-.45, 2.35), xlabel="x (giả lập)",
               ylabel="y (giả lập)")
        ax.grid(alpha=.2)
    axes[0].plot(x, center + noise, "o-", color=COLORS["bad"], linewidth=1.8,
                 label="Một mẫu vẫn có thể rung")
    axes[0].set_title("Cùng mode không tự bảo đảm đường trơn")
    axes[1].plot(x, center, "o-", color=COLORS["left"], linewidth=2,
                 label="Đường tâm có thể trơn")
    axes[1].set_title("Độ trơn cần cơ chế hoặc đánh giá riêng")
    for ax in axes:
        ax.legend(loc="upper left", fontsize=9)
    fig.text(.5, .01, "Ví dụ giả lập: các vị trí vẫn được lấy từ Gaussian ở từng thời điểm.",
             ha="center", color="#64748b")
    save(fig, "06_same_mode_not_smoothness.png")


if __name__ == "__main__":
    pipelines()
    possible_futures()
    mode_matrix()
    sampled_paths()
    marginal_versus_joint()
    caveat()
    print(f"Saved 6 figures in {OUT}")
