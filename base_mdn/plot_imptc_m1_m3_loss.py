#!/usr/bin/env python3
"""Plot IMPTC train/validation NLL for the completed M=1 and M=3 runs.

Run from the repository root:
    .venv/bin/python base_mdn/plot_imptc_m1_m3_loss.py

The script only reads history.csv files; it does not load or train models.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = {
    1: PROJECT_ROOT / "results/trained_models/base_mdn/imptc/m1_peds_imptc/runs/imptc_m1_seed2024/history.csv",
    3: PROJECT_ROOT / "results/trained_models/base_mdn/imptc/default_peds_imptc/runs/imptc_baseline_seed2024/history.csv",
}
DEFAULT_OUTPUT = PROJECT_ROOT / "results/ablations/imptc_num_gaussians/m1_m3_training_nll.png"


def load_history(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing training history: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"epoch", "train_nll", "validation_nll", "is_best"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Missing required columns or rows in {path}")
    history = {
        key: [float(row[key]) for row in rows]
        for key in ("train_nll", "validation_nll")
    }
    history["epoch"] = [int(row["epoch"]) for row in rows]
    if history["epoch"] != list(range(1, len(rows) + 1)):
        raise ValueError(f"Epochs must be consecutive and start at 1: {path}")
    best_rows = [row for row in rows if row["is_best"].strip().lower() == "true"]
    if not best_rows:
        raise ValueError(f"No is_best=True row in {path}")
    history["best_epoch"] = int(best_rows[-1]["epoch"])
    min_val_epoch = history["epoch"][history["validation_nll"].index(min(history["validation_nll"]))]
    if history["best_epoch"] != min_val_epoch:
        raise ValueError(f"Best flag and minimum validation NLL disagree in {path}")
    return history


def plot_histories(histories, output, zoom_start):
    fig, (full_ax, zoom_ax) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=False,
        gridspec_kw={"height_ratios": [1, 1.1]}, constrained_layout=True,
    )
    colors = {
        1: {"train": "#12659A", "validation": "#72B6DD"},
        3: {"train": "#BE5A22", "validation": "#F2A16D"},
    }
    for ax in (full_ax, zoom_ax):
        for m in (1, 3):
            data = histories[m]
            ax.plot(data["epoch"], data["train_nll"], color=colors[m]["train"],
                    linestyle="-", linewidth=1.65, label=f"M={m} train")
            ax.plot(data["epoch"], data["validation_nll"], color=colors[m]["validation"],
                    linestyle="-", linewidth=1.65, label=f"M={m} validation")
        ax.set_ylabel("Negative log-likelihood (NLL)")
        ax.grid(alpha=0.22)
        ax.axhline(0, color="#777777", linewidth=0.75, alpha=0.5)
        for m in (1, 3):
            best_epoch = histories[m]["best_epoch"]
            best_nll = histories[m]["validation_nll"][best_epoch - 1]
            ax.scatter([best_epoch], [best_nll], s=48, color=colors[m]["validation"],
                       edgecolor="white", linewidth=1.0, zorder=5)
        best_epochs = {histories[m]["best_epoch"] for m in (1, 3)}
        for best_epoch in sorted(best_epochs):
            ax.axvline(best_epoch, color="#343a40", linestyle=":",
                       linewidth=1.5, alpha=0.85, zorder=2)

    final_epoch = min(histories[1]["epoch"][-1], histories[3]["epoch"][-1])
    full_ax.set_title("Entire training history (epochs 1–{:d})".format(final_epoch), loc="left")
    full_ax.set_xlim(1, final_epoch)
    full_ax.legend(ncol=4, loc="upper right", frameon=False, fontsize=9)

    zoom_ax.set_title(f"Later training (epochs {zoom_start}–{final_epoch})", loc="left")
    zoom_ax.set_xlim(zoom_start, final_epoch)
    zoom_values = [
        value
        for data in histories.values()
        for key in ("train_nll", "validation_nll")
        for epoch, value in zip(data["epoch"], data[key])
        if zoom_start <= epoch <= final_epoch
    ]
    zoom_min, zoom_max = min(zoom_values), max(zoom_values)
    padding = max((zoom_max - zoom_min) * 0.07, 0.05)
    zoom_ax.set_ylim(zoom_min - padding, zoom_max + padding)
    best_epochs = {histories[m]["best_epoch"] for m in (1, 3)}
    for i, best_epoch in enumerate(sorted(best_epochs)):
        models = ", ".join(f"M={m}" for m in (1, 3)
                           if histories[m]["best_epoch"] == best_epoch)
        zoom_ax.annotate(
            f"Best epoch ({models}): {best_epoch}",
            xy=(best_epoch, 0.96 - 0.09 * i),
            xycoords=("data", "axes fraction"),
            xytext=(7, 0), textcoords="offset points",
            ha="left", va="top", fontsize=9, color="#343a40",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2},
        )
    zoom_ax.set_xlabel("Epoch")
    fig.suptitle("IMPTC · M=1 vs M=3: training and validation NLL", fontsize=15)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-history", type=Path, default=DEFAULT_HISTORY[1])
    parser.add_argument("--m3-history", type=Path, default=DEFAULT_HISTORY[3])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--zoom-start", type=int, default=100)
    args = parser.parse_args()

    histories = {
        1: load_history(args.m1_history),
        3: load_history(args.m3_history),
    }
    final_epoch = min(histories[1]["epoch"][-1], histories[3]["epoch"][-1])
    if not 1 < args.zoom_start < final_epoch:
        parser.error(f"--zoom-start must be between 2 and {final_epoch - 1}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_histories(histories, args.output, args.zoom_start)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
