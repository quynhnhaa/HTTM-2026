#!/usr/bin/env python3
"""Generate presentation-ready figures from an instrumented MDN training run.

This script never trains or evaluates the model. It only reads artifacts already
saved by ExperimentTracker: history.csv, metrics/history.csv, fixed inputs, and
fixed-sample predictions.
"""

import argparse
import ast
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


COLORS = {
    "past": "#0072B2",
    "truth": "#111111",
    "prediction": "#D55E00",
    "expected": "#CC79A7",
    "component": ["#009E73", "#E69F00", "#56B4E9"],
    "train": "#0072B2",
    "validation": "#D55E00",
}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def numeric(rows, key):
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def load_npz(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return np.load(path, allow_pickle=False)


def short_sample_id(sample_id):
    return f"fixed-{str(sample_id).rsplit(':', 1)[-1]}"


def mixture_mean(prediction):
    return np.sum(prediction["pi"][..., None] * prediction["mu"], axis=2)


def total_covariance(pi, mu, covariance):
    mean = np.sum(pi[..., None] * mu, axis=-2)
    delta = mu - mean[..., None, :]
    between = delta[..., :, :, None] * delta[..., :, None, :]
    return np.sum(pi[..., None, None] * (covariance + between), axis=-3)


def ellipse_patch(mean, covariance, probability, **kwargs):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    radius = math.sqrt(-2.0 * math.log(1.0 - probability))
    width, height = 2.0 * radius * np.sqrt(eigenvalues)
    return Ellipse(mean, width, height, angle=angle, **kwargs)


def ellipse_boundary(mean, covariance, probability=0.68, points=180):
    """Return the exact probability-ellipse boundary for axis-limit calculation."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    radius = math.sqrt(-2.0 * math.log(1.0 - probability))
    angles = np.linspace(0.0, 2.0 * np.pi, points)
    unit_circle = np.stack([np.cos(angles), np.sin(angles)])
    transform = eigenvectors @ np.diag(radius * np.sqrt(eigenvalues))
    return (mean[:, None] + transform @ unit_circle).T


def set_equal_limits(axes, point_sets, padding=0.08):
    points = np.concatenate([np.asarray(points).reshape(-1, 2) for points in point_sets], axis=0)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    span = max(float((maximum - minimum).max()), 1e-3) * (1.0 + 2.0 * padding)
    for ax in np.asarray(axes).flat:
        ax.set_xlim(center[0] - span / 2.0, center[0] + span / 2.0)
        ax.set_ylim(center[1] - span / 2.0, center[1] + span / 2.0)
        ax.set_aspect("equal", adjustable="box")


def draw_real_gmm(ax, pi, mu, covariance, probability=0.68, annotate=True, legend=False):
    """Draw actual saved components, preserving raw component index and color."""
    boundaries = []
    offsets = [(-10, 12), (10, -18), (10, 24)]
    horizontal_alignment = ["right", "left", "left"]
    markers = ["o", "s", "^"]
    for component in range(len(pi)):
        color = COLORS["component"][component]
        alpha = float(pi[component])
        ax.add_patch(ellipse_patch(
            mu[component], covariance[component], probability,
            facecolor=color, edgecolor=color, alpha=0.16,
            linewidth=2.0,
        ))
        label = f"Component {component + 1}: π={alpha:.3f}" if legend else None
        ax.scatter(
            *mu[component], color=color, marker=markers[component], s=58,
            edgecolor="white", linewidth=0.6, zorder=6, label=label,
        )
        if annotate:
            ax.annotate(
                f"π{component + 1}={alpha:.2f}", mu[component],
                xytext=offsets[component], textcoords="offset points",
                fontsize=8, color=color, weight="bold",
                ha=horizontal_alignment[component],
            )
        boundaries.append(ellipse_boundary(mu[component], covariance[component], probability))
    return boundaries


def draw_trajectory_context(ax, X, y, timestep, show_labels=True):
    ax.plot(
        X[:, 0], X[:, 1], "-o", color=COLORS["past"], linewidth=2.0,
        markersize=3.0, label="Observed trajectory" if show_labels else None,
        zorder=3,
    )
    ax.scatter(
        X[-1, 0], X[-1, 1], s=85, color=COLORS["past"], marker="D",
        edgecolor="white", linewidth=0.8, label="Last observed position" if show_labels else None,
        zorder=7,
    )
    ax.plot(
        y[:, 0], y[:, 1], "-o", color=COLORS["truth"], linewidth=1.8,
        markersize=2.6, label="Ground-truth future" if show_labels else None,
        zorder=3,
    )
    ax.scatter(
        y[timestep, 0], y[timestep, 1], s=130, color=COLORS["truth"],
        marker="*", edgecolor="white", linewidth=0.7,
        label=f"Ground truth @ t=+{(timestep + 1) * 0.1:.1f}s" if show_labels else None,
        zorder=8,
    )
    ax.set_xlabel("X position [m]")
    ax.set_ylabel("Y position [m]")
    ax.grid(alpha=0.22)
    ax.set_aspect("equal", adjustable="box")


def figure_gmm_single_timestep(inputs, prediction, sample_index, timestep, delta_t, output_dir):
    X = inputs["X"][sample_index]
    y = inputs["y"][sample_index]
    pi = prediction["pi"][sample_index, timestep]
    mu = prediction["mu"][sample_index, timestep]
    covariance = prediction["covariance"][sample_index, timestep]
    seconds = (timestep + 1) * delta_t

    fig, ax = plt.subplots(figsize=(9.2, 8.0))
    draw_trajectory_context(ax, X, y, timestep, show_labels=True)
    boundaries = draw_real_gmm(ax, pi, mu, covariance, probability=0.68, annotate=True, legend=True)
    set_equal_limits([ax], [X[:, :2], y, *boundaries], padding=0.08)
    ax.set_title(f"GMM Prediction at t = +{seconds:.1f} s\nK = {len(pi)} · 68% probability ellipse per component")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"gmm_single_timestep.{suffix}", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_gmm_along_future(inputs, prediction, sample_index, timesteps, delta_t, output_dir):
    X = inputs["X"][sample_index]
    y = inputs["y"][sample_index]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.0), sharex=True, sharey=True)
    all_points = [X[:, :2], y]
    for ax, timestep in zip(axes.flat, timesteps):
        draw_trajectory_context(ax, X, y, timestep, show_labels=False)
        boundaries = draw_real_gmm(
            ax,
            prediction["pi"][sample_index, timestep],
            prediction["mu"][sample_index, timestep],
            prediction["covariance"][sample_index, timestep],
            probability=0.68,
            annotate=True,
            legend=False,
        )
        all_points.extend(boundaries)
        ax.set_title(f"t = +{(timestep + 1) * delta_t:.1f} s")
    set_equal_limits(axes, all_points, padding=0.08)
    handles = [
        plt.Line2D([0], [0], color=COLORS["past"], marker="o", label="Observed trajectory"),
        plt.Line2D([0], [0], color=COLORS["truth"], marker="o", label="Ground-truth future"),
        plt.Line2D([0], [0], color=COLORS["truth"], marker="*", linestyle="None", markersize=11, label="Ground truth at selected time"),
    ] + [
        plt.Line2D([0], [0], color=COLORS["component"][k], marker=["o", "s", "^"][k], linestyle="None", label=f"Gaussian component {k + 1}")
        for k in range(3)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("GMM uncertainty at representative future times\nAll K=3 components · actual best-checkpoint output", fontsize=14)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94), h_pad=2.0)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"gmm_along_future.{suffix}", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_gmm_training_progress(inputs, predictions, sample_index, timestep, delta_t, output_dir):
    X = inputs["X"][sample_index]
    y = inputs["y"][sample_index]
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.5), sharex=True, sharey=True)
    all_points = [X[:, :2], y]
    for ax, (label, prediction) in zip(axes.flat, predictions.items()):
        draw_trajectory_context(ax, X, y, timestep, show_labels=False)
        boundaries = draw_real_gmm(
            ax,
            prediction["pi"][sample_index, timestep],
            prediction["mu"][sample_index, timestep],
            prediction["covariance"][sample_index, timestep],
            probability=0.68,
            annotate=True,
            legend=False,
        )
        all_points.extend(boundaries)
        ax.set_title(label)
    set_equal_limits(axes, all_points, padding=0.06)
    for ax in axes[0]:
        ax.set_xlabel("")
    fig.suptitle(
        f"GMM parameters during training at t = +{(timestep + 1) * delta_t:.1f} s\n"
        "Same sample, timestep, axis limits and 68% component ellipses",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.2)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"gmm_training_progress.{suffix}", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_gmm_notes(inputs, prediction, sample_index, timestep, delta_t, output_dir):
    pi = prediction["pi"][sample_index, timestep]
    mu = prediction["mu"][sample_index, timestep]
    sigma = prediction["sigma"][sample_index, timestep]
    rho = prediction["rho"][sample_index, timestep]
    covariance = prediction["covariance"][sample_index, timestep]
    lines = [
        "# GMM visualization notes",
        "",
        f"- Sample ID: `{str(inputs['sample_ids'][sample_index])}`",
        f"- Checkpoint: `best`, epoch {int(prediction['epoch'])}",
        f"- Future timestep: zero-based index {timestep}, one-based step {timestep + 1}",
        f"- Forecast time: `+{(timestep + 1) * delta_t:.1f} s` (`delta_t = {delta_t:.1f} s`)",
        f"- Gaussian components: `K = {len(pi)}`",
        "- Ellipse: exact 68% probability ellipse of each 2D Gaussian component; radius factor `sqrt(chi2.ppf(0.68, df=2)) = sqrt(-2 ln(1-0.68))`.",
        "- Coordinate system: repository human-ego-centric x-y coordinates, metres.",
        "- No representative predicted trajectory is drawn. Component means at one timestep are not interpreted as trajectories.",
        "",
        "## Actual saved parameters at the selected timestep",
        "",
        "| Component | pi | mu_x [m] | mu_y [m] | sigma_x [m] | sigma_y [m] | rho |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in range(len(pi)):
        lines.append(
            f"| {k + 1} | {pi[k]:.8f} | {mu[k,0]:.8f} | {mu[k,1]:.8f} | "
            f"{sigma[k,0]:.8f} | {sigma[k,1]:.8f} | {rho[k]:.8f} |"
        )
    lines.extend([
        "",
        "## Parameterization",
        "",
        "The network saves raw `mu_x`, `mu_y`, `raw_sigma_x`, `raw_sigma_y`, `raw_rho`, and `alpha` values for every component at every future timestep.",
        "",
        "```text",
        "pi      = softmax(alpha, component_dimension)",
        "sigma_x = exp(raw_sigma_x)",
        "sigma_y = exp(raw_sigma_y)",
        "rho     = tanh(raw_rho)",
        "Sigma   = [[sigma_x^2, rho*sigma_x*sigma_y],",
        "           [rho*sigma_x*sigma_y, sigma_y^2]]",
        "```",
        "",
        "## Actual covariance matrices",
        "",
    ])
    for k in range(len(pi)):
        lines.extend([
            f"Component {k + 1}:",
            "",
            "```text",
            np.array2string(covariance[k], precision=8, suppress_small=False),
            "```",
            "",
        ])
    lines.extend([
        "## Distribution represented",
        "",
        "At the selected future timestep, the MDN defines:",
        "",
        "```text",
        "p(y_t | observed trajectory) = sum_{k=1..K} pi_k(t) N(y_t | mu_k(t), Sigma_k(t))",
        "```",
        "",
        "The implementation defines a separate bivariate mixture at each of the 48 future timesteps. It does not establish one joint mixture distribution over the complete future trajectory.",
    ])
    (output_dir / "gmm_visualization_notes.md").write_text("\n".join(lines) + "\n")


def add_component_ellipses(ax, pi, mu, covariance, probability=0.68, annotate=True):
    # GMM component labels are rank labels for this panel. Mixture components do
    # not have a persistent identity across independently saved checkpoints.
    order = np.argsort(pi)[::-1]
    for rank, component in enumerate(order):
        color = COLORS["component"][rank % len(COLORS["component"])]
        alpha = float(pi[component])
        ax.add_patch(
            ellipse_patch(
                mu[component], covariance[component], probability,
                facecolor=color, edgecolor=color, alpha=0.12 + 0.22 * alpha,
                linewidth=1.8,
            )
        )
        ax.scatter(*mu[component], color=color, marker=["o", "s", "^"][rank], s=34, zorder=4)
        if annotate:
            ax.annotate(
                f"π={alpha:.2f}", mu[component], xytext=(4, 4 + rank * 9),
                textcoords="offset points", fontsize=7, color=color,
            )


def plot_context(ax, X, y, prediction=None, title=None):
    ax.plot(X[:, 0], X[:, 1], "-o", color=COLORS["past"], markersize=2.5, linewidth=1.7, label="Observed")
    ax.plot(y[:, 0], y[:, 1], "-o", color=COLORS["truth"], markersize=2.2, linewidth=1.6, label="Ground truth")
    if prediction is not None:
        ax.plot(prediction[:, 0], prediction[:, 1], "-o", color=COLORS["prediction"], markersize=2.2, linewidth=1.6, label="Mixture mean")
    ax.scatter(X[-1, 0], X[-1, 1], s=45, color=COLORS["past"], edgecolor="white", linewidth=0.8, zorder=5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22)
    if title:
        ax.set_title(title)


def save_figure(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_loss(history, output_dir):
    epochs = numeric(history, "epoch")
    train = numeric(history, "train_nll")
    validation = numeric(history, "validation_nll")
    best_index = int(np.argmin(validation))
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.plot(epochs, train, color=COLORS["train"], linewidth=1.2, label="Train NLL")
    ax.plot(epochs, validation, color=COLORS["validation"], linewidth=1.0, alpha=0.85, label="Validation NLL")
    ax.scatter(epochs[best_index], validation[best_index], color=COLORS["validation"], s=50, zorder=4)
    ax.annotate(
        f"Best: epoch {int(epochs[best_index])}\nNLL={validation[best_index]:.4f}",
        (epochs[best_index], validation[best_index]), xytext=(12, 22),
        textcoords="offset points", arrowprops={"arrowstyle": "->", "color": COLORS["validation"]},
    )
    ax.set(title="Training and validation negative log-likelihood", xlabel="Epoch", ylabel="NLL")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    save_figure(fig, output_dir / "01_training_validation_nll.png")


def figure_metrics(metric_rows, output_dir):
    epochs = numeric(metric_rows, "epoch")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True)
    panels = [
        (axes[0, 0], [("ravg_percent", "Ravg"), ("rmin_percent", "Rmin")], "Reliability", "Score [%]"),
        (axes[0, 1], [("minade20_m", "minADE20"), ("minfde20_m", "minFDE20")], "Best-of-20 displacement", "Error [m]"),
        (axes[1, 0], [("s68_m2_per_s", "S68"), ("s95_m2_per_s", "S95")], "Sharpness (corrected area)", "Sharpness [m²/s]"),
        (axes[1, 1], [("asaee_m_per_s", "ASAEE")], "Average sampled argmax error", "ASAEE [m/s]"),
    ]
    styles = [COLORS["train"], COLORS["validation"]]
    for ax, series, title, ylabel in panels:
        for idx, (key, label) in enumerate(series):
            values = numeric(metric_rows, key)
            ax.plot(epochs, values, "-o", color=styles[idx], linewidth=1.6, markersize=4, label=label)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    fig.suptitle("Official validation metrics during training", fontsize=14)
    fig.tight_layout()
    save_figure(fig, output_dir / "02_official_metrics_by_epoch.png")


def choose_cases(inputs, prediction):
    predicted = mixture_mean(prediction)
    truth = inputs["y"]
    ade = np.linalg.norm(predicted - truth, axis=-1).mean(axis=-1)
    covariance = total_covariance(prediction["pi"], prediction["mu"], prediction["covariance"])
    uncertainty = np.trace(covariance, axis1=-2, axis2=-1).mean(axis=-1)
    # Avoid using a nearly stationary pedestrian as the visually trivial good
    # case. The threshold is only a presentation filter; ADE still ranks cases.
    future_displacement = np.linalg.norm(truth[:, -1] - truth[:, 0], axis=-1)
    moving = np.flatnonzero(future_displacement >= 0.25)
    good = int(moving[np.argmin(ade[moving])]) if len(moving) else int(np.argmin(ade))
    failure = int(np.argmax(ade))
    candidates = [idx for idx in range(len(ade)) if idx not in {good, failure}]
    uncertain = max(candidates, key=lambda idx: uncertainty[idx]) if candidates else int(np.argmax(uncertainty))
    return {"good": good, "uncertain": uncertain, "failure": failure}, ade, uncertainty


def figure_fixed_samples(inputs, prediction, output_dir):
    predicted = mixture_mean(prediction)
    sample_ids = inputs["sample_ids"].astype(str)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for idx, ax in enumerate(axes.flat):
        error = np.linalg.norm(predicted[idx] - inputs["y"][idx], axis=-1).mean()
        plot_context(ax, inputs["X"][idx], inputs["y"][idx], predicted[idx], f"{short_sample_id(sample_ids[idx])} | ADE={error:.2f} m")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Eight fixed validation samples — best checkpoint", fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96), h_pad=2.8)
    save_figure(fig, output_dir / "03_fixed_samples_best_trajectories.png")


def figure_gmm_evolution(inputs, predictions, sample_index, timestep, output_dir):
    labels = list(predictions)
    X = inputs["X"][sample_index]
    y = inputs["y"][sample_index]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.2), sharex=True, sharey=True)
    for ax, label in zip(axes.flat, labels):
        pred = predictions[label]
        pi = pred["pi"][sample_index, timestep]
        mu = pred["mu"][sample_index, timestep]
        covariance = pred["covariance"][sample_index, timestep]
        ax.plot(X[:, 0], X[:, 1], color=COLORS["past"], linewidth=1.2, alpha=0.7)
        ax.plot(y[:, 0], y[:, 1], color=COLORS["truth"], linewidth=1.2, alpha=0.7)
        ax.scatter(*y[timestep], color=COLORS["truth"], marker="*", s=80, zorder=6, label="Ground truth")
        add_component_ellipses(ax, pi, mu, covariance, probability=0.68)
        expected = np.sum(pi[:, None] * mu, axis=0)
        ax.scatter(*expected, color=COLORS["expected"], marker="X", s=45, zorder=6)
        ax.set_title(label)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("y [m]")
    for ax in axes[-1]:
        ax.set_xlabel("x [m]")
    sample_id = str(inputs["sample_ids"][sample_index])
    fig.suptitle(f"GMM evolution — {sample_id}, future t+{(timestep + 1) * 0.1:.1f}s (68% component ellipses)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output_dir / "04_gmm_components_across_epochs.png")


def figure_uncertainty_horizons(inputs, prediction, sample_index, horizons, output_dir):
    X = inputs["X"][sample_index]
    y = inputs["y"][sample_index]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), sharex=True, sharey=True)
    for ax, timestep in zip(axes.flat, horizons):
        pi = prediction["pi"][sample_index, timestep]
        mu = prediction["mu"][sample_index, timestep]
        covariance = prediction["covariance"][sample_index, timestep]
        ax.plot(X[:, 0], X[:, 1], color=COLORS["past"], linewidth=1.2, alpha=0.65)
        ax.plot(y[:, 0], y[:, 1], color=COLORS["truth"], linewidth=1.2, alpha=0.65)
        ax.scatter(*y[timestep], color=COLORS["truth"], marker="*", s=75, zorder=6)
        add_component_ellipses(ax, pi, mu, covariance, probability=0.68, annotate=False)
        ax.set_title(f"t+{(timestep + 1) * 0.1:.1f}s")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("y [m]")
    for ax in axes[-1]:
        ax.set_xlabel("x [m]")
    sample_id = str(inputs["sample_ids"][sample_index])
    fig.suptitle(f"Predicted uncertainty over the forecast horizon — {sample_id}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output_dir / "05_uncertainty_across_horizons.png")


def figure_learning_over_time(inputs, predictions, sample_index, output_dir):
    labels = list(predictions)
    X = inputs["X"][sample_index]
    y = inputs["y"][sample_index]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.0), sharex=True, sharey=True)
    for ax, label in zip(axes.flat, labels):
        predicted = mixture_mean(predictions[label])[sample_index]
        error = np.linalg.norm(predicted - y, axis=-1).mean()
        plot_context(ax, X, y, predicted, f"{label} | ADE={error:.2f} m")
    for ax in axes[0]:
        ax.set_xlabel("")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False)
    sample_id = str(inputs["sample_ids"][sample_index])
    fig.suptitle(f"The same fixed sample across training — {sample_id}", fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96), h_pad=2.8)
    save_figure(fig, output_dir / "06_same_sample_across_epochs.png")


def figure_cases(inputs, prediction, cases, ade, uncertainty, output_dir):
    predicted = mixture_mean(prediction)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    titles = {"good": "Good case", "uncertain": "Difficult / uncertain case", "failure": "Failure case"}
    for ax, (case, idx) in zip(axes, cases.items()):
        plot_context(
            ax, inputs["X"][idx], inputs["y"][idx], predicted[idx],
            f"{titles[case]}\n{short_sample_id(inputs['sample_ids'][idx])}\nADE={ade[idx]:.2f} m | tr(Cov)={uncertainty[idx]:.2f} m²",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Representative fixed-sample cases — best checkpoint", fontsize=14)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    save_figure(fig, output_dir / "07_good_uncertain_failure_cases.png")


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    default_run = project_root / "results/trained_models/base_mdn/imptc/default_peds_imptc/runs/imptc_baseline_seed2024"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=default_run, help="Instrumented run directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Destination directory; defaults to <run-dir>/figures")
    parser.add_argument("--sample-id", default=None, help="Fixed sample ID for evolution figures; defaults to the most uncertain sample")
    parser.add_argument("--future-step", type=int, default=24, choices=range(1, 49), metavar="1..48", help="One-based future timestep for GMM figures; default 24 = +2.4 s")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (args.output_dir or run_dir / "figures").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    history = read_csv(run_dir / "history.csv")
    metric_rows = read_csv(run_dir / "metrics/history.csv")
    resolved_config = json.loads((run_dir / "resolved_config.json").read_text())
    delta_t = float(resolved_config["model_params"]["delta_t"])
    inputs = load_npz(run_dir / "fixed_samples/inputs.npz")
    prediction_dir = run_dir / "fixed_samples/predictions"
    best = load_npz(prediction_dir / "best.npz")

    cases, ade, uncertainty = choose_cases(inputs, best)
    sample_ids = inputs["sample_ids"].astype(str).tolist()
    if args.sample_id is None:
        sample_index = cases["uncertain"]
    else:
        if args.sample_id not in sample_ids:
            raise ValueError(f"Unknown sample ID {args.sample_id!r}; choose one of {sample_ids}")
        sample_index = sample_ids.index(args.sample_id)

    checkpoint_files = [
        ("Epoch 1", "epoch_0001.npz"),
        ("Epoch 100", "epoch_0100.npz"),
        ("Epoch 500", "epoch_0500.npz"),
        ("Epoch 1000", "epoch_1000.npz"),
        ("Epoch 1500", "epoch_1500.npz"),
        ("Epoch 2000", "epoch_2000.npz"),
        ("Epoch 2500", "epoch_2500.npz"),
        (f"Best (epoch {int(best['epoch'])})", "best.npz"),
    ]
    evolution = {label: load_npz(prediction_dir / filename) for label, filename in checkpoint_files}
    progress_files = [
        ("Epoch 1", "epoch_0001.npz"),
        ("Epoch 100", "epoch_0100.npz"),
        ("Epoch 300", "epoch_0300.npz"),
        ("Epoch 500", "epoch_0500.npz"),
        ("Epoch 1000", "epoch_1000.npz"),
        ("Epoch 1500", "epoch_1500.npz"),
        (f"Best (epoch {int(best['epoch'])})", "best.npz"),
        ("Final (epoch 2500)", "epoch_2500.npz"),
    ]
    progress_predictions = {label: load_npz(prediction_dir / filename) for label, filename in progress_files}

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "savefig.pad_inches": 0.08,
    })

    figure_loss(history, output_dir)
    figure_metrics(metric_rows, output_dir)
    figure_fixed_samples(inputs, best, output_dir)
    figure_gmm_evolution(inputs, evolution, sample_index, args.future_step - 1, output_dir)
    figure_uncertainty_horizons(inputs, best, sample_index, [7, 15, 23, 31, 39, 47], output_dir)
    figure_learning_over_time(inputs, evolution, sample_index, output_dir)
    figure_cases(inputs, best, cases, ade, uncertainty, output_dir)
    selected_timestep = args.future_step - 1
    representative_seconds = [1.0, 2.0, 3.0, 4.0]
    representative_timesteps = [int(round(seconds / delta_t)) - 1 for seconds in representative_seconds]
    if any(step < 0 or step >= int(resolved_config["model_params"]["forecast_horizon"]) for step in representative_timesteps):
        raise ValueError("Representative future times fall outside the configured forecast horizon")
    figure_gmm_single_timestep(inputs, best, sample_index, selected_timestep, delta_t, output_dir)
    figure_gmm_along_future(inputs, best, sample_index, representative_timesteps, delta_t, output_dir)
    figure_gmm_training_progress(inputs, progress_predictions, sample_index, selected_timestep, delta_t, output_dir)
    write_gmm_notes(inputs, best, sample_index, selected_timestep, delta_t, output_dir)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_for_case_analysis": f"best (epoch {int(best['epoch'])})",
        "evolution_sample_id": sample_ids[sample_index],
        "gmm_future_step": args.future_step,
        "gmm_future_seconds": round(args.future_step * delta_t, 1),
        "component_label_policy": "sorted by descending pi independently in each panel",
        "case_selection_pool": "8 fixed validation samples",
        "cases": {
            case: {
                "sample_id": sample_ids[idx],
                "mixture_mean_ade_m": float(ade[idx]),
                "mean_predictive_covariance_trace_m2": float(uncertainty[idx]),
            }
            for case, idx in cases.items()
        },
        "figures": sorted(path.name for path in output_dir.glob("*.png")),
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
