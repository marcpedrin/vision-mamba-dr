"""
Report figures, run after training and evaluation:

    python -m eval.plots --config configs/default.yaml

    training_curves.png, confusion_*.png, examples_*.png -> report/figures/
    Anything whose input doesnt exist yet is skipped.

"""

import argparse
import csv
import json
import os

import matplotlib

#no window, just save files (works on kaggle), has to be before pyplot
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from data.eyepacs import load_rgb_image
from data.transforms import CropToRetina, PadToSquare
from vim.config import load_config

#train always blue, val always orange
TRAIN_COLOUR = "#2a78d6"
VAL_COLOUR = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
#light -> dark blue for the confusion matrix
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"]
)

GRADE_LABELS = ["0", "1", "2", "3", "4"]
GRADE_NAMES = ["none", "mild", "moderate", "severe", "prolif."]


def _style_axis(ax):
    """No top/right border, faint grid, whole number epochs."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK_SECONDARY)
    ax.spines["bottom"].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def read_log(log_csv):
    """log.csv -> {column: [floats]}"""
    columns = {}
    with open(log_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key, value in row.items():
                columns.setdefault(key, []).append(float(value))
    return columns


def plot_training_curves(log_csv, out_path):
    """Loss, accuracy and val QWK per epoch, 3 panels."""
    log = read_log(log_csv)
    epochs = log["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    #loss
    ax = axes[0]
    ax.plot(epochs, log["train_loss"], color=TRAIN_COLOUR, linewidth=2, label="train")
    ax.plot(epochs, log["val_loss"], color=VAL_COLOUR, linewidth=2, label="validation")
    ax.set_title("Cross-entropy loss", color=INK)
    ax.set_xlabel("epoch", color=INK_SECONDARY)
    ax.legend(frameon=False)
    _style_axis(ax)

    #accuracy
    ax = axes[1]
    ax.plot(epochs, log["train_acc"], color=TRAIN_COLOUR, linewidth=2, label="train")
    ax.plot(epochs, log["val_acc"], color=VAL_COLOUR, linewidth=2, label="validation")
    ax.set_title("Accuracy", color=INK)
    ax.set_xlabel("epoch", color=INK_SECONDARY)
    ax.legend(frameon=False)
    _style_axis(ax)

    #val QWK with the best epoch marked
    ax = axes[2]
    ax.plot(epochs, log["val_qwk"], color=VAL_COLOUR, linewidth=2)
    best = int(np.argmax(log["val_qwk"]))
    ax.plot([epochs[best]], [log["val_qwk"][best]], marker="o", markersize=8,
            color=VAL_COLOUR, markeredgecolor="white", markeredgewidth=2)
    best_point = (epochs[best], log["val_qwk"][best])
    ax.annotate(f"best {best_point[1]:.3f}", best_point,
                textcoords="offset points", xytext=(0, 10), ha="center", color=INK)
    #headroom so the label doesnt hit the title
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.2 * (high - low))
    ax.set_title("Validation QWK", color=INK)
    ax.set_xlabel("epoch", color=INK_SECONDARY)
    _style_axis(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(matrix, title, out_path):
    """Confusion matrix, colour = share of the true grade (rows sum to 100%)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    #max(.., 1) so an empty grade doesnt divide by zero
    row_totals = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    shares = matrix / row_totals

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    image = ax.imshow(shares, cmap=BLUE_RAMP, vmin=0.0, vmax=1.0)

    #"share (count)" in each cell, white text on dark cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text_colour = "white" if shares[i, j] > 0.55 else INK
            ax.text(j, i, f"{shares[i, j]:.0%}\n({int(matrix[i, j])})",
                    ha="center", va="center", fontsize=8, color=text_colour)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(GRADE_LABELS)
    ax.set_yticklabels([f"{g} {n}" for g, n in zip(GRADE_LABELS, GRADE_NAMES)])
    ax.set_xlabel("predicted grade", color=INK_SECONDARY)
    ax.set_ylabel("true grade", color=INK_SECONDARY)
    ax.set_title(title, color=INK)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="share of true grade")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def read_predictions(pred_csv):
    """predictions csv -> [(path, true, pred)]"""
    rows = []
    with open(pred_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row["path"], int(row["true"]), int(row["pred"])))
    return rows


def plot_example_predictions(pred_csv, title, out_path, num_examples=8, seed=0):
    """Grid of images with true vs predicted, half right half wrong (seeded)."""
    rows = read_predictions(pred_csv)
    rng = np.random.RandomState(seed)

    correct = [r for r in rows if r[1] == r[2]]
    wrong = [r for r in rows if r[1] != r[2]]
    rng.shuffle(correct)
    rng.shuffle(wrong)
    n_wrong = min(num_examples // 2, len(wrong))
    n_correct = num_examples - n_wrong
    chosen = wrong[:n_wrong] + correct[:n_correct]
    if not chosen:
        print(f"No predictions in {pred_csv}; skipping {out_path}")
        return

    #same framing as the model but bigger so lesions are visible
    crop = CropToRetina()
    pad = PadToSquare()

    cols = 4
    #ceiling division
    rows_needed = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows_needed, cols, figsize=(2.6 * cols, 3.0 * rows_needed),
                             layout="constrained")
    #flat list even with one row
    axes = np.array(axes).reshape(-1)

    for ax, (path, true, pred) in zip(axes, chosen):
        image = pad(crop(load_rgb_image(path, draft_size=512))).resize((256, 256))
        ax.imshow(image)
        verdict = "correct" if true == pred else "wrong"
        ax.set_title(f"true {true}, predicted {pred}\n({verdict})", fontsize=9,
                     color=INK)
        ax.axis("off")

    #hide unused cells
    for ax in axes[len(chosen):]:
        ax.axis("off")

    fig.suptitle(title, color=INK)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    """Make every figure whose inputs exist, returns the files written."""
    parser = argparse.ArgumentParser(description="Make the report figures")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.override)

    fig_dir = cfg.eval.figures_dir
    os.makedirs(fig_dir, exist_ok=True)
    results_dir = os.path.dirname(cfg.eval.results_json) or "."
    written = []

    #training curves
    log_csv = os.path.join(cfg.train.output_dir, "log.csv")
    if os.path.exists(log_csv):
        out = os.path.join(fig_dir, "training_curves.png")
        plot_training_curves(log_csv, out)
        written.append(out)
    else:
        print(f"No training log at {log_csv}; skipping training curves")

    #confusion matrices
    if os.path.exists(cfg.eval.results_json):
        with open(cfg.eval.results_json, encoding="utf-8") as f:
            results = json.load(f)["results"]
        titles = {"eyepacs_test": "EyePACS test split", "messidor2": "Messidor-2"}
        for key, title in titles.items():
            if key in results:
                out = os.path.join(fig_dir, f"confusion_{key}.png")
                plot_confusion_matrix(results[key]["confusion_matrix"], title, out)
                written.append(out)
    else:
        print(f"No results at {cfg.eval.results_json}; skipping confusion matrices")

    #example grids
    for key, title in (("eyepacs_test", "EyePACS test split"),
                       ("messidor2", "Messidor-2")):
        pred_csv = os.path.join(results_dir, f"predictions_{key}.csv")
        if os.path.exists(pred_csv):
            out = os.path.join(fig_dir, f"examples_{key}.png")
            plot_example_predictions(pred_csv, f"Example predictions: {title}", out)
            written.append(out)

    for path in written:
        print(f"Saved {path}")
    return written


if __name__ == "__main__":
    main()
