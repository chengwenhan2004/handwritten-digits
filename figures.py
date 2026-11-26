"""从实验保存的划分与预测绘制报告用图。运行方式：python figures.py"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix


RESULTS = Path(__file__).resolve().parent / "results"
DIGITS = np.arange(10)
TEXT = "#23343C"
MUTED = "#4F6269"
TEAL = "#2E7070"
GRID = "#D6DFE0"
ORANGE = "#9F431D"


def set_style():
    """优先选择本机已安装的中文字体；缺少中文字体时改用英文标签。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    choices = ("Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Noto Sans SC", "SimHei")
    font = next((name for name in choices if name in available), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": [font, "DejaVu Sans"], "font.size": 10,
        "text.color": TEXT, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.unicode_minus": False, "savefig.facecolor": "white",
    })
    return font != "DejaVu Sans"


def read_records(filename):
    with (RESULTS / filename).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def save(figure, filename):
    figure.savefig(RESULTS / filename, dpi=240)
    plt.close(figure)


def plot_profiles(images, records, chinese):
    indices = np.array([int(row["index"]) for row in records])
    labels = np.array([int(row["true"]) for row in records])
    counts = np.bincount(labels, minlength=10)
    figure = plt.figure(figsize=(8.0, 3.15))
    grid = figure.add_gridspec(2, 10, left=.076, right=.985, bottom=.08,
                               top=.89, height_ratios=(1.55, 1), hspace=.66)
    ax = figure.add_subplot(grid[0, :])
    figure.text(.077, .935, "训练集样本数" if chinese else "Training samples",
                fontsize=12, fontweight="bold")
    ax.bar(DIGITS, counts, width=.55, color=TEAL)
    for label, count in enumerate(counts):
        ax.text(label, count + 5, str(count), ha="center", va="bottom", fontsize=10)
    ax.set(xlim=(-.5, 9.5), ylim=(0, max(counts) * 1.2),
           xticks=DIGITS, yticks=[0, 50, 100, 150])
    ax.tick_params(length=0, pad=4)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=.6)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    figure.text(.077, .355, "各类数字的平均图像" if chinese else "Mean image by class",
                fontsize=12, fontweight="bold")
    for label in DIGITS:
        tile = figure.add_subplot(grid[1, label])
        tile.imshow(images[indices[labels == label]].mean(axis=0),
                    cmap="Greys", interpolation="nearest", vmin=0, vmax=16)
        tile.set(xticks=[], yticks=[])
        tile.set_xlabel(str(label), labelpad=2)
        for spine in tile.spines.values():
            spine.set_visible(True)
            spine.set_color(GRID)
            spine.set_linewidth(.6)
    save(figure, "digit_profiles.png")


def plot_confusion(records, chinese):
    true = [int(row["true"]) for row in records]
    pred = [int(row["svm"]) for row in records]
    counts = confusion_matrix(true, pred, labels=DIGITS)
    # 只用颜色区分三种格子；每种情况的样本数由格内数字给出。
    status = np.where(counts == 0, 0, np.where(np.eye(10, dtype=bool), 1, 2))
    colors = ["#F5F7F7", "#D2E7E3", "#F8D7BF"]
    figure, ax = plt.subplots(figsize=(4.8, 3.65))
    figure.subplots_adjust(left=.15, right=.96, top=.96, bottom=.23)
    ax.imshow(status, cmap=ListedColormap(colors), interpolation="nearest",
              vmin=0, vmax=2, aspect="auto")
    for row, col in zip(*np.nonzero(counts)):
        ax.text(col, row, str(counts[row, col]), ha="center", va="center",
                fontsize=10, color=TEAL if row == col else ORANGE, fontweight="bold")
    ax.set(xticks=DIGITS, yticks=DIGITS,
           xlabel="预测数字" if chinese else "Predicted digit",
           ylabel="真实数字" if chinese else "True digit")
    ax.set_xticks(np.arange(-.5, 10), minor=True)
    ax.set_yticks(np.arange(-.5, 10), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.15)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    labels = ["识别正确", "错分", "0 个样本"] if chinese else ["Correct", "Incorrect", "Zero cases"]
    figure.legend(handles=[Patch(facecolor=colors[i], label=label)
                           for i, label in zip((1, 2, 0), labels)],
                  loc="lower center", ncol=3, frameon=False, handlelength=1.15)
    save(figure, "confusion.png")


def plot_errors(images, records, chinese):
    errors = sorted((row for row in records if row["true"] != row["svm"]),
                    key=lambda row: int(row["index"]))
    # 给出固定、可复查的样本选择顺序。误差较多时最多画前八个。
    examples = errors[:8]
    figure = plt.figure(figsize=(8., 1.75 if len(examples) <= 4 else 3.3))
    if not examples:
        figure.text(.5, .5, "测试集全部识别正确" if chinese else "All test predictions correct",
                    ha="center", va="center")
    else:
        cols = min(4, len(examples))
        rows = (len(examples) + cols - 1) // cols
        grid = figure.add_gridspec(rows, cols, left=.02, right=.98, bottom=.30 if rows==1 else .16,
                                   top=.83 if rows==1 else .91, hspace=1.17)
        for pos, example in enumerate(examples):
            idx, actual, svm, knn = (int(example[k]) for k in ("index", "true", "svm", "knn"))
            ax = figure.add_subplot(grid[pos // cols, pos % cols])
            ax.imshow(images[idx], cmap="Greys", interpolation="nearest", vmin=0, vmax=16)
            ax.set(xticks=[], yticks=[])
            ax.set_title((f"样本 {idx}" if chinese else f"Sample {idx}"), fontsize=10, pad=4)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(GRID)
                spine.set_linewidth(.7)
            ax.text(.5, -.12, (f"真实 {actual}   SVM {svm}" if chinese else f"True {actual}   SVM {svm}"),
                    transform=ax.transAxes, ha="center", va="top", color=ORANGE, fontweight="bold")
            ax.text(.5, -.37, f"kNN {knn}", transform=ax.transAxes,
                    ha="center", va="top", color=MUTED)
    save(figure, "error_examples.png")


def main():
    chinese = set_style()
    images = load_digits().images
    train = read_records("training.csv")
    test = read_records("predictions.csv")
    plot_profiles(images, train, chinese)
    plot_confusion(test, chinese)
    plot_errors(images, test, chinese)


if __name__ == "__main__":
    main()
