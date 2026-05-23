from __future__ import annotations  # noqa: I001

from pathlib import Path
from typing import TYPE_CHECKING

import fire
import matplotlib.pyplot as plt
import numpy as np

from common import DPI, STYLES  # noqa: I001
from common import read_csv_log, normalize_steps  # noqa: I001

# Enable LaTeX rendering for text
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"""\usepackage{bm}
\usepackage{amsmath}
\usepackage{amssymb}
"""
plt.rcParams.update({"font.size": 12})

FIGSIZE = (10, 4)
PE_M_KEYS = ["PE (Position)", "PE (RotVec)"]
COLORS = {"ideal": "#F1750F", "estimate": "#D426C2", "hline": "#66706e"}
LABELS = {"ideal": r"Marker-based", "estimate": r"DR-based"}
XLIM = [0, 80]
XTICKS = np.linspace(XLIM[0], XLIM[1], 5).astype(np.int64)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def plot_position_errors(
    ctrls: list[str], all_metrics: list[dict[str, tuple[NDArray, NDArray]]]
) -> None:
    # constants
    YMAX = [5.0, 5.0, 25.0]
    YTICKS = [np.linspace(-y, y, 9).round(2) for y in YMAX]

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=DPI)
    axes = axes.flatten()

    # plot position errors
    key = PE_M_KEYS[0]
    for ctrl, metrics in zip(ctrls, all_metrics):
        if key not in metrics:
            continue
        steps, values = metrics[key]
        values *= 1e2  # convert to cm
        for i, ax in enumerate(axes):
            ax.plot(
                steps,
                values[:, i],
                label=LABELS[ctrl] if i == 0 else None,
                linewidth=STYLES["linewidth"],
                alpha=STYLES["alpha"],
                color=COLORS[ctrl],
            )
            ax.axhline(
                0.5,
                linestyle="--",
                linewidth=STYLES["linewidth"],
                alpha=0.25,
                color=COLORS["hline"],
            )
            ax.axhline(
                -0.5,
                linestyle="--",
                linewidth=STYLES["linewidth"],
                alpha=0.25,
                color=COLORS["hline"],
            )

    # Configure plot
    fig.supxlabel(r"Time [sec]")
    fig.supylabel(r"Position Error [cm]")
    axes[0].legend(loc="lower center")
    for i, (ax, ax_l) in enumerate(zip(axes, ["X", "Y", "Z"])):
        ax.set_title(f"{ax_l}-Axis")
        ax.grid(True)
        ax.set_xlim(XLIM)
        ax.set_ylim([-YMAX[i], YMAX[i]])
        ax.set_xticks(XTICKS)
        ax.set_yticks(YTICKS[i])
        ax.get_yaxis().offsetText.set_visible(False)  # disable automatic offset
    plt.tight_layout()

    # Save plot
    save_path = Path(__file__).parent / "figures"
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / "pe_position_eye.png", dpi=DPI)
    plt.show()


def plot_rotation_errors(
    ctrls: list[str], all_metrics: list[dict[str, tuple[NDArray, NDArray]]]
) -> None:
    # constants
    YMAX = [5.0, 5.0, 5.0]
    YTICKS = [np.linspace(-y, y, 9).round(2) for y in YMAX]

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=DPI)
    axes = axes.flatten()

    # plot rotation errors
    key = PE_M_KEYS[1]
    for ctrl, metrics in zip(ctrls, all_metrics):
        if key not in metrics:
            continue
        steps, values = metrics[key]
        values = np.rad2deg(values)  # convert to degrees
        for i, ax in enumerate(axes):
            ax.plot(
                steps,
                values[:, i],
                label=LABELS[ctrl] if i == 0 else None,
                linewidth=STYLES["linewidth"],
                alpha=STYLES["alpha"],
                color=COLORS[ctrl],
            )
            ax.axhline(
                1.0,
                linestyle="--",
                linewidth=STYLES["linewidth"],
                alpha=0.25,
                color=COLORS["hline"],
            )
            ax.axhline(
                -1.0,
                linestyle="--",
                linewidth=STYLES["linewidth"],
                alpha=0.25,
                color=COLORS["hline"],
            )

    # Configure plot
    fig.supxlabel(r"Time [sec]")
    fig.supylabel(r"Rotation Error [deg]")
    axes[0].legend(loc="lower center")
    for i, (ax, ax_l) in enumerate(zip(axes, ["X", "Y", "Z"])):
        ax.set_title(f"{ax_l}-Axis")
        ax.grid(True)
        ax.set_xlim(XLIM)
        ax.set_ylim([-YMAX[i], YMAX[i]])
        ax.set_xticks(XTICKS)
        ax.set_yticks(YTICKS[i])
        ax.get_yaxis().offsetText.set_visible(False)  # disable automatic offset
    plt.tight_layout()

    # Save plot
    save_path = Path(__file__).parent / "figures"
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / "pe_rotation_eye.png", dpi=DPI)
    plt.show()


def main(p_id: str, p_est: str):
    paths = [p_id, p_est]
    ctrls = ["ideal", "estimate"]
    all_readings = []
    for path in paths:
        readings = read_csv_log(path, filters=["PE"])
        readings = normalize_steps(readings)
        all_readings.append(readings)
    plot_position_errors(ctrls, all_readings)
    plot_rotation_errors(ctrls, all_readings)


if __name__ == "__main__":
    fire.Fire(main)
