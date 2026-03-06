from __future__ import annotations  # noqa: I001

from pathlib import Path
from typing import TYPE_CHECKING

import fire
import matplotlib.pyplot as plt
import numpy as np

from common import COLORS, DPI, LABELS, STYLES  # noqa: I001
from common import read_csv_log, normalize_steps  # noqa: I001

# Enable LaTeX rendering for text
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"""\usepackage{bm}
\usepackage{amsmath}
\usepackage{amssymb}
"""
plt.rcParams.update({"font.size": 12})

FIGSIZE = (10, 3)
PE_M_KEYS = ["PE (Position)", "PE (RotVec)"]

if TYPE_CHECKING:
    from numpy.typing import NDArray


def plot_position_errors(
    ctrls: list[str], all_metrics: list[dict[str, tuple[NDArray, NDArray]]]
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=DPI)
    axes = axes.flatten()

    # plot position errors
    key = PE_M_KEYS[0]
    max_abs_v = -np.inf
    for ctrl, metrics in zip(ctrls, all_metrics):
        if key not in metrics:
            continue
        steps, values = metrics[key]
        for i, ax in enumerate(axes):
            ax.plot(
                steps,
                values[:, i],
                label=LABELS[ctrl] if i == 0 else None,
                linewidth=STYLES["linewidth"],
                alpha=STYLES["alpha"],
                color=COLORS[ctrl],
            )
        max_abs_v = max(max_abs_v, np.max(np.abs(values)))
    min_v, max_v = -max_abs_v, max_abs_v

    # Configure plot
    fig.supxlabel(r"Time (sec)")
    fig.supylabel(r"Position Error (m)")
    axes[0].legend()
    for ax, ax_l in zip(axes, ["X", "Y", "Z"]):
        ax.set_title(f"{ax_l}-Axis")
        ax.grid(True)
        ax.set_xlim([0, 60])
        ax.set_ylim([round(min_v - 0.1, 2), round(max_v + 0.1, 2)])
        ax.set_xticks(np.linspace(0, 60, 6).astype(np.int64))
        ax.set_yticks(np.round(np.linspace(min_v - 0.1, max_v + 0.1, 7), 2))
        ax.get_yaxis().offsetText.set_visible(False)  # disable automatic offset
    plt.tight_layout()

    # Save plot
    save_path = Path(__file__).parent / "figures"
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / "pe_position.png", dpi=DPI)
    plt.show()


def plot_rotation_errors(
    ctrls: list[str], all_metrics: list[dict[str, tuple[NDArray, NDArray]]]
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=DPI)
    axes = axes.flatten()

    # plot rotation errors
    key = PE_M_KEYS[1]
    max_abs_v = -np.inf
    for ctrl, metrics in zip(ctrls, all_metrics):
        if key not in metrics:
            continue
        steps, values = metrics[key]
        for i, ax in enumerate(axes):
            ax.plot(
                steps,
                values[:, i],
                label=LABELS[ctrl] if i == 0 else None,
                linewidth=STYLES["linewidth"],
                alpha=STYLES["alpha"],
                color=COLORS[ctrl],
            )
        max_abs_v = max(max_abs_v, np.max(np.abs(values)))
    min_v, max_v = -max_abs_v, max_abs_v

    # Configure plot
    fig.supxlabel(r"Time (sec)")
    fig.supylabel(r"Rotation Error (rad)")
    axes[0].legend()
    for ax, ax_l in zip(axes, ["X", "Y", "Z"]):
        ax.set_title(f"{ax_l}-Axis")
        ax.grid(True)
        ax.set_xlim([0, 60])
        ax.set_ylim([round(min_v - 0.1, 2), round(max_v + 0.1, 2)])
        ax.set_xticks(np.linspace(0, 60, 6).astype(np.int64))
        ax.set_yticks(np.round(np.linspace(min_v - 0.1, max_v + 0.1, 7), 2))
        ax.get_yaxis().offsetText.set_visible(False)  # disable automatic offset
    plt.tight_layout()

    # Save plot
    save_path = Path(__file__).parent / "figures"
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / "pe_rotation.png", dpi=DPI)
    plt.show()


def main(paths: list[str], ctrls: list[str]):
    all_readings = []
    for path in paths:
        readings = read_csv_log(path, filters=["PE"])
        readings = normalize_steps(readings)
        all_readings.append(readings)
    plot_position_errors(ctrls, all_readings)
    plot_rotation_errors(ctrls, all_readings)


if __name__ == "__main__":
    fire.Fire(main)
