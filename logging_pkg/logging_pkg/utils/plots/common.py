from __future__ import annotations

from ast import literal_eval
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

# Configuration variables
COLORS = {"pbvs": "#F1750F", "ibvs": "#D426C2"}
LABELS = {"pbvs": r"PBVS", "ibvs": r"IBVS"}
STYLES = {"linewidth": 1.5, "alpha": 0.8}
DPI = 600

if TYPE_CHECKING:
    from numpy.typing import NDArray


def read_csv_log(path: str, filters: list[str] | None = None) -> dict[str, tuple[NDArray, NDArray]]:
    """Reads a generated log from a `CSVLogger` and groups all entries of the same metric.

    For each metric within the CSV file, its entries are grouped into a tuple of 2 NumPy arrays.
    The first is a 1D array containing the `step` for each entry. The other is a 2D array
    containing the `value` of each metric whose shape is (N, D), where D is the dimenionsality of
    the logged metric.

    Args:
        path: Path to the CSV log file to read data from.
        filters: Optional filters for metrics.

    Returns:
        Dictionary whose keys are the names of the metrics and each entry is a tuple containing
            that metric's step and value entries as NumPy arrays.
    """
    path: Path = Path(path)
    assert path.exists() and path.suffix == ".csv"
    df = pd.read_csv(path)
    df.value = df.value.apply(literal_eval)
    out = {}
    for name, group in df.groupby("name"):
        if filters is not None and not any(f in name for f in filters):
            continue
        step = group["step"].to_numpy()  # (N,)
        value = np.vstack(group["value"].to_numpy())  # (N, D)
        out[name] = (step, value)
    return out


def normalize_steps(
    metrics: dict[str, tuple[NDArray, NDArray]],
) -> dict[str, tuple[NDArray, NDArray]]:
    """Normalizes a metrics' step entries such that the minimum values of all steps is zero.

    Args:
        metrics: Dictionary of metrics containing step entries to modify.

    Returns:
        Modified dictionary of metrics with normalized steps.
    """
    min_step = min([np.min(v[0]) for v in metrics.values()])
    for k, v in metrics.items():
        metrics[k] = (v[0] - min_step, v[1])
    return metrics
