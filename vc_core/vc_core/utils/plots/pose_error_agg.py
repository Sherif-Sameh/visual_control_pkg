from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal

import fire
import numpy as np
import pandas as pd

MULTIPLIERS = {"m": 1, "cm": 1e2, "mm": 1e3}
METRICS = ["PE (Position)", "PE (RotVec)"]


def parse_float_list(value: str | list[float]) -> np.ndarray:
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
    else:
        parsed = value
    return np.asarray(parsed, dtype=np.float64)


def load_metric_arrays(directory: str | Path, pattern: str = "*.csv") -> dict[str, np.ndarray]:
    directory = Path(directory)
    # Discover CSV files
    csv_files = sorted(directory.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {directory}")
    # Accumulate parsed rows per metric
    metric_data: dict[str, list[np.ndarray]] = {col: [] for col in METRICS}
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        # Keep only metrics of interest
        filtered = df[df["name"].isin(METRICS)]
        for metric_name, group in filtered.groupby("name"):
            parsed_rows = group["value"].apply(parse_float_list)
            metric_data[metric_name].extend(parsed_rows.tolist())
    # Convert lists of vectors into stacked (N, D) arrays
    result = {metric: np.vstack(rows) for metric, rows in metric_data.items() if rows}
    return result


def main(
    dir: str, prec: int = 2, pu: Literal["m", "cm", "mm"] = "cm", ru: Literal["rad", "deg"] = "deg"
) -> None:
    arrays = load_metric_arrays(directory=dir)
    mean, std = {}, {}
    # Compute MAE and its standard deviation
    for k, v in arrays.items():
        # Scale each metric appropriately
        if k == "PE (Position)":
            v *= MULTIPLIERS[pu]
        elif ru == "deg":
            v = np.rad2deg(v)
        abs_v = np.abs(v)
        mean[k] = np.mean(abs_v, axis=0)
        std[k] = np.std(abs_v, axis=0)
    # Output metric values
    for m in METRICS:
        unit = pu if m == "PE (Position)" else ru
        print(f"\n{m} aggregated over {arrays[m].shape[0]} samples:")
        print(f"\tMean: {np.round(mean[m], decimals=prec)} {unit}")
        print(f"\tStd: {np.round(std[m], decimals=prec)} {unit}")


if __name__ == "__main__":
    fire.Fire(main)
