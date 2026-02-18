from pathlib import Path

import pandas as pd

from .base import Logger


class CSVLogger(Logger):
    """CSV-based logger that saves metrics to a CSV file using Pandas.

    **Warning**: If the given CSV file path already exists, it will be overriden.

    Args:
        n_log: Interval for logging in steps. Defaults to 1 (i.e. log everything).
        n_flush: Interval for flushing logger in steps. Defaults to 1 (i.e. flush instantly).
        path: Path to the CSV file where logs will be saved.
        filter: Optional filter string to filter metrics for logging.
    """

    def __init__(
        self,
        *,
        n_log: int = 1,
        n_flush: int = 1,
        filter: str | None = None,
        path: str | Path,
    ):
        super().__init__(n_log=n_log, n_flush=n_flush, filter=filter)
        self._path = Path(path) if isinstance(path, str) else path
        assert self._path.suffix == ".csv", (
            f"CSVLogger requires a .csv file extension. Got {self._path.suffix} instead."
        )
        self._path = self._path.resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            self._path.unlink()  # Remove old logs

    def flush(self) -> None:
        """Saves stored logs to a CSV file."""
        if not self._log:
            return
        # Separate the individual entries of arrays
        for step in self._log.keys():
            step_exp = {}
            for name, value in self._log[step].items():
                step_exp.update({f"{name}_{i}": v for (i, v) in enumerate(value)})
            self._log[step] = step_exp
        df = pd.DataFrame.from_dict(self._log, orient="index")
        df.to_csv(self._path, mode="a", header=not self._path.exists())
        self._log.clear()
