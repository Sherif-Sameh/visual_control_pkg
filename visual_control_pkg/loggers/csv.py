from datetime import datetime
from pathlib import Path

import pandas as pd

from .base import Logger


class CSVLogger(Logger):
    """CSV-based logger that saves metrics to a CSV file using Pandas.

    Args:
        n_log: Interval for logging in steps. Defaults to 1 (i.e. log everything).
        n_flush: Interval for flushing logger in steps. Defaults to 1 (i.e. flush instantly).
        filter: Optional filter strings to filter metrics for logging.
        dir: Path to the directory where logs will be saved.
    """

    def __init__(
        self,
        *,
        n_log: int = 1,
        n_flush: int = 1,
        filter: str | list[str] | None = None,
        dir: str | Path,
    ):
        super().__init__(n_log=n_log, n_flush=n_flush, filter=filter)
        self._dir = Path(dir) if isinstance(dir, str) else dir
        self._dir = self._dir.resolve()
        self._dir.mkdir(parents=True, exist_ok=True)
        self.restart()

    def flush(self) -> None:
        """Saves stored logs to a CSV file."""
        if not self._log:
            return
        # Convert nested dict structure to per-row records
        records = []
        for step, metrics in self._log.items():
            for name, value in metrics.items():
                records.append({"name": name, "step": step, "value": value.tolist()})
        df = pd.DataFrame(records, copy=False)
        df.to_csv(self._path, index=False, mode="a", header=not self._path.exists())
        self._log.clear()

    def restart(self) -> None:
        """Restart the logger without reinitialization.

        Flushes all existing logs before designating a new CSV file for logging. The filename is
        set using the current data-time string.
        """
        super().restart()
        datetime_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        self._path = self._dir / f"{datetime_str}.csv"
