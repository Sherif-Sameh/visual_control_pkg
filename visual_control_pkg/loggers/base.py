from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Logger(ABC):
    """Base abstract class for loggers.

    Args:
        n_log: Interval for logging in steps. Defaults to 1 (i.e. log everything).
        n_flush: Interval for flushing logger in steps. Defaults to 1 (i.e. flush instantly).
        filter: Optional filter string to filter metrics for logging.
    """

    def __init__(self, *, n_log: int = 1, n_flush: int = 1, filter: str | None = None):
        assert n_log <= n_flush, (
            "Logging interval must be less than or equal to interval for flushing logger."
        )
        assert n_flush % n_log == 0, (
            "Flushing interval must be a multiple of the logging interval."
        )
        self._n_log = n_log
        self._n_flush = n_flush
        self._filter = filter
        self._log = defaultdict(dict)
        self._count = 0

    @abstractmethod
    def flush(self) -> None:
        """Flushes stored logs to the logging output destination."""
        pass

    def log(self, step: float, metrics: dict[str, NDArray]) -> None:
        """Logs all tracked metrics at the given step.

        **Note**: Logs are flushed automatically when the set interval is reached.

        Args:
            step: Current step for logging metrics. Gets rounded to 3 digits.
            metrics: Dictionary mapping metric names to their ndarray values.
        """
        self._count += 1
        if self._count % self._n_log == 0:
            self._log[round(step, 3)] = self.filter(metrics)
        if self._count % self._n_flush == 0:
            self.flush()
            self._count = 0

    def filter(self, metrics: dict[str, NDArray]) -> dict[str, NDArray]:
        """Filters the provided metrics based on the logger's filter string.

        Args:
            metrics: Dictionary mapping metric names to their ndarray values.

        Returns:
            Filtered dictionary of metrics.
        """
        if self._filter is None:
            return metrics
        return {k: v for k, v in metrics.items() if self._filter in k}

    def restart(self) -> None:
        """Restart the logger without reinitialization.

        Flushes all existing logs.
        """
        self.flush()
