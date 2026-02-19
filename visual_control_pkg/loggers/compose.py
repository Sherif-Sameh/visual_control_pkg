from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

from .base import Logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ComposeLogger(Logger):
    """Composes mutliple loggers into a single logger.

    Args:
        loggers: List of loggers to compose.
    """

    def __init__(self, *, loggers: list[Logger]):
        super().__init__(n_log=1, n_flush=1, filter=None)
        self._loggers = loggers

    def flush(self) -> None:
        """Does nothing.

        Should not be called as flushing is managed by each logger individually according to its
        configuration
        """
        warn(
            "flush() method of ComposeLogger should not be called. Flushing is managed by each"
            " logger individually according to its own configuration."
        )

    def log(self, step: float, metrics: dict[str, NDArray]) -> None:
        """Logs all tracked metrics for all composed loggers at the given step.

        **Note**: Individual logs are flushed automatically when the set interval is reached.

        Args:
            step: Current step for logging metrics. Gets rounded to 3 digits.
            metrics: Dictionary mapping metric names to their ndarray values.
        """
        for logger in self._loggers:
            logger.log(step, metrics)

    def restart(self) -> None:
        """Restart all composed loggers."""
        for logger in self._loggers:
            logger.restart()
