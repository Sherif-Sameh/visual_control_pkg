from __future__ import annotations
from typing import TYPE_CHECKING

from .base import Logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ROSWrapperLogger(Logger):
    """ROS wrapper for loggers.

    Wraps the given logger to allow multiple `log()` calls without incrementing the logger's
    internal counter. This is needed as with ROS-based logging, calls to `log()` have to be made
    multiple times with different metrics due to the asynchronous nature of ROS.

    Args:
        n_hold: Interval for incrementing the wrapped logger's counter.
        logger: Logger instance to wrap.
    """

    def __init__(self, *, n_hold: int, logger: Logger):
        super().__init__(n_log=1, n_flush=1, filter=None)
        self._n_hold = n_hold
        self._logger = logger

    def flush(self) -> None:
        """Flushes the wrapped logger."""
        self._logger.flush()

    def log(self, step: float, metrics: dict[str, NDArray]) -> None:
        """Logs all tracked metrics at the given step.

        Will hold the wrapped logger's counter from incrementing for n_hold steps.

        Args:
            step: Current step for logging metrics. Gets rounded to 3 digits.
            metrics: Dictionary mapping metric names to their ndarray values.
        """
        self._count += 1
        self._logger.log(step, metrics)
        if self._count % self._n_hold == 0:
            self._count = 0
        else:
            self._logger._count -= 1

    def restart(self) -> None:
        """Restarts the wrapped logger."""
        self._logger.flush()
