from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ROSWrapperLogger(Logger):
    """ROS wrapper for loggers.

    Wraps the given logger to allow multiple `log()` calls without changing the value of the
    logger's internal counter. During these calls, metrics are stored within a temporary log,
    while also combining metrics with the same step together in order to prevent them from
    overriding each other. After, these calls, the wrapper will empty its temporary log into the
    logger's log records and increment the logger's counter once.

    This approach is needed as with ROS-based logging, calls to `log()` have to be made multiple
    times with different metrics due to the asynchronous nature of ROS.

    Args:
        n_hold: Interval for accumulating metrics before sending them into to the wrapped logger.
            Alternatively, the number of `log()` calls per a single `log()` call of the wrapped
            logger.
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

        Args:
            step: Current step for logging metrics. Gets rounded to 3 digits.
            metrics: Dictionary mapping metric names to their ndarray values.
        """
        self._count += 1
        self._log[round(step, 3)] |= metrics
        if self._count % self._n_hold == 0:
            for step, metrics_combined in self._log.items():
                logger_count_pre = self._logger._count
                self._logger.log(step, metrics_combined)
                self._logger._count = logger_count_pre
            self._count = 0
            self._log.clear()
            self._logger._count += 1

    def restart(self) -> None:
        """Restarts the wrapped logger."""
        self._logger.restart()

    def close(self) -> None:
        """Close the wrapped logger."""
        self._logger.close()
