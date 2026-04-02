from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from .base import Logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MemoryLogger(Logger):
    """Memory-based logger that stores logs until they're manually flushed.

    Args:
        n_log: Interval for logging in steps. Defaults to 1 (i.e. log everything).
        filter: Optional filter strings to filter metrics for logging.
    """

    def __init__(self, *, n_log: int = 1, filter: str | list[str] | None = None):
        n_flush = (n := 2**32 - 1) - n % n_log
        super().__init__(n_log=n_log, n_flush=n_flush, filter=filter)

    def flush(self) -> dict[float, dict[str, NDArray]]:
        """Returns stored logs unmodifed and clears logs."""
        if not self._log:
            return {}
        log = deepcopy(self._log)
        self._log.clear()
        return log
