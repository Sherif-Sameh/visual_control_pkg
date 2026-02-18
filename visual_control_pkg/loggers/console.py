from dataclasses import dataclass
from functools import partial

import numpy as np

from .base import Logger


class ConsoleLogger(Logger):
    """Console-based logger that prints metrics to the console.

    Args:
        n_log: Interval for logging in steps. Defaults to 1 (i.e. log everything).
        n_flush: Interval for flushing logger in steps. Defaults to 1 (i.e. flush instantly).
        filter: Optional filter string to filter metrics for logging.
        config: Optioanl configuration for printing NumPy arrays to the console.
    """

    @dataclass
    class ArrayPrintOptions:
        """Configuration for printing NumPy arrays to the console."""

        max_line_width: int = 80
        precision: int = 4
        suppress_small: bool = False
        separator: str = "  "
        sign: str = "+"
        floatmode: str = "maxprec"

    def __init__(
        self,
        *,
        n_log: int = 1,
        n_flush: int = 1,
        filter: str | None = None,
        config: ArrayPrintOptions = ArrayPrintOptions(),
    ):
        super().__init__(n_log=n_log, n_flush=n_flush, filter=filter)
        self._arr2str = partial(
            np.array2string,
            max_line_width=config.max_line_width,
            precision=config.precision,
            suppress_small=config.suppress_small,
            separator=config.separator,
            sign=config.sign,
            floatmode=config.floatmode,
        )

    def flush(self) -> None:
        """Prints stored logs to the console."""
        for step, metrics in self._log.items():
            print(f"\nStep {step:.2f}:")
            for name, value in metrics.items():
                print(f"\t{name}: {self._arr2str(value)}")
        self._log.clear()
