from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import wandb

from .base import Logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


class WandBLogger(Logger):
    """WandB-based logger that saves metrics to a WandB project.

    Args:
        n_log: Interval for logging in steps. Defaults to 1 (i.e. log everything).
        n_flush: Interval for flushing logger in steps. Defaults to 1 (i.e. flush instantly).
        filter: Optional filter strings to filter metrics for logging.
        config: WandB configuration.
    """

    @dataclass
    class WandBConfig:
        entity: str | None = None
        project: str | None = None
        group: str | None = None
        dir: str | None = None
        config: dict[str, Any] = field(default_factory=lambda: {})

    def __init__(
        self,
        *,
        n_log: int = 1,
        n_flush: int = 1,
        filter: str | list[str] | None = None,
        config: WandBConfig = WandBConfig(),
    ):
        super().__init__(n_log=n_log, n_flush=n_flush, filter=filter)
        self._config = config

        # Find WandB API environment variable
        assert "WANDB_API_KEY" in os.environ, "WANDB_API_KEY environment variable is not set."
        wandb_api_key = os.environ["WANDB_API_KEY"]
        wandb_api_key = str(wandb_api_key) if not isinstance(wandb_api_key, str) else wandb_api_key

        # Initialize WandB run
        wandb.login(key=wandb_api_key)
        self.restart()

    def flush(self) -> None:
        """Saves stored logs to WandB project and run."""
        if not self._log:
            return

        # Log existing at each step
        for step in self._log.keys():
            self._run.log(self._log[step])
        self._log.clear()

    def log(self, step: float, metrics: dict[str, NDArray]) -> None:
        """Logs all tracked metrics at the given step.

        **Note**: Logs are flushed automatically when the set interval is reached.

        Entries of NumPy arrays are separated into individual metrics since WandB's `log()` method
        does not accept NumPy array values.

        Args:
            step: Current step for logging metrics. Gets rounded to 3 digits.
            metrics: Dictionary mapping metric names to their ndarray values.
        """
        self._count += 1
        if self._count % self._n_log == 0:
            filtered_metrics = self.filter(metrics)
            if filtered_metrics:
                filtered_metrics = self._split_metrics(filtered_metrics)
                filtered_metrics["logger_step"] = step
                self._log[round(step, 3)] = filtered_metrics
        if self._count % self._n_flush == 0:
            self.flush()
            self._count = 0

    def restart(self) -> None:
        """Restart the logger without reinitialization.

        Flushes all existing logs and finishes current WandB run before starting a new run.
        """
        self.flush()
        self._run = wandb.init(
            entity=self._config.entity,
            project=self._config.project,
            group=self._config.group,
            dir=self._config.dir,
            config=self._config.config,
            reinit="finish_previous",
        )

    def close(self) -> None:
        """Close the logger and finish active WandB run cleanly."""
        super().close()
        if self._run:
            self._run.finish()

    @staticmethod
    def _split_metrics(metrics: dict[str, NDArray]) -> dict[str, float]:
        return {
            f"{k}/{i}" if v.shape[0] > 1 else k: v[i]
            for k, v in metrics.items()
            for i in range(v.shape[0])
        }
