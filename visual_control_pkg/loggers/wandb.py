import os
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import wandb

from .base import Logger


class WandBLogger(Logger):
    """WandB-based logger that saves metrics to a WandB project.

    Args:
        n_log: Interval for logging in steps. Defaults to 1 (i.e. log everything).
        n_flush: Interval for flushing logger in steps. Defaults to 1 (i.e. flush instantly).
        path: Path to the CSV file where logs will be saved.
        config: WandB configuration.
    """

    @dataclass
    class WandBConfig:
        entity: str
        project: str
        group: str
        dir: str | Path
        config: dict[str, Any] = field(default_factory=lambda: {})

    def __init__(
        self,
        *,
        n_log: int = 1,
        n_flush: int = 1,
        filter: str | None = None,
        config: WandBConfig,
    ):
        super().__init__(n_log=n_log, n_flush=n_flush, filter=filter)
        self._config = config
        self._metric_names = []

        # Find WandB API environment variable
        assert "WANDB_API_KEY" in os.environ, (
            "WANDB_API_KEY environment variable is not set."
        )
        wandb_api_key = os.environ["WANDB_API_KEY"]
        wandb_api_key = (
            str(wandb_api_key) if not isinstance(wandb_api_key, str) else wandb_api_key
        )

        # Initialize WandB run
        wandb.login(key=wandb_api_key)
        self.restart()

    def __del__(self):
        if self._run:
            self._run.finish()

    def flush(self) -> None:
        """Saves stored logs to WandB project and run."""
        if not self._log:
            return
        # Prepare logs for WandB
        # 1) Add step entry to dicts
        # 2) Define any new metrics
        # 3) Separate array entries
        for step in self._log.keys():
            step_mod = {"logger_step": step}
            for name, value in self._log[step].items():
                for i, v in enumerate(value):
                    metric_name = f"{name}/{i}"
                    if metric_name not in self._metric_names:
                        self._run.define_metric(metric_name, step_metric="logger_step")
                        self._metric_names.append(metric_name)
                    step_mod[metric_name] = v
            self._log[step] = step_mod

        # Log existing at each step
        for step in self._log.keys():
            self._run.log(self._log[step], commit=False)
        self._run.log({}, commit=True)
        self._log.clear()

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
