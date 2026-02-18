from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Metric

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ComposeMetric(Metric):
    """Composes multiple metrics into a single metric.

    Args:
        metrics: List of metrics to compose.
    """

    def __init__(self, *, metrics: list[Metric]):
        super().__init__(name="ComposedMetric", argname="")
        self._metrics = metrics
        self.reset()

    def compute(self) -> dict[str, NDArray]:
        """Computes and returns the values of all composed metrics.

        Returns:
            Dictionary mapping metric names to their computed ndarray values.
        """
        return {metric.name: metric.compute() for metric in self._metrics}

    def reset(self) -> None:
        """Resets all composed metrics' internal states."""
        for metric in self._metrics:
            metric.reset()

    def update(self, **kwargs) -> None:
        """Updates all composed metrics' internal states based on input data.

        Args:
            **kwargs: Keyword arguments containing input ndarray data to update metric with.
        """
        for metric in self._metrics:
            metric.update(**kwargs)
