from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from .base import Metric

if TYPE_CHECKING:
    from numpy.typing import NDArray


class FunctionalMetric(Metric):
    """Functional metric.

    The metric wraps any Metric instance with a given function that is applied to input values
    before they are passed onto the base metric's `update()` method.

    Args:
        name: Name for the metric.
        metric: Base metric to wrap.
        func: Function to apply to input values before calling the `update()` method.
    """

    def __init__(self, *, name: str, metric: Metric, func: Callable[[NDArray], NDArray]):
        super().__init__(name=name, argname=metric.argname)
        self._metric = metric
        self._func = func

    def compute(self) -> NDArray:
        """Computes and returns the metric value based on the internal state.

        Returns:
            Array containing the computed metric value.
        """
        return self._metric.compute()

    def reset(self) -> None:
        """Resets the metric's internal state."""
        self._metric.reset()

    def update(self, **kwargs) -> None:
        """Updates the metric's internal state based on input data.

        The input value is transformed through the metric's function then passed onto the base
        metric's own `update()` method.

        Args:
            **kwargs: Keyword arguments containing input ndarray data to update metric with.
        """
        value: NDArray | None = kwargs.get(self._argname)
        if value is None:
            return  # There's nothing to update.
        value = self._func(value)
        self._metric.update(**{self._argname: value})
