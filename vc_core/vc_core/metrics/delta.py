from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Metric

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DeltaMetric(Metric):
    """Delta metric.

    The metric wraps any Metric instance to track the change in its input values. The metric keeps
    track of the previous input values as part of its internal state. When `update()` is called,
    the Δ-input is passed to the wrapped metric instead of the actual inputs.

    Args:
        name: Name for the metric.
        argname: Name of the argument to compute Δ changes from during updates.
        metric: Base metric to wrap.
        default: Default value to return when no changes have been recorded yet.
    """

    def __init__(self, *, name: str, argname: str, metric: Metric, default: NDArray):
        super().__init__(name=name, argname=argname)
        self._metric = metric
        self._metric.argname = f"delta_{argname}"
        self._default = default
        self._state = None
        self._count = 0

    @Metric.argname.setter
    def argname(self, value: str) -> None:
        """Sets the name of the argument of the metric and its wrapped metric."""
        self._argname = value
        self._metric.argname = f"delta_{value}"

    def compute(self) -> NDArray:
        """Computes and returns the metric value based on the internal state.

        Returns:
            Array containing the computed metric value.
        """
        if self._count == 0:
            return self._default
        return self._metric.compute()

    def reset(self) -> None:
        """Resets the metric's internal state."""
        self._metric.reset()
        self._state = None
        self._count = 0

    def update(self, **kwargs) -> None:
        """Updates the metric's internal state based on input data.

        The input value is captured and its Δ change is computed before passing that change as
        input to the wrapped metric.

        Args:
            **kwargs: Keyword arguments containing input ndarray data to update metric with.
        """
        value: NDArray | None = kwargs.get(self._argname)
        if value is None:
            return  # There's nothing to update.
        if self._state is None:
            self._state = value
            return  # No change to pass to wrapped metric
        self._metric.update(**{f"delta_{self._argname}": value - self._state})
        self._state = value
        self._count += 1
