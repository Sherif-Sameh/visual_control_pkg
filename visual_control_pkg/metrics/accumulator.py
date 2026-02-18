from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .base import Metric

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AccumulatorMetric(Metric):
    """Accumulator metric.

    The metric accumulates values provided during updates. Then, the metric value is computed
    by reducing those accumulated values according to the chosen reduction method.

    Args:
        name: Name for the metric.
        argname: Name of the argument to accumulate from during metric updates.
        red: Reduction method for computing metric. Must be one of "sum", "mean", "cnt".
    """

    def __init__(self, *, name: str, argname: str, red: Literal["sum", "mean", "cnt"]):
        assert red in ["sum", "mean", "cnt"], f"Reduction method {red} not supported."
        super().__init__(name=name, argname=argname)
        self._red = red
        self.reset()

    def compute(self) -> NDArray:
        """Computes and returns the metric value by reducing the accumulated state.

        Returns:
            Array containing the reduced state according to the set reduction method.
                Shape is (D,). where D is the dimensionality of the tracked metric.
        """
        if self._state is None:
            return np.array([float("nan")])
        match self._red:
            case "sum":
                return self._state
            case "mean":
                return self._state / self._count
            case "cnt":
                return np.array([self._count])

    def reset(self) -> None:
        """Resets the internal state and count to their default values."""
        self._state = None
        self._count = 0

    def update(self, **kwargs) -> None:
        """Updates the internal state and count with the provided value.

        **Important**: The provided value can only be 2D. It is reduced with `.sum(axis=0)` before
        being added to the existing state. Once the internal state is initialized with the first
        input array, shapes of inputs from following updates must be broadcastable to the internal
        state's shape.

        Args:
            **kwargs: Keyword arguments containing input ndarray data to update metric with.
        """
        value: NDArray | None = kwargs.get(self._argname)
        if value is None:
            return  # There's nothing to update.
        assert value.ndim == 2, f"Input array ndim must be 2. Got ndim = {value.ndim}"
        if self._state is None:
            self._state = np.zeros(value.shape[1])
        self._state += value.sum(axis=0)
        self._count += value.shape[0]
