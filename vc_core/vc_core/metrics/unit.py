from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import Metric

if TYPE_CHECKING:
    from numpy.typing import NDArray


class UnitMetric(Metric):
    """Unit metric.

    The metric does not alter input values at all during updates. It only stores the latest input
    in its internal state and returns it when the metric's value is queried.
    Args:
        name: Name for the metric.
        argname: Name of the argument to track during metric updates.
    """

    def __init__(self, *, name: str, argname: str):
        super().__init__(name=name, argname=argname)
        self._state = None

    def compute(self) -> NDArray:
        """Returns the last stored input value in its internal state.

        Returns:
            Array containing the last stored input value. Shape is (D,). where D is the
                dimensionality of the tracked metric.
        """
        if self._state is None:
            return np.array([np.nan])
        return self._state

    def reset(self) -> None:
        """Resets the metric's internal state."""
        self._state = None

    def update(self, **kwargs) -> None:
        """Updates the internal state with the provided value.

        Args:
            **kwargs: Keyword arguments containing input ndarray data to update metric with.
        """
        value: NDArray | None = kwargs.get(self._argname)
        if value is None:
            return  # There's nothing to update.
        self._state = value
