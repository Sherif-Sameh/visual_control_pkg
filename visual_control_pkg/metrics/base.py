from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Metric(ABC):
    """Base abstract class for tracking metrics.

    Based on the flax.nnx.Metric class.

    Args:
        name: Name for the metric.
        argname: Name of the argument to compute metric from during updates.
    """

    def __init__(self, *, name: str, argname: str):
        self._name = name
        self._argname = argname

    @property
    def name(self) -> str:
        """Returns the name of the metric."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Sets the name of the metric."""
        self._name = value

    @property
    def argname(self) -> str:
        """Returns the name of the argument of the metric."""
        return self._argname

    @argname.setter
    def argname(self, value: str) -> None:
        """Sets the name of the argument of the metric."""
        self._argname = value

    @abstractmethod
    def compute(self) -> NDArray:
        """Computes and returns the metric value based on the internal state.

        Returns:
            Array containing the computed metric value.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the metric's internal state."""
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """Updates the metric's internal state based on input data.

        Args:
            **kwargs: Keyword arguments containing input ndarray data to update metric with.
        """
        pass
