from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Objective(ABC):
    """Base abstract class for objective functions.

    Objective functions are expected to work with vector inputs and outputs. If an input is 1D, it
    should be extended into 2D before operating on it. The added axis should always be axis 0.

    All objective functions are expected to reduce an input's dimension into a single value
    (i.e., an input with shape (N, M) should result in an output of shape (N, 1)).

    Args:
        name: Name for the objective.
    """

    def __init__(self, *, name: str):
        self._name = name

    @property
    def name(self) -> str:
        """Returns the name of the metric."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Sets the name of the metric."""
        self._name = value

    @abstractmethod
    def __call__(self, array: NDArray) -> NDArray:
        """Compute the objective function for the given inputs.

        Args:
            array: input array. Shape is (N, M) or (M,) if input is 1D.

        Returns:
            Objective values. Shape is (N, 1).
        """
        pass
