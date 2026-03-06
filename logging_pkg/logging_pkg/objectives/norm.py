from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .base import Objective

if TYPE_CHECKING:
    from numpy.typing import NDArray


class NormObjective(Objective):
    """L-norm objective function.

    Args:
        name: Name for the objective.
        ord: Order of the norm. Only L1, L2 and L∞ are supported. Defaults to L2 norm.
    """

    def __init__(self, *, name: str, ord: Literal[1, 2, "inf"] = 2):
        super().__init__(name=name)
        self._ord = ord if ord != "inf" else np.inf

    def __call__(self, array: NDArray) -> NDArray:
        """Compute the L-norm objective function for the given inputs.

        Args:
            array: input array. Shape is (N, M) or (M,) if input is 1D.

        Returns:
            L-norm values. Shape is (N, 1).
        """
        array = np.atleast_2d(array)
        return np.linalg.norm(array, ord=self._ord, axis=1, keepdims=True)
