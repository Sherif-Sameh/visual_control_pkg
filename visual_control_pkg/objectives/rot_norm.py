from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
from scipy.spatial.transform import Rotation as R

from .base import Objective

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RotNormObjective(Objective):
    """L-norm objective function for rotations.

    Makes use of SciPy's `scipy.spatial.transform.Rotation` module to convert from the given input
    rotation representation into a rotation vector that a vector L-norm can be applied to.

    Inputs are expected to have a dimensionality of M, where M is the flattened
    dimensionality of the input rotation representation.

    Args:
        name: Name for the objective.
        ord: Order of the norm. Only L1, L2 and L∞ are supported. Defaults to L2 norm.
        repr: Input rotation representation. One of "euler", "mat", "quat" or "rotvec".
            Defaults to "quat".
        scalar_first: Optional argument for conversion from quaternions to pass to `R.from_quat()`.
            Defaults to False.
        seq: Optional argument for conversion from Euler angles to pass to `R.from_euler()`.
            Defaults to "xyz".
    """

    def __init__(
        self,
        *,
        name: str,
        ord: Literal[1, 2, "inf"] = 2,
        repr: Literal["euler", "mat", "quat", "rotvec"] = "quat",
        scalar_first: bool = False,
        seq: str = "xyz",
    ):
        super().__init__(name=name)
        self._ord = ord if ord != "inf" else np.inf
        self._to_rotvec = self._build_to_rotvec(repr, scalar_first, seq)

    def __call__(self, array: NDArray) -> NDArray:
        """Compute the L-norm objective function for the rotation vector derived from the given
        inputs.

        Args:
            array: input array. Shape is (N, M) or (M,) if input is 1D.

        Returns:
            Rotation vector L-norm values. Shape is (N, 1).
        """
        array = np.atleast_2d(array)
        rotvec = self._to_rotvec(array)
        return np.linalg.norm(rotvec, ord=self._ord, axis=1, keepdims=True)

    @staticmethod
    def _build_to_rotvec(
        repr: Literal["euler", "mat", "quat", "rotvec"], scalar_first: bool, seq: str
    ) -> Callable[[NDArray], NDArray]:
        match repr:
            case "euler":
                return lambda x: R.from_euler(seq, x).as_rotvec()
            case "mat":
                return lambda x: R.from_matrix(x.reshape(-1, 3, 3)).as_rotvec()
            case "quat":
                return lambda x: R.from_quat(x, scalar_first=scalar_first).as_rotvec()
            case "rotvec":
                return lambda x: R.from_rotvec(x).as_rotvec()
