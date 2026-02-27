from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
from scipy.spatial.transform import Rotation as R

from .norm import Objective

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TfNormObjective(Objective):
    """L-norm objective function for poses.

    Makes use of SciPy's `scipy.spatial.transform.Rotation` module to convert the given input
    rotation representation into a `Rotation` instance. Then, the L-norm is computed between a
    fixed set of points before and after applying the derived rigid-body transformation.

    Inputs are expected to have a dimensionality of (3 + M), where M is the flattened
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
        self._to_rot = self._build_to_rot(repr, scalar_first, seq)

    def __call__(self, array: NDArray) -> NDArray:
        """Compute the L-norm objective function for the the given poses.

        The L-norm is computed between a set of fixed points before and after applying the derived
        rigid-body transformation from the given input. The points that are transformed are a fixed
        vector of ones of shape (N, 3).

        Args:
            array: input array. Shape is (N, M) or (M,) if input is 1D.

        Returns:
            Objective values. Shape is (N, 1).
        """
        array = np.atleast_2d(array)
        pos, rot = array[:, :3], self._to_rot(array[:, 3:])
        pts = np.ones((array.shape[0], 3))
        pts_tf = rot.apply(pts) + pos
        return np.linalg.norm(pts - pts_tf, ord=self._ord, axis=1, keepdims=True)

    @staticmethod
    def _build_to_rot(
        repr: Literal["euler", "mat", "quat", "rotvec"], scalar_first: bool, seq: str
    ) -> Callable[[NDArray], R]:
        match repr:
            case "euler":
                return lambda x: R.from_euler(seq, x)
            case "mat":
                return lambda x: R.from_matrix(x.reshape(-1, 3, 3))
            case "quat":
                return lambda x: R.from_quat(x, scalar_first=scalar_first)
            case "rotvec":
                return R.from_rotvec
