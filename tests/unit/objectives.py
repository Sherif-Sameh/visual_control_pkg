from typing import Literal

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from visual_control_pkg.objectives import NormObjective, RotNormObjective, TfNormObjective


@pytest.mark.unit
@pytest.mark.parametrize("ord", [1, 2, "inf"])
def test_norm_objective(ord: Literal[1, 2, "inf"]) -> None:
    norm = NormObjective(name="norm", ord=ord)

    # 2D input
    N, M = 10, 3
    array = np.ones((N, M))
    out = norm(array)
    assert out.ndim == 2, f"Expected output for a ({N}, {M}) input to be 2D. Got {out.ndim}D."
    assert out.shape == (N, 1), (
        f"Expected output for a ({N}, {M}) input to be ({N}, 1). Got {out.shape}."
    )

    # 2D input with singular second dimension
    N, M = 10, 1
    array = np.ones((N, M))
    out = norm(array)
    assert out.ndim == 2, f"Expected output for a ({N}, {M}) input to be 2D. Got {out.ndim}D."
    assert out.shape == (N, 1), (
        f"Expected output for a ({N}, {M}) input to be ({N}, 1). Got {out.shape}."
    )

    # 1D input
    M = 3
    array = np.ones(M)
    out = norm(array)
    assert out.ndim == 2, f"Expected output for a ({M},) input to be 2D. Got {out.ndim}D."
    assert out.shape == (1, 1), f"Expected output for a ({M},) input to be (1, 1). Got {out.shape}."

    # 1D input with singular dimension
    M = 1
    array = np.ones(M)
    out = norm(array)
    assert out.ndim == 2, f"Expected output for a ({M},) input to be 2D. Got {out.ndim}D."
    assert out.shape == (1, 1), f"Expected output for a ({M},) input to be (1, 1). Got {out.shape}."


@pytest.mark.unit
@pytest.mark.parametrize("repr", ["euler", "mat", "quat", "rotvec"])
def test_rot_norm_objective(repr: Literal["euler", "mat", "quat", "rotvec"]) -> None:
    rot_norm = RotNormObjective(name="norm", ord=2, repr=repr)

    # 2D input
    N = 10
    rot = R.random(N)
    match repr:
        case "euler":
            array = rot.as_euler("xyz")
        case "mat":
            array = rot.as_matrix().reshape(-1, 9)
        case "quat":
            array = rot.as_quat()
        case "rotvec":
            array = rot.as_rotvec()
    out = rot_norm(array)
    assert out.ndim == 2, f"Expected output for {N} {repr} inputs to be 2D. Got {out.ndim}D."
    assert out.shape == (N, 1), (
        f"Expected output for {N} {repr} inputs to be ({N}, 1). Got {out.shape}."
    )
    assert np.allclose(out[:, 0], rot.magnitude()), (
        "Rotation vector L2-norm values don't match expected values."
    )

    # 1D input
    rot = rot[0]
    array = array[0]
    out = rot_norm(array)
    assert out.ndim == 2, f"Expected output for a single {repr} input to be 2D. Got {out.ndim}D."
    assert out.shape == (1, 1), (
        f"Expected output for a single {repr} input to be (1, 1). Got {out.shape}."
    )
    assert np.allclose(out[:, 0], rot.magnitude()), (
        "Rotation vector L2-norm values don't match expected values."
    )


@pytest.mark.unit
@pytest.mark.parametrize("repr", ["euler", "mat", "quat", "rotvec"])
def test_tf_norm_objective(repr: Literal["euler", "mat", "quat", "rotvec"]) -> None:
    tf_norm = TfNormObjective(name="norm", ord=2, repr=repr)

    # 2D input
    N = 10
    pos_array = np.ones((N, 3))
    rot = R.random(N)
    match repr:
        case "euler":
            rot_array = rot.as_euler("xyz")
        case "mat":
            rot_array = rot.as_matrix().reshape(-1, 9)
        case "quat":
            rot_array = rot.as_quat()
        case "rotvec":
            rot_array = rot.as_rotvec()
    array = np.concatenate([pos_array, rot_array], axis=1)
    out = tf_norm(array)
    assert out.ndim == 2, (
        f"Expected output for {N} poses (rot={repr}) inputs to be 2D. Got {out.ndim}D."
    )
    assert out.shape == (N, 1), (
        f"Expected output for {N} poses (rot={repr}) inputs to be ({N}, 1). Got {out.shape}."
    )

    # 1D input
    array = array[0]
    out = tf_norm(array)
    assert out.ndim == 2, (
        f"Expected output for a single pose (rot={repr}) input to be 2D. Got {out.ndim}D."
    )
    assert out.shape == (1, 1), (
        f"Expected output for {N} poses (rot={repr}) inputs to be (1, 1). Got {out.shape}."
    )
