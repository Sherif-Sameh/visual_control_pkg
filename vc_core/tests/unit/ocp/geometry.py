import casadi as ca
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from vc_core.ocp.model.geometry import quat_apply, quat_diff, quat_dot, quat_inv, quat_mult


@pytest.mark.unit
def test_quat_inv() -> None:
    # generate random unit quaternion
    q = R.random().as_quat(scalar_first=True)

    # compute inverse with CasADi function
    q_dm = ca.DM(q)
    q_inv_casadi = dm_to_np(quat_inv(q_dm))

    # compute inverse with SciPy
    r = R.from_quat(q, scalar_first=True)
    r_inv = r.inv()
    q_inv_scipy = r_inv.as_quat(scalar_first=True)

    # compare both together
    assert np.allclose(q_inv_casadi, q_inv_scipy, atol=1e-8)


@pytest.mark.unit
def test_quat_mult() -> None:
    # generate random unit quaternions
    q1 = R.random().as_quat(scalar_first=True)
    q2 = R.random().as_quat(scalar_first=True)

    # compute quaternion product with CasADi function
    q1_dm = ca.DM(q1)
    q2_dm = ca.DM(q2)
    q_mult_casadi = dm_to_np(quat_mult(q1_dm, q2_dm))

    # compute quaternion product with SciPy
    r1 = R.from_quat(q1, scalar_first=True)
    r2 = R.from_quat(q2, scalar_first=True)
    r_mult = r1 * r2
    q_mult_scipy = r_mult.as_quat(scalar_first=True)

    # compare both together
    assert np.allclose(q_mult_casadi, q_mult_scipy, atol=1e-8)


@pytest.mark.unit
def test_quat_apply() -> None:
    # generate random unit quaternion and vector
    q = R.random().as_quat(scalar_first=True)
    v = np.random.randn(3)

    # compute quaternion-vector product with CasADi function
    q_dm = ca.DM(q)
    v_dm = ca.DM(v)
    v_rot_casadi = dm_to_np(quat_apply(q_dm, v_dm))

    # compute quaternion-vector product with SciPy
    r = R.from_quat(q, scalar_first=True)
    v_rot_scipy = r.apply(v)

    # compare both together
    assert np.allclose(v_rot_casadi, v_rot_scipy, atol=1e-8)


@pytest.mark.unit
def test_quat_diff() -> None:
    # generate random unit quaternions
    q = R.random().as_quat(scalar_first=True)
    q_ref = R.random().as_quat(scalar_first=True)

    # compute quaternion difference with CasADi function
    q_dm = ca.DM(q)
    q_ref_dm = ca.DM(q_ref)
    diff_casadi = dm_to_np(quat_diff(q_dm, q_ref_dm))

    # compute quaternion difference with SciPy
    r = R.from_quat(q, scalar_first=True)
    r_ref = R.from_quat(q_ref, scalar_first=True)
    r_err = r_ref.inv() * r
    q_err_scipy = r_err.as_quat(scalar_first=True)
    if q_err_scipy[0] < 0:
        q_err_scipy = -q_err_scipy
    diff_scipy = q_err_scipy[1:]

    # compare both together
    assert np.allclose(diff_casadi, diff_scipy, atol=1e-8)


@pytest.mark.unit
def test_quat_dot():
    # generate random unit quaternion and tangent vector
    q = R.random().as_quat(scalar_first=True)
    w = np.random.randn(3)

    # compute new q from q_dot using CasADi function
    q_dm = ca.DM(q)
    w_dm = ca.DM(w)
    q_dot_casadi = dm_to_np(quat_dot(q_dm, w_dm))
    q_new_casadi = q + 0.01 * q_dot_casadi
    q_new_casadi /= np.linalg.norm(q_new_casadi)
    if q_new_casadi[0] < 0:
        q_new_casadi = -q_new_casadi

    # compute new q from tangent vector using SciPy
    r = R.from_quat(q, scalar_first=True)
    tan = R.from_rotvec(0.01 * w)
    r_new = r * tan
    q_new_scipy = r_new.as_quat(scalar_first=True)
    if q_new_scipy[0] < 0:
        q_new_scipy = -q_new_scipy

    # compare both together
    assert np.allclose(q_new_casadi, q_new_scipy, atol=1e-8)


def dm_to_np(x: ca.DM) -> NDArray:
    return np.array(x.full()).squeeze()
