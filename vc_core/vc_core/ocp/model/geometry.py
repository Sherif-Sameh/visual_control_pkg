"""
Geometry helper functions using CasADi.
"""

import casadi as ca


def quat_inv(q: ca.SX) -> ca.DM:
    """Compute the inverse of a unit quaternion `q`.

    Assumes scalar-first (i.e. [w, x, y, z]) convention.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ca.vertcat(w, -x, -y, -z)


def quat_mult(q1: ca.SX, q2: ca.SX) -> ca.DM:
    """Compute the Hamilton product of two quaternions `q1` and `q2`.

    Assumes scalar-first (i.e. [w, x, y, z]) convention.
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    return ca.vertcat(
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def quat_dot(q: ca.SX, w: ca.SX) -> ca.DM:
    """Compute the quaternion time-derivative for unit quaternion `q` and tangent velocity vector `w`.

    Assumes scalar-first (i.e. [w, x, y, z]) convention.
    """
    return 0.5 * quat_mult(q, ca.vertcat(0, w))


def quat_apply(q: ca.SX, p: ca.SX) -> ca.DM:
    """Rotate a 3D vector `p` by a unit quaternion `q`.

    Assumes scalar-first (i.e. [w, x, y, z]) convention.
    """
    p_quat = ca.vertcat(0, p)
    q_inv = quat_inv(q)
    p_rot = quat_mult(quat_mult(q, p_quat), q_inv)
    return p_rot[1:4]
