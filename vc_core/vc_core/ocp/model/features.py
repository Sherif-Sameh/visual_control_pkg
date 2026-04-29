"""
Visual features helper functions using CasADi.
"""

import casadi as ca

from .geometry import quat_apply, quat_inv, quat_mult


def project(tp: ca.SX, tq: ca.SX, tcp: ca.SX, tcq: ca.SX, fp: ca.SX) -> tuple[ca.SX, ca.SX]:
    """Apply perspective camera projection to input feature world points `fp`.

    Assumes scalar-first (i.e. [w, x, y, z]) convention.

    Args:
        tp: Position of the tool wrt world frame.
        tq: Orientation (quaternion) of the tool wrt world frame.
        tcp: Position of the camera wrt tool frame.
        tcq: Orientation (quaternion) of the camera wrt tool frame.
        fp: Position of the features in the world frame.

    Returns:
        Tuple of two symbolic expressions. The first consists of the position of the features in
        the image plane. That is applying perspective division to the X and Y feature coordinates
        after transforming `fp` to the camera frame. The second is the depth Z in the camera frame.
    """
    cp = tp + quat_apply(tq, tcp)
    cq = quat_mult(tq, tcq)
    cfp = quat_apply(quat_inv(cq), fp - cp)
    return ca.vertcat(cfp[0] / cfp[2], cfp[1] / cfp[2]), cfp[2]
