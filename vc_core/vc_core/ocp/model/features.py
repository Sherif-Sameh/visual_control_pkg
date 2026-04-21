"""
Visual features helper functions using CasADi.
"""

import casadi as ca

from .geometry import quat_apply, quat_inv


def project(cp: ca.SX, cq: ca.SX, fp: ca.SX) -> tuple[ca.SX, ca.SX]:
    """Apply perspective camera projection to input feature world points `fp`.

    Assumes scalar-first (i.e. [w, x, y, z]) convention.

    Args:
        cp: Position of the camera wrt world frame.
        cq: Orientation (quaternion) of the camera wrt world frame.
        fp: Position of the features in the world frame.

    Returns:
        Tuple of two symbolic expressions. The first consists of the position of the features in
        the image plane. That is applying perspective division to the X and Y feature coordinates
        after transforming `fp` to the camera frame. The second is the depth Z in the camera frame.
    """
    cfp = quat_apply(quat_inv(cq), fp - cp)
    return ca.vertcat(cfp[0] / cfp[2], cfp[1] / cfp[2]), cfp[2]


def feature_dot(u: ca.SX, s: ca.SX, Z: ca.SX) -> ca.SX:
    """Compute the feature velocities in the image plane from camera twist `u`.

    Refer to `vc_core.ocp.model.features.project` for computing `s` and `Z`.

    Args:
        u: Camera twist vector (`v`, `w`).
        s: Feature projections in the image plane.
        Z: Depth of features in the camera coordinate frame.

    Returns:
        Feature velocity vector `s_dot`.
    """
    x, y = s[0], s[1]
    interaction = ca.horzcat(
        ca.vertcat(-1.0 / Z, 0.0),
        ca.vertcat(0.0, -1.0 / Z),
        ca.vertcat(x / Z, y / Z),
        ca.vertcat(x * y, 1 + y * y),
        ca.vertcat(-(1.0 + x * x), -x * y),
        ca.vertcat(y, -x),
    )
    return interaction @ u
