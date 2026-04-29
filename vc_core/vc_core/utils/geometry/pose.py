from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from geometry_msgs.msg import Pose, Transform
    from numpy.typing import NDArray

    PoseType: TypeAlias = tuple[NDArray, R]


def from_pose_gm(pose_gm: Pose) -> PoseType:
    """Initialize pose tuple from geometry_msgs/Pose."""
    pos, ori = pose_gm.position, pose_gm.orientation
    tvec = np.array([pos.x, pos.y, pos.z])
    rot = R.from_quat([ori.x, ori.y, ori.z, ori.w])
    return tvec, rot


def from_transform_gm(transform_gm: Transform) -> PoseType:
    """Initialize pose tuple from geometry_msgs/Transform."""
    pos, ori = transform_gm.translation, transform_gm.rotation
    tvec = np.array([pos.x, pos.y, pos.z])
    rot = R.from_quat([ori.x, ori.y, ori.z, ori.w])
    return tvec, rot


def from_pose_ndarray(pose_np: NDArray, scalar_first: bool = False) -> PoseType:
    """Initialize pose tuple from a flattened NumPy pose array."""
    tvec = np.copy(pose_np[..., :3])
    rot = R.from_quat(pose_np[..., 3:], scalar_first=scalar_first)
    return tvec, rot


def pose_inv(pose01: PoseType) -> PoseType:
    """Invert pose of frame 1 wrt frame 0.

    Returns:
        Pose of frame 0 wrt frame 1.
    """
    rot = pose01[1].inv()
    tvec = -rot.apply(pose01[0])
    return (tvec, rot)


def pose_mult(pose01: PoseType, pose12: PoseType) -> PoseType:
    """Combine poses of frame 1 wrt frame 0 and frame 2 wrt frame 1.

    Returns:
        Pose of frame 2 wrt frame 0.
    """
    tvec02 = pose01[0] + pose01[1].apply(pose12[0])
    rot02 = pose01[1] * pose12[1]
    return tvec02, rot02


def pose_sub(pose01: PoseType, pose02: PoseType) -> PoseType:
    """Subtract pose of frame 1 wrt frame 0 from pose of frame 2 wrt frame 0.

    Returns:
        Pose of frame 2 wrt frame 1.
    """
    return pose_mult(pose_inv(pose01), pose02)


def pose_adj(pose01: PoseType) -> NDArray:
    """Get the adjoint matrix of pose of frame 1 wrt frame 0.

    The adjoint matrix transforms twists expressed in frame 1 to frame 0.

    Returns:
        Adjoint matrix of frame 1 wrt frame 0. Shape is (6, 6).
    """
    tvec, rot = pose01
    rmat = rot.as_matrix()
    tvec_skew = np.array([[0, -tvec[2], tvec[1]], [tvec[2], 0, -tvec[0]], [-tvec[1], tvec[0], 0]])
    adj = np.zeros((6, 6))
    adj[0:3, 0:3] = rmat
    adj[3:6, 3:6] = rmat
    adj[3:6, 0:3] = tvec_skew @ rmat
    return adj
