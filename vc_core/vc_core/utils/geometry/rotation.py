from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

_EPS = 1e-8


def apply_tangent_rotation(rmat: Tensor, tan: Tensor) -> Tensor:
    """Apply tangent rotation to a rotation matrix using the SO3 exponential map.

    Tangent vectors are clamped to [-1, 1] then scaled by `π` before applying them.

    Args:
        rmat: Rotation matrix to apply rotation to. Shape is (N, 3, 3).
        tan: Tangent space rotation vector. Shape is (N, 3).

    Returns:
        Updated rotation matrix after applying tangent rotation. Shape is (N, 3, 3).
    """
    # Compute delta rotation matrix using exact exponential map
    tan = torch.clamp(tan, -1, 1) * torch.pi
    delta = rotvec_to_matrix(tan)
    # Update rotation matrix using delta
    rmat = torch.bmm(rmat.expand_as(delta), delta)
    return rmat


def rotvec_to_matrix(rvec: Tensor) -> Tensor:
    """Convert rotation vector to a rotation matrix via Rodrigues' formula.

    Args:
        rvec: Rotation vector to convert. Shape is (..., 3).

    Returns:
        Rotation matrix representation of input rotation vector. Shape is (..., 3, 3).
    """
    angle = rvec.norm(dim=-1, keepdim=True).unsqueeze(-1)  # (..., 1, 1)
    axis = rvec / (angle.squeeze(-1) + _EPS)  # (..., 3)
    skew = _skew(axis)  # (..., 3, 3)
    eye = torch.eye(3, device=rvec.device, dtype=rvec.dtype).expand_as(skew)
    axis_axis_t = torch.bmm(axis.unsqueeze(-1).flatten(0, -3), axis.unsqueeze(-2).flatten(0, -3))
    axis_axis_t = axis_axis_t.reshape(*rvec.shape[:-1], 3, 3)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    return cos_a * eye + sin_a * skew + (1 - cos_a) * axis_axis_t


def _skew(vec: Tensor) -> Tensor:
    """Get a batched skew-symmetric matrix from given vector.

    Args:
        vec: Input 3D vector. Shape is (..., 3).

    Returns:
        Skew-symmetric matrix of input vector. Shape is (..., 3, 3).
    """
    *batch, _ = vec.shape
    zero = torch.zeros(*batch, device=vec.device, dtype=vec.dtype)
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    row0 = torch.stack([zero, -z, y], dim=-1)
    row1 = torch.stack([z, zero, -x], dim=-1)
    row2 = torch.stack([-y, x, zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)
