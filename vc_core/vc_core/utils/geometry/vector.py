from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor


def get_tangent_basis(vec: Tensor) -> tuple[Tensor, Tensor]:
    """Get the orthonormal basis for the tangent plane of the given vector using Gram-Schmidt.

    Args:
        vec: Direction unit vector. Can be unnormalized. Shape is (N, 3).

    Returns:
        tuple of orthonormal basis vectors that define the given vector's tangent space. Shape
        of each is (N, 3).
    """
    # Ensure that direction is a unit vector
    vec = F.normalize(vec, dim=-1)
    # Pick arbitrary directions for first vector
    e1 = torch.tensor([1.0, 0.0, 0.0], device=vec.device).expand_as(vec)
    e1_alt = torch.tensor([0.0, 1.0, 0.0], device=vec.device).expand_as(vec)
    parallel_mask = (vec * e1).sum(dim=-1, keepdim=True).abs() > 0.99
    e1 = torch.where(parallel_mask, e1_alt, e1)
    # Make first basis vector normal to given vector
    e1_proj_on_vec = (e1 * vec).sum(dim=-1, keepdim=True) * vec
    e1 = e1 - e1_proj_on_vec
    e1 = F.normalize(e1, dim=-1)
    # Get second basis vector that's normal to both vectors
    e2 = torch.cross(vec, e1, dim=-1)
    return e1, e2


def apply_tangent_rotation(vec: Tensor, tan: Tensor, basis: Tensor) -> Tensor:
    """Apply tangent rotation to a unit direction vector.

    Tangent vectors are clamped to [-1, 1] then scaled by `π` before applying them. A first-order
    approximation is used for the exponential map.

    Args:
        vec: Direction unit vector to apply rotation to. Shape is (N, 3).
        tan: Tangent space delta vector. Shape is (N, 2).
        basis: Orthonormal basis vectors for tangent space. Shape is (N, 3, 2).

    Returns:
        Updated unit direction vector after applying tangent rotation. Shape is (N, 3).
    """
    # Compute delta tangent vector in 3D
    tan = torch.clamp(tan, -1, 1) * torch.pi
    delta = tan[:, :1] * basis[:, :, 0] + tan[:, 1:2] * basis[:, :, 1]
    # Update rotation using first-order approx of exponential map
    vec_rot = vec + delta
    vec_rot = F.normalize(vec_rot, dim=-1)
    return vec_rot


def apply_tangent_rotation_exact(vec: Tensor, tan: Tensor, basis: Tensor) -> Tensor:
    """
    Apply exact tangent rotation to a unit direction vector using the exponential map.

    Tangent vectors are clamped to [-1, 1] then scaled by `π` before applying them.

    Args:
        vec: Direction unit vector to apply rotation to. Shape is (N, 3).
        tan: Tangent space delta vector. Shape is (N, 2).
        basis: Orthonormal basis vectors for tangent space. Shape is (N, 3, 2).

    Returns:
        Updated unit direction vector after applying tangent rotation. Shape is (N, 3).
    """
    eps = 1e-6
    # Compute delta tangent vector in 3D
    tan = torch.clamp(tan, -1, 1) * torch.pi
    delta = tan[:, :1] * basis[:, :, 0] + tan[:, 1:2] * basis[:, :, 1]
    # Compute rotation magnitude and its sinc and cosine
    theta = torch.linalg.norm(delta, dim=-1, keepdim=True)
    mask = (theta > eps).float()
    sinc = mask * (torch.sin(theta) / (theta + eps)) + (1.0 - mask)
    cosine = torch.cos(theta)
    # Update rotation using exact exponential map
    vec_rot = cosine * vec + sinc * delta
    vec_rot = F.normalize(vec_rot, dim=-1)
    return vec_rot


def get_rotation_from_z(z_dir: Tensor) -> Tensor:
    """Construct a rotation matrix from the direction of the Z-axis using Gram-Schmidt.

    Args:
        z_dir: Direction of the Z-axis. Shape is (N, 3).

    Returns:
        Complete rotation matrix contructed from Z-axis. Shape is (N, 3, 3).
    """
    # Get orthonormal tangent basis to current Z-axis
    x_dir, y_dir = get_tangent_basis(z_dir)
    # Combine into rotation matrix
    r_matrix = torch.stack([x_dir, y_dir, z_dir], dim=-1)  # (B, 3, 3)
    return r_matrix
