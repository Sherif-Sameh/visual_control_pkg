from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor


class CylinderModel(nn.Module):
    """PyTorch model for estimating the pose and geometry of a cylinder.

    Only 5 DOFs of the relative pose of the cylinder as observed by a camera can be estimated.
    These 5 DOFs are represented through the position of the cylinder and the orientation of its
    vertical axis (Z-axis) relative to the camera.

    The position of the cylinder is parameterized by zero-centering it around the initial guess and
    scaling its magnitude by a set scaling factor. The Z-axis is parameterized through offsets in
    the 2D tangent space of the initial guess for its orientation scaled by `π`. The orthonormal
    basis for the tangent space is derived through Gram-Schmidt and so is the X and Y-axes to form
    a full right-handed rotation matrix.

    The geometry of the cylinder is parameterized through radial and height offsets that can be
    applied to a cylinder mesh (e.g., see `vc_core.dr.mesh.CylinderMesh`). These offsets are also
    scaled by the initial radius and height of the cylinder respectively.

    Args:
        pos: Initial guess for the position of the cylinder. Shape is (N, 3) or (3,).
        z_dir: Initial guess for the direction of the Z-axis. Can be unnormalized. Shape is (N, 3)
            or (3,).
        radius: Optional initial guess for the radius of the cylinder. Shape is (N,) or (1,). If
            set to `None`, the radius is not estimated and the offset is always zero. Default value
            is `None`.
        height: Optional initial guess for the height of the cylinder. Shape is (N,) or (1,). If
            set to `None`, the height is not estimated and the offset is always zero. Default value
            is `None`.
        n_rep: Number of times to repeat the cylinder's parameters. Ignored if any of the other
            input parameters is repeated. Default value is 1.
        scale: Optional scale factor for position offsets. Default value is 1.0.
    """

    def __init__(
        self,
        pos: Tensor,
        z_dir: Tensor,
        *,
        radius: Tensor | None = None,
        height: Tensor | None = None,
        n_rep: int = 1,
        scale: float = 1.0,
    ):
        super().__init__()
        self.scale = scale
        device = pos.device
        n_rep = self._get_n_rep(pos, z_dir, radius, height, n_rep)
        est_radius, est_height = radius is not None, height is not None
        # preprocess all input tensors
        pos = pos.repeat((n_rep, 1)) if pos.ndim == 1 else pos
        z_dir = z_dir.repeat((n_rep, 1)) if z_dir.ndim == 1 else z_dir
        z_dir = F.normalize(z_dir, dim=-1)
        z_basis = torch.stack(self._get_tangent_basis(z_dir), dim=-1)
        radius = torch.ones(n_rep, dtype=torch.float32, device=device) if radius is None else radius
        radius = radius.repeat(n_rep) if radius.shape[0] == 1 else radius
        height = torch.ones(n_rep, dtype=torch.float32, device=device) if height is None else height
        height = height.repeat(n_rep) if height.shape[0] == 1 else height
        # register buffers
        self.register_buffer("pos_init", pos)
        self.register_buffer("z_dir_init", z_dir)
        self.register_buffer("z_basis", z_basis)
        self.register_buffer("radius", radius)
        self.register_buffer("height", height)
        # create parameters
        self.pos_offset = nn.Parameter(torch.zeros_like(pos))
        self.z_tan = nn.Parameter(torch.zeros((n_rep, 2), dtype=torch.float32, device=device))
        self.r_offset = nn.Parameter(torch.zeros_like(radius), requires_grad=est_radius)
        self.h_offset = nn.Parameter(torch.zeros_like(height), requires_grad=est_height)

    @property
    def n_rep(self) -> int:
        """Get the number of copies of the cylinder's parameters."""
        return self.pos_offset.shape[0]

    def forward(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Construct the cylinder's pose and geometry offsets and return them.

        Returns:
            tuple containing four tensors. The first is the position of the cylinder (N, 3). The
            second is the rotation matrix of the cylinder (N, 3, 3). Then, the radial and height
            offset vectors of the cylinder respectively (N,).
        """
        # unnormalize position
        pos = self.pos_offset * self.scale + self.pos_init
        # apply tangent rotation to z-axis and create rotation matrix
        z_dir = self._apply_tangent_rotation(self.z_dir_init, self.z_tan, self.z_basis)
        rot = self._get_rotation_from_z(z_dir)
        # unnormalize radius and height offsets
        r_offset = self.r_offset * self.radius
        h_offset = self.h_offset * self.height
        return pos, rot, r_offset, h_offset

    @staticmethod
    def _get_n_rep(
        pos: Tensor, z_dir: Tensor, radius: Tensor | None, height: Tensor | None, n_rep: int
    ) -> int:
        if pos.ndim == 2:
            return pos.shape[0]
        if z_dir.ndim == 2:
            return z_dir.shape[0]
        if radius is not None and radius.shape[0] > 1:
            return radius.shape[0]
        if height is not None and height.shape[0] > 1:
            return height.shape[0]
        return n_rep

    @staticmethod
    def _get_tangent_basis(vec: Tensor) -> tuple[Tensor, Tensor]:
        """Get the orthonormal basis for the tangent plane of the given vector using Gram-Schmidt.

        Args:
            vec: Direction unit vector. Shape is (N, 3).

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

    @staticmethod
    def _apply_tangent_rotation(vec: Tensor, tan: Tensor, basis: Tensor) -> Tensor:
        """Apply tangent rotation to a unit direction vector.

        Tangent vectors are clamped to [-1, 1] then scaled by `π` before applying them. Also, for
        the exponential map, a first order approximation is used.

        Args:
            vec: Direction unit vector to apply rotation to. Shape is (N, 3).
            tan: Tangent space delta vector. Shape is (N, 2).
            basis: Orthonormal basis vectors for tangent space. Shape of each is (N, 3, 2).

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

    @staticmethod
    def _get_rotation_from_z(z_dir: Tensor) -> Tensor:
        """Construct a rotation matrix from the direction of the Z-axis using Gram-Schmidt.

        Args:
            z_dir: Direction of the Z-axis. Shape is (N, 3).

        Returns:
            Complete rotation matrix contructed from Z-axis. Shape is (N, 3, 3).
        """
        # Get orthonormal tangent basis to current Z-axis
        x_dir, y_dir = CylinderModel._get_tangent_basis(z_dir)
        # Combine into rotation matrix
        r_matrix = torch.stack([x_dir, y_dir, z_dir], dim=-1)  # (B, 3, 3)
        return r_matrix


class CylinderSplitParamModel(CylinderModel):
    """Extends `CylinderModel` to split parameter copies into individual `torch.nn.Parameter`.

    Args:
        pos: Initial guess for the position of the cylinder. Shape is (N, 3) or (3,).
        z_dir: Initial guess for the direction of the Z-axis. Can be unnormalized. Shape is (N, 3)
            or (3,).
        radius: Optional initial guess for the radius of the cylinder. Shape is (N,) or (1,). If
            set to `None`, the radius is not estimated and the offset is always zero. Default value
            is `None`.
        height: Optional initial guess for the height of the cylinder. Shape is (N,) or (1,). If
            set to `None`, the height is not estimated and the offset is always zero. Default value
            is `None`.
        n_rep: Number of times to repeat the cylinder's parameters. Ignored if any of the other
            input parameters is repeated. Default value is 1.
        scale: Optional scale factor for position offsets. Default value is 1.0.
    """

    def __init__(
        self,
        pos: Tensor,
        z_dir: Tensor,
        *,
        radius: Tensor | None = None,
        height: Tensor | None = None,
        n_rep: int = 1,
        scale: float = 1.0,
    ):
        super().__init__(pos, z_dir, radius=radius, height=height, n_rep=n_rep, scale=scale)
        # split each parameter to a list of parameters
        self.pos_offset_list = nn.ParameterList([nn.Parameter(t.clone()) for t in self.pos_offset])
        self.z_tan_list = nn.ParameterList([nn.Parameter(t.clone()) for t in self.z_tan])
        self.r_offset_list = nn.ParameterList(
            [
                nn.Parameter(t.clone(), requires_grad=self.r_offset.requires_grad)
                for t in self.r_offset
            ]
        )
        self.h_offset_list = nn.ParameterList(
            [
                nn.Parameter(t.clone(), requires_grad=self.h_offset.requires_grad)
                for t in self.h_offset
            ]
        )
        del self.pos_offset, self.z_tan, self.r_offset, self.h_offset

    @property
    def n_rep(self) -> int:
        """Get the number of copies of the cylinder's parameters."""
        return len(self.pos_offset_list)

    def forward(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Construct the cylinder's pose and geometry offsets and return them.

        Returns:
            tuple containing four tensors. The first is the position of the cylinder (N, 3). The
            second is the rotation matrix of the cylinder (N, 3, 3). Then, the radial and height
            offset vectors of the cylinder respectively (N,).
        """
        # stack parameter copies
        self.pos_offset = torch.stack(list(self.pos_offset_list), dim=0)
        self.z_tan = torch.stack(list(self.z_tan_list), dim=0)
        self.r_offset = torch.stack(list(self.r_offset_list), dim=0)
        self.h_offset = torch.stack(list(self.h_offset_list), dim=0)
        return super().forward()
