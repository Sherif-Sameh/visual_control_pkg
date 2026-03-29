from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from pytorch3d.structures import Meshes

from .base import Mesh

if TYPE_CHECKING:
    from pytorch3d.renderer.mesh import TexturesBase
    from torch import LongTensor, Tensor


class CylinderMesh(Mesh):
    """Cylinder mesh class for PyTorch3D.

    The coordinate system for the cylinder is such that the origin (0, 0, 0) lies at the center
    of the cylinder's top face and the Z-axis is aligned with the axis of the cylinder.

    Args:
        radius: Radius of the cylinder in meters.
        height: Height of the cylinder in meters.
        resolution: The circle will be split into `resolution` segments. Defaults value is 20.
        split: The `height` will be split into `split` segments. Default value is 4.
        texture: Optional texture to apply to mesh. Default value is `None`.
    """

    def __init__(
        self,
        radius: float,
        height: float,
        *,
        resolution: int = 20,
        split: int = 4,
        texture: TexturesBase | None = None,
    ):
        nn.Module.__init__()
        assert radius > 0 and height > 0

        self._resolution = resolution
        self._split = split
        # Initialize mesh from cylinder parameters
        vertices, face_idxs = self._init_cylinder_mesh(radius, height)
        self._mesh = Meshes([vertices], [face_idxs], textures=texture)

    def forward(
        self, r_offset: Tensor, h_offset: Tensor, texture: TexturesBase | None = None
    ) -> Meshes:
        vertices = self._mesh.verts_packed()
        offset = torch.empty_like(vertices)
        # set offset for two center vertices
        offset[:2] = 0
        offset[1, 2] = -h_offset
        # set offset for radial vertices
        angles = torch.linspace(0, 2 * torch.pi, self._resolution + 1)[:-1]
        z_offsets = -torch.arange(self._split + 1, device=vertices.device) * h_offset / self._split
        angle_grid, z_offset_grid = torch.meshgrid(angles, z_offsets, indexing="ij")
        offset[2:, 0] = r_offset * torch.cos(angle_grid).flatten()
        offset[2:, 1] = r_offset * torch.sin(angle_grid).flatten()
        offset[2:, 2] = z_offset_grid.flatten()
        return super().forward(offset, texture)

    def _init_cylinder_mesh(self, radius: float, height: float) -> tuple[Tensor, LongTensor]:
        """Initialize a mesh from a cylinder.

        The cylinder is split twice, once radially from [0, 2pi) and another vertically from
        [0, height]. The mesh is therefore made up of `resolution` * (`split` + 1) vertices spread
        along the side faces of the cylinder and an additional 2 vertices at the centers of the top
        and bottom faces.

        Triangles on the top and bottom faces are created by connecting each 2 following vertices
        along the edges of these faces to the center vertix of each face. Side faces are created by
        splitting the quads connecting matching successive pairs of points at different height
        levels into two triangles each all along the height of the cylinder.

        Args:
            radius: Radius of the cylinder in meters.
            height: Height of the cylinder in meters.

        Returns:
            tuple of vertices and face indices tensors. The first tensor has shape (N, 3),
            where N = `resolution` * (`split` + 1) + 2. The second tensor has shape (F, 3),
            where F = `resolution` * (`split` + 1) * 2.
        """
        # Radial and vertical segments
        resolution, split = self._resolution, self._split
        angles = torch.linspace(0, 2 * torch.pi, resolution + 1)[:-1]
        z_coords = torch.linspace(0.0, -height, split + 1)

        # Create vertices
        angle_grid, z_grid = torch.meshgrid(angles, z_coords, indexing="ij")
        x_grid = radius * torch.cos(angle_grid)
        y_grid = radius * torch.sin(angle_grid)
        vertices = torch.cat(
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, -height]),
            torch.stack([x_grid, y_grid, z_grid], dim=-1).view(-1, 3),
            dim=0,
        )

        # Create triangles for top and bottom faces
        face_idxs = []
        idxs = torch.cat((torch.arange(resolution), torch.tensor([0]))) + 2
        face_idxs.append(
            torch.stack((torch.zeros(resolution, dtype=torch.long), idxs[:-1], idxs[1:]), dim=-1)
        )
        idxs += resolution * split
        face_idxs.append(
            torch.stack((torch.ones(resolution, dtype=torch.long), idxs[1:], idxs[:-1]), dim=-1)
        )

        # Create triangles for the side faces
        idxs = torch.cat((torch.arange(resolution), torch.tensor([0]))) + 2
        for _ in range(split):
            # break down qauds between two heights into triangles
            face_idxs.append(torch.stack((idxs[:-1], idxs[1:], idxs[:-1] + resolution), dim=-1))
            idxs += resolution
            face_idxs.append(torch.stack((idxs[:-1], idxs[1:], idxs[1:] - resolution), dim=-1))
        face_idxs = torch.cat(face_idxs, dim=0)
        return vertices, face_idxs
