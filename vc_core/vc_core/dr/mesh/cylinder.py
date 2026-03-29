from __future__ import annotations

from functools import partial
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
        n_rep: Number of times to repeat the wrapped mesh. Default value is 1.
    """

    def __init__(
        self,
        radius: float,
        height: float,
        *,
        resolution: int = 20,
        split: int = 4,
        texture: TexturesBase | None = None,
        n_rep: int = 1,
    ):
        nn.Module.__init__()
        assert radius > 0 and height > 0
        # Initialize mesh from cylinder parameters
        vertices, face_idxs = self._init_cylinder_mesh(radius, height, resolution, split)
        mesh = Meshes([vertices], [face_idxs], textures=texture)
        self._mesh = mesh.extend(n_rep)
        # Create batched func for converting cylinder offsets to mesh vertex offsets
        self._cyl_to_vert_offset_fn = torch.vmap(
            partial(self._cyl_to_vert_offset, resolution=resolution, split=split)
        )

    def forward(
        self, r_offset: Tensor, h_offset: Tensor, texture: TexturesBase | None = None
    ) -> Meshes:
        """Apply radial and height offsets and new texture if given to cylinder meshes.

        Args:
            r_offset: Radial offsets to apply to each cylinder mesh. Shape is (n_rep,).
            h_offset: Height offsets to apply to each cylinder mesh. Shape is (n_rep,).
            texture: Optional texture to apply to the new mesh before returning it. Default value
                is `None`.

        Returns:
            New meshes created by applying given vertex offsets and texture if given.
        """
        vertex_offsets = self._cyl_to_vert_offset_fn(r_offset, h_offset).view(-1, 3)
        return super().forward(vertex_offsets, texture)

    @staticmethod
    def _init_cylinder_mesh(
        self, radius: float, height: float, resolution: int, split: int
    ) -> tuple[Tensor, LongTensor]:
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

    @staticmethod
    def _cyl_to_vert_offset(
        r_offset: Tensor, h_offset: Tensor, resolution: int, split: int
    ) -> Tensor:
        """Convert cylinder-wide radial and height offsets to per-vertex mesh offsets.

        Args:
            r_offset: Radial offset in meters. Zero dimensional tensor.
            h_offset: Height offset in meters. Zero dimensional tensor.
            resolution: Resolution used in creating the cylinder mesh.
            split: Split used in creating the cylinder mesh.

        Returns:
            Vertex offsets for cylinder mesh. Shape is (N, 3), where N is the number of vertices
            in the cylinder mesh.
        """
        device = r_offset.device
        # set offset for two center vertices
        off_t = torch.zeros(3, dtype=torch.float32, device=device)
        off_b = torch.cat([torch.zeros(2, dtype=torch.float32, device=device), -h_offset[None]])
        # set offset for radial vertices
        angles = torch.linspace(0, 2 * torch.pi, resolution + 1)[:-1]
        z_offsets = -torch.arange(split + 1, device=device) * h_offset / split
        angle_grid, z_offset_grid = torch.meshgrid(angles, z_offsets, indexing="ij")
        off_r = torch.stack(
            [
                r_offset * torch.cos(angle_grid).flatten(),
                r_offset * torch.sin(angle_grid).flatten(),
                z_offset_grid.flatten(),
            ],
            dim=-1,
        )
        return torch.cat([off_t[None], off_b[None], off_r])
