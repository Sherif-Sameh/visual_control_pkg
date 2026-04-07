from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import LongTensor, Tensor


def init_cylinder_mesh(
    radius: float, height: float, resolution: int, split: int
) -> tuple[Tensor, LongTensor]:
    """Initialize a mesh representing a cylinder from its dimensions.

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
    z_grid, angle_grid = torch.meshgrid(z_coords, angles, indexing="ij")
    x_grid = radius * torch.cos(angle_grid)
    y_grid = radius * torch.sin(angle_grid)
    vertices = torch.cat(
        (
            torch.tensor([0.0, 0.0, 0.0]).view(1, 3),
            torch.tensor([0.0, 0.0, -height]).view(1, 3),
            torch.stack([x_grid, y_grid, z_grid], dim=-1).view(-1, 3),
        ),
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


def cyl_to_vert_offset(r_offset: Tensor, h_offset: Tensor, resolution: int, split: int) -> Tensor:
    """Convert cylinder-wide radial and height offsets to per-vertex mesh offsets.

    This function can be wrapped by `torch.vmap` to vectorize it for batched inputs.

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
    angles = torch.linspace(0, 2 * torch.pi, resolution + 1, device=device)[:-1]
    z_offsets = -torch.arange(split + 1, device=device) * h_offset / split
    z_offset_grid, angle_grid = torch.meshgrid(z_offsets, angles, indexing="ij")
    off_r = torch.stack(
        [
            r_offset * torch.cos(angle_grid).flatten(),
            r_offset * torch.sin(angle_grid).flatten(),
            z_offset_grid.flatten(),
        ],
        dim=-1,
    )
    return torch.cat([off_t[None], off_b[None], off_r])
