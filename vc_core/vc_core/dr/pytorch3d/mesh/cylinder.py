from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from pytorch3d.structures import Meshes

from vc_core.dr.common.mesh.cylinder import cyl_to_vert_offset, init_cylinder_mesh

from .base import Mesh

if TYPE_CHECKING:
    from pytorch3d.renderer.mesh import TexturesBase
    from torch import Tensor


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
        nn.Module.__init__(self)
        assert radius > 0 and height > 0
        # Initialize mesh from cylinder parameters
        vertices, face_idxs = init_cylinder_mesh(radius, height, resolution, split)
        mesh = Meshes([vertices], [face_idxs], textures=texture)
        self._mesh = mesh.extend(n_rep)
        # Create batched func for converting cylinder offsets to mesh vertex offsets
        self._cyl_to_vert_offset_fn = torch.vmap(
            partial(cyl_to_vert_offset, resolution=resolution, split=split)
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
