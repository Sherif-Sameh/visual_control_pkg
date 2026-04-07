from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
from kaolin.rep import SurfaceMesh

from vc_core.dr.common.mesh.cylinder import cyl_to_vert_offset, init_cylinder_mesh

from .base import Mesh

if TYPE_CHECKING:
    from torch import Tensor


class CylinderMesh(Mesh):
    """Cylinder mesh class for Kaolin.

    The coordinate system for the cylinder is such that the origin (0, 0, 0) lies at the center
    of the cylinder's top face and the Z-axis is aligned with the axis of the cylinder.

    Args:
        radius: Radius of the cylinder in meters.
        height: Height of the cylinder in meters.
        resolution: The circle will be split into `resolution` segments. Defaults value is 20.
        split: The `height` will be split into `split` segments. Default value is 4.
        n_rep: Number of times to repeat the wrapped mesh. Default value is 1.
        kwargs: Other arguments to pass to the `kaolin.rep.SurfaceMesh` constructor.
    """

    def __init__(
        self,
        radius: float,
        height: float,
        *,
        resolution: int = 20,
        split: int = 4,
        n_rep: int = 1,
        **kwargs,
    ):
        assert radius > 0 and height > 0
        # Initialize mesh from cylinder parameters
        vertices, faces = init_cylinder_mesh(radius, height, resolution, split)
        mesh = SurfaceMesh(vertices, faces, **kwargs)
        super().__init__(mesh, n_rep=n_rep)
        # Create batched func for converting cylinder offsets to mesh vertex offsets
        self._cyl_to_vert_offset_fn = torch.vmap(
            partial(cyl_to_vert_offset, resolution=resolution, split=split)
        )

    def forward(
        self, r_offset: Tensor, h_offset: Tensor, offsets: dict[str, Tensor] = {}
    ) -> SurfaceMesh:
        """Apply radial, height and other attribute offsets to cylinder meshes if given.

        Args:
            r_offset: Radial offsets to apply to each cylinder mesh. Shape is (n_rep,).
            h_offset: Height offsets to apply to each cylinder mesh. Shape is (n_rep,).
            offsets: Optional dictionary of offset tensors to apply to other mesh attributes.

        Returns:
            New mesh created by applying given offsets and copying unchanged attributes from the
            original mesh.
        """
        offsets["vertices"] = self._cyl_to_vert_offset_fn(r_offset, h_offset)
        return super().forward(offsets)

    __call__ = forward
