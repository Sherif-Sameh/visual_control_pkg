from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from pytorch3d.renderer.mesh import TexturesBase
    from pytorch3d.structures import Meshes
    from torch import Tensor, device


class Mesh(nn.Module):
    """Base class for wrapping PyTorch3D Meshes.

    Extends `torch.nn.Module`. The `forward()` method is used to apply vertex offsets and
    optionally new textures to the wrapped mesh and return it.

    Args:
        mesh: PyTorch3D mesh to wrap.
    """

    def __init__(self, mesh: Meshes):
        super().__init__()
        self._mesh = mesh

    @property
    def mesh(self) -> Meshes:
        """Get a copy of the cloned mesh."""
        return self._mesh.clone()

    def forward(self, offset: Tensor, texture: TexturesBase | None = None) -> Meshes:
        """Applying offsets to mesh and new texture if given.

        Args:
            offset: Offsets to apply to mesh vertices. Shape is (N, 3), where N is the number of
                vertices in the original mesh.
            texture: Optional texture to apply to the new mesh before returning it. Default value
                is `None`.

        Returns:
            New mesh created by applying given vertex offsets and texture if given.
        """
        new_mesh = self._mesh.offset_verts(offset)
        if texture is not None:
            new_mesh.textures = texture
        return new_mesh

    def to(self, device: str | device) -> Mesh:
        self._mesh = self._mesh.to(device, copy=False)
        return self
