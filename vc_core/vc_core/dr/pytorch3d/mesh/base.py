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
    optionally new textures to the wrapped mesh.

    Args:
        mesh: PyTorch3D mesh to wrap.
        n_rep: Number of times to repeat the wrapped mesh. Default value is 1.
    """

    def __init__(self, mesh: Meshes, *, n_rep: int = 1):
        super().__init__()
        self._mesh = mesh.extend(n_rep)

    def __len__(self) -> int:
        """Get length of the wrapped mesh."""
        return len(self._mesh)

    @property
    def mesh(self) -> Meshes:
        """Get a reference to the wrapped mesh."""
        return self._mesh

    @mesh.setter
    def mesh(self, value: Meshes) -> None:
        """Set the value of the wrapped mesh."""
        self._mesh = value

    def forward(self, offset: Tensor, texture: TexturesBase | None = None) -> Meshes:
        """Applying offsets to mesh and new texture if given.

        Args:
            offset: Offsets to apply to mesh vertices. Shape should match that of `mesh.verts_packed`
                (N * n_rep, 3), where N is the number of vertices in the original mesh.
            texture: Optional texture to apply to the new mesh before returning it. Default value
                is `None`.

        Returns:
            New meshes created by applying given vertex offsets and texture if given.
        """
        new_mesh = self._mesh.offset_verts(offset)
        if texture is not None:
            new_mesh.textures = texture
        return new_mesh

    def to(self, device: str | device) -> "Mesh":
        self._mesh = self._mesh.to(device, copy=False)
        return self
