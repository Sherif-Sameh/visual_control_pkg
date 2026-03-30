from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from pytorch3d.renderer.mesh.shader import ShaderBase

if TYPE_CHECKING:
    from pytorch3d.renderer.mesh.rasterizer import Fragments
    from pytorch3d.structures import Meshes
    from torch import Tensor, device


class ComposeShader(ShaderBase):
    """Compose multiple shaders into a single shader.

    The outputs of the composed shaders are concatenated along their channel dimension (dim=-1) in
    the same order that they are given in.

    Args:
        shaders: List of shaders to compose.
    """

    def __init__(self, shaders: list[ShaderBase]):
        super().__init__()
        self._shaders = shaders

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> Tensor:
        """Render outputs of all composed shaders and return combined output.

        Shader outputs are concatenated along their channel dimension (dim=-1) in the same order
        that they were given in.
        """
        return torch.cat([shader(fragments, meshes, **kwargs) for shader in self._shaders], dim=-1)

    def to(self, device: str | device) -> "ComposeShader":
        """Move composed shaders to device."""
        self = super().to(device)
        for i in range(len(self._shaders)):
            self._shaders[i] = self._shaders[i].to(device)
        return self
