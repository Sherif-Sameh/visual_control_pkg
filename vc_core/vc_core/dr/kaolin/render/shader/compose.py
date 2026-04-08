from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import Shader

if TYPE_CHECKING:
    from kaolin.rep import SurfaceMesh
    from torch import LongTensor, Tensor, device


class ComposeShader(Shader):
    """Compose multiple shaders into a single shader.

    The outputs of the composed shaders are concatenated along their channel dimension (dim=-1) in
    the same order that they are given in.

    Args:
        shaders: List of shaders to compose.
    """

    def __init__(self, shaders: list[Shader]):
        self._shaders = shaders

    def forward(
        self, features_image: Tensor, faces_image: LongTensor, mesh: SurfaceMesh, **kwargs
    ) -> Tensor:
        """Render outputs of all composed shaders and return combined output.

        Shader outputs are concatenated along their channel dimension (dim=-1) in the same order
        that they were given in.
        """
        return torch.cat(
            [shader(features_image, faces_image, mesh, **kwargs) for shader in self._shaders],
            dim=-1,
        )

    def to(self, device: str | device) -> "ComposeShader":
        """Move composed shaders to device."""
        for i in range(len(self._shaders)):
            self._shaders[i] = self._shaders[i].to(device)
        return self
