from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

from .base import Shader

if TYPE_CHECKING:
    from kaolin.rep import SurfaceMesh
    from torch import Tensor, device

    from ..rasterizer import Fragments


class ComposeShader(Shader):
    """Compose multiple shaders into a single shader.

    The outputs of the composed shaders are concatenated along their channel dimension (dim=-1) in
    the same order that they are given in.

    Args:
        shaders: List of shaders to compose.
    """

    def __init__(self, shaders: list[Shader]):
        self._shaders = shaders
        self._pre_hook = self._build_pre_hook(shaders)
        self._idxs = np.cumsum([0] + [shader.feature_dim for shader in shaders])
        self._feature_dim = np.max(self._idxs)

    def forward(self, fragments: Fragments, mesh: SurfaceMesh, **kwargs) -> Tensor:
        """Render outputs of all composed shaders and return combined output.

        Shader outputs are concatenated along their channel dimension (dim=-1) in the same order
        that they were given in.
        """
        out = []
        for i, shader in enumerate(self._shaders):
            fragments_shader = fragments._replace(
                features_image=(
                    fragments.features_image[0],
                    fragments.features_image[1][..., self._idxs[i] : self._idxs[i + 1]],
                )
            )
            out.append(shader(fragments_shader, mesh, **kwargs))
        return torch.cat(out, dim=-1)

    __call__ = forward

    def to(self, device: str | device) -> "ComposeShader":
        """Move composed shaders to device."""
        for i in range(len(self._shaders)):
            self._shaders[i] = self._shaders[i].to(device)
        return self

    def _build_pre_hook(
        self, shaders: list[Shader]
    ) -> Callable[[Tensor, SurfaceMesh], Tensor] | None:
        """Build composed pre-hook function."""
        pre_hooks = [shader.pre_hook for shader in shaders if shader.pre_hook is not None]

        def _pre_hook(face_vertices_camera: Tensor, mesh: SurfaceMesh) -> Tensor:
            return torch.cat(
                [pre_hook(face_vertices_camera, mesh) for pre_hook in pre_hooks], dim=-1
            )

        pre_hook = _pre_hook if len(pre_hooks) > 0 else None
        return pre_hook
