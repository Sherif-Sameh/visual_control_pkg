from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import Shader

if TYPE_CHECKING:
    from kaolin.render.camera import Camera
    from kaolin.rep import SurfaceMesh
    from torch import Tensor

    from ..rasterizer import Fragments


class HardDepthShader(Shader):
    """Hard depth shader for Kaolin rendering.

    Since the shader is hard, gradients can only propagate from the foreground pixels.

    The shader's pre-hook function must be registered with the rasterizer to enable depth
    rasterization outputs.

    Args:
        cameras: Optional batched cameras. Default value is `None`.
    """

    def __init__(self, cameras: Camera | None = None) -> None:
        super().__init__(cameras=cameras)
        self._pre_hook = self._pre_hook_fn
        self._feature_dim = 1

    def forward(self, fragments: Fragments, mesh: SurfaceMesh, **kwargs) -> Tensor:
        """Render hard depth from rasterization outputs and mesh.

        Args:
            fragments: Rasterization outputs.
            mesh: Batched representation of meshes to render.
            kwargs: Optional overrides for rendering parameters. These include `cameras`, and `zfar`.
                `cameras` can override the value passed to the constructor. `zfar` can override the
                value provided by the camera.

        Returns:
            Rendering output. Shape is (B, H, W, 1).
        """
        # apply overrides
        zfar = kwargs.get("zfar", kwargs.get("cameras", self._cameras).intrinsics.far)

        # apply rendering
        invalid = (fragments.faces_image < 0).unsqueeze(-1)
        depth = torch.where(invalid, zfar, fragments.features_image[1])
        return depth

    __call__ = forward

    @staticmethod
    def _pre_hook_fn(face_vertices_camera: Tensor, mesh: SurfaceMesh) -> Tensor:
        """Pre-hook function for depth shader."""
        return face_vertices_camera[..., -1:]
