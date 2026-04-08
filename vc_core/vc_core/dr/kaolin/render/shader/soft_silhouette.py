from __future__ import annotations

from typing import TYPE_CHECKING

import kaolin as kal

from .base import BlendParams, Shader

if TYPE_CHECKING:
    from kaolin.rep import SurfaceMesh
    from torch import Tensor

    from ..rasterizer import Fragments


class SoftSilhouetteShader(Shader):
    """Soft silhouette shader for Kaolin rendering.

    The method used to blend pixels is described in the paper `"Learning to Predict 3D Objects with an
    Interpolation-based Differentiable Renderer"` (https://arxiv.org/abs/1908.01210).

    Blending only affects background pixels, foreground pixels backprop gradients through interpolation
    rather than smoothing.

    Args:
        blend_params: Optional parameters for alpha blending. Initialized with defaults if not set.
            Default value is `None`.
    """

    def __init__(self, blend_params: BlendParams | None = None) -> None:
        blend_params = blend_params if blend_params is not None else BlendParams()
        super().__init__(blend_params=blend_params)

    def forward(self, fragments: Fragments, mesh: SurfaceMesh, **kwargs) -> Tensor:
        """Render soft silhouette mask from rasterization outputs and mesh.

        Args:
            fragments: Rasterization outputs.
            mesh: Batched representation of meshes to render.
            kwargs: Optional overrides for rendering parameters. These include `blend_params`,
                which can override the value passed to the constructor.

        Returns:
            Soft silhouette output. Shape is (B, H, W, 1).
        """
        # apply overrides
        blend_params: BlendParams = kwargs.get("blend_params", self._blend_params)

        # apply rendering
        soft_mask = kal.render.mesh.dibr_soft_mask(
            fragments.face_vertices_image,
            fragments.faces_image,
            sigmainv=blend_params.sigmainv,
            boxlen=blend_params.boxlen,
            knum=blend_params.knum,
            multiplier=blend_params.multiplier,
        )
        return soft_mask

    __call__ = forward
