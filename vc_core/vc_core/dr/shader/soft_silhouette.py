from __future__ import annotations

from typing import TYPE_CHECKING

import pytorch3d.renderer.mesh.shader as shader

if TYPE_CHECKING:
    from pytorch3d.renderer.blending import BlendParams
    from pytorch3d.renderer.mesh.rasterizer import Fragments
    from pytorch3d.structures import Meshes
    from torch import Tensor


class SoftSilhouetteShader(shader.SoftSilhouetteShader):
    """Extends the PyTorch3D `SoftSilhouetteShader` to only return the alpha channel from its output.

    By default, the PyTorch3D `SoftSilhouetteShader` will return a 4-channel output of RGBA values.
    However, those RGB values are invalid anyways since they're blended from a tensor of all ones
    not the true RGB values of the rasterized fragments.

    For more on the shader, refer to `pytorch3d.renderer.mesh.shader.SoftSilhouetteShader`.

    Args:
        blend_params: Parameters for sigmoid alpha blending.
    """

    def __init__(self, blend_params: BlendParams | None = None) -> None:
        super().__init__(blend_params=blend_params)

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> Tensor:
        """Render the silhouette using sigmoid alpha blending for given meshes.

        Args:
            fragments: Fragments returned by the rasterizer.
            meshes: Unused by silhouette shader.

        Returns:
            Rendered silhouette image (alpha channel only). Shape is (N, H, W, 1).
        """
        img_out = super().forward(fragments, meshes, **kwargs)
        return img_out[..., 3:]
