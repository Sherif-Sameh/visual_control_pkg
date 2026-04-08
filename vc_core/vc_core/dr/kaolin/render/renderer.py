from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaolin.rep import SurfaceMesh
    from torch import Tensor, device

    from .rasterizer import MeshRasterizer
    from .shader import Shader


class MeshRenderer:
    """Renderer for batched meshes using Kaolin."""

    def __init__(self, rasterizer: MeshRasterizer, shader: Shader):
        # Set attributes
        self._rasterizer = rasterizer
        self._shader = shader

        # Register shader pre-rasterization hook with rasterizer
        if self._shader.pre_hook is not None:
            self._rasterizer.register_hook(self._shader.pre_hook)

    def forward(self, mesh: SurfaceMesh, **kwargs) -> Tensor:
        """Rasterize mesh and render shader outputs.

        Args:
            mesh: Batched representation of meshes to rasterize.
            kwargs: Optional arguments to pass to rasterizer and shader `forward()` methods.

        Returns:
            Rendering output. Shape is (B, H, W, shader_dim).
        """
        fragments = self._rasterizer(mesh, *kwargs)
        output = self._shader(fragments, mesh, **kwargs)
        return output

    __call__ = forward

    def to(self, device: str | device) -> "MeshRenderer":
        self._rasterizer = self._rasterizer.to(device=device)
        self._shader = self._shader.to(device=device)
        return self
