from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from kaolin.render.lighting import SgLightingParameters

if TYPE_CHECKING:
    from kaolin.render.camera import Camera
    from kaolin.rep import SurfaceMesh
    from torch import Tensor, device

    from ..rasterizer import Fragments


class Shader(ABC):
    """Base shader class for Kaolin rendering.

    Args:
        device: Optional device for default initializations. Default value is `cpu`.
        cameras: Optional batched cameras if needed by shader. Default value is `None`.
        lights: Optional lighting parameters if neeeded by shader. Default value is `None`.
    """

    def __init__(
        self,
        device: str | device = "cpu",
        cameras: Camera | None = None,
        lights: SgLightingParameters | None = None,
    ):
        self._cameras = cameras
        self._lights = lights if lights is not None else SgLightingParameters().to(device=device)

    @abstractmethod
    def forward(self, fragments: Fragments, mesh: SurfaceMesh, **kwargs) -> Tensor:
        """Render shader output from rasterization outputs and mesh.

        Args:
            fragments: Rasterization outputs.
            mesh: Batched representation of meshes to render.
            kwargs: Optional overrides for rendering parameters.

        Returns:
            Rendering output. Shape is (B, H, W, shader_dim).
        """
        pass

    __call__ = forward

    def to(self, device: str | device) -> "Shader":
        if self._cameras is not None:
            self._cameras = self._cameras.to(device=device)
        self._lights = self._lights.to(device=device)
        return self
