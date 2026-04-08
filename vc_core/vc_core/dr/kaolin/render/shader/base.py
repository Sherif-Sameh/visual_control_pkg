from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from kaolin.render.camera import Camera
    from kaolin.render.lighting import SgLightingParameters
    from kaolin.rep import SurfaceMesh
    from torch import Tensor, device

    from ..rasterizer import Fragments


class Shader(ABC):
    """Base shader class for Kaolin rendering.

    Derived classes must provide a definition for the `forward()` method to produce their rendering
    outputs. Optionally, they can provide pre-rasterization hook function by setting their
    `_pre_hook` attribute. If they do, they must also set the `_feature_dim` attribute appropriately
    with the dimensionality of the features they added to the rasterizer's `face_features`.

    Args:
        cameras: Optional batched cameras if needed by shader. Default value is `None`.
        lights: Optional lighting parameters if needed by shader. Default value is `None`.
    """

    def __init__(self, cameras: Camera | None = None, lights: SgLightingParameters | None = None):
        self._cameras = cameras
        self._lights = lights
        self._pre_hook = None
        self._feature_dim = 0

    @property
    def feature_dim(self) -> int:
        """Get the feature dimension added to `face_features` by the registered pre-hook."""
        return self._feature_dim

    @property
    def pre_hook(self) -> Callable[[Tensor, SurfaceMesh], Tensor] | None:
        """Get the pre-rasterization hook function for the shader."""
        return self._pre_hook

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
        if self._lights is not None:
            self._lights = self._lights.to(device=device)
        return self
