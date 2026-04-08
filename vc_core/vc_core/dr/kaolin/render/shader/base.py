from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from kaolin.render.camera import Camera
    from kaolin.render.lighting import SgLightingParameters
    from kaolin.rep import SurfaceMesh
    from torch import Tensor, device

    from ..rasterizer import Fragments


@dataclass
class BlendParams:
    """Blending parameters for rendering soft mask.

    All default parameter values match those of the `kaolin.render.mesh.dibr_soft_mask` function.
    """

    sigmainv: float = 7000.0
    """Smoothness term for computing the soft mask, the higher the sharper. Default value is 7000."""

    boxlen: float = 0.02
    """Margin over bounding box of faces which will threshold which pixels will be influenced by the face.
    Default value is 0.02.
    """

    knum: int = 30
    """Maximum number of faces that can influence one pixel. Default value is 30."""

    multiplier: int = 1000
    """Muliplier for coordinates used internally to avoid numeric issues. Default value is 1000."""


class Shader(ABC):
    """Base shader class for Kaolin rendering.

    Derived classes must provide a definition for the `forward()` method to produce their rendering
    outputs. Optionally, they can provide pre-rasterization hook function by setting their
    `_pre_hook` attribute. If they do, they must also set the `_feature_dim` attribute appropriately
    with the dimensionality of the features they added to the rasterizer's `face_features`.

    Args:
        cameras: Optional batched cameras if needed by shader. Default value is `None`.
        lights: Optional lighting parameters if needed by shader. Default value is `None`.
        blend_params: Optional parameters for alpha blending if needed by shader. Default value is
            `None`.
    """

    def __init__(
        self,
        cameras: Camera | None = None,
        lights: SgLightingParameters | None = None,
        blend_params: BlendParams | None = None,
    ):
        self._cameras = cameras
        self._lights = lights
        self._blend_params = blend_params
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
