from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

import kaolin as kal
import torch
from kaolin.render.lighting import SgLightingParameters

from vc_core.dr.kaolin.utils import generate_pinhole_rays_batched

from .base import Shader

if TYPE_CHECKING:
    from kaolin.render.camera import Camera
    from kaolin.rep import SurfaceMesh
    from torch import Tensor, device

    from ..rasterizer import Fragments


class HardColorAmbientShader(Shader):
    """Hard color shader considering only ambient lighting for Kaolin rendering.

    Since the shader is hard, gradients can only propagate from the foreground pixels.

    For efficiency, we restrict ourselves to meshes with a single material per mesh instance.
    That is `mesh.materials[0][0]["map_Kd"]`. Also, the light is assumed to be the same across
    all mesh instances.

    The shader's pre-hook function must be registered with the rasterizer to enable uv
    rasterization outputs.

    Args:
        ambient: Optional color for ambient lightings. Default value is `None`.
        raw_texture: Whether mesh texture is uint8 (H, W, 3) and needs conversion or is already
            processed. If `False`, the texture must be float [0, 1] (3, H, W). Default value is
            `True`.
        uvs_origin: Whether the origin of mesh uvs follows the OpenGL (bottom-left) or Kaolin
            (top-left) convention. Default value is `OpenGL`.
    """

    def __init__(
        self,
        ambient: Tensor | None = None,
        raw_texture: bool = True,
        uvs_origin: Literal["OpenGL", "Kaolin"] = "OpenGL",
    ) -> None:
        self._ambient = ambient if ambient is not None else torch.ones(3, dtype=torch.float32)
        assert self._ambient.ndim == 1, f"Light parameters must be 1-dim, got {self._ambient.ndim}."
        self._get_albedo = _build_get_albedo(raw_texture, uvs_origin)
        self._pre_hook = self._pre_hook_fn
        self._feature_dim = 2

    def forward(self, fragments: Fragments, mesh: SurfaceMesh, **kwargs) -> Tensor:
        """Render hard color shader with ambient lighting from rasterization outputs and mesh.

        Args:
            fragments: Rasterization outputs.
            mesh: Batched representation of meshes to render.
            kwargs: Optional overrides for rendering parameters. These include `ambient`, which can
                override the value passed to the constructor.

        Returns:
            Rendering output. Shape is (B, H, W, 3).
        """
        # apply overrides
        ambient = kwargs.get("ambient", self._ambient).view(3)

        # apply rendering
        albedo = self._get_albedo(fragments.features_image[1], mesh)
        mask = fragments.faces_image != -1
        color = torch.zeros_like(albedo)
        color[mask] = torch.clamp(albedo[mask] * ambient, 0.0, 1.0)
        return color

    __call__ = forward

    def to(self, device: str | device) -> "HardColorAmbientShader":
        self._ambient = self._ambient.to(device=device)
        return self

    @staticmethod
    def _pre_hook_fn(face_vertices_camera: Tensor, mesh: SurfaceMesh) -> Tensor:
        """Pre-hook function for hard color ambient shader to register mesh uvs."""
        return mesh.face_uvs


class HardColorDiffuseSH9Shader(Shader):
    """Hard color shader considering only diffuse lighting for Kaolin rendering.

    Light is modeled through Spherical Harmonics of base 3 and is parameterized by its
    direction vector in Cartesian space. For more on the SH-based lighting model, refer
    to `kaolin.render.lighting.sh9_diffuse`.

    Since the shader is hard, gradients can only propagate from the foreground pixels.

    For efficiency, we restrict ourselves to meshes with a single material per mesh instance.
    That is `mesh.materials[0][0]["map_Kd"]`. Also, the light is assumed to be the same across
    all mesh instances.

    The shader's pre-hook function must be registered with the rasterizer to enable uv
    rasterization outputs.

    Args:
        azimuth: Optional azimuth angles for light direction. Default value is `None`.
        elevation: Optional elevation angles for light direction. Default value is `None`.
        direction: Optional unit vectors for light direction. Default value is `None`.
        degrees: Whether the angles are specified in degrees or radians.
        raw_texture: Whether mesh texture is uint8 (H, W, 3) and needs conversion or is already
            processed. If `False`, the texture must be float [0, 1] (3, H, W). Default value is
            `True`.
        uvs_origin: Whether the origin of mesh uvs follows the OpenGL (bottom-left) or Kaolin
            (top-left) convention. Default value is `OpenGL`.
    """

    def __init__(
        self,
        azimuth: Tensor | None = None,
        elevation: Tensor | None = None,
        direction: Tensor | None = None,
        degrees: bool = True,
        raw_texture: bool = True,
        uvs_origin: Literal["OpenGL", "Kaolin"] = "OpenGL",
    ) -> None:
        if direction is None:
            azimuth = azimuth if azimuth is not None else torch.zeros(1, dtype=torch.float32)
            elevation = elevation if elevation is not None else torch.zeros_like(azimuth)
            if degrees:
                azimuth = torch.pi / 180.0 * azimuth
                elevation = torch.pi / 180.0 * elevation
            direction = torch.stack(kal.ops.coords.spherical2cartesian(azimuth, elevation), dim=-1)
            direction = direction.squeeze(0)
        self._direction = direction
        assert self._direction.ndim == 1, (
            f"Light parameters must be 1-dim, got {self._direction.ndim}."
        )
        self._get_albedo = _build_get_albedo(raw_texture, uvs_origin)
        self._pre_hook = _register_uvs_and_normals
        self._feature_dim = 5

    def forward(self, fragments: Fragments, mesh: SurfaceMesh, **kwargs) -> Tensor:
        """Render hard color shader with diffuse (SH9) lighting from rasterization outputs and mesh.

        Args:
            fragments: Rasterization outputs.
            mesh: Batched representation of meshes to render.
            kwargs: Optional overrides for rendering parameters. These include `direction`, which can
                override the value passed to the constructor.

        Returns:
            Rendering output. Shape is (B, H, W, 3).
        """
        # apply overrides
        direction = kwargs.get("direction", self._direction).view(3)

        # apply rendering
        uv_map = fragments.features_image[1][..., :2]
        normal_map = fragments.features_image[1][..., 2:]
        albedo = self._get_albedo(uv_map, mesh)
        mask = fragments.faces_image != -1
        color = torch.zeros_like(albedo)
        color[mask] = torch.clamp(
            kal.render.lighting.sh9_diffuse(direction, normal_map[mask], albedo[mask]), 0.0, 1.0
        )
        return color

    __call__ = forward

    def to(self, device: str | device) -> "HardColorAmbientShader":
        self._direction = self._direction.to(device=device)
        return self


class HardColorDiffuseSGFittedShader(Shader):
    """Hard color shader considering only diffuse lighting for Kaolin rendering.

    Light is modeled through fitted approximation of Spherical Gaussians and is parameterized
    by its direction vector in Cartesian space and SG parameters. For more on the fitted SG-based
    lighting model, refer to `kaolin.render.lighting.sg_diffuse_fitted`.

    Since the shader is hard, gradients can only propagate from the foreground pixels.

    For efficiency, we restrict ourselves to meshes with a single material per mesh instance.
    That is `mesh.materials[0][0]["map_Kd"]`. Also, the light is assumed to be the same across
    all mesh instances but can be modelled with multiple SGs.

    The shader's pre-hook function must be registered with the rasterizer to enable uv
    rasterization outputs.

    Args:
        azimuth: Optional azimuth angles for SG light directions. Default value is `None`.
        elevation: Optional elevation angles for SG light directions. Default value is `None`.
        degrees: Whether the angles are specified in degrees or radians.
        lights: Optional SG lighting parameters. Default value is `None`.
        raw_texture: Whether mesh texture is uint8 (H, W, 3) and needs conversion or is already
            processed. If `False`, the texture must be float [0, 1] (3, H, W). Default value is
            `True`.
        uvs_origin: Whether the origin of mesh uvs follows the OpenGL (bottom-left) or Kaolin
            (top-left) convention. Default value is `OpenGL`.
    """

    def __init__(
        self,
        azimuth: Tensor | None = None,
        elevation: Tensor | None = None,
        degrees: bool = True,
        lights: SgLightingParameters | None = None,
        raw_texture: bool = True,
        uvs_origin: Literal["OpenGL", "Kaolin"] = "OpenGL",
    ) -> None:
        super().__init__()
        lights = lights if lights is not None else SgLightingParameters()
        if azimuth is not None and elevation is not None:
            azimuth = azimuth.view(-1, 3)
            elevation = elevation.view(-1, 3)
            if degrees:
                azimuth = torch.pi / 180.0 * azimuth
                elevation = torch.pi / 180.0 * elevation
            direction = torch.stack(kal.ops.coords.spherical2cartesian(azimuth, elevation), dim=-1)
            lights = SgLightingParameters(lights.amplitude, direction, lights.sharpness)
        self._lights = lights
        self._get_albedo = _build_get_albedo(raw_texture, uvs_origin)
        self._pre_hook = _register_uvs_and_normals
        self._feature_dim = 5

    def forward(self, fragments: Fragments, mesh: SurfaceMesh, **kwargs) -> Tensor:
        """Render hard color shader with diffuse (fitted SG) lighting from rasterization outputs and mesh.

        Args:
            fragments: Rasterization outputs.
            mesh: Batched representation of meshes to render.
            kwargs: Optional overrides for rendering parameters. These include `lights`, which can
                override the value passed to the constructor.

        Returns:
            Rendering output. Shape is (B, H, W, 3).
        """
        # apply overrides
        lights: SgLightingParameters = kwargs.get("lights", self._lights)

        # apply rendering
        uv_map = fragments.features_image[1][..., :2]
        normal_map = fragments.features_image[1][..., 2:]
        albedo = self._get_albedo(uv_map, mesh)
        mask = fragments.faces_image != -1
        color = torch.zeros_like(albedo)
        color[mask] = torch.clamp(
            kal.render.lighting.sg_diffuse_fitted(
                lights.amplitude, lights.direction, lights.sharpness, normal_map[mask], albedo[mask]
            ),
            0.0,
            1.0,
        )
        return color

    __call__ = forward


class HardColorSpecularSGFittedShader(Shader):
    """Hard color shader considering diffuse and specular lighting for Kaolin rendering.

    Light is modeled through fitted approximation of Spherical Gaussians and is parameterized
    by its direction vector in Cartesian space and SG parameters. Material properties outside the
    diffuse albedo include the specular albedo and material roughness of the mesh. Both of these
    can be passed during initialization or forwad passes. For more on the fitted SG-based lighting
    model, refer to `kaolin.render.lighting.sg_diffuse_fitted`. For more on the calculation of the
    specular lighting component, refer to `kaolin.render.lighting.sg_warp_specular_term`.

    Since the shader is hard, gradients can only propagate from the foreground pixels.

    For efficiency, we restrict ourselves to meshes with a single material per mesh instance.
    That is `mesh.materials[0][0]["map_Kd"]`. Mesh specular albedo and surface roughness are
    assumed to be global properties, not spatially-varying. Also, the light is assumed to be
    the same across all mesh instances but can be modelled with multiple SGs.

    The shader's pre-hook function must be registered with the rasterizer to enable uv
    rasterization outputs.

    Args:
        azimuth: Optional azimuth angles for SG light directions. Default value is `None`.
        elevation: Optional elevation angles for SG light directions. Default value is `None`.
        degrees: Whether the angles are specified in degrees or radians.
        spec_albedo: Optional specular albedo. Default value is `None`.
        roughness: Optional material roughness. Default value is `None`.
        cameras: Optional batched cameras. Default value is `None`.
        lights: Optional SG lighting parameters. Default value is `None`.
        raw_texture: Whether mesh texture is uint8 (H, W, 3) and needs conversion or is already
            processed. If `False`, the texture must be float [0, 1] (3, H, W). Default value is
            `True`.
        uvs_origin: Whether the origin of mesh uvs follows the OpenGL (bottom-left) or Kaolin
            (top-left) convention. Default value is `OpenGL`.
    """

    def __init__(
        self,
        azimuth: Tensor | None = None,
        elevation: Tensor | None = None,
        degrees: bool = True,
        spec_albedo: Tensor | None = None,
        roughness: Tensor | None = None,
        cameras: Camera | None = None,
        lights: SgLightingParameters | None = None,
        raw_texture: bool = True,
        uvs_origin: Literal["OpenGL", "Kaolin"] = "OpenGL",
    ) -> None:
        super().__init__(cameras=cameras)
        lights = lights if lights is not None else SgLightingParameters()
        if azimuth is not None and elevation is not None:
            azimuth = azimuth.view(-1, 3)
            elevation = elevation.view(-1, 3)
            if degrees:
                azimuth = torch.pi / 180.0 * azimuth
                elevation = torch.pi / 180.0 * elevation
            direction = torch.stack(kal.ops.coords.spherical2cartesian(azimuth, elevation), dim=-1)
            lights = SgLightingParameters(lights.amplitude, direction, lights.sharpness)
        self._lights = lights
        self._spec_albedo = spec_albedo if spec_albedo is not None else torch.ones(3)
        self._roughness = roughness if roughness is not None else torch.ones(1)
        self._get_albedo = _build_get_albedo(raw_texture, uvs_origin)
        self._pre_hook = _register_uvs_and_normals
        self._feature_dim = 5

    def forward(self, fragments: Fragments, mesh: SurfaceMesh, **kwargs) -> Tensor:
        """Render hard color shader with diffuse and specular (fitted SG) lighting from
        rasterization outputs and mesh.

        Args:
            fragments: Rasterization outputs.
            mesh: Batched representation of meshes to render.
            kwargs: Optional overrides for rendering parameters. These include `cameras`, `lights`,
                `spec_albedo` and `roughness`, all of which can override the values passed to the
                constructor.

        Returns:
            Rendering output. Shape is (B, H, W, 3).
        """
        # apply overrides
        cameras: Camera = kwargs.get("cameras", self._cameras)
        lights: SgLightingParameters = kwargs.get("lights", self._lights)
        spec_albedo: Tensor = kwargs.get("spec_albedo", self._spec_albedo).view(1, 1, 1, 3)
        roughness: Tensor = kwargs.get("roughness", self._roughness).view(1, 1, 1)

        # apply rendering
        uv_map = fragments.features_image[1][..., :2]
        normal_map = fragments.features_image[1][..., 2:]
        albedo = self._get_albedo(uv_map, mesh)
        mask = fragments.faces_image != -1
        spec_albedo = spec_albedo.expand_as(albedo)
        roughness = roughness.expand_as(mask)
        _, rays_dir = generate_pinhole_rays_batched(cameras)
        rays_dir = -rays_dir.view_as(albedo)
        color = torch.zeros_like(albedo)
        color[mask] += kal.render.lighting.sg_diffuse_fitted(
            lights.amplitude, lights.direction, lights.sharpness, normal_map[mask], albedo[mask]
        )
        color[mask] += kal.render.lighting.sg_warp_specular_term(
            lights.amplitude,
            lights.direction,
            lights.sharpness,
            normal_map[mask],
            roughness[mask],
            rays_dir[mask],
            spec_albedo[mask],
        )
        color[mask] = torch.clamp(color[mask], 0.0, 1.0)
        return color

    __call__ = forward

    def to(self, device: str | device) -> "HardColorSpecularSGFittedShader":
        if self._cameras is not None:
            self._cameras = self._cameras.to(device=device)
        if self._lights is not None:
            self._lights = self._lights.to(device=device)
        if self._spec_albedo is not None:
            self._spec_albedo = self._spec_albedo.to(device=device)
        if self._roughness is not None:
            self._roughness = self._roughness.to(device=device)
        return self


# region Helpers


def _build_get_albedo(
    raw_texture: bool, uvs_origin: Literal["OpenGL", "Kaolin"]
) -> Callable[[Tensor, SurfaceMesh], Tensor]:
    """Build function to get albedo from rasterization output and mesh texture."""
    is_opengl = uvs_origin == "OpenGL"

    def get_albedo(uv_map: Tensor, mesh: SurfaceMesh) -> Tensor:
        """Get albedo from rasterization output and mesh texture."""
        # get and process mesh texture
        texture = mesh.materials[0][0]["map_Kd"]
        if raw_texture:
            texture = _convert_raw_texture(texture)

        # process mesh rasterized uvs
        if is_opengl:
            uv_map[..., 1] = 1 - uv_map[..., 1]

        # sample texture
        B = uv_map.shape[0]
        albedo = kal.render.mesh.texture_mapping(uv_map, texture[None].expand([B, -1, -1, -1]))
        return albedo

    return get_albedo


def _convert_raw_texture(texture: Tensor) -> Tensor:
    """Convert a raw uint8 (H, W, 3) texture to a float (3, H, W) texture."""
    return texture.float().permute(2, 0, 1) / 255.0


def _register_uvs_and_normals(face_vertices_camera: Tensor, mesh: SurfaceMesh) -> Tensor:
    """Pre-hook function for diffuse/specular shaders to register mesh uvs and normals."""
    return torch.cat([mesh.face_uvs, mesh.face_normals], dim=-1)
