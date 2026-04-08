from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple

import kaolin as kal

if TYPE_CHECKING:
    from kaolin.render.camera import Camera
    from kaolin.rep import SurfaceMesh
    from torch import LongTensor, Tensor, device


@dataclass
class RasterizationSettings:
    """Configuration for mesh rasterizer.

    All default parameter values match those of the `kaolin.render.mesh.rasterize` function.
    """

    image_size: int | tuple[int, int] = (256, 256)
    """Image height and width in pixels. Must be multiples of 8. Default value is 256."""

    multiplier: int = 1000
    """Muliplier for coordinates used internally to avoid numeric issues. Default value is 1000."""

    eps: float = 1e-8
    """Epsilon value used during normalization of barycentric coordinates. Default value is 1e-8."""

    backend: Literal["cuda", "nvdiffrast", "nvdiffrast_fwd"] = "cuda"
    """Backend used for the rasterization. Default value is `cuda` (i.e. DIB-R)."""


class Fragments(NamedTuple):
    """Named tuple grouping all outputs of rasterization."""

    face_vertices_camera: Tensor
    """Face vertices in the camera frame. Shape is (B, F, 3, 3)."""

    face_vertices_image: Tensor
    """Face vertices in the image plane (in NDC). Shape is (B, F, 3, 2)."""

    features_image: Tensor
    """Rendered features of shape corresponding to the `face_features` attribute of the input mesh.
    Shape is (B, H, W, num_features).
    """

    faces_image: LongTensor
    """Rendered face indices where -1 corresponds to `None` (i.e. pixels that don't intersect any
    faces). Shape is (B, H, W).
    """


class MeshRasterizer:
    """Rasterizer for batched meshes using Kaolin's rasterization methods.

    Args:
        cameras: Optional batched cameras to transform points from world/mesh -> ndc frame. If not
            set, then it must be passed to the `forward` method. Default value is `None`.
        raster_settings: Optional parameters for rasterization. Initialized with defaults if not
            set. Default value is `None`.
    """

    def __init__(
        self, cameras: Camera | None = None, raster_settings: RasterizationSettings | None = None
    ):
        self._cameras = cameras
        self._settings = RasterizationSettings() if raster_settings is None else raster_settings
        if isinstance(self._settings.image_size, int):
            self._settings.image_size = (self._settings.image_size, self._settings.image_size)
        assert self._settings.image_size[0] % 8 == 0, "Image height must be a multiple of 8."
        assert self._settings.image_size[1] % 8 == 0, "Image width must be a multiple of 8."

    def forward(self, mesh: SurfaceMesh, **kwargs) -> Fragments:
        """Rasterize input batched meshes using Kaolin.

        Args:
            mesh: Batched representation of meshes to rasterize.
            kwargs: Optional overrides for rasterization parameters. These include `cameras`, `R`
                and `T`. `cameras` can override the instance passed to the contructor. `R` can
                override the rotation part only of the view matrix. `T` can override the
                translation part only of the view matrix.

        Returns:
            Named tuple of rasterization outputs.
        """
        # apply overrides
        cameras: Camera = kwargs.get("cameras", self._cameras)
        R: Tensor = kwargs.get("R", cameras.extrinsics.R).view(-1, 3, 3)
        T: Tensor = kwargs.get("T", cameras.extrinsics.t).view(-1, 3, 1)

        # prepare mesh vertices
        vertices_camera = kal.render.camera.rotate_translate_points(mesh.vertices, R, T)
        vertices_ndc = cameras.intrinsics.transform(vertices_camera)
        face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(vertices_camera, mesh.faces)
        face_vertices_image = kal.ops.mesh.index_vertices_by_faces(
            vertices_ndc[..., :2], mesh.faces
        )
        face_vertices_z = face_vertices_camera[..., -1]

        # apply rasterization
        features_image, faces_image = kal.render.mesh.rasterize(
            self._settings.image_size[0],
            self._settings.image_size[1],
            face_vertices_z,
            face_vertices_image,
            mesh.face_features,
            multiplier=self._settings.multiplier,
            eps=self._settings.eps,
            backend=self._settings.backend,
        )
        return Fragments(face_vertices_camera, face_vertices_image, features_image, faces_image)

    __call__ = forward

    def to(self, device: str | device) -> "MeshRasterizer":
        if self._cameras is not None:
            self._cameras = self._cameras.to(device=device)
        return self
