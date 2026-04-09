from __future__ import annotations

from typing import TYPE_CHECKING

import kaolin as kal
import torch
from kaolin.render.camera import CameraFOV
from kaolin.render.camera.raygen import _to_ndc_coords, generate_centered_pixel_coords

if TYPE_CHECKING:
    from kaolin.render.camera import Camera
    from torch import Tensor, device


def camera_position_from_spherical_angles(
    distance: float | Tensor,
    elevation: float | Tensor,
    azimuth: float | Tensor,
    degrees: bool = True,
    device: str | device = "cpu",
) -> Tensor:
    """
    Calculate the location of the camera based on the distance away from the target point, the
    elevation and azimuth angles.

    All inputs are converted to tensors and broadcast against each other.

    Args:
        distance: Distance of the camera from the object.
        elevation: Elevation angle of the camera relative to the object.
        azimuth: Azimuth angle of the camera relative to the object.
        degrees: Whether the angles are specified in degrees or radians.
        device: Device for tensors to be placed on.

    Returns:
        Camera position relative to the object. Shape is (N, 3).
    """
    args = [
        torch.tensor(arg) if not torch.is_tensor(arg) else arg
        for arg in [distance, elevation, azimuth]
    ]
    args = [arg.view(-1, 1).to(device=device) for arg in args]
    dist, elev, azim = args
    if degrees:
        elev = torch.pi / 180.0 * elev
        azim = torch.pi / 180.0 * azim
    x, y, z = kal.ops.coords.spherical2cartesian(azim, elev, distance=dist)
    camera_position = torch.cat([x, y, z], dim=1)
    return camera_position


def generate_pinhole_rays_batched(
    cameras: Camera, coords_grid: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Batched variant of the `generate_pinhole_rays` function in `kaolin.render.camera.raygen`.

    For efficient batching, we generate rays only for one camera then apply each camera's
    extrinsics to transform them to world frame.

    This function assumes that the principal point (the pinhole location) is specified by a
    displacement (camera.x0, camera.y0) in pixel coordinates from the center of the image.

    The Kaolin camera class does not enforce a coordinate space for how the principal point is specified,
    so users will need to make sure that the correct principal point conventions are followed for
    the cameras passed into this function.

    Args:
        cameras: Batched camera object to generate rays for.
        coords_grid: Pixel grid of ray-intersecting coordinates of shape `(H, W, 2)`. Coordinates
            integer parts represent the pixel :math:`(i, j)` coords, and the fraction part of
            `[0,1]` represents the location within the pixel itself. For example, a coordinate of
            `(0.5, 0.5)` represents the center of the top-left pixel. Shared across all cameras.

    Returns:
        tuple of tensors representing the generated pinhole rays for the camera, as ray origins and
            ray direction tensors respectively of shape `(B, HxW, 3)` each.
    """
    if coords_grid is None:
        pixel_y, pixel_x = generate_centered_pixel_coords(
            cameras.width, cameras.height, device=cameras.device
        )
    else:
        assert cameras.device == coords_grid.device, (
            f"Expected camera and coords_grid to be on the same device, "
            f"but found {cameras.device} and {coords_grid.device}."
        )
        pixel_y, pixel_x = coords_grid[..., 0], coords_grid[..., 1]

    # coords_grid should remain immutable (a new tensor is implicitly created here)
    pixel_x = pixel_x.to(device=cameras.device, dtype=cameras.dtype)
    pixel_y = pixel_y.to(device=cameras.device, dtype=cameras.dtype)

    # Account for principal point (offsets from the center)
    pixel_x = pixel_x - cameras.x0[0]
    pixel_y = pixel_y + cameras.y0[0]

    # pixel values are now in range [-1, 1], both tensors are of shape res_y x res_x
    pixel_x, pixel_y = _to_ndc_coords(pixel_x, pixel_y, cameras)

    ray_dir = torch.stack(
        (
            pixel_x * cameras.tan_half_fov(CameraFOV.HORIZONTAL)[0],
            -pixel_y * cameras.tan_half_fov(CameraFOV.VERTICAL)[0],
            -torch.ones_like(pixel_x),
        ),
        dim=-1,
    )
    ray_dir = ray_dir.reshape(-1, 3)  # Flatten grid rays to 1D array
    ray_orig = torch.zeros_like(ray_dir)

    # Transform from camera to world coordinates
    ray_orig, ray_dir = cameras.extrinsics.inv_transform_rays(ray_orig, ray_dir)
    ray_dir /= torch.linalg.norm(ray_dir, dim=-1, keepdim=True)
    return ray_orig, ray_dir
