from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn.functional as F
from kaolin.render.camera import CameraFOV
from kaolin.render.camera.raygen import _to_ndc_coords, generate_centered_pixel_coords

if TYPE_CHECKING:
    from kaolin.render.camera import Camera
    from torch import Tensor, device


def look_at_view_transform(
    dist: float | Tensor,
    elev: float | Tensor,
    azim: float | Tensor,
    degrees: bool = True,
    eye: Sequence[float] | Tensor | None = None,
    at=((0, 0, 0),),  # (1, 3)
    up=((0, 1, 0),),  # (1, 3)
    device: str | device = "cpu",
) -> tuple[Tensor, Tensor]:
    """
    This function returns a rotation and translation matrix to apply the 'Look At'
    transformation from world -> view coordinates

    Camera utility function from `pytorch3d` [0]. The function is modified from the original
    `pytorch3d` version such that the rotation matrix follows the OpenGL convention such that
    the camera's Z-axis points away from the scene.

    All inputs are converted to tensors and broadcast against each other.

    Args:
        dist: Distance of the camera from the object.
        elev: Angle in degrees or radians. This is the angle between the vector from the object to
            the camera, and the horizontal plane y = 0 (xz-plane).
        azim: Angle in degrees or radians. The vector from the object to the camera is projected
            onto a horizontal plane y = 0. azim is the angle between the projected vector and a
            reference vector at (0, 0, 1) on the reference plane (the horizontal plane).
        degrees: Boolean flag to indicate if the elevation and azimuth angles are specified in
            degrees or radians.
        eye: Position of the camera(s) in world coordinates. If eye is not `None`, it will override
            the camera position derived from dist, elev, azim.
        up: Direction of the x axis in the world coordinate system.
        at: Position of the object(s) in world coordinates.

    Returns:
        2-element tuple containing

        - **R**: the rotation to apply to the points to align with the camera.
        - **T**: the translation to apply to the points to align with the camera.

    References:
    [0] https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/renderer/cameras.html#look_at_view_transform
    """
    if eye is not None:
        args = [torch.tensor(arg) if not torch.is_tensor(arg) else arg for arg in [eye, at, up]]
        args = [arg.view(-1, 3).to(dtype=torch.float32, device=device) for arg in args]
        eye, at, up = args
        C = eye
    else:
        args = [torch.tensor(arg) if not torch.is_tensor(arg) else arg for arg in [at, up]]
        args = [arg.view(-1, 3).to(dtype=torch.float32, device=device) for arg in args]
        at, up = args
        C = (
            camera_position_from_spherical_angles(dist, elev, azim, degrees=degrees, device=device)
            + at
        )

    R = look_at_rotation(C, at, up, device=device)
    T = -torch.bmm(R, C[:, :, None])[:, :, 0]
    return R, T


def look_at_rotation(
    camera_position: Sequence[float] | Tensor,
    at=((0, 0, 0),),
    up=((0, 1, 0),),
    device: str | device = "cpu",
) -> Tensor:
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.

    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.

    Camera utility function from `pytorch3d` [0]. The function is modified from the original
    `pytorch3d` version such that the rotation matrix follows the OpenGL convention such that
    the camera's Z-axis points away from the scene.

    All inputs are converted to tensors and broadcast against each other.

    Args:
        camera_position: Position of the camera in world coordinates
        at: Position of the object in world coordinates
        up: Vector specifying the up direction in the world coordinate frame.

    Returns:
        R: (N, 3, 3) batched rotation matrices

    References:
    [0]: https://pytorch3d.readthedocs.io/en/latest/modules/renderer/cameras.html#pytorch3d.renderer.cameras.look_at_rotation
    """
    args = [
        torch.tensor(arg) if not torch.is_tensor(arg) else arg for arg in [camera_position, at, up]
    ]
    args = [arg.view(-1, 3).to(dtype=torch.float32, device=device) for arg in args]
    camera_position, at, up = args
    z_axis = -F.normalize(at - camera_position, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(dim=1, keepdim=True)
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.stack((x_axis, y_axis, z_axis), dim=1)
    return R


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

    Camera utility function from `pytorch3d` [0].

    All inputs are converted to tensors and broadcast against each other.

    Args:
        distance: Distance of the camera from the object.
        elevation: Elevation angle of the camera relative to the object.
        azimuth: Azimuth angle of the camera relative to the object.
        degrees: Whether the angles are specified in degrees or radians.
        device: Device for tensors to be placed on.

    Returns:
        Camera position relative to the object. Shape is (N, 3).

    References:
    [0] https://pytorch3d.readthedocs.io/en/latest/modules/renderer/cameras.html#pytorch3d.renderer.cameras.camera_position_from_spherical_angles
    """
    args = [
        torch.tensor(arg) if not torch.is_tensor(arg) else arg
        for arg in [distance, elevation, azimuth]
    ]
    args = [arg.view(-1, 1).to(dtype=torch.float32, device=device) for arg in args]
    dist, elev, azim = args
    if degrees:
        elev = torch.pi / 180.0 * elev
        azim = torch.pi / 180.0 * azim
    x = dist * torch.cos(elev) * torch.sin(azim)
    y = dist * torch.sin(elev)
    z = dist * torch.cos(elev) * torch.cos(azim)
    camera_position = torch.cat([x, y, z], dim=1)
    return camera_position


def transform_from_rotation_translation(rot: Tensor, trans: Tensor) -> Tensor:
    """Combine rotation and translation into a 4x4 homogeneous transformation matrix.

    Args:
        rot: Rotation matrices. Shape is (..., 3, 3).
        trans: Translation vectors. Shape is (..., 3).

    Returns:
        homogeneous transformation matrices. Shape is (..., 4, 4).
    """
    device = rot.device
    H = torch.cat([rot, trans.unsqueeze(-1)], dim=-1)
    bottom_row = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    bottom_row = bottom_row.expand_as(H[..., 0, :]).unsqueeze(-2)
    return torch.cat([H, bottom_row], dim=-2)


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
