from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor, device


def camera_position_from_spherical_angles(
    distance: float,
    elevation: float,
    azimuth: float,
    degrees: bool = True,
    device: str | device = "cpu",
) -> Tensor:
    """
    Calculate the location of the camera based on the distance away from the target point, the
    elevation and azimuth angles.

    From `pytorch3d.renderer.cameras`.

    Args:
        distance: Distance of the camera from the object.
        elevation: Elevation angle of the camera relative to the object.
        azimuth: Azimuth angle of the camera relative to the object.
        degrees: Whether the angles are specified in degrees or radians.
        device: Device for new tensors to be placed on.

    Returns:
        Camera position relative to the object. Shape is (1, 3).
    """
    args = [torch.tensor(arg).view(1, 1) for arg in [distance, elevation, azimuth]]
    dist, elev, azim = args
    if degrees:
        elev = torch.pi / 180.0 * elev
        azim = torch.pi / 180.0 * azim
    x = dist * torch.cos(elev) * torch.sin(azim)
    y = dist * torch.sin(elev)
    z = dist * torch.cos(elev) * torch.cos(azim)
    camera_position = torch.cat([x, y, z], dim=1).to(device=device)
    return camera_position
