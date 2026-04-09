from __future__ import annotations

from typing import TYPE_CHECKING

import kaolin as kal
import torch

if TYPE_CHECKING:
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
