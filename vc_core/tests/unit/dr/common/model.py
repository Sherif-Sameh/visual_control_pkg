from __future__ import annotations

import pytest
import torch
from pytorch3d.renderer import look_at_view_transform

from vc_core.dr.common.model import CylinderModel

Devices = [torch.device("cpu")]
Devices = Devices + [torch.device("cuda")] if torch.cuda.is_available() else Devices


@pytest.mark.unit
@pytest.mark.parametrize("device", Devices)
def test_cylinder_model(device: torch.device) -> None:
    # Create cylinder params
    height = 1.0
    n_rep = 10

    # Create cylinder model
    distance, elevation, azimuth = 1, 50, 30
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
    pos = torch.randn((n_rep, 3), dtype=torch.float32, device=device) + T
    z_dir = torch.randn((n_rep, 3), dtype=torch.float32, device=device) + R[..., -1]
    height = torch.randn((n_rep,), dtype=torch.float32, device=device) + height
    model = CylinderModel(pos, z_dir, radius=None, height=height, n_rep=n_rep)

    # Test `forward()` method
    pos_m, rot_m, r_off_m, h_off_m = model()
    assert pos_m.requires_grad and rot_m.requires_grad
    assert not r_off_m.requires_grad and h_off_m.requires_grad
    assert torch.allclose(pos, pos_m)
    assert torch.allclose(rot_m[..., -1], z_dir / torch.linalg.norm(z_dir, dim=-1, keepdim=True))
    assert torch.allclose(r_off_m, torch.zeros_like(r_off_m))
    assert torch.allclose(h_off_m, torch.zeros_like(h_off_m))
