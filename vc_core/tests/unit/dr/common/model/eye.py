from __future__ import annotations

import pytest
import torch

from vc_core.dr.common.model import EyePoseModel
from vc_core.dr.kaolin.utils import look_at_view_transform

Devices = [torch.device("cpu")]
Devices = Devices + [torch.device("cuda")] if torch.cuda.is_available() else Devices


@pytest.mark.unit
@pytest.mark.parametrize("device", Devices)
def test_eye_pose_model(device: torch.device) -> None:
    # Create eye pose model
    n_rep = 10
    distance, elevation, azimuth = 1, 50, 30
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
    T_sigma = torch.tensor([0.05, 0.1, 0.2], device=device)
    model = EyePoseModel(T[0], R[0, :, -1], T_sigma, 0.05, n_rep=n_rep)

    # Test `forward()` method
    pos_m, rot_m = model()
    assert pos_m.requires_grad and rot_m.requires_grad
    assert pos_m.shape == (n_rep, 3)
    assert rot_m.shape == (n_rep, 3, 3)

    # Test `resample_params()` method
    model.resample_params(pos_m[0], rot_m[0, :, -1])
    pos_m_new, rot_m_new = model()
    assert pos_m_new.requires_grad and rot_m_new.requires_grad
    assert pos_m_new.shape == (n_rep, 3)
    assert rot_m_new.shape == (n_rep, 3, 3)
    assert not torch.allclose(pos_m, pos_m_new)
    assert not torch.allclose(rot_m, rot_m_new)
