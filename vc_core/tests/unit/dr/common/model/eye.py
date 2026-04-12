from __future__ import annotations

from itertools import product

import pytest
import torch

from vc_core.dr.common.model import (
    EyePoseModel,
    EyePoseTextureHashEncoderModel,
    EyePoseTextureMipmapModel,
    EyePoseTextureModel,
    HashEncoder2DCfg,
)
from vc_core.dr.kaolin.utils import look_at_view_transform

Devices = [torch.device("cpu")]
Devices = Devices + [torch.device("cuda")] if torch.cuda.is_available() else Devices
Modes = ["nearest", "bilinear", "nearest-exact"]


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
    pos_m, rot_m, amb_m = model()
    assert all([m.requires_grad for m in [pos_m, rot_m, amb_m]])
    assert pos_m.shape == (n_rep, 3)
    assert rot_m.shape == (n_rep, 3, 3)
    assert amb_m.shape == (3,)

    # Test `resample_params()` method
    model.resample_params(pos_m[0], rot_m[0, :, -1])
    pos_m_new, rot_m_new, amb_m_new = model()
    assert all([m.requires_grad for m in [pos_m_new, rot_m_new, amb_m_new]])
    assert pos_m_new.shape == (n_rep, 3)
    assert rot_m_new.shape == (n_rep, 3, 3)
    assert amb_m_new.shape == (3,)
    assert not torch.allclose(pos_m, pos_m_new)
    assert not torch.allclose(rot_m, rot_m_new)
    assert torch.allclose(amb_m, amb_m_new)


@pytest.mark.unit
@pytest.mark.parametrize("device", Devices)
def test_eye_pose_texture_model(device: torch.device) -> None:
    # Create eye pose texture model
    H, W = 256, 256
    n_view = 3
    distance, elevation, azimuth = 1, 50, 30
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
    text_rgb = torch.full((3,), 0.8, device=device)
    model = EyePoseTextureModel(T[0], R[0, :, -1], (H, W), text_rgb, n_view=n_view)

    # Test `forward()` method
    pos_m, rot_m, text_m = model()
    assert all([m.requires_grad for m in [pos_m, rot_m, text_m]])
    assert pos_m.shape == (n_view, 3)
    assert rot_m.shape == (n_view, 3, 3)
    assert text_m.shape == (3, H, W)
    assert torch.allclose(text_m, text_rgb.view(3, 1, 1))


@pytest.mark.unit
@pytest.mark.parametrize("device,mode", product(Devices, Modes))
def test_eye_pose_texture_mipmap_model(device: torch.device, mode: str) -> None:
    # Create eye pose texture model
    H, W = 256, 256
    n_view, n_level = 3, 5
    distance, elevation, azimuth = 1, 50, 30
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
    text_rgb = torch.full((3,), 0.8, device=device)
    model = EyePoseTextureMipmapModel(
        T[0], R[0, :, -1], (H, W), text_rgb, n_view=n_view, n_level=n_level, mode=mode
    )

    # Test `forward()` method
    pos_m, rot_m, text_m = model()
    assert all([m.requires_grad for m in [pos_m, rot_m, text_m]])
    assert pos_m.shape == (n_view, 3)
    assert rot_m.shape == (n_view, 3, 3)
    assert text_m.shape == (3, H, W)
    assert torch.allclose(text_m, text_rgb.view(3, 1, 1))


@pytest.mark.unit
@pytest.mark.parametrize("device", Devices)
def test_eye_pose_texture_hash_encoder_model(device: torch.device) -> None:
    # Create eye pose texture model
    n_view = 3
    res = 512
    enc_cfg = HashEncoder2DCfg(finest_res=res)
    mlp_n_layer, mlp_n_feature = 2, 64
    distance, elevation, azimuth = 1, 50, 30
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
    model = EyePoseTextureHashEncoderModel(
        T[0],
        R[0, :, -1],
        n_view=n_view,
        enc_cfg=enc_cfg,
        mlp_n_layer=mlp_n_layer,
        mlp_n_feature=mlp_n_feature,
    )

    # Test `forward()` method
    pos_m, rot_m, text_m = model()
    assert all([m.requires_grad for m in [pos_m, rot_m, text_m]])
    assert pos_m.shape == (n_view, 3)
    assert rot_m.shape == (n_view, 3, 3)
    assert text_m.shape == (3, res, res)
