from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    look_at_view_transform,
)
from pytorch3d.renderer.mesh.shader import HardDepthShader

from vc_core.dr.pytorch3d.mesh import CylinderMesh
from vc_core.dr.pytorch3d.shader import ComposeShader, SoftSilhouetteShader

Devices = [torch.device("cpu")]
Devices = Devices + [torch.device("cuda")] if torch.cuda.is_available() else Devices


@pytest.mark.unit
@pytest.mark.parametrize("device", Devices)
def test_soft_silhouette_shader(device: torch.device) -> None:
    # Create cylinder meshes
    radius, height = 0.2, 1.0
    n_rep = 10
    mesh_repeat = CylinderMesh(radius, height, n_rep=n_rep).to(device)

    # Create renderer using soft silhouette shader
    img_size = 256
    camera = FoVPerspectiveCameras(device=device)
    raster_settings = RasterizationSettings(image_size=img_size, blur_radius=0, faces_per_pixel=1)
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=0, gamma=0)).to(device),
    )

    # Test `forward()` method
    distance, elevation, azimuth = 1, 50, 30
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
    R.requires_grad, T.requires_grad = True, True
    silhouette: torch.Tensor = silhouette_renderer(mesh_repeat.mesh, R=R, T=T)
    assert silhouette.shape == (n_rep, img_size, img_size, 1)
    assert silhouette.requires_grad

    # store output visualization
    path = Path(__file__).parent / "outputs"
    path.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(5, 5))
    img = plt.imshow(silhouette[0].detach().cpu())
    plt.colorbar(img)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(path / "soft_silhouette.png", dpi=150)
    plt.close()


@pytest.mark.unit
@pytest.mark.parametrize("device", Devices)
def test_compose_shader(device: torch.device) -> None:
    # Create cylinder meshes
    radius, height = 0.2, 1.0
    n_rep = 10
    mesh_repeat = CylinderMesh(radius, height, n_rep=n_rep).to(device)

    # Create renderer using composed silhouette and depth shader
    img_size = 256
    camera = FoVPerspectiveCameras(device=device, znear=0.1, zfar=10)
    raster_settings = RasterizationSettings(image_size=img_size, blur_radius=0, faces_per_pixel=1)
    shader = ComposeShader(
        [
            SoftSilhouetteShader(blend_params=BlendParams(sigma=0, gamma=0)),
            HardDepthShader(cameras=camera),
        ]
    ).to(device)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings), shader=shader
    )

    # Test `forward()` method
    distance, elevation, azimuth = 1, 50, 30
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
    R.requires_grad, T.requires_grad = True, True
    image: torch.Tensor = renderer(mesh_repeat.mesh, R=R, T=T)
    assert image.shape == (n_rep, img_size, img_size, 2)
    assert image.requires_grad

    # store output visualization
    path = Path(__file__).parent / "outputs"
    path.mkdir(exist_ok=True, parents=True)
    silhouette = image.detach().cpu()[0, :, :, 0]
    depth = image.detach().cpu()[0, :, :, 1]
    depth /= depth.max()
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(silhouette)
    axes[0].set_title("Silhouette")
    axes[0].grid(False)
    axes[1].imshow(depth, norm="log")
    axes[1].set_title("Normalized Depth (Log-Scale) ")
    axes[1].grid(False)
    plt.tight_layout()
    plt.savefig(path / "compose.png", dpi=150)
    plt.close()
