from __future__ import annotations

from itertools import product

import matplotlib.pyplot as plt
import pytest
import torch
from kaolin.render.camera import Camera

from vc_core.dr.kaolin.mesh import CylinderMesh
from vc_core.dr.kaolin.render import (
    BlendParams,
    ComposeShader,
    HardDepthShader,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftSilhouetteShader,
)
from vc_core.dr.kaolin.utils import camera_position_from_spherical_angles

Devices = [torch.device("cpu")]
Devices = Devices + [torch.device("cuda")] if torch.cuda.is_available() else Devices
Backends = ["cuda", "nvdiffrast", "nvdiffrast_fwd"]


@pytest.mark.unit
@pytest.mark.parametrize("device,backend", product(Devices, Backends))
def test_soft_silhouette_shader(device: torch.device, backend: str) -> None:
    if device.type == "cpu":
        return  # Kaolin doesn't support CPU rendering
    # Create cylinder meshes
    radius, height = 0.2, 1.0
    n_rep = 10
    mesh_repeat = CylinderMesh(radius, height, split=10, n_rep=n_rep).to(device)
    F = mesh_repeat.mesh.vertices.shape[1]
    mesh_repeat.mesh.vertex_features = torch.zeros([n_rep, F, 1], device=device)

    # Setup camera
    img_size = 256
    distance, elevation, azimuth = 1.0, 50, 30
    eye = camera_position_from_spherical_angles(distance, elevation, azimuth)
    at, up = torch.zeros(1, 3), torch.tensor([0.0, 1.0, 0.0]).view(1, 3)
    camera = Camera.from_args(
        eye=eye,
        at=at,
        up=up,
        fov=60 * torch.pi / 180,
        width=img_size,
        height=img_size,
        dtype=torch.float32,
    )
    camera = Camera.cat([camera] * n_rep).to(device=device)

    # Create renderer using soft silhouette shader
    raster_settings = RasterizationSettings(image_size=img_size, backend=backend)
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=BlendParams()),
    ).to(device=device)

    # Test `forward()` method
    R = camera.extrinsics.R.clone()
    T = camera.extrinsics.t.clone()
    R.requires_grad, T.requires_grad = True, True
    silhouette: torch.Tensor = silhouette_renderer(mesh_repeat.mesh, R=R, T=T)
    assert silhouette.shape == (n_rep, img_size, img_size, 1)
    assert silhouette.requires_grad


@pytest.mark.unit
@pytest.mark.parametrize("device,backend", product(Devices, Backends))
def test_hard_depth_shader(device: torch.device, backend: str) -> None:
    if device.type == "cpu":
        return  # Kaolin doesn't support CPU rendering
    # Create cylinder meshes
    radius, height = 0.2, 1.0
    n_rep = 10
    mesh_repeat = CylinderMesh(radius, height, split=10, n_rep=n_rep).to(device)
    F = mesh_repeat.mesh.vertices.shape[1]
    mesh_repeat.mesh.vertex_features = torch.zeros([n_rep, F, 0], device=device)

    # Setup camera
    img_size = 256
    distance, elevation, azimuth = 1.0, 50, 30
    eye = camera_position_from_spherical_angles(distance, elevation, azimuth)
    at, up = torch.zeros(1, 3), torch.tensor([0.0, 1.0, 0.0]).view(1, 3)
    camera = Camera.from_args(
        eye=eye,
        at=at,
        up=up,
        fov=60 * torch.pi / 180,
        width=img_size,
        height=img_size,
        near=0.1,
        far=10.0,
        dtype=torch.float32,
    )
    camera = Camera.cat([camera] * n_rep).to(device=device)

    # Create renderer using hard depth shader
    raster_settings = RasterizationSettings(image_size=img_size, backend=backend)
    depth_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=HardDepthShader(cameras=camera),
    ).to(device=device)

    # Test `forward()` method
    R = camera.extrinsics.R.clone()
    T = camera.extrinsics.t.clone()
    R.requires_grad, T.requires_grad = True, True
    depth: torch.Tensor = depth_renderer(mesh_repeat.mesh, R=R, T=T)
    assert depth.shape == (n_rep, img_size, img_size, 1)
    assert depth.requires_grad


@pytest.mark.unit
@pytest.mark.parametrize("device,backend", product(Devices, Backends))
def test_compose_shader(device: torch.device, backend: str) -> None:
    if device.type == "cpu":
        return  # Kaolin doesn't support CPU rendering
    # Create cylinder meshes
    radius, height = 0.2, 1.0
    n_rep = 10
    mesh_repeat = CylinderMesh(radius, height, split=10, n_rep=n_rep).to(device)
    F = mesh_repeat.mesh.vertices.shape[1]
    mesh_repeat.mesh.vertex_features = torch.zeros([n_rep, F, 0], device=device)

    # Setup camera
    img_size = 256
    distance, elevation, azimuth = 1.0, 50, 30
    eye = camera_position_from_spherical_angles(distance, elevation, azimuth)
    at, up = torch.zeros(1, 3), torch.tensor([0.0, 1.0, 0.0]).view(1, 3)
    camera = Camera.from_args(
        eye=eye,
        at=at,
        up=up,
        fov=60 * torch.pi / 180,
        width=img_size,
        height=img_size,
        near=0.1,
        far=10.0,
        dtype=torch.float32,
    )
    camera = Camera.cat([camera] * n_rep).to(device=device)

    # Create renderer using composed silhouette and hard depth shaders
    raster_settings = RasterizationSettings(image_size=img_size, backend=backend)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=ComposeShader(
            [SoftSilhouetteShader(blend_params=BlendParams()), HardDepthShader(cameras=camera)]
        ),
    ).to(device=device)

    # Test `forward()` method
    R = camera.extrinsics.R.clone()
    T = camera.extrinsics.t.clone()
    R.requires_grad, T.requires_grad = True, True
    image: torch.Tensor = renderer(mesh_repeat.mesh, R=R, T=T)
    assert image.shape == (n_rep, img_size, img_size, 2)
    assert image.requires_grad

    # visualize output
    silhouette = image.detach().cpu()[0, :, :, 0]
    depth = image.detach().cpu()[0, :, :, 1]
    depth /= depth.max()
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(silhouette)
    axes[0].set_title("Silhouette")
    axes[0].grid(False)
    axes[1].imshow(depth)
    axes[1].set_title("Normalized Depth")
    axes[1].grid(False)
    plt.tight_layout()
    plt.show()
