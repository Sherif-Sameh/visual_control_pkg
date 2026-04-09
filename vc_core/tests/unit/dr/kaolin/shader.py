from __future__ import annotations

import logging
from itertools import product
from pathlib import Path

import kaolin as kal
import matplotlib.pyplot as plt
import pytest
import torch
from kaolin.render.camera import Camera

from vc_core.dr.kaolin.mesh import CylinderMesh, ObjMesh
from vc_core.dr.kaolin.render import (
    BlendParams,
    ComposeShader,
    HardColorAmbientShader,
    HardColorDiffuseSGFittedShader,
    HardColorDiffuseSH9Shader,
    HardColorSpecularSGFittedShader,
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
    n_rep = 2
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
@pytest.mark.parametrize("device,backend", product(Devices, Backends))
def test_hard_depth_shader(device: torch.device, backend: str) -> None:
    if device.type == "cpu":
        return  # Kaolin doesn't support CPU rendering
    # Create cylinder meshes
    radius, height = 0.2, 1.0
    n_rep = 2
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

    # store output visualization
    path = Path(__file__).parent / "outputs"
    path.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(5, 5))
    img = plt.imshow(depth[0].detach().cpu())
    plt.colorbar(img)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(path / "hard_depth.png", dpi=150)
    plt.close()


@pytest.mark.unit
@pytest.mark.parametrize("device,backend", product(Devices, Backends))
def test_hard_color_ambient_shader(caplog, device: torch.device, backend: str) -> None:
    if device.type == "cpu":
        return  # Kaolin doesn't support CPU rendering
    # Create obj meshes
    n_rep = 2
    scale = 20.0
    path = Path(__file__).parent / "samples/eye_mesh/eye.obj"
    with caplog.at_level(logging.ERROR):
        mesh_repeat = ObjMesh(path, with_materials=True, with_normals=True, n_rep=n_rep).to(device)
    mesh_repeat.mesh.vertices *= scale
    F = mesh_repeat.mesh.vertices.shape[1]
    mesh_repeat.mesh.vertex_features = torch.zeros([n_rep, F, 0], device=device)

    # Setup camera
    img_size = 256
    distance, elevation, azimuth = 1.0, 60, 30
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

    # Create renderer using hard color ambient shader
    raster_settings = RasterizationSettings(image_size=img_size, backend=backend)
    color_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=HardColorAmbientShader(ambient=None, raw_texture=True, uvs_origin="OpenGL"),
    ).to(device=device)

    # Test `forward()` method
    ambient = torch.ones(3, requires_grad=True, device=device) * 0.85
    color: torch.Tensor = color_renderer(mesh_repeat.mesh, ambient=ambient)
    assert color.shape == (n_rep, img_size, img_size, 3)
    assert color.requires_grad

    # store output visualization
    path = Path(__file__).parent / "outputs"
    path.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(color[0].detach().cpu())
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(path / "hard_color_ambient.png", dpi=150)
    plt.close()


@pytest.mark.unit
@pytest.mark.parametrize("device,backend", product(Devices, Backends))
def test_hard_color_diffuse_sh9_shader(caplog, device: torch.device, backend: str) -> None:
    if device.type == "cpu":
        return  # Kaolin doesn't support CPU rendering
    # Create obj meshes
    n_rep = 2
    scale = 20.0
    path = Path(__file__).parent / "samples/eye_mesh/eye.obj"
    with caplog.at_level(logging.ERROR):
        mesh_repeat = ObjMesh(path, with_materials=True, with_normals=True, n_rep=n_rep).to(device)
    mesh_repeat.mesh.vertices *= scale
    F = mesh_repeat.mesh.vertices.shape[1]
    mesh_repeat.mesh.vertex_features = torch.zeros([n_rep, F, 0], device=device)

    # Setup camera
    img_size = 256
    distance, elevation, azimuth = 1.0, 60, 30
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

    # Setup lights
    azimuth = torch.tensor([-35.0], device=device)
    elevation = torch.tensor([60.0], device=device)
    intensity = torch.ones(3, device=device) * 0.85

    # Create renderer using hard color diffuse (SH9) shader
    raster_settings = RasterizationSettings(image_size=img_size, backend=backend)
    color_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=HardColorDiffuseSH9Shader(
            azimuth=azimuth,
            elevation=elevation,
            intensity=intensity,
            degrees=True,
            raw_texture=True,
            uvs_origin="OpenGL",
        ),
    ).to(device=device)

    # Test `forward()` method
    intensity = intensity.clone()
    intensity.requires_grad = True
    color: torch.Tensor = color_renderer(mesh_repeat.mesh, intensity=intensity)
    assert color.shape == (n_rep, img_size, img_size, 3)
    assert color.requires_grad

    # store output visualization
    path = Path(__file__).parent / "outputs"
    path.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(color[0].detach().cpu())
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(path / "hard_color_diffuse_sh9.png", dpi=150)
    plt.close()


@pytest.mark.unit
@pytest.mark.parametrize("device,backend", product(Devices, Backends))
def test_hard_color_diffuse_sg_fitted_shader(caplog, device: torch.device, backend: str) -> None:
    if device.type == "cpu":
        return  # Kaolin doesn't support CPU rendering
    # Create obj meshes
    n_rep = 2
    scale = 20.0
    path = Path(__file__).parent / "samples/eye_mesh/eye.obj"
    with caplog.at_level(logging.ERROR):
        mesh_repeat = ObjMesh(path, with_materials=True, with_normals=True, n_rep=n_rep).to(device)
    mesh_repeat.mesh.vertices *= scale
    F = mesh_repeat.mesh.vertices.shape[1]
    mesh_repeat.mesh.vertex_features = torch.zeros([n_rep, F, 0], device=device)

    # Setup camera
    img_size = 256
    distance, elevation, azimuth = 1.0, 60, 30
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

    # Setup lights
    azimuth = torch.tensor([-35.0] * 2, device=device)
    elevation = torch.tensor([60.0] * 2, device=device)
    amplitude = torch.ones(3, device=device) * 4.0
    lights = kal.render.lighting.SgLightingParameters(
        amplitude=torch.stack([amplitude, torch.zeros_like(amplitude)], dim=0)
    )

    # Create renderer using hard color diffuse (SH9) shader
    raster_settings = RasterizationSettings(image_size=img_size, backend=backend)
    color_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=HardColorDiffuseSGFittedShader(
            azimuth=azimuth,
            elevation=elevation,
            lights=lights,
            degrees=True,
            raw_texture=True,
            uvs_origin="OpenGL",
        ),
    ).to(device=device)

    # Test `forward()` method
    lights = color_renderer._shader._lights.to(device=device)  # clone lights
    lights.amplitude.requires_grad = True
    lights.direction.requires_grad = True
    color: torch.Tensor = color_renderer(mesh_repeat.mesh, lights=lights)
    assert color.shape == (n_rep, img_size, img_size, 3)
    assert color.requires_grad

    # store output visualization
    path = Path(__file__).parent / "outputs"
    path.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(color[0].detach().cpu())
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(path / "hard_color_diffuse_sg_fitted.png", dpi=150)
    plt.close()


@pytest.mark.unit
@pytest.mark.parametrize("device,backend", product(Devices, Backends))
def test_hard_color_specular_sg_fitted_shader(caplog, device: torch.device, backend: str) -> None:
    if device.type == "cpu":
        return  # Kaolin doesn't support CPU rendering
    # Create obj meshes
    n_rep = 2
    scale = 20.0
    path = Path(__file__).parent / "samples/eye_mesh/eye.obj"
    with caplog.at_level(logging.ERROR):
        mesh_repeat = ObjMesh(path, with_materials=True, with_normals=True, n_rep=n_rep).to(device)
    mesh_repeat.mesh.vertices *= scale
    F = mesh_repeat.mesh.vertices.shape[1]
    mesh_repeat.mesh.vertex_features = torch.zeros([n_rep, F, 0], device=device)

    # Setup camera
    img_size = 256
    distance, elevation, azimuth = 1.0, 60, 30
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

    # Setup additional material properties for specular lighting
    spec_albedo = torch.ones(3, device=device) * 0.15
    roughness = torch.tensor([0.3], device=device)

    # Setup lights
    azimuth = torch.tensor([-35.0] * 2, device=device)
    elevation = torch.tensor([60.0] * 2, device=device)
    amplitude = torch.ones(3, device=device) * 4.0
    lights = kal.render.lighting.SgLightingParameters(
        amplitude=torch.stack([amplitude, torch.zeros_like(amplitude)], dim=0)
    )

    # Create renderer using hard color diffuse (SH9) shader
    raster_settings = RasterizationSettings(image_size=img_size, backend=backend)
    color_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=HardColorSpecularSGFittedShader(
            azimuth=azimuth,
            elevation=elevation,
            spec_albedo=spec_albedo,
            roughness=roughness,
            cameras=camera,
            lights=lights,
            degrees=True,
            raw_texture=True,
            uvs_origin="OpenGL",
        ),
    ).to(device=device)

    # Test `forward()` method
    spec_albedo = spec_albedo.clone()
    roughness = roughness.clone()
    spec_albedo.requires_grad, roughness.requires_grad = True, True
    color: torch.Tensor = color_renderer(
        mesh_repeat.mesh, spec_albedo=spec_albedo, roughness=roughness
    )
    assert color.shape == (n_rep, img_size, img_size, 3)
    assert color.requires_grad

    # store output visualization
    path = Path(__file__).parent / "outputs"
    path.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(color[0].detach().cpu())
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(path / "hard_color_specular_sg_fitted.png", dpi=150)
    plt.close()


@pytest.mark.unit
@pytest.mark.parametrize("device,backend", product(Devices, Backends))
def test_compose_shader(device: torch.device, backend: str) -> None:
    if device.type == "cpu":
        return  # Kaolin doesn't support CPU rendering
    # Create cylinder meshes
    radius, height = 0.2, 1.0
    n_rep = 2
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
    axes[1].imshow(depth)
    axes[1].set_title("Normalized Depth")
    axes[1].grid(False)
    plt.tight_layout()
    plt.savefig(path / "compose.png", dpi=150)
    plt.close()
