from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pytest
import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)

from vc_core.dr.pytorch3d.mesh import CylinderMesh, ObjMesh

if TYPE_CHECKING:
    from pytorch3d.structures import Meshes

Devices = [torch.device("cpu")]
Devices = Devices + [torch.device("cuda")] if torch.cuda.is_available() else Devices


@pytest.mark.unit
def test_obj_mesh() -> None:
    # Download test .obj model
    url = "https://dl.fbaipublicfiles.com/pytorch3d/data/teapot/teapot.obj"
    path = Path(__file__).parent / "teapot.obj"
    urllib.request.urlretrieve(url, path)

    # Create meshes from .obj file
    n_rep = 10
    mesh_single = ObjMesh(path, n_rep=1)
    mesh_repeat = ObjMesh(path, n_rep=n_rep)

    # Test `len()` method and `mesh` property
    assert len(mesh_single) == len(mesh_single.mesh) == 1
    assert len(mesh_repeat) == len(mesh_repeat.mesh) == 10

    # Test `to()` method
    if torch.device("cuda") in Devices:
        for d in ["cpu", "cuda"]:
            device = torch.device(d)
            mesh_single = mesh_single.to(device)
            mesh_repeat = mesh_repeat.to(d)
            assert mesh_single.mesh.device.type == d
            assert mesh_repeat.mesh.device.type == d

    # Test `forward()` method without texture
    offsets_single = torch.randn_like(mesh_single.mesh.verts_packed(), requires_grad=True)
    offsets_repeat = torch.randn_like(mesh_repeat.mesh.verts_packed(), requires_grad=True)
    meshes_single: Meshes = mesh_single(offsets_single)
    meshes_repeat: Meshes = mesh_repeat(offsets_repeat)
    for vert in meshes_single.verts_list():
        assert vert.requires_grad
    for vert in meshes_repeat.verts_list():
        assert vert.requires_grad
    assert meshes_single.textures is None
    assert meshes_repeat.textures is None

    # Test `forward()` method with a texture
    rgb = torch.ones_like(mesh_single.mesh.verts_packed())[None]
    texture_single = TexturesVertex(rgb)
    texture_repeat = TexturesVertex(rgb.repeat(n_rep, 1, 1))
    meshes_single: Meshes = mesh_single(offsets_single, texture=texture_single)
    meshes_repeat: Meshes = mesh_repeat(offsets_repeat, texture=texture_repeat)
    assert meshes_single.textures is texture_single
    assert meshes_repeat.textures is texture_repeat

    # Remove .obj model
    path.unlink(missing_ok=True)


@pytest.mark.unit
def test_cylinder_mesh() -> None:
    # Create cylinder meshes
    radius, height = 0.2, 1.0
    n_rep = 10
    mesh_single = CylinderMesh(radius, height, n_rep=1)
    mesh_repeat = CylinderMesh(radius, height, n_rep=n_rep)

    # Test `len()` method and `mesh` property
    assert len(mesh_single) == len(mesh_single.mesh) == 1
    assert len(mesh_repeat) == len(mesh_repeat.mesh) == 10

    # Test `to()` method
    if torch.device("cuda") in Devices:
        for d in ["cpu", "cuda"]:
            device = torch.device(d)
            mesh_single = mesh_single.to(device)
            mesh_repeat = mesh_repeat.to(d)
            assert mesh_single.mesh.device.type == d
            assert mesh_repeat.mesh.device.type == d

    # Test `forward()` method without texture
    device = mesh_single.mesh.device
    r_offset = torch.full((n_rep,), -0.1, dtype=torch.float32, device=device, requires_grad=True)
    h_offset = torch.full((n_rep,), 0.5, dtype=torch.float32, device=device, requires_grad=True)
    meshes_single: Meshes = mesh_single(r_offset[:1], h_offset[:1])
    meshes_repeat: Meshes = mesh_repeat(r_offset, h_offset)
    for mesh in [meshes_single, meshes_repeat]:
        r = torch.linalg.norm(mesh[0].verts_packed()[2:, :2], dim=-1)
        h = torch.abs(
            torch.max(mesh[0].verts_packed()[:, 2]) - torch.min(mesh[0].verts_packed()[:, 2])
        )
        assert torch.allclose(r, torch.full_like(r, radius + r_offset[0].item()))
        assert torch.allclose(h, torch.full_like(h, height + h_offset[0].item()))
        for vert in mesh.verts_list():
            assert vert.requires_grad
        assert mesh.textures is None

    # Test `forward()` method with a texture
    rgb = torch.ones_like(mesh_single.mesh.verts_packed())[None]
    rgb[:, :, 2] = 0
    texture_single = TexturesVertex(rgb)
    texture_repeat = TexturesVertex(rgb.repeat(n_rep, 1, 1))
    meshes_single: Meshes = mesh_single(r_offset[:1], h_offset[:1], texture=texture_single)
    meshes_repeat: Meshes = mesh_repeat(r_offset, h_offset, texture=texture_repeat)
    assert meshes_single.textures is texture_single
    assert meshes_repeat.textures is texture_repeat
    meshes_single = meshes_single.detach()
    meshes_repeat = meshes_repeat.detach()

    # visualize all four meshes
    images = render_meshes([mesh_single.mesh, mesh_repeat.mesh, meshes_single, meshes_repeat])
    _, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    for img, ax in zip(images, axes):
        ax.imshow(img.cpu()[..., :3])
        ax.grid(False)
    plt.tight_layout()
    plt.show()


def render_meshes(meshes: list[Meshes]) -> torch.Tensor:
    """Render given meshes using a renderer based on the HardPhongShader."""
    if len(meshes) == 0:
        return
    device = meshes[0].device
    # Setup common renderer
    camera = FoVPerspectiveCameras(device=device)
    raster_settings = RasterizationSettings(image_size=256, blur_radius=0.0, faces_per_pixel=1)
    lights = PointLights(device=device, location=((2.0, 2.0, 2.0),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=camera, lights=lights),
    )

    # Render all meshes from common viewpoint
    distance, elevation, azimuth = 1.5, 50, 30
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
    images = []
    for mesh in meshes:
        if mesh.textures is None:
            white = torch.ones_like(mesh[0].verts_packed())[None]
            mesh.textures = TexturesVertex(white.repeat((len(mesh), 1, 1)))
        images.append(phong_renderer(mesh[0], R=R, T=T))
    images = torch.cat(images, dim=0)
    return images
