from __future__ import annotations

import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import kaolin as kal
import matplotlib.pyplot as plt
import pytest
import torch

from vc_core.dr.kaolin.mesh import CylinderMesh, ObjMesh

if TYPE_CHECKING:
    from kaolin.rep import SurfaceMesh

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
            assert mesh_single.mesh.vertices.device.type == d
            assert mesh_repeat.mesh.vertices.device.type == d

    # Test `forward()` method without texture
    offsets_single = torch.randn_like(mesh_single.mesh.vertices, requires_grad=True)
    offsets_repeat = torch.randn_like(mesh_repeat.mesh.vertices, requires_grad=True)
    meshes_single: SurfaceMesh = mesh_single({"vertices": offsets_single})
    meshes_repeat: SurfaceMesh = mesh_repeat({"vertices": offsets_repeat})
    assert meshes_single.vertices.requires_grad
    assert meshes_repeat.vertices.requires_grad
    assert meshes_single.vertex_colors is None
    assert meshes_repeat.vertex_colors is None

    # Assign colors to the mesh vertices
    rgb = torch.ones_like(meshes_single.vertices)
    meshes_single.vertex_colors = rgb
    meshes_repeat.vertex_colors = rgb.repeat((n_rep, 1, 1))
    meshes_single.vertices = mesh_single.mesh.vertices
    meshes_repeat.vertices = mesh_repeat.mesh.vertices
    mesh_single.mesh = deepcopy(meshes_single.detach())
    mesh_repeat.mesh = deepcopy(meshes_repeat.detach())

    # Test `forward()` method with a texture
    offset_rgb = torch.randn_like(rgb)
    meshes_single: SurfaceMesh = mesh_single(
        {"vertices": offsets_single, "vertex_colors": offset_rgb}
    )
    meshes_repeat: SurfaceMesh = mesh_repeat(
        {"vertices": offsets_repeat, "vertex_colors": offset_rgb.repeat((n_rep, 1, 1))}
    )
    assert not torch.allclose(mesh_single.mesh.vertex_colors, meshes_single.vertex_colors)
    assert not torch.allclose(mesh_repeat.mesh.vertex_colors, meshes_repeat.vertex_colors)

    # Remove .obj model
    path.unlink(missing_ok=True)


@pytest.mark.unit
def test_cylinder_mesh() -> None:
    # Create cylinder meshes
    radius, height = 0.2, 1.0
    n_rep = 10
    mesh_single = CylinderMesh(radius, height, n_rep=1, split=10)
    mesh_repeat = CylinderMesh(radius, height, n_rep=n_rep, split=10)

    # Test `len()` method and `mesh` property
    assert len(mesh_single) == len(mesh_single.mesh) == 1
    assert len(mesh_repeat) == len(mesh_repeat.mesh) == 10

    # Test `to()` method
    if torch.device("cuda") in Devices:
        for d in ["cpu", "cuda"]:
            device = torch.device(d)
            mesh_single = mesh_single.to(device)
            mesh_repeat = mesh_repeat.to(d)
            assert mesh_single.mesh.vertices.device.type == d
            assert mesh_repeat.mesh.vertices.device.type == d

    # Test `forward()` method without texture
    device = mesh_single.mesh.vertices.device
    r_offset = torch.full((n_rep,), -0.1, dtype=torch.float32, device=device, requires_grad=True)
    h_offset = torch.full((n_rep,), 0.5, dtype=torch.float32, device=device, requires_grad=True)
    meshes_single: SurfaceMesh = mesh_single(r_offset[:1], h_offset[:1])
    meshes_repeat: SurfaceMesh = mesh_repeat(r_offset, h_offset)
    for mesh in [meshes_single, meshes_repeat]:
        r = torch.linalg.norm(mesh.vertices[:, 2:, :2], dim=-1)
        h = torch.abs(
            torch.amax(mesh.vertices[..., 2], dim=1) - torch.amin(mesh.vertices[..., 2], dim=1)
        )
        assert torch.allclose(r, torch.full_like(r, radius + r_offset[0].item()))
        assert torch.allclose(h, torch.full_like(h, height + h_offset[0].item()))
        assert mesh.vertices.requires_grad
        assert mesh.vertex_colors is None

    # Assign colors to the mesh vertices
    rgb = torch.ones_like(meshes_single.vertices)
    rgb[..., 2] = 0
    meshes_single.vertex_colors = rgb
    meshes_repeat.vertex_colors = rgb.repeat((n_rep, 1, 1))
    meshes_single.vertices = mesh_single.mesh.vertices
    meshes_repeat.vertices = mesh_repeat.mesh.vertices
    mesh_single.mesh = deepcopy(meshes_single.detach())
    mesh_repeat.mesh = deepcopy(meshes_repeat.detach())

    # Test `forward()` method with a texture
    offset_rgb = torch.randn_like(rgb)
    meshes_single: SurfaceMesh = mesh_single(
        r_offset[:1], h_offset[:1], {"vertex_colors": offset_rgb}
    )
    meshes_repeat: SurfaceMesh = mesh_repeat(
        r_offset, h_offset, {"vertex_colors": offset_rgb.repeat((n_rep, 1, 1))}
    )
    assert not torch.allclose(mesh_single.mesh.vertex_colors, meshes_single.vertex_colors)
    assert not torch.allclose(mesh_repeat.mesh.vertex_colors, meshes_repeat.vertex_colors)
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


def render_meshes(meshes: list[SurfaceMesh]) -> torch.Tensor:
    """Render given meshes using a renderer based on the default Kaolin settings."""
    if len(meshes) == 0:
        return
    device = meshes[0].vertices.device
    # Setup common renderer
    camera = kal.render.easy_render.default_camera(256).to(device=device)
    lighting = kal.render.easy_render.default_lighting().to(device=device)

    # Render all meshes from common viewpoint
    images = []
    for mesh in meshes:
        if mesh.vertex_colors is None:
            white = torch.ones_like(mesh.vertices)
            mesh.vertex_colors = white
        # Normalize so it is easy to set up default camera
        mesh.vertices = kal.ops.pointcloud.center_points(mesh.vertices, normalize=False)
        render_res = kal.render.easy_render.render_mesh(camera, mesh[0], lighting=lighting)
        img = render_res[kal.render.easy_render.RenderPass.render].clamp(0, 1)
        images.append(img)
    images = torch.cat(images, dim=0)
    return images
