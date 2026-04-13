from __future__ import annotations

import logging
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from kaolin.render.camera import Camera
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from vc_core.dr.common.losses import build_combined_loss_fn
from vc_core.dr.common.model import EyePoseTextureModel
from vc_core.dr.kaolin.mesh import ObjMesh
from vc_core.dr.kaolin.optim import EyePoseTextureOptimizer
from vc_core.dr.kaolin.render import (
    BlendParams,
    ComposeShader,
    HardColorAmbientShader,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftSilhouetteShader,
)
from vc_core.dr.kaolin.utils import look_at_view_transform
from vc_core.loggers import MemoryLogger

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

Devices = [torch.device("cpu")]
Devices = Devices + [torch.device("cuda")] if torch.cuda.is_available() else Devices
Backends = ["cuda", "nvdiffrast"]


@pytest.mark.unit
@pytest.mark.parametrize("device,backend", product(Devices, Backends))
def test_eye_pose_optimizer(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture,
    device: torch.device,
    backend: str,
) -> None:
    if device.type == "cpu":
        return  # Kaolin doesn't support CPU rendering
    # Create obj meshes
    n_view = 9
    path = Path(__file__).parents[1] / "samples/eye_mesh/eye.obj"
    mesh = ObjMesh(path, with_materials=True, with_normals=True, n_rep=n_view)
    F = mesh.mesh.vertices.shape[1]
    mesh.mesh.vertex_features = torch.zeros([n_view, F, 0])
    mesh.mesh.face_uvs = 1 - mesh.mesh.face_uvs
    mesh.mesh.materials[0][0]["map_Kd"] = (
        mesh.mesh.materials[0][0]["map_Kd"].float().permute(2, 0, 1) / 255.0
    )
    with caplog.at_level(logging.ERROR):
        mesh = mesh.to(device)
        mesh.mesh.materials[0][0]["map_Kd"] = mesh.mesh.materials[0][0]["map_Kd"].cuda()
    distance = 0.3
    elevation, azimuth = torch.meshgrid(
        torch.linspace(-25.0, 25.0, 3), torch.linspace(-15.0, 15.0, 3), indexing="xy"
    )
    elevation, azimuth = elevation.flatten().to(device=device), azimuth.flatten().to(device=device)
    R_gt, T_gt = look_at_view_transform(distance, elevation, azimuth, device=device)

    # Create eye pose texture model
    noise = torch.bernoulli(torch.full_like(elevation, 0.5))
    noise[noise == 0] = -1
    R, T = look_at_view_transform(
        distance, elevation + 10 * noise, azimuth + 10 * noise, device=device
    )
    T = T + torch.tensor([0.05, -0.05, -0.1], device=device) * distance
    text_rgb = torch.zeros(3, device=device)
    model = EyePoseTextureModel(
        T, R[:, :, -1], res=256, text_rgb=text_rgb, n_view=n_view, scale=0.2
    )
    model = torch.compile(model.to(device=device))
    _, _, texture_init = model()
    texture_init = texture_init.detach()

    # Setup camera
    img_size = 176
    cameras = Camera.from_args(
        eye=torch.tensor([10.0, 0.0, 0.0]),
        at=torch.tensor([0.0, 0.0, 0.0]),
        up=torch.tensor([0.0, 1.0, 0.0]),
        focal_x=733.0,
        height=img_size,
        width=img_size,
        dtype=torch.float32,
    )
    cameras = Camera.cat([cameras] * n_view).to(device=device)

    # Create silhouette + ambient color renderer
    blend_params = BlendParams(sigmainv=1 / 3e-5, boxlen=0.01, knum=10)
    raster_settings = RasterizationSettings(image_size=img_size, backend=backend)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=None, raster_settings=raster_settings),
        shader=ComposeShader(
            [
                SoftSilhouetteShader(blend_params=blend_params),
                HardColorAmbientShader(raw_texture=False, uvs_origin="Kaolin"),
            ]
        ),
    ).to(device=device)

    # Render initial and target inputs
    renderer_gt = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=None,
            raster_settings=RasterizationSettings(image_size=img_size, backend=backend),
        ),
        shader=ComposeShader(
            [
                SoftSilhouetteShader(BlendParams(sigmainv=1e6, boxlen=0, knum=1)),
                HardColorAmbientShader(raw_texture=False, uvs_origin="Kaolin"),
            ]
        ),
    ).to(device=device)
    initial = renderer_gt(mesh.mesh, cameras=cameras, R=R, T=T).detach()
    target = renderer_gt(mesh.mesh, cameras=cameras, R=R_gt, T=T_gt).detach()
    assert initial.shape == (n_view, img_size, img_size, 4)
    assert target.shape == (n_view, img_size, img_size, 4)

    # Create temporary silhouette-based optimizer
    n_iter = 100
    lr = 0.01
    optim = EyePoseTextureOptimizer(
        mesh,
        model,
        renderer,
        torch.compile(
            build_combined_loss_fn(["mse_loss"], [slice(1)], device=device, reduction="mean")
        ),
        lr=lr,
        lr_sched_cfg=EyePoseTextureOptimizer.LRSchedulerCfg(
            CosineAnnealingLR, {"T_max": n_iter, "eta_min": 1e-6}
        ),
    )

    # Run silhouette-based optimizer to correct position offsets
    optim.optimize(target, n_iter=n_iter, n_iter_text=0, cameras=cameras)

    # Create eye pose texture optimizer
    n_iter, n_iter_text = 300, 20
    lr = 0.05
    optim = EyePoseTextureOptimizer(
        mesh,
        model,
        renderer,
        torch.compile(
            build_combined_loss_fn(
                ["l1_loss", "masked_loss"],
                [slice(1), slice(4)],
                weights=[0.5, 0.5],
                device=device,
                reduction="mean",
                dim=None,
                kwargs=[{}, {"inner_fn_name": "l1_loss"}],
            )
        ),
        lr=lr,
        lr_sched_cfg=EyePoseTextureOptimizer.LRSchedulerCfg(
            CosineAnnealingWarmRestarts, {"T_0": 100, "eta_min": 3e-4}
        ),
    )

    # Run optimizer for a number of iterations
    logger = MemoryLogger(n_log=5, filter="loss")
    T, R, texture = optim.optimize(
        target, n_iter=n_iter, n_iter_text=n_iter_text, logger=logger, cameras=cameras
    )
    mesh_final = mesh({}, texture=texture)
    final = renderer_gt(mesh_final, cameras=cameras, R=R, T=T).detach()
    T_err = torch.linalg.norm(T_gt - T, dim=-1)
    z_dir_err = torch.acos((R_gt[..., -1] * R[..., -1]).sum(dim=-1))
    with capsys.disabled():
        print(f"\nBackend: {backend}, GT Distance: {distance:.2f}m")
        print(f"\tPosition Errors (m): {T_err.cpu()}")
        print(f"\tZ direction Errors (deg): {torch.rad2deg(z_dir_err).cpu()}")

    # Store initial, final and target visualizations
    path = Path(__file__).parent / "outputs/texture"
    path.mkdir(parents=True, exist_ok=True)
    for state, name in zip([initial, final, target], ["initial", "final", "target"]):
        _, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            ax.imshow(state[i, ..., 1:].cpu())
            ax.grid(False)
            ax.set_title(f"Elevation: {elevation[i].item():.1f}, Azimuth: {azimuth[i].item():.1f}")
        plt.tight_layout()
        plt.savefig(path / f"{name}_{backend}.png")
        plt.close()

    # visualize loss history and output textures vs target
    log = logger.flush()
    loss = np.array([np.min(v["loss"]) for v in log.values()])
    textures = [mesh.mesh.materials[0][0]["map_Kd"], texture_init, texture]
    _, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    axes[0].plot(list(log.keys()), loss)
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss History")
    for i, ax in enumerate(axes[1:]):
        ax.imshow(textures[i].permute(1, 2, 0).cpu())
        ax.grid(False)
    axes[1].set_title("Target")
    axes[2].set_title("Initial")
    axes[3].set_title("Final")
    plt.tight_layout()
    plt.savefig(path / f"result_{backend}.png")
    plt.close()
