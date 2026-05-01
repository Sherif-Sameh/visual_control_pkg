from __future__ import annotations

import logging
from itertools import product
from pathlib import Path

import kaolin as kal
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from kaolin.render.camera import Camera
from scipy.spatial.transform import Rotation
from torch.optim.lr_scheduler import CosineAnnealingLR

from vc_core.dr.common.losses import build_combined_loss_fn
from vc_core.dr.common.model import EyePoseModel
from vc_core.dr.kaolin.mesh import EyeObjMesh
from vc_core.dr.kaolin.optim import EyePoseOptimizer
from vc_core.dr.kaolin.render import (
    BlendParams,
    ComposeShader,
    HardColorAmbientShader,
    HardColorDiffuseSGFittedShader,
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
Distances = [0.6, 0.4, 0.2]


@pytest.mark.unit
@pytest.mark.parametrize("device,backend,distance", product(Devices, Backends, Distances))
def test_eye_pose_optimizer(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture,
    device: torch.device,
    backend: str,
    distance: float,
) -> None:
    if device.type == "cpu":
        return  # Kaolin doesn't support CPU rendering
    # Create obj meshes
    n_rep = 9
    path = Path(__file__).parents[1] / "samples/eye_mesh/eye.obj"
    mesh = EyeObjMesh(path, n_rep=n_rep)
    F = mesh.mesh.vertices.shape[1]
    mesh.mesh.vertex_features = torch.zeros([n_rep, F, 0])
    mesh.mesh.face_uvs = 1 - mesh.mesh.face_uvs
    mesh.mesh.materials[0][0]["map_Kd"] = (
        mesh.mesh.materials[0][0]["map_Kd"].float().permute(2, 0, 1) / 255.0
    )
    with caplog.at_level(logging.ERROR):
        mesh = mesh.to(device)
        mesh.mesh.materials[0][0]["map_Kd"] = mesh.mesh.materials[0][0]["map_Kd"].cuda()
    elevation, azimuth = -25.0, 15.0
    R_gt, T_gt = look_at_view_transform(distance, elevation, azimuth, device=device)

    # Create eye pose model
    elevation, azimuth = 0.0, 0.0
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
    T = T + torch.tensor([0.025, -0.025, -0.05], device=device) * distance
    pos, rot = T[0], R[0]
    pos_sigma, z_tan_range = 0.0, 0.005
    model = EyePoseModel(pos, rot, pos_sigma, z_tan_range, n_rep=n_rep, scale=0.1)
    model = torch.compile(model.to(device=device))

    # Setup camera
    img_size = 256
    cameras = Camera.from_args(
        eye=torch.tensor([10.0, 0.0, 0.0]),
        at=torch.tensor([0.0, 0.0, 0.0]),
        up=torch.tensor([0.0, 1.0, 0.0]),
        focal_x=733.0,
        height=img_size,
        width=img_size,
        dtype=torch.float32,
    )
    cameras = Camera.cat([cameras] * n_rep).to(device=device)

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

    # Render target inputs
    elevation, azimuth = 30.0, 10.0
    lights = kal.render.lighting.SgLightingParameters(amplitude=3.0, sharpness=2.0)
    renderer_gt = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=None,
            raster_settings=RasterizationSettings(image_size=img_size, backend=backend),
        ),
        shader=ComposeShader(
            [
                SoftSilhouetteShader(BlendParams(sigmainv=1e6, boxlen=0, knum=1)),
                HardColorDiffuseSGFittedShader(
                    azimuth, elevation, lights, raw_texture=False, uvs_origin="Kaolin"
                ),
            ]
        ),
    ).to(device=device)
    target = renderer_gt(mesh.mesh, cameras=cameras, R=R_gt, T=T_gt).detach()
    assert target.shape == (n_rep, img_size, img_size, 4)

    # Create eye pose optimizer
    n_iter = 100
    lr = 0.025
    optim = EyePoseOptimizer(
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
                dim=(1, 2, 3),
                kwargs=[{}, {"inner_fn_name": "l1_loss"}],
            )
        ),
        lr=lr,
        lr_sched_cfg=EyePoseOptimizer.LRSchedulerCfg(
            CosineAnnealingLR, {"T_max": n_iter, "eta_min": 1e-5}
        ),
    )

    # Run optimizer for a number of iterations
    logger_first = MemoryLogger(n_log=1)
    logger_all = MemoryLogger(n_log=10)
    optim.optimize(target, n_iter=1, logger=logger_first, cameras=cameras)
    T, R = optim.optimize(target, n_iter=n_iter, logger=logger_all, cameras=cameras)
    T_err = torch.linalg.norm(T_gt[0] - T, dim=0)
    R_err = Rotation.from_matrix((R_gt[0] @ R.T).cpu().numpy()).magnitude()
    with capsys.disabled():
        print(f"\nBackend: {backend}, GT Distance: {distance:.2f}m")
        print(f"\tPosition Error: {T_err}m")
        print(f"\tRotation Error: {np.rad2deg(R_err)}deg")

    # visualize loss history and outputs vs target
    log_first = logger_first.flush()[1]
    log_all = logger_all.flush()
    loss = np.array([np.min(v["loss"]) for v in log_all.values()])
    image_init = log_first["output"][np.argmin(log_first["loss"])]
    image_final = log_all[n_iter]["output"][np.argmin(log_all[n_iter]["loss"])]
    _, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    axes[0].plot(list(log_all.keys()), loss)
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss History")
    axes[1].imshow(target[0, :, :, 1:].cpu())
    axes[1].grid(False)
    axes[1].set_title("Target")
    axes[2].imshow(image_init[:, :, 1:])
    axes[2].grid(False)
    axes[2].set_title("Initial")
    axes[3].imshow(image_final[:, :, 1:])
    axes[3].grid(False)
    axes[3].set_title("Final")
    plt.tight_layout()
    plt.show()
