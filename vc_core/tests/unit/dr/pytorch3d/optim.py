from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    look_at_view_transform,
)
from pytorch3d.renderer.mesh.shader import HardDepthShader, SoftDepthShader
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.utils import cameras_from_opencv_projection
from torch.optim.lr_scheduler import CosineAnnealingLR

from vc_core.dr.losses import wrap_combined_loss_fn
from vc_core.dr.pytorch3d.mesh import CylinderMesh
from vc_core.dr.pytorch3d.model import CylinderModel, CylinderSplitParamModel
from vc_core.dr.pytorch3d.optim import CylinderMultiLROptimizer, CylinderOptimizer
from vc_core.dr.pytorch3d.shader import ComposeShader, SoftSilhouetteShader
from vc_core.loggers import MemoryLogger

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

Devices = [torch.device("cpu")]
Devices = Devices + [torch.device("cuda")] if torch.cuda.is_available() else Devices


@pytest.mark.unit
@pytest.mark.parametrize(
    "device,cls",
    [t for d in Devices for t in [(d, CylinderOptimizer), (d, CylinderMultiLROptimizer)]],
)
def test_cylinder_optimizer(device: torch.device, cls: CylinderOptimizer) -> None:
    if device.type == "cpu":
        return  # Extremely slow to run test on CPU
    # Create cylinder meshes
    radius, height = 0.003, 0.025
    n_rep = 8
    mesh = CylinderMesh(radius, height, n_rep=n_rep).to(device)
    R_gt = quaternion_to_matrix(torch.tensor([[0.988, 0.152, 0.0, 0.0]])).to(device=device)
    T_gt = torch.tensor([[0.0, 0.006, 0.113]]).to(device=device)

    # Create cylinder model
    distance, elevation, azimuth = 0.125, 135.0, 0.0
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
    pos_sigma, z_dir_sigma, height_sigma = 0.005, 0.1, 0.002
    pos = torch.randn((n_rep, 3), dtype=torch.float32, device=device) * pos_sigma + T
    z_dir = torch.randn((n_rep, 3), dtype=torch.float32, device=device) * z_dir_sigma + R[..., -1]
    height = torch.randn((n_rep,), dtype=torch.float32, device=device) * height_sigma + height
    model_cls = CylinderModel if cls == CylinderOptimizer else CylinderSplitParamModel
    model = model_cls(pos, z_dir, radius=None, height=height, n_rep=n_rep, scale=0.1)

    # Create silhouette + depth renderer
    camera = cameras_from_opencv_projection(
        R=torch.zeros(1, 3, 3),
        tvec=torch.zeros(1, 3),
        camera_matrix=torch.tensor([[[733.0, 0.0, 128.0], [0.0, 733.0, 128.0], [0.0, 0.0, 1.0]]]),
        image_size=torch.tensor([[256, 256]]),
    ).to(device=device)
    blend_params = BlendParams(sigma=1e-5, gamma=1e-4)
    raster_settings = RasterizationSettings(
        image_size=256,
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
        faces_per_pixel=10,
        perspective_correct=True,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=None, raster_settings=raster_settings),
        shader=ComposeShader(
            [
                SoftSilhouetteShader(blend_params=blend_params),
                SoftDepthShader(blend_params=blend_params),
            ]
        ).to(device=device),
    )

    # Render target inputs
    renderer_gt = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=None,
            raster_settings=RasterizationSettings(
                image_size=256, blur_radius=0, faces_per_pixel=1, perspective_correct=True
            ),
        ),
        shader=HardDepthShader(device=device),
    )
    target = renderer_gt(mesh.mesh, R=R_gt, T=T_gt, cameras=camera, zfar=1.0).detach()
    target = torch.cat([(target < 1.0).float(), target], dim=-1)

    # Create cylinder model optimizer
    n_iter = 100
    lr = 0.05 if cls == CylinderOptimizer else sample_log_uniform(5e-3, 0.1, n_rep).tolist()
    optim = cls(
        mesh,
        model,
        renderer,
        torch.compile(
            wrap_combined_loss_fn(["mse_loss"], [slice(2)], device=device, reduction="mean")
        ),
        lr=lr,
        lr_sched_cfg=CylinderOptimizer.LRSchedulerCfg(
            CosineAnnealingLR, {"T_max": n_iter, "eta_min": 3e-4}
        ),
    )

    # Run optimizer for a number of iterations
    logger_first = MemoryLogger(n_log=1)
    logger_all = MemoryLogger(n_log=10)
    optim.optimize(target, n_iter=1, logger=logger_first, cameras=camera, zfar=1.0)
    optim.optimize(target, n_iter=n_iter, logger=logger_all, cameras=camera, zfar=1.0)

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
    axes[1].imshow(target[0, :, :, 0].cpu())
    axes[1].grid(False)
    axes[1].set_title("Target")
    axes[2].imshow(image_init[:, :, 0])
    axes[2].grid(False)
    axes[2].set_title("Initial")
    axes[3].imshow(image_final[:, :, 0])
    axes[3].grid(False)
    axes[3].set_title("Final")
    plt.tight_layout()
    plt.show()


def sample_log_uniform(low: float, high: float, size=None):
    sample = np.random.uniform(np.log(low), np.log(high), size)
    return np.exp(sample)
