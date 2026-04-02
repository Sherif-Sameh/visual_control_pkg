from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    look_at_view_transform,
)
from torch.optim.lr_scheduler import StepLR

from vc_core.dr.pytorch3d.mesh import CylinderMesh
from vc_core.dr.pytorch3d.model import CylinderModel, CylinderSplitParamModel
from vc_core.dr.pytorch3d.optim import CylinderMultiLROptimizer, CylinderOptimizer
from vc_core.dr.pytorch3d.shader import SoftSilhouetteShader
from vc_core.loggers import MemoryLogger

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
    radius, height = 0.2, 1.0
    n_rep = 10
    mesh = CylinderMesh(radius, height, n_rep=n_rep).to(device)

    # Create cylinder model
    distance, elevation, azimuth = 2.5, 50, 15
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
    pos_off = torch.tensor([[0.1, -0.1, -0.1]], dtype=torch.float32, device=device)
    z_dir_off = torch.tensor([[0.05, 0.05, -0.1]], dtype=torch.float32, device=device)
    height_off = -0.1
    pos = torch.randn((n_rep, 3), dtype=torch.float32, device=device) + T + pos_off
    z_dir = torch.randn((n_rep, 3), dtype=torch.float32, device=device) + R[..., -1] + z_dir_off
    height = torch.randn((n_rep,), dtype=torch.float32, device=device) + height + height_off
    model_cls = CylinderModel if cls == CylinderOptimizer else CylinderSplitParamModel
    model = model_cls(pos, z_dir, radius=None, height=height, n_rep=n_rep)

    # Create silhouette renderer
    camera = FoVPerspectiveCameras(device=device)
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster_settings = RasterizationSettings(
        image_size=256,
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
        faces_per_pixel=10,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=None, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params).to(device),
    )

    # Create cylinder model optimizer
    def loss_fn(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(input, target, reduction="none").mean(dim=(1, 2, 3))

    lr = 0.05 if cls == CylinderOptimizer else sample_log_uniform(1e-2, 0.1, n_rep).tolist()
    optim = cls(
        mesh,
        model,
        renderer,
        loss_fn,
        lr=lr,
        lr_sched_cfg=CylinderOptimizer.LRSchedulerCfg(StepLR, {"step_size": 25, "gamma": 0.7}),
    )

    # Run optimizer for a number of iterations
    n_iter = 100
    target = renderer(mesh.mesh, R=R, T=T, cameras=camera).detach()
    logger = MemoryLogger(n_log=10)
    logger._count = -1
    optim.optimize(target, n_iter=n_iter, logger=logger, cameras=camera)

    # visualize loss history and outputs vs target
    log = logger.flush()
    iters = list(log.keys())
    loss = np.array([np.min(v["loss"]) for v in log.values()])
    image_init = log[iters[0]]["output"][np.argmin(log[iters[0]]["loss"])]
    image_final = log[iters[-1]]["output"][np.argmin(log[iters[-1]]["loss"])]
    _, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    axes[0].plot(iters, loss)
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
