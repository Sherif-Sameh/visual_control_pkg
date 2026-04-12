from __future__ import annotations

from itertools import product

import kaolin as kal
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from kaolin.render.camera import Camera
from torch.optim.lr_scheduler import CosineAnnealingLR

from vc_core.dr.common.losses import wrap_combined_loss_fn
from vc_core.dr.common.model import CylinderModel, CylinderSplitParamModel
from vc_core.dr.kaolin.mesh import CylinderMesh
from vc_core.dr.kaolin.optim import CylinderMultiLROptimizer, CylinderOptimizer
from vc_core.dr.kaolin.render import (
    BlendParams,
    ComposeShader,
    HardDepthShader,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftSilhouetteShader,
)
from vc_core.dr.kaolin.utils import look_at_view_transform, transform_from_rotation_translation
from vc_core.loggers import MemoryLogger

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

Devices = [torch.device("cpu")]
Devices = Devices + [torch.device("cuda")] if torch.cuda.is_available() else Devices
Backends = ["cuda", "nvdiffrast"]
Optimizers = [CylinderOptimizer, CylinderMultiLROptimizer]


@pytest.mark.unit
@pytest.mark.parametrize("device,backend,cls", product(Devices, Backends, Optimizers))
def test_cylinder_optimizer(
    capsys: pytest.CaptureFixture, device: torch.device, backend: str, cls: CylinderOptimizer
) -> None:
    if device.type == "cpu":
        return  # Kaolin doesn't support CPU rendering
    # Create cylinder meshes
    radius, height = 0.003, 0.025
    n_rep = 8
    mesh = CylinderMesh(radius, height, split=3, n_rep=n_rep)
    F = mesh.mesh.vertices.shape[1]
    mesh.mesh.vertex_features = torch.zeros([n_rep, F, 0], device=device)
    mesh = mesh.to(device)
    R_gt = kal.math.quat.rot33_from_quat(torch.tensor([[0.152, 0.0, 0.0, 0.988]])).to(device=device)
    R_off = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], device=device)
    R_gt = R_gt @ R_off[None]
    T_gt = torch.tensor([[0.0, 0.006, -0.113]]).to(device=device)

    # Create cylinder model
    distance, elevation, azimuth = 0.125, 135.0, 0.0
    R, T = look_at_view_transform(distance, elevation, azimuth)
    pos_sigma, z_dir_sigma, height_sigma = 0.005, 0.1, 0.002
    pos = torch.randn((n_rep, 3), dtype=torch.float32) * pos_sigma + T
    z_dir = torch.randn((n_rep, 3), dtype=torch.float32) * z_dir_sigma + R[..., -1]
    height = torch.randn((n_rep,), dtype=torch.float32) * height_sigma + height
    model_cls = CylinderModel if cls == CylinderOptimizer else CylinderSplitParamModel
    model = model_cls(pos, z_dir, radius=None, height=height, n_rep=n_rep, scale=0.1)
    model = model.to(device=device)

    # Setup camera
    view_matrix = transform_from_rotation_translation(R, T)
    cameras = Camera.from_args(
        view_matrix=view_matrix, focal_x=733.0, height=256, width=256, dtype=torch.float32
    )
    cameras = Camera.cat([cameras] * n_rep).to(device=device)

    # Create silhouette + depth renderer
    blend_params = BlendParams(sigmainv=1 / 1e-4, boxlen=0.02, knum=10)
    raster_settings = RasterizationSettings(image_size=256, backend=backend)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=ComposeShader(
            [SoftSilhouetteShader(blend_params=blend_params), HardDepthShader(cameras=cameras)]
        ).to(device=device),
    )

    # Render target inputs
    renderer_gt = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, raster_settings=RasterizationSettings(image_size=256, backend=backend)
        ),
        shader=HardDepthShader(cameras=cameras),
    ).to(device=device)
    target = renderer_gt(mesh.mesh, R=R_gt, T=T_gt, zfar=1.0).detach()
    target = torch.cat([(target < 1.0).float(), target], dim=-1)

    # Create cylinder model optimizer
    n_iter = 100
    lr = 0.01 if cls == CylinderOptimizer else sample_log_uniform(5e-3, 0.1, n_rep).tolist()
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
    optim.optimize(target, n_iter=1, logger=logger_first, zfar=1.0)
    T, R, _, _ = optim.optimize(target, n_iter=n_iter, logger=logger_all, zfar=1.0)
    T_err = torch.linalg.norm(T_gt[0] - T, dim=0)
    z_dir_err = torch.acos(torch.dot(R_gt[0, :, -1], R[:, -1]))
    with capsys.disabled():
        print(f"\nOptimizer: {cls.__name__}, Backend: {backend}")
        print(f"\tPosition Error: {T_err}m")
        print(f"\tZ direction Error: {torch.rad2deg(z_dir_err)}deg")

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
