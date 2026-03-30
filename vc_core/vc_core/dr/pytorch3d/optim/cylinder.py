from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence

import torch

from .base import Optimizer

if TYPE_CHECKING:
    from pytorch3d.renderer import MeshRenderer
    from torch import Tensor

    from vc_core.dr.pytorch3d.mesh import Mesh
    from vc_core.dr.pytorch3d.model import CylinderModel
    from vc_core.loggers.base import Logger


class CylinderOptimizer(Optimizer):
    """Optimizer for optimizing a cylinder's pose and geometry using differentiable rendering.

    This optimizer treats all parameters of the cylinder model as a single parameter group. If
    multiple copies of the cylinder's parameters exist, for example for multiple initializations,
    the copy that yields the min loss value is returned and the min loss is used for early stopping.

    Args:
        mesh: Mesh to use for rendering.
        model: Model of learnable parameters for cylinder.
        renderer: Renderer to render output images.
        loss_fn: Loss function. Order of arguments is input then target.
        lr: Learning rate for Adam optimizer. Default value is 1e-2.
        lr_sched_cfg: Optional configuration for learning rate scheduler. Default value is `None`.
    """

    def optimize(
        self,
        target: Tensor,
        *,
        n_iter: int = 10,
        eps: float | None = None,
        logger: Logger | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run the optimizer for a number of iterations to update model.

        Args:
            target: Target images for loss computation. Shape is (B, H, W, C).
            n_iter: Number of iterations to perform.
            eps: Optional threshold for early stopping. Default value is `None`.
            logger: Optional logger for logging loss and output values. Default value is `None`.

        Returns:
            Output learned parameters after optimization.
        """
        eps = -torch.inf if eps is None else eps
        # Reset optimizer and LR scheduler
        optim = self.reset()
        sched = (
            self._sched_cfg.cls(optim, **self._sched_cfg.kwargs)
            if self._sched_cfg is not None
            else None
        )
        # optimization loop
        for epoch in range(n_iter):
            pos, rot, r_off, h_off = self._model()
            meshes = self._mesh(r_off, h_off)
            images = self._renderer(meshes, T=pos, R=rot)
            loss = self._loss_fn(images, target)
            optim.zero_grad()
            loss.mean().backward()
            optim.step()
            if logger is not None:
                logger.log(
                    epoch, {"output": images.detach().numpy(), "loss": loss.detach().numpy()}
                )
            if loss.detach().min() <= eps:
                break
            if sched is not None:
                sched.step()
        # get final parameters
        min_idx = torch.argmin(loss.detach(), dim=0)
        pos, rot, r_off, h_off = self._model()
        return pos[min_idx], rot[min_idx], r_off[min_idx], h_off[min_idx]


class CylinderMultiLROptimizer(CylinderOptimizer):
    """Optimizer for optimizing a cylinder's pose and geometry using differentiable rendering.

    This optimizer allows for setting different learning rates for each copy of the cylinder's
    parameters. This allows for implementing the multi-learning rate strategy described in the
    paper titled `Diff-DOPE: Differentiable Deep Object Pose Estimation`.

    Args:
        mesh: Mesh to use for rendering.
        model: Model of learnable parameters for cylinder.
        renderer: Renderer to render output images.
        loss_fn: Loss function. Order of arguments is input then target.
        lr: Learning rate/s for Adam optimizer. Default value is 1e-2.
        lr_sched_cfg: Optional configuration for learning rate scheduler. Default value is `None`.
    """

    def __init__(
        self,
        mesh: Mesh,
        model: CylinderModel,
        renderer: MeshRenderer,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        *,
        lr: float | Sequence[float] = 1e-2,
        lr_sched_cfg: Optimizer.LRSchedulerCfg | None = None,
    ):
        lr = [lr] * model.n_rep if isinstance(lr, float) else lr
        assert len(lr) == model.n_rep
        super().__init__(mesh, model, renderer, loss_fn, lr=lr, lr_sched_cfg=lr_sched_cfg)
        self._param_groups = [
            {
                "params": [
                    model.pos_offset[i],
                    model.z_dir[i],
                    model.r_offset[i],
                    model.h_offset[i],
                ],
                "lr": lr[i],
            }
            for i in range(model.n_rep)
        ]

    def reset(self) -> torch.optim.Adam:
        """Reset and return a new instance the internal optimizer."""
        return torch.optim.Adam(self._param_groups)
