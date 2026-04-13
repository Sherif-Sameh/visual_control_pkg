from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence

import torch

from .base import Optimizer

if TYPE_CHECKING:
    from torch import Tensor

    from vc_core.dr.common.model import EyePoseModel, EyePoseTextureModel
    from vc_core.dr.kaolin.mesh import Mesh
    from vc_core.dr.kaolin.render import MeshRenderer
    from vc_core.loggers.base import Logger


class EyePoseOptimizer(Optimizer):
    """Optimizer for optimizing an eye mesh's pose using differentiable rendering (Kaolin).

    This optimizer treats all parameters of the model as a single parameter group. If multiple
    copies of the pose parameters exist, for example for multiple initializations, the copy that
    yields the min loss value is returned and the min loss is used for early stopping.

    Args:
        mesh: Mesh to use for rendering.
        model: Model of learnable parameters for eye pose.
        renderer: Renderer to render output images.
        loss_fn: Loss function. Order of arguments is input then target.
        tan_norm_w: Weight for norm loss penalty of model tangent offsets. Default value is 5e-4.
        lr: Learning rate for Adam optimizer. Default value is 1e-2.
        lr_sched_cfg: Optional configuration for learning rate scheduler. Default value is `None`.
    """

    def __init__(
        self,
        mesh: Mesh,
        model: EyePoseModel,
        renderer: MeshRenderer,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        *,
        tan_norm_w: float = 5e-4,
        lr: float | Sequence[float] = 1e-2,
        lr_sched_cfg: Optimizer.LRSchedulerCfg | None = None,
    ):
        self.tan_norm_w = tan_norm_w
        super().__init__(mesh, model, renderer, loss_fn, lr=lr, lr_sched_cfg=lr_sched_cfg)

    def optimize(
        self,
        target: Tensor,
        *,
        n_iter: int = 10,
        eps: float | None = None,
        logger: Logger | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Run the optimizer for a number of iterations to update model (Kaolin).

        Args:
            target: Target images for loss computation. Shape is (B, H, W, C).
            n_iter: Number of iterations to perform.
            eps: Optional threshold for early stopping. Default value is `None`.
            logger: Optional logger for logging loss and output values. Default value is `None`.
            kwargs: Optional kwargs to pass to the renderer's `forward()` method.

        Returns:
            Output detached learned parameters after optimization.
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
        for iter in range(n_iter):
            pos, rot = self._model()
            meshes = self._mesh({})
            images = self._renderer(meshes, T=pos, R=rot, **kwargs)
            loss = self._loss_fn(images, target)
            loss += self.tan_norm_w * torch.linalg.norm(self._model.z_tan, dim=-1)
            optim.zero_grad()
            loss.sum().backward()
            optim.step()
            if logger is not None:
                logger.log(
                    iter + 1,
                    {"output": images.detach().cpu().numpy(), "loss": loss.detach().cpu().numpy()},
                )
            if loss.detach().min() <= eps:
                break
            if sched is not None:
                sched.step()
        # get final parameters
        min_idx = torch.argmin(loss.detach(), dim=0)
        pos, rot = self._model()
        pos_min, rot_min = pos[min_idx].detach(), rot[min_idx].detach()
        return pos_min, rot_min

    def resample_model_params(self, pos: Tensor, z_dir: Tensor, **kwargs) -> None:
        """Resample the model's pose parameters around a new pose.

        Args:
            pos: Mean position to sample around. Shape is (3,).
            z_dir: Z-axis direction to sample around. Shape is (2,)
            **kwargs: Optional arguments to pass to the model's `resample_params()` method.
        """
        self._model.resample_params(pos, z_dir, **kwargs)


class EyePoseTextureOptimizer(Optimizer):
    """Optimizer for optimizing an eye mesh's pose and texture using differentiable rendering (Kaolin).

    This optimizer treats all parameters of the model as a single parameter group. Unlike
    `vc_core.dr.kaolin.optim.EyePoseOptimizer`, batched pose parameters are not treated as copies.
    Instead they are treated as pose estimates for different views. Therefore, the loss is reduced
    over all of them and early stopping depends on that combined loss rather than the min.

    Args:
        mesh: Mesh to use for rendering.
        model: Model of learnable parameters for eye pose and texture.
        renderer: Renderer to render output images.
        loss_fn: Loss function. Order of arguments is input then target.
        lr: Learning rate for Adam optimizer. Default value is 1e-2.
        lr_sched_cfg: Optional configuration for learning rate scheduler. Default value is `None`.
    """

    def __init__(
        self,
        mesh: Mesh,
        model: EyePoseTextureModel,
        renderer: MeshRenderer,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        *,
        lr: float | Sequence[float] = 1e-2,
        lr_sched_cfg: Optimizer.LRSchedulerCfg | None = None,
    ):
        super().__init__(mesh, model, renderer, loss_fn, lr=lr, lr_sched_cfg=lr_sched_cfg)

    def optimize(
        self,
        target: Tensor,
        *,
        n_iter: int = 10,
        eps: float | None = None,
        logger: Logger | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the optimizer for a number of iterations to update model (Kaolin).

        Args:
            target: Target images for loss computation. Shape is (B, H, W, C).
            n_iter: Number of iterations to perform.
            eps: Optional threshold for early stopping. Default value is `None`.
            logger: Optional logger for logging loss and output values. Default value is `None`.
            kwargs: Optional kwargs to pass to the renderer's `forward()` method.

        Returns:
            Output detached learned parameters after optimization.
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
        for iter in range(n_iter):
            pos, rot, texture = self._model()
            meshes = self._mesh({}, texture=texture)
            images = self._renderer(meshes, T=pos, R=rot, **kwargs)
            loss = self._loss_fn(images, target).sum()
            optim.zero_grad()
            loss.backward()
            optim.step()
            if logger is not None:
                logger.log(
                    iter + 1,
                    {"output": images.detach().cpu().numpy(), "loss": loss.detach().cpu().numpy()},
                )
            if loss <= eps:
                break
            if sched is not None:
                sched.step()
        # get final parameters
        pos, rot, texture = [m.detach() for m in self._model()]
        return pos, rot, texture
