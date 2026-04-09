from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Sequence, TypeVar

import torch
from torch.optim.lr_scheduler import LRScheduler

if TYPE_CHECKING:
    import torch.nn as nn
    from pytorch3d.renderer import MeshRenderer
    from torch import Tensor

    from vc_core.dr.pytorch3d.mesh import Mesh
    from vc_core.loggers.base import Logger

T = TypeVar("T", bound=LRScheduler)


class Optimizer(ABC):
    """Base optimizer for optimization using differentiable rendering (PyTorch3D).

    The optimizer makes use of the `torch.optim.Adam` optimizer internally.

    Args:
        mesh: Mesh to use for rendering.
        model: Model of learnable parameters.
        renderer: Renderer to render output images.
        loss_fn: Loss function. Order of arguments is input then target.
        lr: Learning rate for Adam optimizer. Default value is 1e-2.
        lr_sched_cfg: Optional configuration for learning rate scheduler. Default value is `None`.
    """

    @dataclass
    class LRSchedulerCfg:
        """Configuration for initializing a learning rate scheduler."""

        cls: type[T]
        """Type of learning scheduler. A subclass of `torch.optim.lr_scheduler.LRScheduler`."""

        kwargs: dict[str, Any]
        """Named arguments to pass to the constructor of the scheduler alongside the optimizer."""

    def __init__(
        self,
        mesh: Mesh,
        model: nn.Module,
        renderer: MeshRenderer,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        *,
        lr: float | Sequence[float] = 1e-2,
        lr_sched_cfg: LRSchedulerCfg | None = None,
    ):
        self._mesh = mesh
        self._model = model
        self._renderer = renderer
        self._loss_fn = loss_fn
        self._lr = lr
        self._sched_cfg = lr_sched_cfg

    @property
    def model(self) -> nn.Module:
        """Get internal model of learnable parameters."""
        return self._model

    @model.setter
    def model(self, value: nn.Module) -> None:
        """Set the internal model of learnable parameters."""
        self._model = value

    @abstractmethod
    def optimize(
        self,
        target: Tensor,
        *,
        n_iter: int = 10,
        eps: float | None = None,
        logger: Logger | None = None,
    ) -> Any:
        """Run the optimizer for a number of iterations to update model.

        Args:
            target: Target images for loss computation. Shape is (B, H, W, C).
            n_iter: Number of iterations to perform.
            eps: Optional threshold for early stopping. Default value is `None`.
            logger: Optional logger for logging loss and output values. Default value is `None`.

        Returns:
            Output learned parameters after optimization.
        """
        pass

    def reset(self) -> torch.optim.Adam:
        """Reset and return a new instance the internal optimizer."""
        lr = self._lr if isinstance(self._lr, float) else self._lr[0]
        return torch.optim.Adam(self._model.parameters(), lr=lr)
