from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor


def wrap_loss_fn(
    fn_name: str, reduction: Literal["sum", "mean"] = "mean", **kwargs
) -> Callable[[Tensor, Tensor], Tensor]:
    """Wrap a loss function from `torch.nn.functional` so that the batch dimension is retained.

    The input should be the name of a loss function from `torch.nn.functional` that accepts input
    and target tensors respectively plus additional fixed kwargs (e.g., `mse_loss`). This function
    is retreived and wrapped such that all dimensions are reduced except for the batch dimension.

    Args:
        fn_name: Name of loss function from `torch.nn.functional` that accepts input and target
            tensors respectively plus additional fixed kwargs (e.g., `mse_loss`).
        reduction: Reduction operation to apply to loss tensor. Default value is `mean`.
        kwargs: Optional kwargs to forward to loss function (e.g., weight).

    Returns:
        Wrapped loss function that reduces all input dimensions but the batch dimension.
    """
    assert hasattr(F, fn_name), f"Function must be from `torch.nn.functional`. Got {fn_name}."
    assert reduction in ["sum", "mean"]
    fn = getattr(F, fn_name)
    reduction_fn = torch.sum if reduction == "sum" else torch.mean

    def loss_fn(input: Tensor, target: Tensor) -> Tensor:
        loss = fn(input, target, reduction="none", **kwargs)
        return reduction_fn(loss, dim=tuple(range(1, input.ndim)))

    return loss_fn
