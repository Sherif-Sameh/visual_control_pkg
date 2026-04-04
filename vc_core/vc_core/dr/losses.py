from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal

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
    assert reduction in ["sum", "mean"]
    if not hasattr(F, fn_name):
        assert f"build_{fn_name}" in globals(), (
            f"Function must be from torch.nn.functional or losses module. Got {fn_name}."
        )
        fn = globals()[f"build_{fn_name}"](reduction="none")
    else:
        fn = getattr(F, fn_name)
    reduction_fn = torch.sum if reduction == "sum" else torch.mean

    def loss_fn(input: Tensor, target: Tensor) -> Tensor:
        loss = fn(input, target, reduction="none", **kwargs)
        return reduction_fn(loss, dim=tuple(range(1, input.ndim)))

    return loss_fn


def build_dice_loss(
    *, reduction: Literal["mean", "sum", "none"] = "mean"
) -> Callable[[Tensor, Tensor], Tensor]:
    """Builds a torch compiled dice loss function for the given reduction method.

    Args:
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'mean':
            the mean of the output is taken. 'sum': the output will be summed. 'none': no reduction
            will be applied. Default value is 'mean'.

    Return:
        Dice loss.
    """

    match reduction:
        case "mean":
            reduction_fn = torch.mean
        case "sum":
            reduction_fn = torch.sum
        case _:
            reduction_fn = lambda x: x  # noqa: E731

    def dice_loss(
        input: Tensor, target: Tensor, reduction: Any = None, smooth: float = 1.0
    ) -> Tensor:
        """Compute the Dice loss with optional smoothing factor.

        Args:
            input: Predicted values in the range [0, 1].
            target Ground truth values. Shape is the same as `input`.
            reduction: Placeholder to match torch.nn.functional signatures. Reduction method set
                from the `reduction` argument to `build_dice_loss` function.
            smooth: Smoothing factor to apply to Dice coefficient, Default value is 1.0.

        Return:
            Dice loss.
        """
        dice = (2 * (input * target) + smooth) / (input + target + smooth)
        loss = 1.0 - dice
        return reduction_fn(loss)

    return torch.compile(dice_loss)
