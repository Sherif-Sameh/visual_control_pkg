from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor


def wrap_combined_loss_fn(
    fn_names: Sequence[str],
    ranges: Sequence[slice],
    weights: Sequence[float] | None = None,
    device: str | torch.device = "cpu",
    reduction: Literal["sum", "mean", "none"] = "mean",
    **kwargs,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Wrap a combined loss function so that the batch dimension is retained.

    If all functions in `fn_names` can be compiled with `torch.compile`, then the returned function
    retains this ability.

    For more on how losses are combined, refer to `vc_core.dr.losses.build_combined_loss_fn`.

    Args:
        fn_names: Loss function names to combine from `vc_core.dr.losses` module or
            `torch.nn.functional` that accepts input and target tensors respectively plus
            additional fixed kwargs (e.g., `mse_loss`).
        ranges: Slices for extracting the relevant views from input and target tensors for each
            loss function. Length must be the same as `fn_names`. The channel dimension that is
            sliced is assumed to be the last.
        weights: Optional weights for combining the individual loss functions. If `None`, then all
            losses are given equal weights of 1.0. Default value is `None`.
        device: Device to move `weights` tensor to. Should the same device as that of the input
            tensors. Default value is `cpu`.
        reduction: Specifies the reduction to apply to the output (mean | sum). If mean,
            the mean of the output is taken. If sum, the output will be summed. Default value is mean.
        kwargs: Optional kwargs to forward to loss function (e.g., weight).

    Returns:
        Wrapped combined loss function that reduces all input dimensions but the batch dimension.
    """
    assert reduction in ["sum", "mean"]
    fn = build_combined_loss_fn(
        fn_names, ranges, weights=weights, device=device, reduction="none", **kwargs
    )
    reduction_fn = _get_reduction_fn(reduction)

    def loss_fn(input: Tensor, target: Tensor) -> Tensor:
        loss = fn(input, target)
        return reduction_fn(loss, dim=tuple(range(1, input.ndim)))

    return loss_fn


def build_combined_loss_fn(
    fn_names: Sequence[str],
    ranges: Sequence[slice],
    weights: Sequence[float] | None = None,
    device: str | torch.device = "cpu",
    reduction: Literal["sum", "mean", "none"] = "mean",
    **kwargs,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Build a torch loss function made up of a combination of other losses.

    If all functions in `fn_names` can be compiled with `torch.compile`, then the returned function
    retains this ability.

    Args:
        fn_names: Loss function names to combine from `vc_core.dr.losses` module or
            `torch.nn.functional` that accepts input and target tensors respectively plus
            additional fixed kwargs (e.g., `mse_loss`).
        ranges: Slices for extracting the relevant views from input and target tensors for each
            loss function. Length must be the same as `fn_names`. The channel dimension that is
            sliced is assumed to be the last.
        weights: Optional weights for combining the individual loss functions. If `None`, then all
            losses are given equal weights of 1.0. Default value is `None`.
        device: Device to move `weights` tensor to. Should the same device as that of the input
            tensors. Default value is `cpu`.
        reduction: Specifies the reduction to apply to the output (none | mean | sum). If mean,
            the mean of the output is taken. If sum, the output will be summed. If none, no
            reduction will be applied. Default value is mean.
        kwargs: Optional kwargs to forward to all loss functions (e.g., weight).

    Returns:
        Combined loss function.
    """
    assert len(fn_names) == len(ranges), "Number of functions and slices must be equal."
    # Build all loss functions
    losses = [build_loss_fn(fn_name, reduction="none", **kwargs) for fn_name in fn_names]
    reduction_fn = _get_reduction_fn(reduction)
    weights = weights if isinstance(weights, list) else [1.0] * len(losses)
    assert len(weights) == len(ranges), "Number of functions and weights must be equal."
    weights = torch.cat(
        [torch.tensor([w] * _get_slice_len(s)) for w, s in zip(weights, ranges)]
    ).to(dtype=torch.float32, device=device)

    def loss_fn(input: Tensor, target: Tensor) -> Tensor:
        loss = (
            torch.cat([fn(input[..., s], target[..., s]) for fn, s in zip(losses, ranges)], dim=-1)
            * weights
        )
        return reduction_fn(loss)

    return loss_fn


def build_loss_fn(
    fn_name: str, reduction: Literal["sum", "mean", "none"] = "mean", **kwargs
) -> Callable[[Tensor, Tensor], Tensor]:
    """Build a torch loss function given its name and reduction method.

    If function defined by `fn_name` can be compiled with `torch.compile`, then the returned function
    retains this ability.

    Args:
        fn_name: Name of loss function from `vc_core.dr.losses` module or `torch.nn.functional`
            that accepts input and target tensors respectively plus additional fixed kwargs
            (e.g., `mse_loss`).
        reduction: Specifies the reduction to apply to the output (none | mean | sum). If mean,
            the mean of the output is taken. If sum, the output will be summed. If none, no
            reduction will be applied. Default value is mean.
        kwargs: Optional kwargs to forward to loss function (e.g., weight).

    Returns:
        Loss function.
    """
    # Search for function inside torch.nn.functional
    if hasattr(F, fn_name):
        return _build_loss_fn_torch(fn_name, reduction=reduction, **kwargs)
    # Else it must be from this module
    assert f"_build_{fn_name}" in globals(), (
        f"Function must be from torch.nn.functional or vc_core.dr.losses modules. Got {fn_name}."
    )
    return globals()[f"_build_{fn_name}"](reduction=reduction, **kwargs)


# region Private


def _build_loss_fn_torch(
    fn_name: str, reduction: Literal["sum", "mean", "none"] = "mean", **kwargs
) -> Callable[[Tensor, Tensor], Tensor]:
    """Build a torch loss function from torch given its name and reduction method.

    Args:
        fn_name: Name of loss function from `torch.nn.functional` that accepts input and target
            tensors respectively plus additional fixed kwargs (e.g., `mse_loss`).
        reduction: Specifies the reduction to apply to the output (none | mean | sum). If mean,
            the mean of the output is taken. If sum, the output will be summed. If none, no
            reduction will be applied. Default value is mean.
        kwargs: Optional kwargs to forward to loss function (e.g., weight).

    Returns:
        Loss function.
    """
    assert hasattr(F, fn_name), f"Function must be from torch.nn.functional. Got {fn_name}."
    assert reduction in ["sum", "mean", "none"], (
        f"Reduction must be one of sum, mean, none. Got {reduction}."
    )
    fn = getattr(F, fn_name)
    return partial(fn, reduction=reduction, **kwargs)


def _build_dice_loss(
    *, reduction: Literal["mean", "sum", "none"] = "mean", smooth: float = 1.0
) -> Callable[[Tensor, Tensor], Tensor]:
    """Builds a torch dice loss function for the given reduction method.

    Args:
        reduction: Specifies the reduction to apply to the output (none | mean | sum). If mean,
            the mean of the output is taken. If sum, the output will be summed. If none, no
            reduction will be applied. Default value is mean.
        smooth: Smoothing factor to apply to Dice coefficient, Default value is 1.0.

    Returns:
        Dice loss.
    """

    def dice_loss(input: Tensor, target: Tensor, smooth: float = 1.0) -> Tensor:
        """Compute the Dice loss with optional smoothing factor.

        Args:
            input: Predicted values in the range [0, 1].
            target Ground truth values. Shape is the same as `input`.
            smooth: Smoothing factor to apply to Dice coefficient, Default value is 1.0.

        Return:
            Dice loss.
        """
        dice = (2 * (input * target) + smooth) / (input + target + smooth)
        loss = 1.0 - dice
        return loss

    return _build_loss_fn(dice_loss, reduction=reduction, smooth=smooth)


def _build_loss_fn(
    fn: Callable[[Tensor, Tensor, Any], Tensor],
    reduction: Literal["sum", "mean", "none"] = "mean",
    **kwargs,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Build a torch loss function around the given function and reduction method.

    Args:
        fn: Function to wrap with given reduction method. The function should accept two tensors
            (input and target respectively) as well as any additional kwargs if needed and return
            the loss computed between them without any reductions applied.
        reduction: Specifies the reduction to apply to the output (none | mean | sum). If mean,
            the mean of the output is taken. If sum, the output will be summed. If none, no
            reduction will be applied. Default value is mean.
        kwargs: Optional kwargs to forward to loss function (e.g., weight).

    Returns:
        Loss function.
    """
    reduction_fn = _get_reduction_fn(reduction)

    def loss_fn(input: Tensor, target: Tensor) -> Tensor:
        loss = fn(input, target, **kwargs)
        return reduction_fn(loss)

    return loss_fn


def _get_reduction_fn(reduction: Literal["sum", "mean", "none"]) -> Callable[[Tensor], Tensor]:
    """Convert reduction literal to reduction function."""
    assert reduction in ["sum", "mean", "none"], (
        f"Reduction must be one of sum, mean, none. Got {reduction}."
    )
    match reduction:
        case "mean":
            return torch.mean
        case "sum":
            return torch.sum
        case _:
            return lambda x: x


def _get_slice_len(range: slice) -> int:
    """Get the length of a slice according to start, stop and step."""
    start = range.start if range.start is not None else 0
    step = range.step if range.step is not None else 1
    return (range.stop - start) // step
