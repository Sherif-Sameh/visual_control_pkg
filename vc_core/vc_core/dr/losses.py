from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor


def wrap_combined_loss_fn(
    fn_names: list[str],
    ranges: list[slice],
    weights: list[float] | None = None,
    device: str | torch.device = "cpu",
    reduction: Literal["sum", "mean"] = "mean",
    kwargs: list[dict[str, Any]] | None = None,
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
        kwargs: Optional sequence of kwargs to forward to loss functions (e.g., weight).

    Returns:
        Wrapped combined loss function that reduces all input dimensions but the batch dimension.
    """
    assert reduction in ["sum", "mean"]
    fn = build_combined_loss_fn(
        fn_names, ranges, weights=weights, device=device, reduction="none", kwargs=kwargs
    )
    reduction_fn = _get_reduction_fn(reduction)

    def loss_fn(input: Tensor, target: Tensor) -> Tensor:
        loss = fn(input, target)
        return reduction_fn(loss, dim=tuple(range(1, input.ndim)))

    return loss_fn


def build_combined_loss_fn(
    fn_names: list[str],
    ranges: list[slice],
    weights: list[float] | None = None,
    device: str | torch.device = "cpu",
    reduction: Literal["sum", "mean", "none"] = "mean",
    kwargs: list[dict[str, Any]] | None = None,
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
        kwargs: Optional sequence of kwargs to forward to loss functions (e.g., weight).

    Returns:
        Combined loss function.
    """
    assert len(fn_names) == len(ranges), "Number of functions and slices must be equal."
    kwargs = kwargs if isinstance(kwargs, list) else [{}] * len(fn_names)
    assert len(kwargs) == len(fn_names), "Number of functions and kwargs must be equal."
    # Build all loss functions
    losses = [build_loss_fn(name, reduction="none", **kw) for name, kw in zip(fn_names, kwargs)]
    reduction_fn = _get_reduction_fn(reduction)
    if weights is not None:
        weights = torch.tensor(weights, dtype=torch.float32, device=device)

    def loss_fn(input: Tensor, target: Tensor) -> Tensor:
        w = torch.ones_like(input) if weights is None else weights
        loss = (
            torch.cat([fn(input[..., s], target[..., s]) for fn, s in zip(losses, ranges)], dim=-1)
            * w
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
                device=device,

        Return:
            Dice loss.
        """
        dice = (2 * (input * target) + smooth) / (input + target + smooth)
        loss = 1.0 - dice
        return loss

    return _build_loss_fn(dice_loss, reduction=reduction, smooth=smooth)


def _build_mncc_loss(
    *, reduction: Literal["mean", "sum", "none"] = "mean", patch_size: int = 13
) -> Callable[[Tensor, Tensor], Tensor]:
    """Build a torch multi-scale NCC function for the given reduction method.

    The implementation of multi-scale normalized cross correlation (mNCC) follows the description given
    in the paper titled: `Intraoperative 2D/3D Image Registration via Differentiable X-ray Rendering`
    (https://arxiv.org/abs/2312.06358).

    mNCC is the combination of two losses: a global NCC and a local one computed over corresponding
    image patches in the two images.

    `reduction` is added for consistency with losses in `torch.nn.functional`. However, the mNCC loss
    cannot be computed without reducing the spatial dimensions since the global and local NCCs will
    have different spatial dimensions. Therefore, those dimensions are reduced and the final loss
    is repeated across them for spatial consistency with other losses.

    Args:
        reduction: Specifies the reduction to apply to the output (none | mean | sum). If mean,
            the mean of the output is taken. If sum, the output will be summed. If none, no
            reduction will be applied. Default value is mean.
        patch_size: Size for the square patches for local NCC. Default value is 13.

    Returns:
        Multi-scale NCC loss.
    """

    def normalize(x: Tensor) -> Tensor:
        """Normalize input by its mean and standard deviation."""
        mu = x.mean(dim=(-3, -2), keepdim=True)
        std = x.std(dim=(-3, -2), keepdim=True, correction=0)
        return (x - mu) / (std + 1e-6)

    def patchify(x: Tensor) -> Tensor:
        """Divide input to square patches and zero-pad as needed."""
        h, w = x.shape[-3:-1]
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        x_p = F.pad(x, (0, 0, 0, pad_w, 0, pad_h), mode="constant", value=0)
        x_p = x_p.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        x_p = x_p.permute(0, 1, 2, 4, 5, 3).contiguous().flatten(1, 2)
        return x_p

    def ncc(x: Tensor, y: Tensor) -> Tensor:
        """Compute NCC between inputs."""
        x_norm, y_norm = normalize(x), normalize(y)
        return (x_norm * y_norm).mean(dim=(-3, -2))

    def mncc_loss(input: Tensor, target: Tensor) -> Tensor:
        """Compute mNCC loss."""
        h, w = input.shape[-3:-1]
        input_patches = patchify(input)  # (B, nP, Ps, Ps, C)
        target_patches = patchify(target)  # (B, nP, Ps, Ps, C)
        global_ncc = ncc(input, target)  # (B, C)
        local_ncc = ncc(input_patches, target_patches).mean(dim=-2)  # (B, C)
        loss = -(global_ncc + local_ncc) / 2
        loss = loss[:, None, None].expand([-1, h, w, -1])
        return loss

    return _build_loss_fn(mncc_loss, reduction=reduction)


def _build_masked_loss(
    *, reduction: Literal["mean", "sum", "none"] = "mean", inner_fn_name: str, **kwargs
) -> Callable[[Tensor, Tensor], Tensor]:
    """Build a torch masked loss wrapped around the given function and reduction method.

    Inputs are assumed to contain the mask at index 0 along their channel (last) dimension.

    Args:
        inner_fn_name: Name of loss function to wrap from `vc_core.dr.losses` module or
            `torch.nn.functional` that accepts input and target tensors respectively plus
            additional fixed kwargs (e.g., `mse_loss`).
        reduction: Specifies the reduction to apply to the output (none | mean | sum). If mean,
            the mean of the output is taken. If sum, the output will be summed. If none, no
            reduction will be applied. Default value is mean.
        kwargs: Optional kwargs to forward to loss function (e.g., weight).

    Returns:
        Masked loss.
    """
    fn = build_loss_fn(inner_fn_name, reduction="none", **kwargs)

    def masked_loss(input: Tensor, target: Tensor) -> Tensor:
        """Computed masked loss."""
        loss = fn(input[..., 1:], target[..., 1:])
        loss = loss * target[..., :1]
        return loss

    return _build_loss_fn(masked_loss, reduction=reduction)


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
