from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from vc_core.utils.geometry.vector import (
    apply_tangent_rotation,
    get_rotation_from_z,
    get_tangent_basis,
)

if TYPE_CHECKING:
    from torch import Tensor


class EyePoseModel(nn.Module):
    """PyTorch model for estimating the pose of an eye mesh.

    Only 5 DOFs of the relative pose as observed by a camera can be estimated.
    These 5 DOFs are represented through the position of the eye and the orientation of its
    vertical axis (Z-axis) relative to the camera.

    The position is parameterized by zero-centering it around the initial guess/previous estimate
    and by scaling its magnitude by a set scaling factor. The Z-axis is parameterized through
    offsets in the 2D tangent space of the initial guess/previous estimate for its orientation
    scaled by `π`. The orthonormal basis for the tangent space is derived through Gram-Schmidt
    and so are the X and Y-axes to form a full right-handed rotation matrix.

    Args:
        pos: Initial guess for the position. Shape is (3,).
        z_dir: Initial guess for the direction of the Z-axis. Can be unnormalized. Shape is (3,).
        pos_sigma: Standard deviation for position noise to add to initial guess/previous estimate
            before starting optimization. If tensor, shape is (3,).
        z_tan_sigma: Standard deviation for Z-axis tangent noise to rotate initial guess/previous
            estimate by before starting optimization. If tensor, shape is (2,).
        n_rep: Number of times to repeat the model's parameters. Default value is 1.
        scale: Optional scale factor for position offsets. Default value is 1.0.
    """

    def __init__(
        self,
        pos: Tensor,
        z_dir: Tensor,
        pos_sigma: float | Tensor,
        z_tan_sigma: float | Tensor,
        *,
        n_rep: int = 1,
        scale: float = 1.0,
    ):
        super().__init__()
        self.scale = scale
        device = pos.device
        # preprocess input tensors
        z_dir = F.normalize(z_dir, dim=-1)
        pos_sigma = pos_sigma if torch.is_tensor(pos_sigma) else torch.tensor([pos_sigma] * 3)
        pos_sigma = pos_sigma.to(dtype=torch.float32, device=device)
        z_tan_sigma = (
            z_tan_sigma if torch.is_tensor(z_tan_sigma) else torch.tensor([z_tan_sigma] * 2)
        )
        z_tan_sigma = z_tan_sigma.to(dtype=torch.float32, device=device)
        assert pos.shape == (3,)
        assert z_dir.shape == (3,)
        assert pos_sigma.shape == (3,)
        assert z_tan_sigma.shape == (2,)
        # sample initial positions and Z-axis directions
        pos, z_dir, z_basis = self._sample_pos_and_z_dir(pos, z_dir, pos_sigma, z_tan_sigma, n_rep)
        # register buffers
        self.register_buffer("pos_init", pos)
        self.register_buffer("z_dir_init", z_dir)
        self.register_buffer("z_basis", z_basis)
        self.register_buffer("pos_sigma", pos_sigma)
        self.register_buffer("z_tan_sigma", z_tan_sigma)
        # create parameters
        self.pos_offset = nn.Parameter(torch.zeros_like(pos))
        self.z_tan = nn.Parameter(torch.zeros((n_rep, 2), dtype=torch.float32, device=device))

    @property
    def n_rep(self) -> int:
        """Get the number of copies of the model's pose parameters."""
        return self.pos_offset.shape[0]

    def forward(self) -> tuple[Tensor, Tensor]:
        """Construct the model's pose estimates return them.

        Returns:
            tuple containing two tensors. First is the position whose shape is (N, 3). Second is
            the rotation matrix whose shape (N, 3, 3).
        """
        # unnormalize position
        pos = self.pos_offset * self.scale + self.pos_init
        # apply tangent rotation to z-axis and create rotation matrix
        z_dir = apply_tangent_rotation(self.z_dir_init, self.z_tan, self.z_basis)
        rot = get_rotation_from_z(z_dir)
        return pos, rot

    @torch.no_grad
    def resample_params(self, pos: Tensor, z_dir: Tensor) -> None:
        """Update reference position and Z-axis direction and resample new parameters.

        Args:
            pos: Position to center new positions around. Shape is (3,).
            z_dir: Z-axis direction to center new directions around. Shape is (3,).
        """
        # resample reference position and Z-axis direction
        pos, z_dir, z_basis = self._sample_pos_and_z_dir(
            pos, z_dir, self.pos_sigma, self.z_tan_sigma, self.n_rep
        )
        # update buffers
        self.pos_init = pos
        self.z_dir_init = z_dir
        self.z_basis = z_basis
        # reset parameters
        self.pos_offset.zero_()
        self.z_tan.zero_()

    @staticmethod
    @torch.no_grad
    def _sample_pos_and_z_dir(
        pos: Tensor, z_dir: Tensor, pos_sigma: Tensor, z_tan_sigma: Tensor, n_samples: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample new positions and Z-axis unit vectors around given vectors.

        Args:
            pos: Mean position to sample around. Shape is (3,).
            z_dir: Z-axis direction to sample around. Shape is (2,)
            pos_sigma: Standard deviation for position noise. Shape is (3,).
            z_tan_sigma: Standard deviation for tangent noise. Shape is (2,).
            n_samples: Number of samples for each sampled vector.

        Returns:
            tuple of tensors. First is the sampled position offsets whose shape is (`n_samples`, 3).
            Second is the sampled Z-axis directions whose shape is (`n_samples`, 3). Third is the
            orthonormal basis for the sampled Z-axis directions whose shape is (`n_samples`, 3, 2).
        """
        # sample positions
        pos_sigma = pos_sigma.view(1, 3).expand([n_samples, -1])
        pos_s = pos + torch.normal(mean=0, std=pos_sigma)

        # compute Z-axis basis and sample in its tangent space
        z_dir = z_dir.view(1, 3)
        z_basis = torch.stack(get_tangent_basis(z_dir), dim=-1)
        z_tan_sigma = z_tan_sigma.view(1, 2).expand([n_samples, -1])
        z_tan_s = torch.normal(mean=0, std=z_tan_sigma)

        # update Z-axis directions and re-compute basis
        z_dir_s = apply_tangent_rotation(z_dir, z_tan_s, z_basis)
        z_basis_s = torch.stack(get_tangent_basis(z_dir_s), dim=-1)
        return pos_s, z_dir_s, z_basis_s


class EyePoseTextureModel(nn.Module):
    """PyTorch model for estimating the pose and texture of an eye mesh.

    Since estimating both camera/eye pose and texture from a single view is an ill-posed
    optimization problem. Multiple views of the same mesh must be used. Then, the learned texture
    must be able to fit all of them for different poses using the same fixed meshs UVs. Therefore,
    the `n_rep` parameter of `EyeModel` is re-purposed for this model as `n_view`.

    Refer to `vc_core.dr.common.model.EyePoseModel` for a detailed description of the eye pose
    parameterization.

    Texture is modeled only through a single full-resolution RGB image of shape (3, H, W). Only a
    single texture is stored and output by the model.

    Args:
        pos: Initial guess for positions. Shape is (N, 3) or (3,).
        z_dir: Initial guess for directions of the Z-axis. Can be unnormalized. Shape is (N, 3)
            or (3,).
        res: Texture resolution (H, W).
        text_rgb: Initial color [0, 1] to apply to texture. Shape is (3,).
        n_view: Number of different views to optimize simultaneously. Ignored if any of the other
            input parameters is 2D. Default value is 2.
        scale: Optional scale factor for position offsets. Default value is 1.0.
        **kwargs: Additional arguments for initializing texture representation.
    """

    def __init__(
        self,
        pos: Tensor,
        z_dir: Tensor,
        res: tuple[int, int] | int,
        text_rgb: Tensor,
        *,
        n_view: int = 2,
        scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.scale = scale
        device = pos.device
        n_view = self._get_n_view(pos, z_dir, n_view)
        # preprocess all inputs
        pos = pos.repeat((n_view, 1)) if pos.ndim == 1 else pos
        pos = pos.to(dtype=torch.float32, device=device)
        z_dir = z_dir.repeat((n_view, 1)) if z_dir.ndim == 1 else z_dir
        z_dir = F.normalize(z_dir, dim=-1).to(dtype=torch.float32, device=device)
        z_basis = torch.stack(get_tangent_basis(z_dir), dim=-1)
        res = (res, res) if isinstance(res, int) else res
        text_rgb = text_rgb.clamp(0, 1).to(dtype=torch.float32, device=device)
        # register buffers
        self.register_buffer("pos_init", pos)
        self.register_buffer("z_dir_init", z_dir)
        self.register_buffer("z_basis", z_basis)
        # create parameters
        self.pos_offset = nn.Parameter(torch.zeros_like(pos))
        self.z_tan = nn.Parameter(torch.zeros((n_view, 2), dtype=torch.float32, device=device))
        self._init_texture(res, text_rgb, **kwargs)

    def forward(self) -> tuple[Tensor, Tensor, Tensor]:
        """Construct the model's pose and texture estimates return them.

        Returns:
            tuple containing three tensors. First is the position whose shape is (N, 3). Second is
            the rotation matrix whose shape (N, 3, 3). Third is the texture [0, 1] whose shape is
            (3, H, W).
        """
        # unnormalize position
        pos = self.pos_offset * self.scale + self.pos_init
        # apply tangent rotation to z-axis and create rotation matrix
        z_dir = apply_tangent_rotation(self.z_dir_init, self.z_tan, self.z_basis)
        rot = get_rotation_from_z(z_dir)
        # get output texture
        texture = self._get_texture()
        return pos, rot, texture

    @staticmethod
    def _get_n_view(pos: Tensor, z_dir: Tensor, n_view: int) -> int:
        if pos.ndim == 2:
            return pos.shape[0]
        if z_dir.ndim == 2:
            return z_dir.shape[0]
        return n_view

    def _init_texture(self, res: tuple[int, int], text_rgb: Tensor, **kwargs) -> None:
        """Initialize texture parameter from inputs.

        Args:
            res: Texture resolution tuple (H, W).
            text_rgb: Initial color [0, 1] to apply to texture. Shape is (3,).
            **kwargs: Additonal arguments for initializing texture.
        """
        # repeat initial RGB color across H and W
        H, W = res
        text_rgb = text_rgb[:, None, None].repeat([1, H, W])
        # create texture parameter
        self.texture = nn.Parameter(text_rgb)

    def _get_texture(self) -> Tensor:
        """Compute output texture from internal representation.

        Returns:
            Texture map [0, 1]. Shape is (3, H, W).
        """
        return torch.clamp(self.texture, 0, 1)


class EyePoseTextureMipmapModel(EyePoseTextureModel):
    """PyTorch model for estimating the pose and texture of an eye mesh.

    Since estimating both camera/eye pose and texture from a single view is an ill-posed
    optimization problem. Multiple views of the same mesh must be used. Then, the learned texture
    must be able to fit all of them for different poses using the same fixed meshs UVs. Therefore,
    the `n_rep` parameter of `EyeModel` is re-purposed for this model as `n_view`.

    Refer to `vc_core.dr.common.model.EyePoseModel` for a detailed description of the eye pose
    parameterization.

    Unlike `vc_core.dr.common.model.EyePoseTextureModel`, a hierarchy/pyramid of textures is used
    to represent the final texture. The approach is based on a similar idea to mipmapping in
    rendering. Each level stores a texture with half the resolution of the previous one, starting
    from the full resolution set by `res`. Textures are combined by upsampling lower-res textures
    then summing all of them. This allows coarse textures to handle low-frequency features, while
    high-res ones focus on higher frequency details. Only the coarsest texture will be initialized
    to `text_rgb`, others are all zero-initialized.

    Args:
        pos: Initial guess for positions. Shape is (N, 3) or (3,).
        z_dir: Initial guess for directions of the Z-axis. Can be unnormalized. Shape is (N, 3)
            or (3,).
        res: Texture resolution (H, W).
        text_rgb: Initial color [0, 1] to apply to texture. Shape is (3,).
        n_view: Number of different views to optimize simultaneously. Ignored if any of the other
            input parameters is 2D. Default value is 2.
        scale: Optional scale factor for position offsets. Default value is 1.0.
        n_level: Number of levels for texture mipmapping. Default value is 3.
        mode: Mode for interpolation using `torch.nn.functional.interpolate`. Default value is
            `nearest`.
    """

    def __init__(
        self,
        pos: Tensor,
        z_dir: Tensor,
        res: tuple[int, int] | int,
        text_rgb: Tensor,
        *,
        n_view: int = 2,
        scale: float = 1.0,
        n_level: int = 2,
        mode: str = "nearest",
    ):
        super().__init__(pos, z_dir, res, text_rgb, n_view=n_view, scale=scale, n_level=n_level)
        self.mode = mode

    def _init_texture(self, res: tuple[int, int], text_rgb: Tensor, **kwargs) -> None:
        """Initialize texture parameter from inputs.

        Args:
            res: Texture resolution tuple (H, W).
            text_rgb: Initial color [0, 1] to apply to texture. Shape is (3,).
            **kwargs: Additonal arguments for initializing texture.
        """
        device = text_rgb.device
        n_level: int = kwargs["n_level"]
        # compute H and W for all texture levels
        H, W = res
        scale = 2 ** torch.arange(0, n_level, device=device).float()
        heights, widths = (H / scale).long(), (W / scale).long()
        assert torch.all(heights > 0) and torch.all(widths > 0)
        # repeat initial RGB color across heights[-1] and widths[-1]
        text_rgb = text_rgb[:, None, None].repeat([1, heights[-1], widths[-1]])
        # create texture parameters
        texts_init = []
        for i, (h, w) in enumerate(zip(heights, widths)):
            if i == (n_level - 1):
                texts_init.append(text_rgb)
                continue
            texts_init.append(torch.zeros((3, h, w), dtype=torch.float32, device=device))
        self.texture = nn.ParameterList([nn.Parameter(text) for text in texts_init])

    def _get_texture(self) -> Tensor:
        """Compute output texture from internal representation.

        Returns:
            Texture map [0, 1]. Shape is (3, H, W).
        """
        n_level = len(self.texture)
        scale = 2 ** torch.arange(1, n_level).float()
        # upsample lower-res textures
        texture = [self.texture[0].unsqueeze(0)]
        for i, s in enumerate(scale):
            texture.append(
                F.interpolate(
                    self.texture[i + 1].unsqueeze(0), scale_factor=s.item(), mode=self.mode
                )
            )
        # sum textures
        texture = torch.cat(texture, dim=0).sum(dim=0)
        return torch.clamp(texture, 0, 1)
