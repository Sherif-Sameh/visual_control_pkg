from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from vc_core.utils.geometry.vector import (
    apply_tangent_rotation,
    get_rotation_from_z,
    get_tangent_basis,
)

from .hash_encoder import HashEncoder2D, HashEncoder2DCfg

if TYPE_CHECKING:
    from torch import Tensor


class EyePoseModel(nn.Module):
    """PyTorch model for estimating the pose of an eye mesh.

    Only 5 DOFs of the relative pose as observed by a camera can be estimated.
    These 5 DOFs are represented through the position of the eye and the orientation of its
    vertical axis (Z-axis) relative to the camera. In addition to these 5 DOFs, a sixth is
    added for scaling ambient light (i.e. diffuse albedo) to allow the model to fit to scenes
    that are uniformly brighter or dimmer than the mesh's own texture (i.e. diffuse albedo).

    The position is parameterized by zero-centering it around the initial guess/previous estimate
    and by scaling its magnitude by a set scaling factor. The Z-axis is parameterized through
    offsets in the 2D tangent space of the initial guess/previous estimate for its orientation
    scaled by `π`. The orthonormal basis for the tangent space is derived through Gram-Schmidt
    and so are the X and Y-axes to form a full right-handed rotation matrix. Ambient light is
    parameterized by zero-centering it around a default value of ones and scaling the learned
    parameter by a constant factor of 0.1.

    Args:
        pos: Initial guess for the position. Shape is (3,).
        z_dir: Initial guess for the direction of the Z-axis. Can be unnormalized. Shape is (3,).
        pos_sigma: Standard deviation for position noise to add to initial guess/previous estimate
            before starting optimization. If tensor, shape is (3,).
        z_tan_range: Range for sampling Z-axis tangent space around initial guess/previous estimate.
            If tensor, shape is (2,).
        n_rep: Number of times to repeat the model's parameters. If > 1, must be a value whose
            square root is exact. Default value is 1.
        scale: Optional scale factor for position offsets. Default value is 1.0.
    """

    AMBIENT_SCALE_FACTOR = 0.1

    def __init__(
        self,
        pos: Tensor,
        z_dir: Tensor,
        pos_sigma: float | Tensor,
        z_tan_range: float | Tensor,
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
        z_tan_range = (
            z_tan_range if torch.is_tensor(z_tan_range) else torch.tensor([z_tan_range] * 2)
        )
        z_tan_range = z_tan_range.to(dtype=torch.float32, device=device)
        assert pos.shape == (3,)
        assert z_dir.shape == (3,)
        assert pos_sigma.shape == (3,)
        assert z_tan_range.shape == (2,)
        # sample initial positions and Z-axis directions
        pos, z_dir, z_basis = self._sample_pos_and_z_dir(pos, z_dir, pos_sigma, z_tan_range, n_rep)
        # register buffers
        self.register_buffer("pos_init", pos)
        self.register_buffer("z_dir_init", z_dir)
        self.register_buffer("z_basis", z_basis)
        self.register_buffer("pos_sigma", pos_sigma)
        self.register_buffer("z_tan_range", z_tan_range)
        # create parameters
        self.pos_offset = nn.Parameter(torch.zeros_like(pos))
        self.z_tan = nn.Parameter(torch.zeros((n_rep, 2), dtype=torch.float32, device=device))
        self.amb_mult = nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device))

    @property
    def n_rep(self) -> int:
        """Get the number of copies of the model's pose parameters."""
        return self.pos_offset.shape[0]

    def forward(self) -> tuple[Tensor, Tensor, Tensor]:
        """Construct the model's pose and ambient light estimates return them.

        Returns:
            tuple containing three tensors. First is the position whose shape is (N, 3). Second is
            the rotation matrix whose shape (N, 3, 3). Third is the ambient light whose shape is (3,).
        """
        device = self.pos_offset.device
        # unnormalize position
        pos = self.pos_offset * self.scale + self.pos_init
        # apply tangent rotation to z-axis and create rotation matrix
        z_dir = apply_tangent_rotation(self.z_dir_init, self.z_tan, self.z_basis)
        rot = get_rotation_from_z(z_dir)
        # compute ambient light intensity
        amb = torch.ones(3, device=device) + self.amb_mult * self.AMBIENT_SCALE_FACTOR
        return pos, rot, amb

    @torch.no_grad
    def resample_params(self, pos: Tensor, z_dir: Tensor, **kwargs) -> None:
        """Update reference position and Z-axis direction and resample new parameters.

        Args:
            pos: Position to center new positions around. Shape is (3,).
            z_dir: Z-axis direction to center new directions around. Shape is (3,).
            **kwargs: Optional overrides for sampling paramters `pos_sigma`, `z_tan_range` and `n_rep`.
        """
        # apply overrides
        pos_sigma = kwargs.get("pos_sigma", self.pos_sigma)
        z_tan_range = kwargs.get("z_tan_range", self.z_tan_range)
        n_rep = kwargs.get("n_rep", self.n_rep)
        # resample reference position and Z-axis direction
        pos, z_dir, z_basis = self._sample_pos_and_z_dir(pos, z_dir, pos_sigma, z_tan_range, n_rep)
        # update buffers
        self.pos_init = pos
        self.z_dir_init = z_dir
        self.z_basis = z_basis
        # reset parameters
        self.pos_offset = nn.Parameter(torch.zeros_like(self.pos_init))
        self.z_tan = nn.Parameter(torch.zeros((n_rep, 2), dtype=torch.float32, device=pos.device))

    @staticmethod
    @torch.no_grad
    def _sample_pos_and_z_dir(
        pos: Tensor, z_dir: Tensor, pos_sigma: Tensor, z_tan_range: Tensor, n_samples: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample new positions and Z-axis unit vectors around given vectors.

        Args:
            pos: Mean position to sample around. Shape is (3,).
            z_dir: Z-axis direction to sample around. Shape is (2,)
            pos_sigma: Standard deviation for position noise. Shape is (3,).
            z_tan_range: Range for sampling tangent space. Shape is (2,).
            n_samples: Number of samples for each sampled vector.

        Returns:
            tuple of tensors. First is the sampled position offsets whose shape is (`n_samples`, 3).
            Second is the sampled Z-axis directions whose shape is (`n_samples`, 3). Third is the
            orthonormal basis for the sampled Z-axis directions whose shape is (`n_samples`, 3, 2).
        """
        if n_samples == 1:
            z_dir = z_dir.view(1, 3)
            return pos.view(1, 3), z_dir, torch.stack(get_tangent_basis(z_dir), dim=-1)
        device = pos.device
        # sample positions
        pos_sigma = pos_sigma.view(1, 3).expand([n_samples, -1])
        pos_s = pos + torch.normal(mean=0, std=pos_sigma)

        # compute Z-axis basis
        z_dir = z_dir.view(1, 3)
        z_basis = torch.stack(get_tangent_basis(z_dir), dim=-1)

        # sample from grid in Z-axis's tangent space
        n_samples_sqrt = int(math.sqrt(n_samples))
        assert n_samples_sqrt**2 == n_samples
        z_tan_0 = torch.linspace(-z_tan_range[0], z_tan_range[0], n_samples_sqrt, device=device)
        z_tan_1 = torch.linspace(-z_tan_range[1], z_tan_range[1], n_samples_sqrt, device=device)
        z_tan_s = torch.stack(torch.meshgrid(z_tan_0, z_tan_1, indexing="xy"), dim=-1).view(-1, 2)

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


class EyePoseTextureHashEncoderModel(EyePoseTextureModel):
    """PyTorch model for estimating the pose and texture of an eye mesh.

    Since estimating both camera/eye pose and texture from a single view is an ill-posed
    optimization problem. Multiple views of the same mesh must be used. Then, the learned texture
    must be able to fit all of them for different poses using the same fixed meshs UVs. Therefore,
    the `n_rep` parameter of `EyeModel` is re-purposed for this model as `n_view`.

    Refer to `vc_core.dr.common.model.EyePoseModel` for a detailed description of the eye pose
    parameterization.

    Uses a 2D multi-resolution hash encoder for embedding inputs to generate the final RGB texture.
    The embedding approach is based on the 3D voxel multi-res embedding presented in the paper
    titled `Instant Neural Graphics Primitives with a Multiresolution Hash Encoding` [0]. A small
    MLP network is used to convert the feature embeddings into RGB values. The embedding are
    evaluated at the grid center locations.

    Args:
        pos: Initial guess for positions. Shape is (N, 3) or (3,).
        z_dir: Initial guess for directions of the Z-axis. Can be unnormalized. Shape is (N, 3)
            or (3,).
        n_view: Number of different views to optimize simultaneously. Ignored if any of the other
            input parameters is 2D. Default value is 2.
        scale: Optional scale factor for position offsets. Default value is 1.0.
        enc_cfg: Configuration for the 2D hash encoder. Default initialized.
        mlp_n_layer: Number of hidden layers for RGB projection MLP. Default value is 2.
        mlp_n_feature: Number of features per hidden layer for RGB projection MLP. Default value is 64.

    References:
    [0] https://arxiv.org/abs/2201.05989
    """

    def __init__(
        self,
        pos: Tensor,
        z_dir: Tensor,
        *,
        n_view: int = 2,
        scale: float = 1.0,
        enc_cfg: HashEncoder2DCfg = HashEncoder2DCfg(),
        mlp_n_layer: int = 2,
        mlp_n_feature: int = 64,
    ):
        super().__init__(
            pos,
            z_dir,
            enc_cfg.finest_res,
            torch.zeros(3, device=pos.device),
            n_view=n_view,
            scale=scale,
            enc_cfg=enc_cfg,
            mlp_n_layer=mlp_n_layer,
            mlp_n_feature=mlp_n_feature,
        )
        self.res = enc_cfg.finest_res
        self.uv_max = enc_cfg.uv_max

    def _init_texture(self, _: tuple[int, int], text_rgb: Tensor, **kwargs) -> None:
        """Initialize texture parameter from inputs.

        Args:
            res: Texture resolution tuple (H, W).
            text_rgb: Initial color [0, 1] to apply to texture. Shape is (3,).
            **kwargs: Additonal arguments for initializing texture.
        """
        device = text_rgb.device
        enc_cfg: HashEncoder2DCfg = kwargs["enc_cfg"]
        mlp_n_layer: int = kwargs["mlp_n_layer"]
        mlp_n_feature: int = kwargs["mlp_n_feature"]
        # initialize hash encoder
        self.enc = HashEncoder2D(enc_cfg).to(device=device)
        # initialize projection mlp
        layers = []
        in_dim, out_dim = self.enc.feature_dim, mlp_n_feature
        for _ in range(mlp_n_layer):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 3))
        self.texture = nn.Sequential(*layers).to(device=device)

    def _get_texture(self) -> Tensor:
        """Compute output texture from internal representation.

        Returns:
            Texture map [0, 1]. Shape is (3, H, W).
        """
        device = self.pos_offset.device
        # sample grid UVs
        linear = torch.linspace(0.0, self.uv_max, self.res + 1, device=device)[:-1]
        grid = torch.stack(torch.meshgrid(linear, linear, indexing="xy"), dim=-1)
        # evaluate embeddings
        embd = self.enc(grid[None])[0]  # (H, W, feature_dim)
        # project to RGB and permute axes
        texture = self.texture(embd)  # (H, W, 3)
        texture = texture.permute(2, 0, 1)
        return torch.clamp(texture, 0, 1)
