from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from vc_core.utils.geometry.rotation import apply_tangent_rotation
from vc_core.utils.geometry.vector import (
    apply_tangent_rotation_exact,
    get_rotation_from_z,
    get_tangent_basis,
)

from .hash_encoder import HashEncoder2D, HashEncoder2DCfg

if TYPE_CHECKING:
    from torch import Tensor


class EyePoseModel(nn.Module):
    """PyTorch model for estimating the pose of an eye mesh.

    The position is parameterized by zero-centering it around the initial guess/previous estimate
    and by scaling its magnitude by a set scaling factor. The rotation is parameterized through
    offsets in the 3D tangent space of the initial guess/previous estimate for its orientation
    scaled by `π`. It is assumed that relative rotation about the Z-axis is almost zero. Therefore,
    tangent rotations are sampled on a grid for XY only and Z is set to zero.

    Args:
        pos: Initial guess for the position. Shape is (3,).
        rot: Initial guess for the rotation. Shape is (3, 3).
        pos_sigma: Standard deviation for position noise to add to initial guess/previous estimate
            before starting optimization. If tensor, shape is (3,).
        z_tan_range: Range for sampling Z-axis tangent space around initial guess/previous estimate.
            If tensor, shape is (2,).
        n_rep: Number of times to repeat the model's parameters. If > 1, must be a value whose
            square root is exact. Default value is 1.
        scale: Optional scale factor for position offsets. Default value is 1.0.
    """

    TAN_SIGMA = 0.015

    def __init__(
        self,
        pos: Tensor,
        rot: Tensor,
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
        pos_sigma = pos_sigma if torch.is_tensor(pos_sigma) else torch.tensor([pos_sigma] * 3)
        pos_sigma = pos_sigma.to(dtype=torch.float32, device=device)
        tan_sigma = torch.tensor([self.TAN_SIGMA, self.TAN_SIGMA, 0.0])
        tan_sigma = tan_sigma.to(dtype=torch.float32, device=device)
        z_tan_range = (
            z_tan_range if torch.is_tensor(z_tan_range) else torch.tensor([z_tan_range] * 2)
        )
        z_tan_range = z_tan_range.to(dtype=torch.float32, device=device)
        assert pos.shape == (3,)
        assert rot.shape == (3, 3)
        assert pos_sigma.shape == (3,)
        assert z_tan_range.shape == (2,)
        # sample initial positions and orientations
        pos, rot = self._sample_pos_and_rot(pos, rot, pos_sigma, z_tan_range, n_rep)
        # register buffers
        self.register_buffer("pos_init", pos)
        self.register_buffer("rot_init", rot)
        self.register_buffer("pos_sigma", pos_sigma)
        self.register_buffer("tan_sigma", tan_sigma)
        self.register_buffer("z_tan_range", z_tan_range)
        # create parameters
        self.pos_offset = nn.Parameter(torch.zeros_like(pos))
        self.rot_tan = nn.Parameter(torch.randn_like(pos) * tan_sigma)

    @property
    def n_rep(self) -> int:
        """Get the number of copies of the model's pose parameters."""
        return self.pos_offset.shape[0]

    def forward(self) -> tuple[Tensor, Tensor]:
        """Construct the model's pose estimates and return them.

        Returns:
            Tuple containing two tensors. First is the position whose shape is (N, 3). Second is
            the rotation matrix whose shape (N, 3, 3).
        """
        # unnormalize position
        pos = self.pos_offset * self.scale + self.pos_init
        # apply tangent rotation
        rot = apply_tangent_rotation(self.rot_init, self.rot_tan)
        return pos, rot

    @torch.no_grad
    def resample_params(self, pos: Tensor, rot: Tensor, **kwargs) -> None:
        """Update reference position and orientation and resample new parameters.

        Args:
            pos: Position to center new positions around. Shape is (3,).
            rot: Rotation to center new orientations around. Shape is (3, 3).
            **kwargs: Optional overrides for sampling paramters `pos_sigma`, `z_tan_range` and `n_rep`.
        """
        # apply overrides
        pos_sigma = kwargs.get("pos_sigma", self.pos_sigma)
        z_tan_range = kwargs.get("z_tan_range", self.z_tan_range)
        n_rep = kwargs.get("n_rep", self.n_rep)
        # resample reference position and orientation
        pos, rot = self._sample_pos_and_rot(pos, rot, pos_sigma, z_tan_range, n_rep)
        # update buffers
        self.pos_init = pos
        self.rot_init = rot
        # reset parameters
        self.pos_offset = nn.Parameter(torch.zeros_like(self.pos_init))
        self.rot_tan = nn.Parameter(torch.randn_like(pos) * self.tan_sigma)

    @staticmethod
    @torch.no_grad
    def _sample_pos_and_rot(
        pos: Tensor, rot: Tensor, pos_sigma: Tensor, z_tan_range: Tensor, n_samples: int
    ) -> tuple[Tensor, Tensor]:
        """Sample new positions and orientations around given values.

        Args:
            pos: Position to sample around. Shape is (3,).
            rot: Rotation to sample around. Shape is (3, 3).
            pos_sigma: Standard deviation for position noise. Shape is (3,).
            z_tan_range: Range for sampling Z-axis tangent space. Shape is (2,).
            n_samples: Number of samples for each sampled vector.

        Returns:
            Tuple of tensors. First is the sampled positions whose shape is (`n_samples`, 3).
            Second is the sampled rotations whose shape is (`n_samples`, 3, 3).
        """
        if n_samples == 1:
            return pos.view(1, 3), rot.view(1, 3, 3)
        device = pos.device
        # sample positions
        pos_sigma = pos_sigma.view(1, 3).expand([n_samples, -1])
        pos_s = pos + torch.normal(mean=0, std=pos_sigma)

        # sample from grid in Z-axis's tangent space
        n_samples_sqrt = int(math.sqrt(n_samples))
        assert n_samples_sqrt**2 == n_samples
        z_tan_0 = torch.linspace(-z_tan_range[0], z_tan_range[0], n_samples_sqrt, device=device)
        z_tan_1 = torch.linspace(-z_tan_range[1], z_tan_range[1], n_samples_sqrt, device=device)
        z_tan_s = torch.stack(torch.meshgrid(z_tan_0, z_tan_1, indexing="xy"), dim=-1).view(-1, 2)

        # complete tangent vector and compute resulting rotations
        tan_s = torch.cat([z_tan_s, torch.zeros_like(z_tan_s[:, :1])], dim=-1)
        rot_s = apply_tangent_rotation(rot, tan_s)
        return pos_s, rot_s


class EyePoseMeshTextureModel(nn.Module):
    """PyTorch model for estimating the pose, mesh and texture of an eye mesh.

    Since estimating all of these parameters from a single view is an ill-posed optimization
    problem. Multiple views of the same mesh must be used. Then, the learned mesh and texture
    must be able to fit all of them for different poses using the same fixed meshs UVs. Therefore,
    the `n_rep` parameter of `EyeModel` is re-purposed for this model as `n_view`.

    Refer to `vc_core.dr.common.model.EyePoseModel` for a detailed description of the eye pose
    parameterization.

    The mesh geometry is paramterized through per-vertex offsets. To account for the scale mismatch
    between mesh vertices and other parameters, vertex offsets are scaled by `scale_mesh`.

    Texture is modeled only through a single full-resolution RGB image of shape (3, `res`, `res`).
    Only a single texture model is stored and used by the model.

    Args:
        pos: Initial guess for positions. Shape is (N, 3) or (3,).
        z_dir: Initial guess for directions of the Z-axis. Can be unnormalized. Shape is (N, 3)
            or (3,).
        n_vertex: Number of vertex offsets to learn for updating the mesh.
        res: Texture resolution.
        text_init: Initial value to apply to texture. Shape is (3,).
        n_view: Number of different views to optimize simultaneously. Ignored if any of the other
            input parameters is 2D. Default value is 2.
        scale: Optional scale factor for position offsets. Default value is 1.0.
        scale_mesh: Optional scale factor for mesh vertex offsets. Default value is 1e-3.
        **kwargs: Additional arguments for initializing texture representation.
    """

    def __init__(
        self,
        pos: Tensor,
        z_dir: Tensor,
        n_vertex: int,
        res: int,
        text_init: Tensor,
        *,
        n_view: int = 2,
        scale: float = 1.0,
        scale_mesh: float = 1e-3,
        **kwargs,
    ):
        super().__init__()
        self.scale = scale
        self.scale_mesh = scale_mesh
        device = pos.device
        n_view = self._get_n_view(pos, z_dir, n_view)
        # preprocess all inputs
        pos = pos.repeat((n_view, 1)) if pos.ndim == 1 else pos
        pos = pos.to(dtype=torch.float32, device=device)
        z_dir = z_dir.repeat((n_view, 1)) if z_dir.ndim == 1 else z_dir
        z_dir = F.normalize(z_dir, dim=-1).to(dtype=torch.float32, device=device)
        z_basis = torch.stack(get_tangent_basis(z_dir), dim=-1)
        vertex_offsets = torch.zeros((n_vertex, 3), dtype=torch.float32, device=device)
        text_init = text_init.to(dtype=torch.float32, device=device)
        # register buffers
        self.register_buffer("pos_init", pos)
        self.register_buffer("z_dir_init", z_dir)
        self.register_buffer("z_basis", z_basis)
        # create parameters
        self.pos_offset = nn.Parameter(torch.zeros_like(pos))
        self.z_tan = nn.Parameter(torch.zeros((n_view, 2), dtype=torch.float32, device=device))
        self.vertex_offsets = nn.Parameter(vertex_offsets)
        self._init_texture(res, text_init, **kwargs)

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
        z_dir = apply_tangent_rotation_exact(self.z_dir_init, self.z_tan, self.z_basis)
        rot = get_rotation_from_z(z_dir)
        # scale vertex offsets
        vertex_offsets = self.vertex_offsets * self.scale_mesh
        # get output texture
        texture = self._get_texture()
        return pos, rot, vertex_offsets, texture

    @staticmethod
    def _get_n_view(pos: Tensor, z_dir: Tensor, n_view: int) -> int:
        if pos.ndim == 2:
            return pos.shape[0]
        if z_dir.ndim == 2:
            return z_dir.shape[0]
        return n_view

    def _init_texture(self, res: int, text_init: Tensor, **kwargs) -> None:
        """Initialize texture parameter from inputs.

        Args:
            res: Texture resolution.
            text_init: Initial value to apply to texture. Shape is (3,).
            **kwargs: Additonal arguments for initializing texture.
        """
        # repeat initial value across H and W
        text_init = text_init[:, None, None].repeat([1, res, res])
        # create texture parameter
        self.texture = nn.Parameter(text_init)

    def _get_texture(self) -> Tensor:
        """Compute output texture from internal representation.

        Returns:
            Texture map [0, 1]. Shape is (3, `res`, `res`).
        """
        return torch.sigmoid(self.texture)


class EyePoseMeshTextureMipmapModel(EyePoseMeshTextureModel):
    """PyTorch model for estimating the pose, mesh and texture of an eye mesh.

    Since estimating all of these parameters from a single view is an ill-posed optimization
    problem. Multiple views of the same mesh must be used. Then, the learned mesh and texture
    must be able to fit all of them for different poses using the same fixed meshs UVs. Therefore,
    the `n_rep` parameter of `EyeModel` is re-purposed for this model as `n_view`.

    Refer to `vc_core.dr.common.model.EyePoseModel` for a detailed description of the eye pose
    parameterization.

    The mesh geometry is paramterized through per-vertex offsets. To account for the scale mismatch
    between mesh vertices and other parameters, vertex offsets are scaled by `scale_mesh`.

    Unlike `vc_core.dr.common.model.EyePoseTextureModel`, a hierarchy/pyramid of textures is used
    to represent the learned texture. The approach is based on a similar idea to mipmapping in
    rendering. Each level stores a texture with half the resolution of the previous one, starting
    from the full resolution set by `res`. Textures are combined by upsampling lower-res textures
    then summing all of them. This allows coarse textures to handle low-frequency features, while
    high-res ones focus on higher frequency details. Only the coarsest texture will be initialized
    to `text_init`, others are all zero-initialized.

    Args:
        pos: Initial guess for positions. Shape is (N, 3) or (3,).
        z_dir: Initial guess for directions of the Z-axis. Can be unnormalized. Shape is (N, 3)
            or (3,).
        n_vertex: Number of vertex offsets to learn for updating the mesh.
        res: Texture resolution.
        text_init: Initial value to apply to texture. Shape is (3,).
        n_view: Number of different views to optimize simultaneously. Ignored if any of the other
            input parameters is 2D. Default value is 2.
        scale: Optional scale factor for position offsets. Default value is 1.0.
        scale_mesh: Optional scale factor for mesh vertex offsets. Default value is 1e-3.
        n_level: Number of levels for texture mipmapping. Default value is 3.
        mode: Mode for interpolation using `torch.nn.functional.interpolate`. Default value is
            `bilinear`.
    """

    def __init__(
        self,
        pos: Tensor,
        z_dir: Tensor,
        n_vertex: int,
        res: int,
        text_init: Tensor,
        *,
        n_view: int = 2,
        scale: float = 1.0,
        scale_mesh: float = 1e-3,
        n_level: int = 2,
        mode: str = "bilinear",
    ):
        super().__init__(
            pos,
            z_dir,
            n_vertex,
            res,
            text_init,
            n_view=n_view,
            scale=scale,
            scale_mesh=scale_mesh,
            n_level=n_level,
        )
        self.mode = mode

    def _init_texture(self, res: int, text_init: Tensor, **kwargs) -> None:
        """Initialize texture parameter from inputs.

        Args:
            res: Texture resolution.
            text_init: Initial value to apply to texture. Shape is (3,).
            **kwargs: Additonal arguments for initializing texture.
        """
        device = text_init.device
        n_level: int = kwargs["n_level"]
        # compute H and W for all texture levels
        scale = 2 ** torch.arange(0, n_level, device=device).float()
        heights, widths = (res / scale).long(), (res / scale).long()
        assert torch.all(heights > 0) and torch.all(widths > 0)
        # repeat initial value across heights[-1] and widths[-1]
        text_init = text_init[:, None, None].repeat([1, heights[-1], widths[-1]])
        # create texture parameters
        texts_init = []
        for i, (h, w) in enumerate(zip(heights, widths)):
            if i == (n_level - 1):
                texts_init.append(text_init)
                continue
            texts_init.append(torch.zeros((3, h, w), dtype=torch.float32, device=device))
        self.texture = nn.ParameterList([nn.Parameter(text) for text in texts_init])

    def _get_texture(self) -> Tensor:
        """Compute output texture from internal representation.

        Returns:
            Texture map [0, 1]. Shape is (3, `res`, `res`).
        """
        n_level = len(self.texture)
        scale = (2 ** torch.arange(1, n_level)).tolist()
        # upsample lower-res textures
        texture = [self.texture[0].unsqueeze(0)]
        for i, s in enumerate(scale):
            texture.append(
                F.interpolate(
                    self.texture[i + 1].unsqueeze(0), scale_factor=float(s), mode=self.mode
                )
            )
        # sum textures
        texture = torch.cat(texture, dim=0).sum(dim=0)
        return torch.sigmoid(texture)


class EyePoseMeshTextureHashEncoderModel(EyePoseMeshTextureModel):
    """PyTorch model for estimating the pose, mesh and texture of an eye mesh.

    Since estimating all of these parameters from a single view is an ill-posed optimization
    problem. Multiple views of the same mesh must be used. Then, the learned mesh and texture
    must be able to fit all of them for different poses using the same fixed meshs UVs. Therefore,
    the `n_rep` parameter of `EyeModel` is re-purposed for this model as `n_view`.

    Refer to `vc_core.dr.common.model.EyePoseModel` for a detailed description of the eye pose
    parameterization.

    The mesh geometry is paramterized through per-vertex offsets. To account for the scale mismatch
    between mesh vertices and other parameters, vertex offsets are scaled by `scale_mesh`.

    Uses a 2D multi-resolution hash encoder for embedding inputs to generate the learned RGB texture.
    The embedding approach is based on the 3D voxel multi-res embedding presented in the paper
    titled `Instant Neural Graphics Primitives with a Multiresolution Hash Encoding` [0]. A small
    MLP network is used to convert the feature embeddings into RGB values. The embedding are
    evaluated at the grid center locations.

    Args:
        pos: Initial guess for positions. Shape is (N, 3) or (3,).
        z_dir: Initial guess for directions of the Z-axis. Can be unnormalized. Shape is (N, 3)
            or (3,).
        n_vertex: Number of vertex offsets to learn for updating the mesh.
        n_view: Number of different views to optimize simultaneously. Ignored if any of the other
            input parameters is 2D. Default value is 2.
        scale: Optional scale factor for position offsets. Default value is 1.0.
        scale_mesh: Optional scale factor for mesh vertex offsets. Default value is 1e-3.
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
        n_vertex: int,
        *,
        n_view: int = 2,
        scale: float = 1.0,
        scale_mesh: float = 1e-3,
        enc_cfg: HashEncoder2DCfg = HashEncoder2DCfg(),
        mlp_n_layer: int = 2,
        mlp_n_feature: int = 64,
    ):
        super().__init__(
            pos,
            z_dir,
            n_vertex,
            enc_cfg.finest_res,
            torch.zeros(3, device=pos.device),
            n_view=n_view,
            scale=scale,
            scale_mesh=scale_mesh,
            enc_cfg=enc_cfg,
            mlp_n_layer=mlp_n_layer,
            mlp_n_feature=mlp_n_feature,
        )
        self.res = enc_cfg.finest_res
        self.uv_max = enc_cfg.uv_max

    def _init_texture(self, _: int, text_init: Tensor, **kwargs) -> None:
        """Initialize texture parameter from inputs.

        Args:
            res: Texture resolution.
            text_init: Initial value to apply to texture. Unused. Shape is (3,).
            **kwargs: Additonal arguments for initializing texture.
        """
        device = text_init.device
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
        return torch.sigmoid(texture)
