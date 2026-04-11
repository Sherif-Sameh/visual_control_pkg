from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import LongTensor, Tensor


@dataclass
class HashEncoder2DCfg:
    """Configuration for multi-resolution 2D hash encoder."""

    finest_res: int = 512
    """Finest grid resolution. Default value is 512."""

    coarsest_res: int = 16
    """Coarsest grid resolution. Default value is 16."""

    n_level: int = 16
    """Number of levels for hash encoding. Number of `nn.Embedding` layers. Default value is 16."""

    log2_hashmap_size: int = 14
    """Hashmap size. `num_embeddings` = 2 ** `log2_hashmap_size`. Default value is 14."""

    n_feature: int = 2
    """Number of features per level (i.e. `embedding_dim`). Default value is 2."""

    uv_max: float = 1000.0
    """Maximum value for UV coordinates. Defautl value is 1000."""


class HashEncoder2D(nn.Module):
    """Multi-resolution 2D hash encoding PyTorch module.

    Implementation and default hyperparameters based on the description given in the original paper
    titled: `Instant Neural Graphics Primitives with a Multiresolution Hash Encoding` [0].

    Args:
        cfg: Hash 2D encoder configuration.

    References:
    [0] https://arxiv.org/abs/2201.05989
    """

    def __init__(self, cfg: HashEncoder2DCfg):
        super().__init__()
        self.hashmap_size = 2**cfg.log2_hashmap_size
        self.n_feature = cfg.n_feature
        self.uv_max = cfg.uv_max
        # Compute geometric progression of resolutions
        b = math.exp((math.log(cfg.finest_res) - math.log(cfg.coarsest_res)) / (cfg.n_level - 1))
        self.resolutions = [int(cfg.coarsest_res * (b**i)) for i in range(cfg.n_level)]
        # Store primes for hashing
        self.primes = (1, 2_654_435_761)
        # Create embedding layer (hashmap) per level
        self.embeddings = nn.ModuleList(
            [nn.Embedding(self.hashmap_size, cfg.n_feature) for _ in range(cfg.n_level)]
        )
        # Initialize embeddings
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, a=-1e-4, b=1e-4)

    @property
    def feature_dim(self) -> int:
        """Get the output feature dimension of concatenated embeddings."""
        n_level = len(self.embeddings)
        return n_level * self.n_feature

    def forward(self, uvs: Tensor) -> Tensor:
        """Compute concatenated mult-res hashmap embeddings for input 2D coordinates.

        Args:
            uvs: Grid uv coordinates [0, `uv_max`] to encode with hash embeddings. Shape is
                (B, H, W, 2).

        Returns:
            Hash embeddings. Shape is (B, H, W, `n_level` * `n_feature`).
        """
        outputs = []
        for level, resolution in enumerate(self.resolutions):
            # Scale UVs to current resolution
            uv_scaled = uvs * resolution / self.uv_max
            # Float grid coordinates within pixel
            uv_floor = torch.floor(uv_scaled).long()
            fu = uv_scaled[..., 0] - uv_floor[..., 0].float()
            fv = uv_scaled[..., 1] - uv_floor[..., 1].float()
            # Integer grid coordinates
            u0 = uv_floor[..., 0]
            v0 = uv_floor[..., 1]
            u1 = u0 + 1
            v1 = v0 + 1
            # Hash the 4 neighboring pixels
            h00 = self._hash_fn(u0, v0)
            h10 = self._hash_fn(u1, v0)
            h01 = self._hash_fn(u0, v1)
            h11 = self._hash_fn(u1, v1)
            # Fetch embeddings
            emb_table = self.embeddings[level]
            v00 = emb_table(h00)
            v10 = emb_table(h10)
            v01 = emb_table(h01)
            v11 = emb_table(h11)
            # Bilinear interpolation
            wu = fu.unsqueeze(-1)
            wv = fv.unsqueeze(-1)
            v0 = v00 * (1 - wu) + v10 * wu
            v1 = v01 * (1 - wu) + v11 * wu
            v = v0 * (1 - wv) + v1 * wv
            outputs.append(v)

        # Concatenate features across levels
        return torch.cat(outputs, dim=-1)

    def _hash_fn(self, u: LongTensor, v: LongTensor) -> LongTensor:
        """Spatial 2D hashing function.

        Args:
            u: Integer u-axis grid coordinates to hash. Shape is (B, H, W).
            v: Integer v-axis grid coordinates to hash. Shape is (B, H, W).

        Returns:
            Hashed indices. Shape is (B, H, W).
        """
        hash = torch.bitwise_xor(u * self.primes[0], v * self.primes[1]) % self.hashmap_size
        return hash
