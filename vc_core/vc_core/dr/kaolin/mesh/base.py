from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING

from kaolin.rep import SurfaceMesh

if TYPE_CHECKING:
    from torch import Tensor, device


class Mesh:
    """Base class for wrapping Kaolin meshes.

    The `forward()` method is used to update any tensor mesh attributes of the wrapped mesh.

    Args:
        mesh: Kaolin mesh to wrap.
        n_rep: Number of times to repeat the wrapped mesh. Default value is 1.
    """

    def __init__(self, mesh: SurfaceMesh, *, n_rep: int = 1):
        super().__init__()
        self._mesh = mesh
        assert mesh.batching != SurfaceMesh.Batching.LIST, "List batching is not supported."
        if mesh.batching == SurfaceMesh.Batching.FIXED and len(mesh) > 1:
            warnings.warn("Ignoring n_rep since mesh is batched with len() > 1.")
        else:
            self._mesh = SurfaceMesh.cat([copy.deepcopy(mesh) for _ in range(n_rep)])
        self._mesh_shallow = copy.copy(self._mesh)

    def __len__(self) -> int:
        """Get length of the wrapped mesh."""
        return len(self._mesh)

    @property
    def mesh(self) -> SurfaceMesh:
        """Get a reference to the wrapped mesh."""
        return self._mesh

    @mesh.setter
    def mesh(self, value: SurfaceMesh) -> None:
        """Set the value of the wrapped mesh."""
        self._mesh = value
        self._mesh_shallow = copy.copy(self._mesh)

    def forward(self, offsets: dict[str, Tensor]) -> SurfaceMesh:
        """Apply offsets to the mesh's tensor attributes.

        **Note**: the attributes of the returned mesh instance should not be modified manually.
        For consistent attributes, they should only be update through this function.

        Args:
            offset: Dictionary of offset tensors to apply to mesh attributes. Keys must match
                the names of the attributes in `kaolin.rep.SurfaceMesh` exactly.

        Returns:
            New mesh created by applying given offsets and copying unchanged attributes from the
            original mesh.
        """
        for k, v in offsets.items():
            setattr(self._mesh_shallow, k, getattr(self._mesh, k) + v)
        return self._mesh_shallow

    __call__ = forward

    def to(self, device: str | device) -> "Mesh":
        self._mesh = self._mesh.to(device)
        self._mesh_shallow = copy.copy(self._mesh)
        return self
