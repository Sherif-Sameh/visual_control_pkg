from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from kaolin.io.obj import import_mesh
from kaolin.rep import SurfaceMesh

from .base import Mesh

if TYPE_CHECKING:
    from torch import Tensor


class EyeObjMesh(Mesh):
    """Mesh class for Kaolin eye meshes initialized from a .obj file.

    Extends `vc_core.dr.kaolin.mesh.Mesh` to provide the following additional functionalities:
    - Filtering the full eye mesh by removing vertices and their corresponding normals that
    wouldn't be visible and re-ordering and filtering faces as needed after loading the raw mesh
    from obj. This controlled by `elev_lim` and `azim_lim`.
    - Projecting mesh vertex offsets to the tangent plane of each vertex according to its normal
    direction.

    This class makes the following assumptions about the eye mesh:

    - Its coordinate frame is set at the center of the eye ball.
    - Its Z-axis points forwards towards the center of the pupil, the X-axis points to the left and
    the Y-axis points vertically upwards.

    Args:
        path: Path to the .obj file to load mesh from.
        elev_lim: Elevation limits (min, max) in degrees for filtering mesh vertices. Default value
            is (-30, 30).
        azim_lim: Azimuth limits (min, max) in degrees for filtering mesh vertices. Default value
            is (-75, 75).
        n_rep: Number of times to repeat the wrapped mesh. Default value is 1.
        kwargs: Other arguments to pass to the `kaolin.io.obj.import_mesh` function.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        elev_lim: tuple[float, float] = (-30.0, 30.0),
        azim_lim: tuple[float, float] = (-75.0, 75.0),
        n_rep: int = 1,
        **kwargs,
    ):
        path = Path(path) if isinstance(path, str) else path
        assert path.exists() and path.suffix == ".obj"
        mesh = import_mesh(str(path), with_materials=True, with_normals=True, **kwargs)
        mesh = self._filter_mesh(mesh, elev_lim, azim_lim)
        mesh.vertex_normals = torch.nn.functional.normalize(mesh.vertex_normals, dim=-1, eps=1e-8)
        super().__init__(mesh, n_rep=n_rep)

    def forward(self, offsets: dict[str, Tensor], texture: Tensor | None = None) -> SurfaceMesh:
        """Apply offsets to the mesh's tensor attributes and override texture if given.

        If mesh vertex offsets are given, they are projected to the tangent plane of the
        corresponding vertex to avoid distorting the mesh along its normals.

        If a texture is given, it's assumed that the underlying mesh has only a single material
        shared by all its instances and stored under the `map_Kd` key in its materials dict.

        **Note**: the attributes of the returned mesh instance should not be modified manually.
        For consistent attributes, they should only be updated through this function.

        Args:
            offset: Dictionary of offset tensors to apply to mesh attributes. Keys must match
                the names of the attributes in `kaolin.rep.SurfaceMesh` exactly.
            texture: Optional texture to override mesh's texture. Default value is `None`.

        Returns:
            New mesh created by applying given offsets and texture and copying unchanged attributes
            from the original mesh.
        """
        for k, v in offsets.items():
            if k == "vertices":
                v = self._project_vertex_offsets(v)
            setattr(self._mesh_shallow, k, getattr(self._mesh, k) + v)
        if texture is not None:
            self._mesh_shallow.materials = [[{"map_Kd": texture}]]
        return self._mesh_shallow

    __call__ = forward

    @staticmethod
    def _filter_mesh(
        mesh: SurfaceMesh, elev_lim: tuple[float, float], azim_lim: tuple[float, float]
    ) -> SurfaceMesh:
        if mesh.batching != SurfaceMesh.Batching.NONE:
            mesh = mesh[0]
        # convert vertices to spherical coordinates
        x, y, z = torch.split(mesh.vertices, 1, dim=-1)
        elev = torch.rad2deg(torch.atan2(y, torch.sqrt(x**2 + z**2)))
        azim = torch.rad2deg(torch.atan2(x, z))
        # compute vertex mask according to spherical coordinates
        elev_lim_pos = torch.where(
            azim > 0,
            elev_lim[1] * (1 - (azim / azim_lim[1]) ** 1.5),
            elev_lim[1] * (1 - (azim / azim_lim[0]) ** 1.5),
        )
        elev_lim_neg = torch.where(
            azim > 0,
            elev_lim[0] * (1 - (azim / azim_lim[1]) ** 1.5),
            elev_lim[0] * (1 - (azim / azim_lim[0]) ** 1.5),
        )
        mask = torch.logical_and(
            torch.logical_and(elev >= elev_lim_neg, elev <= elev_lim_pos),
            torch.logical_and(azim >= azim_lim[0], azim <= azim_lim[1]),
        ).squeeze()
        # filter vertices and vertex attributes according to spherical coordinates
        vertices = mesh.vertices[mask]
        normals = mesh.normals[mask]
        uvs = mesh.uvs[mask]
        # re-order and filter pruned faces
        indices = torch.full((mesh.vertices.shape[0],), -1, dtype=torch.long)
        indices[mask] = torch.arange(mask.sum())
        face_mask = mask[mesh.faces].all(dim=1)

        def get_new_faces(face_idx: torch.LongTensor) -> torch.LongTensor:
            face_mask = mask[face_idx].all(dim=1)
            return indices[face_idx[face_mask]]

        # update mesh
        mesh.vertices = vertices
        mesh.normals = normals
        mesh.uvs = uvs
        mesh.faces = get_new_faces(mesh.faces)
        mesh.face_normals_idx = get_new_faces(mesh.face_normals_idx)
        mesh.face_uvs_idx = get_new_faces(mesh.face_uvs_idx)
        mesh.material_assignments = mesh.material_assignments.long()[face_mask]
        mesh.material_assignments = mesh.material_assignments.to(torch.int16)
        assert mesh.check_sanity()
        return mesh

    def _project_vertex_offsets(self, vertex_offsets: Tensor) -> Tensor:
        """Project mesh vertex offsets to their respective tangent planes.

        Vertex offsets are broadcast to the batch dimension of the mesh if needed.

        Args:
            vertex_offsets: Mesh vertex offsets. Shape is (N, 3) or (B, N, C).

        Returns:
            Projected mesh vertex offsets. Shape is (B, N, 3).
        """
        vertex_normals = self._mesh.vertex_normals
        vertex_offsets_proj = torch.sum(vertex_offsets * vertex_normals, dim=-1, keepdim=True)
        return vertex_offsets - vertex_offsets_proj * vertex_normals
