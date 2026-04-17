from pathlib import Path

import torch
from kaolin.io.obj import import_mesh
from kaolin.rep import SurfaceMesh

from .base import Mesh


class EyeObjMesh(Mesh):
    """Mesh class for Kaolin eye meshes initialized from a .obj file.

    Extends `vc_core.dr.kaolin.mesh.Mesh` to allow filtering the full eye mesh by removing
    vertices and their corresponding normals that wouldn't be visible and re-ordering and
    filtering faces as needed after loading the raw mesh from obj.

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
        super().__init__(mesh, n_rep=n_rep)

    @staticmethod
    def _filter_mesh(
        mesh: SurfaceMesh, elev_lim: tuple[float, float], azim_lim: tuple[float, float]
    ) -> SurfaceMesh:
        """"""
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
