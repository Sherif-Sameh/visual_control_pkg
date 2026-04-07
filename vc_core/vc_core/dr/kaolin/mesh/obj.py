from pathlib import Path

from kaolin.io.obj import import_mesh

from .base import Mesh


class ObjMesh(Mesh):
    """Mesh class for Kaolin meshes initialized from a .obj file.

    Args:
        path: Path to the .obj file to load mesh from.
        with_materials: Load materials. Default value is `False`.
        with_normals: Load normals. Default value is `False`.
        n_rep: Number of times to repeat the wrapped mesh. Default value is 1.
        kwargs: Other arguments to pass to the `kaolin.io.obj.import_mesh` function.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        with_materials: bool = False,
        with_normals: bool = False,
        n_rep: int = 1,
        **kwargs,
    ):
        path = Path(path) if isinstance(path, str) else path
        assert path.exists() and path.suffix == ".obj"
        mesh = import_mesh(
            str(path), with_materials=with_materials, with_normals=with_normals, **kwargs
        )
        super().__init__(mesh, n_rep=n_rep)
