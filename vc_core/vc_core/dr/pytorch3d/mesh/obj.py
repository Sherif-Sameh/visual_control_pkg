from pathlib import Path

import torch.nn as nn
from pytorch3d.io import load_objs_as_meshes

from .base import Mesh


class ObjMesh(Mesh):
    """Mesh class for PyTorch3D meshes initialized from a .obj file.

    Args:
        path: Path to the .obj file to load mesh from.
        load_textures: Load material properties from associated .mtl file.
        n_rep: Number of times to repeat the wrapped mesh. Default value is 1.
    """

    def __init__(self, path: str | Path, *, load_textures: bool = False, n_rep: int = 1):
        nn.Module.__init__(self)
        path = Path(path) if isinstance(path, str) else path
        assert path.exists() and path.suffix == ".obj"
        mesh = load_objs_as_meshes([str(path)], load_textures=load_textures)
        self._mesh = mesh.extend(n_rep)
