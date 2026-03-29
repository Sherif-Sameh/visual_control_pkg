from pathlib import Path

import torch.nn as nn
from pytorch3d.io import load_objs_as_meshes

from .base import Mesh


class ObjMesh(Mesh):
    """Mesh class for PyTorch3D meshes initialized from a .obj file.

    Args:
        path: Path to the .obj file to load mesh from.
        load_textures: Load material properties from associated .mtl file.
    """

    def __init__(self, path: str | Path, load_textures: bool = False):
        nn.Module.__init__()
        path = Path(path) if isinstance(path, str) else path
        assert path.exists() and path.suffix == ".obj"
        self._mesh = load_objs_as_meshes([str(path)], load_textures=load_textures)
