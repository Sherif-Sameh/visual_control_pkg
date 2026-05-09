from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from kaolin.render.camera import Camera
from torchvision.transforms.functional import to_pil_image

from vc_core.dr.kaolin.mesh import EyeObjMesh, ObjMesh
from vc_core.dr.kaolin.render import (
    BlendParams,
    ComposeShader,
    HardColorAmbientShader,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftSilhouetteShader,
)
from vc_core.dr.kaolin.utils import look_at_view_transform, transform_from_rotation_translation

if TYPE_CHECKING:
    from vc_core.dr.kaolin.mesh import Mesh

logging.getLogger("kaolin.rep.surface_mesh").setLevel(logging.ERROR)


### Configuration ###
BACKEND = "nvdiffrast"
assert torch.cuda.is_available()
DEVICE = torch.device("cuda")
FULL_MESH = True
ELEV_LIM = (-25.0, 25.0)
AZIM_LIM = (-80.0, 80.0)
AMBIENT_SCALE = 1.2
RES = (576, 1024)
CAMERA_COORDS = (0.03, 7.5, 7.5)
PATH = Path(__file__).parents[3] / "tests/unit/dr/kaolin/samples/eye_mesh"
assert PATH.exists() and PATH.is_dir()


### Helpers ###
def get_mesh() -> Mesh:
    if FULL_MESH:
        path = PATH / "eye_low.obj"
        mesh = ObjMesh(path, with_materials=True, with_normals=True)
    else:
        path = PATH / "eye.obj"
        mesh = EyeObjMesh(path, elev_lim=ELEV_LIM, azim_lim=AZIM_LIM)
    F = mesh.mesh.faces.shape[0]
    mesh.mesh.face_features = torch.zeros([1, F, 3, 0])
    mesh.mesh.face_uvs = 1 - mesh.mesh.face_uvs
    mesh.mesh.materials[0][0]["map_Kd"] = (
        mesh.mesh.materials[0][0]["map_Kd"].float().permute(2, 0, 1) / 255.0
    )
    return mesh.to(device=DEVICE)


def get_camera() -> Camera:
    height, width = RES
    distance, elevation, azimuth = CAMERA_COORDS
    R, T = look_at_view_transform(distance, elevation, azimuth)
    view_matrix = transform_from_rotation_translation(R, T)
    camera = Camera.from_args(
        view_matrix=view_matrix,
        fov=60 * torch.pi / 180,
        height=height,
        width=width,
        near=0.1,
        far=10.0,
        dtype=torch.float32,
    )
    camera = Camera.cat([camera] * 1).to(device=DEVICE)
    return camera


def get_renderer() -> MeshRenderer:
    rasterizer = MeshRasterizer(
        cameras=get_camera(), raster_settings=RasterizationSettings(image_size=RES, backend=BACKEND)
    )
    shader = ComposeShader(
        [
            SoftSilhouetteShader(BlendParams(sigmainv=1e6, boxlen=0, knum=1)),
            HardColorAmbientShader(
                ambient=torch.ones(3) * AMBIENT_SCALE, raw_texture=False, uvs_origin="Kaolin"
            ),
        ]
    )
    renderer = MeshRenderer(rasterizer, shader).to(device=DEVICE)
    return renderer


### Main ###
@torch.inference_mode
def main() -> None:
    mesh = get_mesh()
    renderer = get_renderer()

    output = renderer(mesh.mesh)[0]
    silhouette = output[:, :, :1]
    rgb = output[:, :, 1:]
    rgba = torch.cat([rgb, silhouette], dim=-1).permute(2, 0, 1)

    path = Path(__file__).parent / "figures"
    path.mkdir(parents=True, exist_ok=True)
    if FULL_MESH:
        filename = "eye_full.png"
    else:
        elev, azim = int(ELEV_LIM[1]), int(AZIM_LIM[1])
        filename = f"eye_partial_{elev}_{azim}.png"
    img = to_pil_image(rgba)
    img.save(str(path / filename))


if __name__ == "__main__":
    main()
