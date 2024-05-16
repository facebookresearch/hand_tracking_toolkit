# pyre-unsafe
from typing import Tuple

import numpy as np

import pytorch3d
import torch
from pytorch3d.renderer import (
    HardPhongShader,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
)
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection

from .camera import PinholePlaneCameraModel


def rasterize_mesh(
    verts_np: np.ndarray,  # num_verts x 3
    faces_np: np.ndarray,  # num_faces x 3
    mesh_color: Tuple[float, float, float],
    camera: PinholePlaneCameraModel,
) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        verts = torch.tensor(verts_np, dtype=torch.float)[None]
        faces = torch.tensor(faces_np, dtype=torch.long)[None]

        verts_rgb = torch.ones_like(verts) * torch.tensor(mesh_color)[None, None]
        textures = TexturesVertex(verts_features=verts_rgb)
        mesh = pytorch3d.structures.Meshes(verts=verts, faces=faces, textures=textures)

        c2w = torch.tensor(camera.T_world_from_eye, dtype=torch.float)
        w2c = torch.linalg.inv(c2w)

        R = w2c[None, :3, :3]
        T = w2c[None, :3, 3]
        K = torch.tensor(camera.uv_to_window_matrix(), dtype=torch.float)[None]
        image_size = (camera.height, camera.width)

        p3d_cam = cameras_from_opencv_projection(
            R, T, K, image_size=torch.tensor(image_size, dtype=torch.long)[None]
        )

        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0,
            faces_per_pixel=1,
        )

        materials = Materials(
            ambient_color=[[0, 0, 0]],
            specular_color=[[1.0, 1.0, 1.0]],
            diffuse_color=[[1.0, 1.0, 1.0]],
            shininess=5,
        )

        lights = PointLights(
            location=torch.tensor(camera.T_world_from_eye)[None, :3, 3],
            ambient_color=((0, 0, 0),),
            diffuse_color=((1.0, 1.0, 1.0),),
            specular_color=((1.0, 1.0, 1.0),),
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=p3d_cam, raster_settings=raster_settings),
            shader=HardPhongShader(
                cameras=p3d_cam,
                lights=lights,
                materials=materials,
            ),
        )

        images = renderer(mesh).numpy()
        image = images[0, ..., :3]
        mask = images[0, ..., 3]

    return image, mask
