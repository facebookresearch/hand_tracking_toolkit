# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

from typing import Optional, Sequence, Tuple

import numpy as np

from .camera import CameraModel


def normalized(vecs: np.ndarray, add_const_to_denom: bool = True) -> np.ndarray:
    # faster than np.linalg.norm
    denom = np.sqrt(vecs[..., 0:1] ** 2 + vecs[..., 1:2] ** 2 + vecs[..., 2:3] ** 2)
    if add_const_to_denom:
        denom += 1e-8
    return vecs / denom


def get_vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    norm = np.zeros_like(vertices)
    tris = vertices[triangles]
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    n = normalized(n)
    # accumulate normals at each vertex
    np.add.at(norm, triangles[:, 0], n)
    np.add.at(norm, triangles[:, 1], n)
    np.add.at(norm, triangles[:, 2], n)
    return normalized(norm)


def barycentric_coords_perspective(
    v: np.ndarray,  # num_faces x 3 (vertices) x 3 (xyz)
    x: np.ndarray,  # num_faces x num_pixels_per_bbox
    y: np.ndarray,  # num_faces x num_pixels_per_bbox
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    n, m = x.shape

    # https://www.cs.umd.edu/~zwicker/courses/computergraphics/04_Rasterization.pdf
    v_mat = v.copy()
    v_mat[..., 0] = (v_mat[..., 0] - cx) / fx * v_mat[..., 2]
    v_mat[..., 1] = (v_mat[..., 1] - cy) / fy * v_mat[..., 2]
    inv_v_mat = np.linalg.inv(v_mat)

    # n x m x 1 x 3 @ n x 1 x 3 x 3 -> n x m x 1 x 3
    pts = np.concatenate(
        # pyre-ignore
        (
            (x.reshape((n, m, 1, 1)) - cx) / fx,
            (y.reshape((n, m, 1, 1)) - cy) / fy,
            np.ones((n, m, 1, 1), dtype=x.dtype),
        ),
        axis=-1,
    )
    bary = pts @ inv_v_mat[:, None, :, :]
    bary = bary[:, :, 0, :]

    # faster than bary.sum(axis=-1, keepdims=True)
    bary_sum = bary[..., 0:1] + bary[..., 1:2] + bary[..., 2:3]
    bary /= bary_sum + 1e-8

    # n x m x 3
    return bary


def dot(v1, v2):
    # faster than (v1 * v2).sum(axis=-1)
    prod = v1 * v2
    return prod[..., 0] + prod[..., 1] + prod[..., 2]


def phong_reflection_model(
    verts: np.ndarray,
    normals: np.ndarray,
    view_pos: np.ndarray,
    light_pos: np.ndarray,
    ambient: Sequence[float],
    diffuse: Sequence[float],
    specular: Sequence[float],
    shininess: float,
) -> np.ndarray:
    color = np.zeros_like(verts)
    ambient_np = np.array(ambient)
    diffuse_np = np.array(diffuse)
    specular_np = np.array(specular)

    # ambinet
    color += ambient_np

    # diffuse
    light_dir = normalized(light_pos - verts)
    ndotl = dot(light_dir, normals)
    color += ndotl[:, None] * diffuse_np[None, :]

    # specular
    reflection_v = normals * ndotl[:, None] * 2 - light_dir
    view_dir = normalized(view_pos - verts)
    vdotr = dot(reflection_v, view_dir).clip(min=0)
    color += np.power(vdotr, shininess)[:, None] * specular_np[None, :]

    return color


def rasterize_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    camera: CameraModel,
    vert_normals: Optional[np.ndarray] = None,
    light_pos: Optional[np.ndarray] = None,
    ambient: Sequence[float] = (0.0, 0.0, 0.0),
    diffuse: Sequence[float] = (1.0, 1.0, 1.0),
    specular: Sequence[float] = (1.0, 1.0, 1.0),
    shininess: float = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    H = camera.height
    W = camera.width

    verts = verts.astype(np.float32)
    faces = faces.astype(np.int32)

    if vert_normals is None:
        vert_normals = get_vertex_normals(verts, faces)
    vert_normals = vert_normals.astype(np.float32)

    verts2d_z = camera.world_to_window3(verts).astype(np.float32)
    z = verts2d_z[:, 2]
    recip_z = 1 / z

    verts2d_z_tri = verts2d_z[faces]
    faces_x, faces_y, faces_z = (
        verts2d_z_tri[..., 0],
        verts2d_z_tri[..., 1],
        verts2d_z_tri[..., 2],
    )

    # # select faces where all three vertices have valid coordinates
    valid_faces = np.all(
        (faces_z > 0)
        & (faces_y >= 0)
        & (faces_y <= H - 1)
        & (faces_x >= 0)
        & (faces_x <= W - 1),
        axis=-1,
    )
    faces = faces[valid_faces]

    # make output buffers
    image = np.zeros((H, W, 3), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.uint8)

    if faces.shape[0] == 0:
        return image, mask

    # Compute the box size for each triangle, take the largest one.
    # This can be suboptimal if one triangle is significantly larger than
    # the other ones after projection, but this step is essential for vectorization
    verts2d_z_tri = verts2d_z[faces]
    min_xy = np.floor(verts2d_z_tri[..., :2].min(1))
    max_xy = np.ceil(verts2d_z_tri[..., :2].max(1))

    box_size = max_xy - min_xy
    max_box_size = np.max(box_size, axis=0)
    max_xy = min_xy + max_box_size[None, :]

    # coordinates in each box
    xgrid, ygrid = np.meshgrid(
        np.arange(max_box_size[0], dtype=np.float32),
        np.arange(max_box_size[1], dtype=np.float32),
    )
    x = min_xy[:, 0:1] + xgrid.flatten()[None, :]
    y = min_xy[:, 1:2] + ygrid.flatten()[None, :]

    bary = barycentric_coords_perspective(
        verts2d_z_tri,
        x,
        y,
        fx=float(camera.f[0]),
        fy=float(camera.f[1]),
        cx=float(camera.c[0]),
        cy=float(camera.c[1]),
    )

    # get attributes at target locations
    xt = x.reshape(-1)
    yt = y.reshape(-1)

    # pyre-ignore
    recip_z_tri = recip_z[faces]
    recip_zt = bary[:, None] @ recip_z_tri[:, None, :, None]
    recip_zt = recip_zt.reshape(-1)

    n_tri = vert_normals[faces]
    nt = bary[:, None] @ n_tri[:, None]
    nt = normalized(nt.reshape((-1, 3)))

    v_tri = verts[faces]
    vt = bary[:, None] @ v_tri[:, None]
    vt = vt.reshape((-1, 3))

    in_boundary = (yt >= 0) & (yt <= H - 1) & (xt >= 0) & (xt <= W - 1)
    # faster than bary.min(axis=-1) > 0
    in_triangle = (bary[..., 0] > 0) & (bary[..., 1] > 0) & (bary[..., 2] > 0)
    valid = in_triangle.reshape(-1) & in_boundary

    vt = vt[valid]
    nt = nt[valid]
    xt = xt[valid]
    yt = yt[valid]
    recip_zt = recip_zt[valid]

    view_pos = camera.eye_to_world(np.zeros(3, dtype=np.float32))
    if light_pos is None:
        # If not provided, light is colocated with the camera
        light_pos = view_pos
    light_pos = light_pos.astype(np.float32)

    color = phong_reflection_model(
        vt,
        nt,
        view_pos=view_pos,
        light_pos=light_pos,
        ambient=ambient,
        diffuse=diffuse,
        specular=specular,
        shininess=shininess,
    )

    # order the target values by 1/z
    order = recip_zt.argsort()
    xt = xt[order].astype(np.int32)
    yt = yt[order].astype(np.int32)
    color = color[order]

    image[yt, xt] = color
    image = (image.clip(min=0, max=1.0) * 255).astype(np.uint8)

    mask[yt, xt] = 1

    return image, mask
