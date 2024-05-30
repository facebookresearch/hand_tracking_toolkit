from typing import Optional, Sequence, Tuple

import numpy as np

from .camera import CameraModel


def normalized(
    vecs: np.ndarray, axis: int = -1, add_const_to_denom: bool = True
) -> np.ndarray:
    denom = np.linalg.norm(vecs, axis=axis, keepdims=True)
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

    bary /= bary.sum(axis=-1, keepdims=True)

    # n x m x 3
    return bary


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
    ndotl = (light_dir * normals).sum(axis=1)
    color += ndotl[:, None] * diffuse_np[None, :]

    # specular
    reflection_v = normals * ndotl[:, None] * 2 - light_dir
    view_dir = normalized(view_pos - verts)
    vdotr = (reflection_v * view_dir).sum(axis=1).clip(min=0)
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
    if vert_normals is None:
        vert_normals = get_vertex_normals(verts, faces)

    verts2d_z = camera.world_to_window3(verts)
    verts2d = verts2d_z[:, :2]
    recip_z = 1 / verts2d_z[:, 2]

    # Compute the box size for each triangle, take the largest one.
    # This can be suboptimal if one triangle is significantly larger than
    # the other ones after projection, but this step is essential for vectorization
    vers2d_tri = verts2d[faces]
    min_xy = np.floor(vers2d_tri.min(1))
    max_xy = np.ceil(vers2d_tri.max(1))
    box_size = max_xy - min_xy
    max_box_size = box_size.max(axis=0)
    max_xy = min_xy + max_box_size[None, :]

    # coordinates in each box
    xgrid, ygrid = np.meshgrid(np.arange(max_box_size[0]), np.arange(max_box_size[1]))
    x = min_xy[:, 0:1] + xgrid.flatten()[None, :]
    y = min_xy[:, 1:2] + ygrid.flatten()[None, :]

    bary = barycentric_coords_perspective(
        verts2d_z[faces],
        x,
        y,
        fx=camera.f[0],
        fy=camera.f[1],
        cx=camera.c[0],
        cy=camera.c[1],
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

    H = camera.height
    W = camera.width
    in_boundary = np.logical_and(
        np.logical_and(yt >= 0, yt <= H - 1),
        np.logical_and(xt >= 0, xt <= W - 1),
    )
    in_triangle = bary.min(axis=-1) > 0
    valid = np.logical_and(in_triangle.reshape(-1), in_boundary)

    vt = vt[valid]
    nt = nt[valid]
    xt = xt[valid]
    yt = yt[valid]
    recip_zt = recip_zt[valid]

    view_pos = camera.eye_to_world(np.zeros(3))
    if light_pos is None:
        # If not provided, light is colocated with the camera
        light_pos = view_pos

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

    image = np.zeros((H, W, 3))
    mask = np.zeros((H, W), dtype=np.uint8)

    image[yt, xt] = color
    image = (image.clip(min=0, max=1.0) * 255).astype(np.uint8)

    mask[yt, xt] = 1

    return image, mask
