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
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from .camera import CameraModel
from .dataset import (
    HandCropData,
    HandData,
    HandPoseCollection,
    HandShapeCollection,
    HandSide,
)
from .hand_models.mano_hand_model import (
    forward_kinematics as mano_forward_kinematics,
    MANOHandModel,
)
from .hand_models.umetrack_hand_model import (
    forward_kinematics as umetrack_forward_kinematics,
)

from .rasterizer import rasterize_mesh

CONTRASTIVE_COLORS = [
    [0, 35, 255],
    [255, 0, 0],
    [0, 145, 255],
    [0, 255, 145],
    [0, 255, 255],
    [182, 255, 0],
    [73, 255, 0],
    [255, 219, 0],
    [73, 0, 255],
    [255, 0, 109],
    [255, 109, 0],
    [0, 255, 35],
    [255, 0, 219],
    [255, 73, 0],
    [0, 255, 219],
    [145, 0, 255],
    [219, 255, 0],
    [0, 182, 255],
    [255, 0, 73],
    [145, 255, 0],
    [255, 0, 182],
]


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Ensure the image is RGB (convert if necessary)"""
    assert len(img.shape) == 2 or len(img.shape) == 3
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]

    assert img.shape[-1] == 1 or img.shape[-1] == 3
    if img.shape[-1] == 1:
        img = np.tile(img, (1, 1, 3))

    return img


def ensure_uint8(img: np.ndarray, check_value_range: bool = False) -> np.ndarray:
    """Ensure the image is [0, 255] ranged (cast if necessary)"""
    if img.dtype == np.uint8:
        return img

    if check_value_range:
        assert (
            img.max() <= 1.0 and img.min() >= 0.0
        ), "A valid image should have values in [0, 1]"

    return (img * 255).astype(np.uint8)


def find_in_frame(pts: np.ndarray, im_w: int, im_h: int) -> np.ndarray:
    """Find all points within the image frame and return the binary mask"""
    return (
        (pts[..., 0] >= 0)
        & (pts[..., 0] < im_w)
        & (pts[..., 1] > 0)
        & (pts[..., 1] < im_h)
    )


def draw_keypoints(
    image: np.ndarray,
    pts: np.ndarray,
    pts_colors: List[List[int]],
    marker_size: int = 1,
) -> np.ndarray:
    image = image.copy()
    image = ensure_rgb(image)
    image = ensure_uint8(image, True)
    im_h, im_w = image.shape[:2]

    pts = np.array(pts)
    valid = (np.abs(pts) < 1e4).all(axis=1)
    pts = pts.astype(np.int32)
    inside = find_in_frame(pts, im_w=im_w, im_h=im_h)

    # Draw points
    for i, (x, y) in enumerate(pts):
        if not (inside[i] and valid[i]):
            continue
        cv2.circle(image, (x, y), round(marker_size), tuple(pts_colors[i]), -1)

    return image


def cat_images(imgs: List[np.ndarray], vertically: bool) -> np.ndarray:
    """
    Concat images (with potentially inconsistent sizes)
    """
    chs = 1 if len(imgs[0].shape) == 2 else imgs[0].shape[2]
    dtype = imgs[0].dtype
    max_w = 0
    max_h = 0
    for im in imgs:
        im_chs = 1 if len(im.shape) == 2 else im.shape[2]
        assert (
            chs == im_chs
        ), f"number of channels is not consistent! - expected {chs} but got {im_chs}!"
        assert (
            im.dtype == dtype
        ), f"data type does not match - expected {dtype} but got {im.dtype}!"
        if vertically:
            max_w = max(max_w, im.shape[1])
            max_h += im.shape[0]
        else:
            max_h = max(max_h, im.shape[0])
            max_w += im.shape[1]

    if chs == 1:
        canvas = 255 * np.ones((max_h, max_w), dtype=dtype)
    else:
        canvas = 255 * np.ones((max_h, max_w, chs), dtype=dtype)

    if vertically:
        start_h = 0
        for im in imgs:
            # pyre-fixme[16]: `int` has no attribute `__setitem__`.
            canvas[start_h : start_h + im.shape[0], : im.shape[1]] = im
            start_h += im.shape[0]
    else:
        start_w = 0
        for im in imgs:
            canvas[: im.shape[0], start_w : start_w + im.shape[1]] = im
            start_w += im.shape[1]

    # pyre-fixme[7]: Expected `ndarray` but got `int`.
    return canvas


def vcat_images(imgs: List[np.ndarray]) -> np.ndarray:
    return cat_images(imgs, vertically=True)


def hcat_images(imgs: List[np.ndarray]) -> np.ndarray:
    return cat_images(imgs, vertically=False)


def get_keypoints_and_mesh(
    hand_pose: Optional[HandPoseCollection],
    hand_shape: Optional[HandShapeCollection],
    mano_model: MANOHandModel,
    pose_type: str = "mano",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    keypoints = verts = faces = None
    default_ret = (keypoints, verts, faces)

    if hand_pose is None or hand_shape is None:
        return default_ret

    if pose_type == "mano":
        if hand_pose.mano is None:
            return default_ret

        return mano_forward_kinematics(
            hand_pose.mano,
            hand_shape.mano_beta,
            mano_model,
        )
    else:  # umetrack
        if hand_pose.umetrack is None:
            return default_ret

        return umetrack_forward_kinematics(
            hand_pose.umetrack,
            hand_shape.umetrack,
            requires_mesh=True,
        )


def visualize_keypoints_and_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    keypoints: np.ndarray,
    hand_side: HandSide,
    image: np.ndarray,
    camera: CameraModel,
    visualize_mesh: bool,
    visualize_keypoints: bool,
    alpha: float = 0.6,
) -> np.ndarray:
    image = ensure_rgb(image)

    if visualize_mesh:
        assert verts is not None and faces is not None
        rendering, mask, _ = rasterize_mesh(
            verts,
            faces,
            camera,
            diffuse=((1.0, 0, 0) if hand_side == HandSide.LEFT else (0, 0, 1.0)),
            shininess=10,
        )

        # blending
        alpha = alpha * mask.astype(np.float32)
        image = (
            # pyre-ignore
            image.astype(np.float32) * (1 - alpha[..., None])
            + rendering.astype(np.float32) * alpha[..., None]
        ).astype(np.uint8)

    if visualize_keypoints:
        keypoints_win = camera.world_to_window3(keypoints)
        # set points with negative z to nan
        keypoints_win[keypoints_win[:, 2] <= 0] = np.nan
        image = draw_keypoints(
            image,
            keypoints_win[:, :2],
            CONTRASTIVE_COLORS,
        )

    return image


def visualize_hand_data(
    data: HandData,
    mano_model: MANOHandModel,
    visualize_mesh: bool,
    visualize_keypoints: bool,
    alpha: float = 0.6,
    pose_type: str = "mano",  # "umetrack"
) -> np.ndarray:
    hand_poses = data.hand_poses
    hand_shape = data.hand_shape

    keypoints_and_meshes = {}
    if hand_poses is not None and hand_shape is not None:
        for hand_side in hand_poses:
            keypoints, verts, faces = get_keypoints_and_mesh(
                hand_poses[hand_side],
                hand_shape,
                mano_model,
                pose_type,
            )
            if keypoints is None:
                continue
            keypoints_and_meshes[hand_side] = (keypoints, verts, faces)

    all_vis = []
    for stream_id, image in data.images.items():
        vis = image
        for hand_side, (keypoints, verts, faces) in keypoints_and_meshes.items():
            vis = visualize_keypoints_and_mesh(
                verts=verts.numpy(),
                faces=faces.numpy(),
                keypoints=keypoints.numpy(),
                hand_side=hand_side,
                image=vis,
                camera=data.cameras[stream_id],
                visualize_mesh=visualize_mesh,
                visualize_keypoints=visualize_keypoints,
                alpha=alpha,
            )

        all_vis.append(vis)

    return hcat_images(all_vis)


def visualize_hand_crop_data(
    data: HandCropData,
    mano_model: MANOHandModel,
    visualize_mesh: bool,
    visualize_keypoints: bool,
    alpha: float = 0.6,
    pose_type: str = "mano",  # "umetrack"
) -> np.ndarray:
    hand_pose = data.hand_pose
    hand_shape = data.hand_shape

    keypoints = verts = faces = None
    if hand_pose is not None and hand_shape is not None:
        keypoints, verts, faces = get_keypoints_and_mesh(
            hand_pose,
            hand_shape,
            mano_model,
            pose_type,
        )

    all_vis = []
    for stream_id, image in data.images.items():
        vis = image
        if (
            hand_pose is not None
            and keypoints is not None
            and verts is not None
            and faces is not None
        ):
            vis = visualize_keypoints_and_mesh(
                verts=verts.numpy(),
                faces=faces.numpy(),
                keypoints=keypoints.numpy(),
                hand_side=hand_pose.hand_side,
                image=vis,
                camera=data.cameras[stream_id],
                visualize_mesh=visualize_mesh,
                visualize_keypoints=visualize_keypoints,
                alpha=alpha,
            )
        all_vis.append(vis)

    return hcat_images(all_vis)
