# pyre-unsafe
from typing import List

import cv2
import numpy as np
import torch

from .camera import CameraModel
from .dataset import HandCropData, HandData, HandPose, HandSide
from .mano_layer import MANOHandModel
from hand_tracking_toolkit.mano_layer import MANOHandModel

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
    pts = pts.astype(int)
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
            canvas[start_h : start_h + im.shape[0], : im.shape[1]] = im
            start_h += im.shape[0]
    else:
        start_w = 0
        for im in imgs:
            canvas[: im.shape[0], start_w : start_w + im.shape[1]] = im
            start_w += im.shape[1]

    return canvas


def vcat_images(imgs: List[np.ndarray]) -> np.ndarray:
    return cat_images(imgs, vertically=True)


def hcat_images(imgs: List[np.ndarray]) -> np.ndarray:
    return cat_images(imgs, vertically=False)


def visualize_hand_pose(
    hand_pose: HandPose,
    mano_beta: np.ndarray,
    mano_layer: MANOHandModel,
    image: np.ndarray,
    camera: CameraModel,
    visualize_mesh: bool,
    visualize_keypoints: bool,
    alpha: float = 0.6,
) -> np.ndarray:
    image = ensure_rgb(image)

    side = hand_pose.hand_side
    verts, keypoints = mano_layer.forward_kinematics(
        torch.tensor(mano_beta),
        torch.tensor(hand_pose.mano_theta),
        torch.tensor(hand_pose.wrist_xform),
        is_right_hand=torch.tensor([side == HandSide.RIGHT]),
    )

    if visualize_mesh:
        if side == HandSide.LEFT:
            faces = mano_layer.mano_layer_left.faces
        else:
            faces = mano_layer.mano_layer_right.faces

        rendering, mask = rasterize_mesh(
            verts.numpy(),
            faces,
            camera,
            diffuse=(0, 0, 1.0) if side == HandSide.LEFT else (1.0, 0, 0),
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
        keypoints_win = camera.world_to_window(keypoints.numpy())
        image = draw_keypoints(
            image,
            keypoints_win,
            CONTRASTIVE_COLORS,
        )

    return image


def visualize_hand_data(
    data: HandData,
    mano_layer: MANOHandModel,
    visualize_mesh: bool,
    visualize_keypoints: bool,
    alpha: float = 0.6,
) -> np.ndarray:
    all_vis = []
    for stream_id, image in data.images.items():
        hand_poses = data.hand_poses
        mano_betas = data.mano_betas
        if hand_poses is None or mano_betas is None:
            continue

        vis = image

        for hand_side, hand_pose in hand_poses.items():
            vis = visualize_hand_pose(
                hand_pose,
                mano_betas[hand_side],
                mano_layer,
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
    mano_layer: MANOHandModel,
    visualize_mesh: bool,
    visualize_keypoints: bool,
    alpha: float = 0.6,
) -> np.ndarray:

    all_vis = []
    for stream_id, image in data.images.items():
        hand_pose = data.hand_pose
        mano_beta = data.mano_beta
        if hand_pose is None or mano_beta is None:
            continue

        vis = visualize_hand_pose(
            hand_pose,
            mano_beta,
            mano_layer,
            image=image,
            camera=data.cameras[stream_id],
            visualize_mesh=visualize_mesh,
            visualize_keypoints=visualize_keypoints,
            alpha=alpha,
        )

        all_vis.append(vis)

    return hcat_images(all_vis)
