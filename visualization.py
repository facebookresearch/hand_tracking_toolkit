# pyre-unsafe
from typing import List

import cv2
import numpy as np
import torch

from .dataset import HandCropData, HandSide

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


def visualize_hand_crop_data(
    data: HandCropData, mano_layer: torch.nn.Module
) -> np.ndarray:
    side = data.hand_pose.hand_side
    _, keypoints = mano_layer.forward_kinematics(
        torch.tensor(data.mano_beta),
        torch.tensor(data.hand_pose.pose),
        torch.tensor(data.hand_pose.global_xform),
        is_right_hand=torch.tensor([side == HandSide.RIGHT]),
    )
    keypoints_win = data.camera.world_to_window(keypoints.numpy())

    keypoints_vis = draw_keypoints(
        data.image,
        keypoints_win,
        CONTRASTIVE_COLORS,
    )
    return keypoints_vis
