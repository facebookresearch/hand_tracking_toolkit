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

"""
Helper methods for metric computation
"""

from typing import Dict, Optional

import numpy as np
import torch

from .hand_models.mano_hand_model import (
    forward_kinematics as mano_forward_kinematics,
    MANOHandModel,
    MANOHandPose,
)

MAX_LANDMARK_ERROR_MM = 50
PCK_THRESHOLDS: torch.Tensor = torch.linspace(0, MAX_LANDMARK_ERROR_MM, 101)
FINGERTIP_IDX = list(range(5))
M_TO_MM = 1000  # meter to mm conversion

# Remove the wrist since MANO's wrist location is very different from UmeTrack's
LANDMARKS_TO_EVAL = [
    0,  # THUMB_FINGERTIP
    1,  # INDEX_FINGER_FINGERTIP
    2,  # MIDDLE_FINGER_FINGERTIP
    3,  # RING_FINGER_FINGERTIP
    4,  # PINKY_FINGER_FINGERTIP
    6,  # THUMB_INTERMEDIATE_FRAME
    7,  # THUMB_DISTAL_FRAME
    8,  # INDEX_PROXIMAL_FRAME
    9,  # INDEX_INTERMEDIATE_FRAME
    10,  # INDEX_DISTAL_FRAME
    11,  # MIDDLE_PROXIMAL_FRAME
    12,  # MIDDLE_INTERMEDIATE_FRAME
    13,  # MIDDLE_DISTAL_FRAME
    14,  # RING_PROXIMAL_FRAME
    15,  # RING_INTERMEDIATE_FRAME
    16,  # RING_DISTAL_FRAME
    17,  # PINKY_PROXIMAL_FRAME
    18,  # PINKY_INTERMEDIATE_FRAME
    19,  # PINKY_DISTAL_FRAME
]


def _safe_div(x, y, eps: float = 1e-6, default_val: float = 0):
    if np.isscalar(x):
        assert np.isscalar(y)
        # pyre-fixme[58]: `<` is not supported for operand types `Union[bool, bytes,
        #  complex, float, int, memoryview, np.generic, str]` and `float`.
        if y < eps:
            return default_val
        else:
            # pyre-fixme[58]: `/` is not supported for operand types `Union[bool,
            #  bytes, complex, float, int, memoryview, np.generic, str]` and
            #  `Union[bool, bytes, complex, float, int, memoryview, np.generic, str]`.
            return x / y

    assert x.shape == y.shape
    z = x / y
    z[y < eps] = default_val
    return z


def _PCK_curve(
    errors: torch.Tensor, mask: torch.Tensor, thresholds: torch.Tensor
) -> torch.Tensor:
    pcks = []
    for thresh in thresholds:
        le_threh = errors <= thresh
        pck = _safe_div((le_threh * mask).sum(dim=-1), mask.sum(dim=-1))
        pcks.append(pck)
    return torch.stack(pcks)


def PCK_curve(
    errors: torch.Tensor,
    thresholds: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    axis: Optional[int] = None,
) -> torch.Tensor:
    """
    Computes the total PCK curve. If the axis is given, computes one PCK curve
    for each element along the given axis.
    e.g., If `errors` is a 100 x 2 x 21 ndarray and axis = 1,
    return a 2 x len(thresholds) matrix.

    Parameters
    ----------
    errors : torch.Tensor
        tensor of errors
    thresholds : torch.Tensor
        Thresholds for computing PCK
    mask : Optional[torch.Tensor], optional
        Mask to filter invalid samples. Shape is the same as `errors`. If None,
        all error samples are assumed valid.
    axis : Optional[int], optional
        See summary, by default None

    Returns
    -------
    torch.Tensor
        PCK curve(s)
    """
    if mask is None:
        mask = torch.ones_like(errors)
    if axis is None:
        return _PCK_curve(errors.reshape(-1), mask.reshape(-1), thresholds)
    N = errors.shape[axis]
    return _PCK_curve(
        torch.movedim(errors, axis, 0).reshape(N, -1),
        torch.movedim(mask, axis, 0).reshape(N, -1),
        thresholds,
    )


def normalized_AUC(
    x: torch.Tensor, y: torch.Tensor, y_max: float = 1.0
) -> torch.Tensor:
    """
    Given curves sharing the same x-axis, computes normalized AUC for each of
    the curves.

    Parameters
    ----------
    x : torch.Tensor
        X-axis ticks represented as a 1-D array
    y : torch.Tensor
        An ndarray with the last dimension being the y-axis ticks
    y_max : float, optional
        Maximum of y value, by default 1.0

    Returns
    -------
    torch.Tensor
        Normalized AUCs with shape = y.shape[:-1]
    """
    out_shape = y.shape[:-1]
    y = y.reshape(-1, y.shape[-1])
    auc = ((x[1:] - x[:-1]).reshape(1, -1) * ((y[..., 1:] + y[..., :-1]) * 0.5)).sum(
        dim=-1
    )
    max_area = (x[-1] - x[0]) * y_max
    return (auc / max_area).reshape(out_shape)


def compute_mpjpe(pred_landmarks: torch.Tensor, gt_landmarks: torch.Tensor) -> float:
    """
    Computes the MPJPE for all landmarks.
    pred_landmarks: N x L x 3, predicted landmarks
    gt_landmarks: N x L x 3, gt landmarks
    where L is the number of Landmarks
    """
    per_landmark_error = torch.sqrt(
        torch.sum(torch.square(pred_landmarks - gt_landmarks), dim=-1)
    )
    mpjpe = torch.mean(per_landmark_error)
    return mpjpe.item()


def compute_mpvpe(pred_vertices: torch.Tensor, gt_vertices: torch.Tensor) -> float:
    """
    Computes the MPVPE for all vertices.
    pred_vertices: N x V, predicted landmarks
    gt_landmarks: N x V, gt landmarks
    where V is the number of vertices
    """
    per_vertex_error = torch.sqrt(
        torch.sum(torch.square(pred_vertices - gt_vertices), dim=-1)
    )
    mpvpe = torch.mean(per_vertex_error)
    return mpvpe.item()


def compute_pck_auc(pred_landmarks: torch.Tensor, gt_landmarks: torch.Tensor) -> float:
    """
    Computes the PCK AUC for all landmarks.
    pred_landmarks: N x L x 3, predicted landmarks
    gt_landmarks: N x L x 3, gt landmarks
    where L is the number of Landmarks
    """
    per_landmark_error = torch.sqrt(
        torch.sum(torch.square(pred_landmarks - gt_landmarks), dim=-1)
    )
    pckc = PCK_curve(per_landmark_error, PCK_THRESHOLDS) * 100
    return float(normalized_AUC(PCK_THRESHOLDS, pckc))


def compute_fingertip_pck_auc(
    pred_landmarks: torch.Tensor, gt_landmarks: torch.Tensor
) -> float:
    """
    Computes the PCK AUC for fingertips.
    pred_landmarks: N x L x 3, predicted landmarks
    gt_landmarks: N x L x 3, gt landmarks
    where L is the number of Landmarks
    """
    per_landmark_error = torch.sqrt(
        torch.sum(torch.square(pred_landmarks - gt_landmarks), dim=-1)
    )
    fingertip_errors = per_landmark_error[:, FINGERTIP_IDX]
    pckc = PCK_curve(fingertip_errors, PCK_THRESHOLDS) * 100
    return float(normalized_AUC(PCK_THRESHOLDS, pckc))


def compute_pose_metrics(
    *,
    pred_pose_params: torch.Tensor,
    pred_wrist_xform: torch.Tensor,
    pred_shape_params: torch.Tensor,
    gt_landmarks: torch.Tensor,
    hand_side: torch.Tensor,
    mano_model: MANOHandModel,
) -> Dict[str, float]:
    """
    Method for computing all the metrics for the pose track

    Args:
        pred_pose_params: N x 15, predicted pose parameters
        pred_wrist_xform: N x 6, predicted global transformation
        pred_shape_params: N x 10, predicted shape parameters
        gt_landmarks: N x 19 x 3, ground truth landmarks
        is_right_hand: N x 1, boolean mask for right hand

    Returns:
        A dictionary with the following keys:
            MPJPE: mean per joint error
            MPVPE: mean per vertex error
            PCK_AUC: Overall PCK AUC
            FINGERTIP_PCK_AUC: PCK AUC for fingertips
    """
    (
        pred_landmarks,
        _,
        _,
    ) = mano_forward_kinematics(
        MANOHandPose(
            hand_side=hand_side,
            mano_theta=pred_pose_params,
            wrist_xform=pred_wrist_xform,
        ),
        pred_shape_params,
        mano_model,
    )

    pred_landmarks = pred_landmarks[:, LANDMARKS_TO_EVAL, :] * M_TO_MM
    gt_landmarks = gt_landmarks[:, LANDMARKS_TO_EVAL, :] * M_TO_MM

    # compute mpjpe
    mpjpe = compute_mpjpe(pred_landmarks, gt_landmarks)
    # compute pck auc
    pck_auc = compute_pck_auc(pred_landmarks, gt_landmarks)
    # compute fingertip pck auc
    fingertip_pck_auc = compute_fingertip_pck_auc(pred_landmarks, gt_landmarks)

    return {
        "MPJPE": mpjpe,
        "PCK_AUC": pck_auc,
        "FINGERTIP_PCK_AUC": fingertip_pck_auc,
    }


def compute_shape_metrics(
    *,
    pred_shape_params: torch.Tensor,
    gt_shape_params: torch.Tensor,
    mano_model: MANOHandModel,
) -> Dict[str, float]:
    """
    Method for computing all the metrics for the shape track

    Args:
        pred_shape_params: N x 10, predicted shape parameters
        gt_shape_params: N x 10, ground truth shape parameters

    Returns:
        A dictionary with the following keys:
            MPVPE: mean per vertex error
    """
    zero_pose = MANOHandPose(
        hand_side=torch.zeros(pred_shape_params.shape[0], dtype=torch.int),
        mano_theta=torch.zeros(
            (pred_shape_params.shape[0], mano_model.num_pose_coeffs),
            dtype=torch.float32,
        ),
        wrist_xform=torch.zeros((pred_shape_params.shape[0], 6), dtype=torch.float32),
    )

    _, pred_mano_vertices, _ = mano_forward_kinematics(
        zero_pose,
        pred_shape_params,
        mano_model,
    )

    _, gt_mano_vertices, _ = mano_forward_kinematics(
        zero_pose,
        gt_shape_params,
        mano_model,
    )

    pred_mano_vertices = pred_mano_vertices * M_TO_MM
    gt_mano_vertices = gt_mano_vertices * M_TO_MM

    assert pred_mano_vertices.shape == gt_mano_vertices.shape

    mpvpe = compute_mpvpe(pred_mano_vertices, gt_mano_vertices)
    return {
        "MPVPE": mpvpe,
    }
