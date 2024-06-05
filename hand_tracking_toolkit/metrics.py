# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Helper methods for metric computation
"""

from typing import Dict, Optional

import numpy as np
import torch

from .mano_layer import MANOHandModel

MAX_LANDMARK_ERROR_MM = 50
PCK_THRESHOLDS: torch.Tensor = torch.linspace(0, MAX_LANDMARK_ERROR_MM, 101)
MANO_FINGERTIP_IDX = list(range(16, 21))
M_TO_MM = 1000  # meter to mm conversion


def _safe_div(x, y, eps: float = 1e-6, default_val: float = 0):
    if np.isscalar(x):
        assert np.isscalar(y)
        if y < eps:
            return default_val
        else:
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
    fingertip_errors = per_landmark_error[:, MANO_FINGERTIP_IDX]
    pckc = PCK_curve(fingertip_errors, PCK_THRESHOLDS) * 100
    return float(normalized_AUC(PCK_THRESHOLDS, pckc))


def compute_pose_metrics(
    pred_pose_params: torch.Tensor,
    gt_pose_params: torch.Tensor,
    pred_global_xform: torch.Tensor,
    gt_global_xform: torch.Tensor,
    pred_shape_params: torch.Tensor,
    gt_shape_params: torch.Tensor,
    is_right_hand: torch.Tensor,
    mano_model_dir: str,
) -> Dict[str, float]:
    """
    Method for computing all the metrics for the pose track

    Args:
        pred_pose_params: N x 15, predicted pose parameters
        gt_pose_params: N x 15, ground truth pose parameters
        pred_global_xform: N x 6, predicted global transformation
        gt_global_xform: N x 6, ground truth global transformation
        pred_shape_params: N x 10, predicted shape parameters
        gt_shape_params: N x 10, ground truth shape parameters
        is_right_hand: N x 1, boolean mask for right hand

    Returns:
        A dictionary with the following keys:
            MPJPE: mean per joint error
            MPVPE: mean per vertex error
            PCK_AUC: Overall PCK AUC
            FINGERTIP_PCK_AUC: PCK AUC for fingertips
    """
    mano_layer = MANOHandModel(mano_model_dir)

    pred_mano_vertices, pred_landmarks = mano_layer.forward_kinematics(
        pred_shape_params,
        pred_pose_params,
        pred_global_xform,
        is_right_hand=is_right_hand,
    )

    gt_mano_vertices, gt_landmarks = mano_layer.forward_kinematics(
        gt_shape_params,
        gt_pose_params,
        gt_global_xform,
        is_right_hand=is_right_hand,
    )

    pred_mano_vertices = pred_mano_vertices * M_TO_MM
    gt_mano_vertices = gt_mano_vertices * M_TO_MM
    pred_landmarks = pred_landmarks * M_TO_MM
    gt_landmarks = gt_landmarks * M_TO_MM

    # compute mpjpe
    mpjpe = compute_mpjpe(pred_landmarks, gt_landmarks)
    # compute pck auc
    pck_auc = compute_pck_auc(pred_landmarks, gt_landmarks)
    # compute fingertip pck auc
    fingertip_pck_auc = compute_fingertip_pck_auc(pred_landmarks, gt_landmarks)
    # compute mpvpe
    mpvpe = compute_mpvpe(pred_mano_vertices, gt_mano_vertices)

    return {
        "MPJPE": mpjpe,
        "MPVPE": mpvpe,
        "PCK_AUC": pck_auc,
        "FINGERTIP_PCK_AUC": fingertip_pck_auc,
    }


def compute_shape_metrics(
    pred_shape_params: torch.Tensor, gt_shape_params: torch.Tensor, mano_model_dir: str
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
    mano_layer = MANOHandModel(mano_model_dir)
    pred_mano_vertices, _ = mano_layer.shape_only_forward_kinematics(pred_shape_params)
    gt_mano_vertices, _ = mano_layer.shape_only_forward_kinematics(gt_shape_params)

    pred_mano_vertices = pred_mano_vertices * M_TO_MM
    gt_mano_vertices = gt_mano_vertices * M_TO_MM

    assert pred_mano_vertices.shape == gt_mano_vertices.shape

    mpvpe = compute_mpvpe(pred_mano_vertices, gt_mano_vertices)
    return {
        "MPVPE": mpvpe,
    }
