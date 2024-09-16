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

# pyre-strict

from typing import List

import torch


NUM_DIGITS: int = 5
DOF_PER_FINGER: int = 4


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    leading_dims = axis_angle.shape[:-1]
    axis_angle = axis_angle.reshape(-1, 3)

    theta = torch.norm(axis_angle, p=2, dim=-1)
    out = torch.zeros((theta.shape[0], 3, 3), dtype=theta.dtype, device=theta.device)
    for i in range(3):
        out[:, i, i] = 1

    small_angle = theta < 1e-6
    theta = theta[~small_angle]
    axis_angle = axis_angle[~small_angle]

    axis = axis_angle / theta[..., None]

    c = torch.cos(theta)
    s = torch.sin(theta)
    kx = axis[..., 0]
    ky = axis[..., 1]
    kz = axis[..., 2]
    kxky = kx * ky
    kxkz = kx * kz
    kykz = ky * kz
    kx2 = kx * kx
    ky2 = ky * ky
    kz2 = kz * kz

    o = torch.stack(
        (
            c + kx2 * (1 - c),
            kxky * (1 - c) - kz * s,
            kxkz * (1 - c) + ky * s,
            kxky * (1 - c) + kz * s,
            c + ky2 * (1 - c),
            kykz * (1 - c) - kx * s,
            kxkz * (1 - c) - ky * s,
            kykz * (1 - c) + kx * s,
            c + kz2 * (1 - c),
        ),
        -1,
    )
    out[~small_angle] = o.reshape(-1, 3, 3)

    return out.reshape(*leading_dims, 3, 3)


def _finger_fk(
    joint_local_xfs: torch.Tensor, parent_transform: torch.Tensor
) -> List[torch.Tensor]:
    """
    each finger consists 4 DoF / Joints,
    and returns 3 transformation frames
    Input:
        joint_local_xfs: (B, 4, 4)
        parent_transform: (B, 4, 4)
    Return:
        transform_mats: (B, 3, 4, 4)
    """
    transform_mats = [parent_transform]
    for i in range(4):
        transform_mats.append(torch.matmul(transform_mats[-1], joint_local_xfs[:, i]))
    return transform_mats[2:]


def _joint_local_transform(
    rotation_axis: torch.Tensor, rest_pose: torch.Tensor, joint_angles: torch.Tensor
) -> torch.Tensor:
    rotation_axis_flat = rotation_axis.reshape(-1, 3)
    rest_pose_flat = rest_pose.reshape(-1, 3)
    joint_angles_flat = joint_angles.reshape(-1)

    angle_axis = rotation_axis_flat * joint_angles_flat.unsqueeze(-1)
    local_transform = torch.eye(4, dtype=angle_axis.dtype, device=angle_axis.device)
    local_transform = local_transform.unsqueeze(dim=0).repeat(angle_axis.shape[0], 1, 1)

    rot_mat = axis_angle_to_matrix(angle_axis)
    translation = rest_pose_flat - torch.matmul(
        rot_mat, rest_pose_flat.unsqueeze(dim=-1)
    ).squeeze(dim=-1)
    local_transform[:, :3, :3] = rot_mat
    local_transform[:, 0:3, 3] = torch.squeeze(translation, dim=-1)

    return local_transform.reshape(*rotation_axis.shape[0:-1], 4, 4)


def _lbs(trans_mats: torch.Tensor, skinned_points: torch.Tensor) -> torch.Tensor:
    """
    Input:
        trans_mats: (B, 17, 4, 4)
        skinned_points: (B, V, 17, 4)
    Return:
        fk_points: (B, V, 4)
    """
    trans_mats = trans_mats.unsqueeze(dim=1)
    skinned_points = skinned_points.unsqueeze(dim=-1)
    fk_points = torch.matmul(trans_mats, skinned_points).sum(dim=2).squeeze(dim=-1)
    return fk_points


def _hand_skinning_transform(
    rotation_axis: torch.Tensor,
    rest_poses: torch.Tensor,
    joint_angles: torch.Tensor,
    wrist_transforms: torch.Tensor,
) -> torch.Tensor:
    """
    Input:
        rotation_axis: (B, 20, 3)
        rest_poses: (B, 20, 3)
        joint_angles: (B, 20)
        wrist_transforms: (B, 4, 4)
    Return:
        skinning_matrices: (B, 17, 4, 4)
    """
    transform_mats = [wrist_transforms] * 2  # [root_transform, wrist_transforms]
    d = DOF_PER_FINGER

    joint_local_xfs = _joint_local_transform(
        rotation_axis[:, 0:20], rest_poses[:, 0:20], joint_angles[:, 0:20]
    )

    for finger_idx in range(NUM_DIGITS):
        transform_mats += _finger_fk(
            joint_local_xfs[:, d * finger_idx : d * finger_idx + d], wrist_transforms
        )
    transform_mats = torch.cat([m.unsqueeze(1) for m in transform_mats], dim=1)
    return transform_mats


def _get_skinned_vertices(
    vertices: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """
    Input:
        vertices: (B, V, 3) or (B, V, 4)
        weights: (B, V, 17)
    Return:
        skinned_vertices: (B, V, 17, 4)
    """
    if vertices.shape[2] == 3:
        n_vertices = vertices.shape[1]
        homo = torch.ones(
            vertices.shape[0],
            n_vertices,
            1,
            dtype=vertices.dtype,
            device=vertices.device,
        )
        vertices = torch.cat([vertices, homo], dim=-1)

    vertices = vertices.unsqueeze(dim=2)
    weights = weights.unsqueeze(dim=-1)
    return vertices * weights


def get_skinning_weights(
    bone_indices: torch.Tensor, bone_weights: torch.Tensor, n_frames: int
) -> torch.Tensor:
    """
    Input:
        bone_indices: (B, V, K)
        bone_weights: (B, V, K)
        n_frames: (or n_bones) Number of frames/bones (17 for hands)
          Note: K is number of bones.
    Return:
        skin_mat: (B, V, n_frames)
    """

    bs = bone_indices.shape[0]
    n_lms = bone_indices.shape[1]
    # Offset all the bones linearly from 0 to (bs*n_lms*n_frames) so that we can directly
    # index into the flattened weight matrix and set the corresponding skinning weights
    flat_idx_offset = torch.arange(0, bs * n_lms, device=bone_indices.device) * n_frames
    bone_flat_idx = bone_indices.long() + flat_idx_offset.reshape(bs, n_lms, 1)
    skin_mat = torch.zeros(
        bs * n_lms * n_frames, device=bone_weights.device, dtype=bone_weights.dtype
    )
    non0_w_mask = bone_weights != 0
    non0_indices = bone_flat_idx[non0_w_mask]
    skin_mat[non0_indices] = bone_weights[non0_w_mask]
    skin_mat = skin_mat.reshape(bs, n_lms, n_frames)

    return skin_mat


def skin_points(
    joint_rest_positions: torch.Tensor,
    joint_rotation_axes: torch.Tensor,
    skin_mat: torch.Tensor,
    joint_angles: torch.Tensor,
    points: torch.Tensor,
    wrist_transforms: torch.Tensor,
) -> torch.Tensor:
    leading_dims = joint_angles.shape[:-1]
    assert joint_rest_positions.shape[:-2] == leading_dims, (
        "Leading dimensions do not match, "
        + f"got {leading_dims} and {joint_rest_positions.shape[:-2]}"
    )

    # This allows querying the product of leading dimensions without making the
    # model specialized to a particular shape
    num_el = (
        torch.flatten(joint_angles, end_dim=-2).shape[0] if len(leading_dims) else 1
    )

    batched_joint_rest_positions = joint_rest_positions.reshape(num_el, -1, 3)

    skin_xfs = _hand_skinning_transform(
        rotation_axis=joint_rotation_axes.reshape(num_el, -1, 3),
        rest_poses=batched_joint_rest_positions,
        joint_angles=joint_angles.reshape(num_el, -1),
        wrist_transforms=wrist_transforms.reshape(num_el, 4, 4),
    )

    verts = _get_skinned_vertices(points.reshape(num_el, -1, 3), skin_mat)
    skinned_vecs = _lbs(skin_xfs, verts)[..., :3]
    skinned_vecs = skinned_vecs.reshape(
        list(leading_dims) + list(skinned_vecs.shape[-2:])
    )
    return skinned_vecs
