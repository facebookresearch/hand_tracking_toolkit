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

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import torch

from .umetrack_skinning import get_skinning_weights, skin_points

NUM_JOINT_FRAMES: int = 1 + 1 + 3 * 5  # root + wrist + finger frames * 5
RIGHT_HAND_INDEX = 1
UMETRACK_TO_CANONICAL_LANDMARK_MAPPING: List[int] = list(range(20))


@dataclasses.dataclass
class UmeTrackHandPose:
    hand_side: torch.Tensor
    joint_angles: torch.Tensor
    wrist_xform: torch.Tensor


@dataclasses.dataclass
class UmeTrackHandModelData:
    joint_rotation_axes: torch.Tensor
    joint_rest_positions: torch.Tensor
    joint_frame_index: torch.Tensor
    joint_parent: torch.Tensor
    joint_first_child: torch.Tensor
    joint_next_sibling: torch.Tensor
    landmark_rest_positions: torch.Tensor
    landmark_rest_bone_weights: torch.Tensor
    landmark_rest_bone_indices: torch.Tensor
    hand_scale: Optional[torch.Tensor] = None
    mesh_vertices: Optional[torch.Tensor] = None
    mesh_triangles: Optional[torch.Tensor] = None
    dense_bone_weights: Optional[torch.Tensor] = None
    joint_limits: Optional[torch.Tensor] = None


def from_json(j: Dict[str, Any]) -> UmeTrackHandModelData:
    model = UmeTrackHandModelData(**{k: torch.tensor(v) for k, v in j.items()})
    MM_TO_M = 0.001
    model.joint_rest_positions *= MM_TO_M
    model.landmark_rest_positions *= MM_TO_M
    if model.mesh_vertices is not None:
        model.mesh_vertices *= MM_TO_M
    return model


def skin_landmarks(
    hand_model: UmeTrackHandModelData,
    joint_angles: torch.Tensor,
    wrist_transforms: torch.Tensor,
) -> torch.Tensor:
    leading_dims = joint_angles.shape[:-1]
    numel = torch.flatten(joint_angles, end_dim=-2).shape[0] if len(leading_dims) else 1
    max_weights = hand_model.landmark_rest_bone_indices.shape[-1]
    skin_mat = get_skinning_weights(
        hand_model.landmark_rest_bone_indices.reshape(numel, -1, max_weights),
        hand_model.landmark_rest_bone_weights.reshape(numel, -1, max_weights),
        NUM_JOINT_FRAMES,
    )
    return skin_points(
        hand_model.joint_rest_positions,
        hand_model.joint_rotation_axes,
        skin_mat,
        joint_angles,
        hand_model.landmark_rest_positions,
        wrist_transforms,
    )


def skin_vertices(
    hand_model: UmeTrackHandModelData,
    joint_angles: torch.Tensor,
    wrist_transforms: torch.Tensor,
) -> torch.Tensor:
    assert hand_model.mesh_vertices is not None, "mesh vertices should not be none"
    assert hand_model.dense_bone_weights is not None, (
        "dense bone weights should not be none"
    )
    vertices = skin_points(
        hand_model.joint_rest_positions,
        hand_model.joint_rotation_axes,
        hand_model.dense_bone_weights,
        joint_angles,
        hand_model.mesh_vertices,
        wrist_transforms,
    )

    leading_dims = joint_angles.shape[:-1]
    vertices = vertices.reshape(list(leading_dims) + list(vertices.shape[-2:]))
    return vertices


def forward_kinematics(
    hand_pose: UmeTrackHandPose,
    hand_model: UmeTrackHandModelData,
    requires_mesh: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Runs UmeTrack forward kinematics on possibly batched data"""

    wrist_xform = hand_pose.wrist_xform.clone()

    is_batched = len(wrist_xform.shape) > 2
    N = wrist_xform.shape[0] if is_batched else 1

    hand_side = hand_pose.hand_side.reshape(N)
    right_hand_mask = hand_side == RIGHT_HAND_INDEX

    wrist_xform = wrist_xform.reshape(N, 4, 4)
    wrist_xform[right_hand_mask, :, 0] *= -1
    if not is_batched:
        wrist_xform = wrist_xform[0]
    landmarks = skin_landmarks(hand_model, hand_pose.joint_angles, wrist_xform)
    landmarks = landmarks[..., UMETRACK_TO_CANONICAL_LANDMARK_MAPPING, :]

    if requires_mesh:
        verts = skin_vertices(hand_model, hand_pose.joint_angles, wrist_xform)

        assert hand_model.mesh_triangles is not None
        faces = hand_model.mesh_triangles.int().reshape(N, -1, 3)
        faces[right_hand_mask] = torch.flip(faces[right_hand_mask], dims=[-1])
        if not is_batched:
            faces = faces[0]
    else:
        verts = None
        faces = None

    return landmarks, verts, faces
