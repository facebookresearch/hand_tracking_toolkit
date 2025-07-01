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

import dataclasses
import os
from typing import Tuple

import numpy as np

import torch
import torch.nn as nn


try:
    import smplx
except ImportError:
    print(
        "INFO: Using MANO requires smplx (See our GitHub repository for more information on its installation)."
    )


# Without this hack loading the .pkl files could fail when using
# a newer version of numpy
if np.__version__ < "2":
    np.bool = np.bool_
    np.int = np.int_
    # `np.float_` was removed in the NumPy 2.0 release.
    np.float64 = np.float_
    np.float = np.float_
    np.complex = np.complex128
    np.object = np.object_
    np.unicode = np.unicode_
    np.str = np.str_

RIGHT_HAND_INDEX = 1
MANO_TO_CANONICAL_LANDMARK_MAPPING = [
    16,
    17,
    18,
    19,
    20,
    0,
    14,
    15,
    1,
    2,
    3,
    4,
    5,
    6,
    10,
    11,
    12,
    7,
    8,
    9,
]


@dataclasses.dataclass
class MANOHandPose:
    hand_side: torch.Tensor
    mano_theta: torch.Tensor
    wrist_xform: torch.Tensor


class MANOHandModel(nn.Module):
    N_VERT = 778
    N_LANDMARKS = 21
    MANO_FINGERTIP_VERT_INDICES = {
        "thumb": 744,
        "index": 320,
        "middle": 443,
        "ring": 554,
        "pinky": 671,
    }
    num_pose_coeffs = 15
    num_shape_params = 10
    dtype = torch.float32

    def __init__(self, mano_model_files_dir: str = ""):
        super().__init__()
        if not mano_model_files_dir:
            return

        mano_left_filename = os.path.join(mano_model_files_dir, "MANO_LEFT.pkl")
        mano_right_filename = os.path.join(mano_model_files_dir, "MANO_RIGHT.pkl")

        self.mano_layer_left = smplx.create(
            mano_left_filename,
            "mano",
            use_pca=True,
            is_rhand=False,
            num_pca_comps=self.num_pose_coeffs,
        )

        self.mano_layer_right = smplx.create(
            mano_right_filename,
            "mano",
            use_pca=True,
            is_rhand=True,
            num_pca_comps=self.num_pose_coeffs,
        )

        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        if (
            torch.sum(
                torch.abs(
                    self.mano_layer_left.shapedirs[:, 0, :]
                    - self.mano_layer_right.shapedirs[:, 0, :]
                )
            )
            < 1
        ):
            self.mano_layer_left.shapedirs[:, 0, :] *= -1

    def forward(
        self,
        mano_beta: torch.Tensor,
        mano_theta: torch.Tensor,
        wrist_xform: torch.Tensor,  # meter
        is_right_hand: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_frames = mano_theta.shape[0]
        device = mano_theta.device
        is_right_hand = is_right_hand.bool()
        is_left_hand = ~is_right_hand

        # Left hand FK
        if torch.any(is_left_hand):
            left_wrist_xform = wrist_xform[is_left_hand]
            left_joint_angles = mano_theta[is_left_hand]
            left_mano_beta = mano_beta[is_left_hand]
            left_mano_output = self.mano_layer_left(
                betas=left_mano_beta.to(self.dtype),
                global_orient=left_wrist_xform[:, :3].to(self.dtype),
                hand_pose=left_joint_angles.to(self.dtype),
                transl=left_wrist_xform[:, 3:].to(self.dtype),
                return_verts=True,  # MANO doesn't return landmarks as well if this is false
            )

        # Right hand FK
        if torch.any(is_right_hand):
            right_wrist_xform = wrist_xform[is_right_hand]
            right_joint_angles = mano_theta[is_right_hand]
            right_mano_beta = mano_beta[is_right_hand]
            right_mano_output = self.mano_layer_right(
                betas=right_mano_beta.to(self.dtype),
                global_orient=right_wrist_xform[:, :3].to(self.dtype),
                hand_pose=right_joint_angles.to(self.dtype),
                transl=right_wrist_xform[:, 3:].to(self.dtype),
                return_verts=True,  # MANO doesn't return landmarks as well if this is false
            )

        # Merge the left and right hand outputs
        out_vertices = torch.zeros(
            (
                num_frames,
                self.N_VERT,
                3,
            ),
            device=device,
        )
        if torch.any(torch.logical_not(is_right_hand)):
            out_vertices[torch.logical_not(is_right_hand)] = left_mano_output.vertices
        if torch.sum(is_right_hand) > 0:
            out_vertices[is_right_hand] = right_mano_output.vertices

        out_landmarks = torch.zeros(
            (
                num_frames,
                self.N_LANDMARKS,
                3,
            ),
            device=device,
        )
        if torch.any(torch.logical_not(is_right_hand)):
            if left_mano_output.joints.shape[1] != self.N_LANDMARKS:
                extra_joints = torch.index_select(
                    left_mano_output.vertices,
                    1,
                    torch.tensor(
                        list(self.MANO_FINGERTIP_VERT_INDICES.values()),
                        dtype=torch.long,
                    ),
                )
                joints = torch.cat([left_mano_output.joints, extra_joints], dim=1)
            else:
                joints = left_mano_output.joints
            out_landmarks[torch.logical_not(is_right_hand)] = joints
        if torch.sum(is_right_hand) > 0:
            if right_mano_output.joints.shape[1] != self.N_LANDMARKS:
                extra_joints = torch.index_select(
                    right_mano_output.vertices,
                    1,
                    torch.tensor(
                        list(self.MANO_FINGERTIP_VERT_INDICES.values()),
                        dtype=torch.long,
                    ),
                )
                joints = torch.cat([right_mano_output.joints, extra_joints], dim=1)
            else:
                joints = right_mano_output.joints
            out_landmarks[is_right_hand] = joints

        assert out_landmarks.shape[1] == self.N_LANDMARKS

        return out_vertices, out_landmarks


def forward_kinematics(
    hand_pose: MANOHandPose,
    mano_beta: torch.Tensor,
    mano_model: MANOHandModel,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run MANO forward kinematics on possibly batached data"""

    is_batched = len(mano_beta.shape) > 1
    N = mano_beta.shape[0] if is_batched else 1

    mano_beta = mano_beta.reshape(N, -1)
    mano_theta = hand_pose.mano_theta.reshape(N, -1)
    wrist_xform = hand_pose.wrist_xform.reshape(N, -1)
    hand_side = hand_pose.hand_side.reshape(N)

    right_hand_mask = hand_side == RIGHT_HAND_INDEX
    verts, landmarks = mano_model(
        mano_beta,
        mano_theta,
        wrist_xform,
        is_right_hand=right_hand_mask,
    )
    faces = torch.zeros(
        (N, mano_model.mano_layer_right.faces.shape[0], 3),
        dtype=torch.int32,
        device=verts.device,
    )
    faces[right_hand_mask] = torch.from_numpy(
        mano_model.mano_layer_right.faces.astype(np.int32)
    ).to(faces.device)
    faces[~right_hand_mask] = torch.from_numpy(
        mano_model.mano_layer_left.faces.astype(np.int32)
    ).to(faces.device)

    landmarks = landmarks[:, MANO_TO_CANONICAL_LANDMARK_MAPPING, :]

    if not is_batched:
        faces = faces[0]
        verts = verts[0]
        landmarks = landmarks[0]

    return landmarks, verts, faces
