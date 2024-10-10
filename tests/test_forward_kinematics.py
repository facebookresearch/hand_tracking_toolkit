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

import dataclasses
import importlib.resources
import json
import os.path
import unittest

import numpy as np
import torch
from hand_tracking_toolkit.hand_models.mano_hand_model import (
    forward_kinematics as mano_forward_kinematics,
    MANOHandModel,
    MANOHandPose,
)
from hand_tracking_toolkit.hand_models.umetrack_hand_model import (
    forward_kinematics as umetrack_forward_kinematics,
    from_json as from_umetrack_hand_model_json,
    UmeTrackHandModelData,
    UmeTrackHandPose,
)

from smplx.utils import MANOOutput


class MockMANOLayer:
    def __init__(self):
        self.faces = np.zeros((123, 3), dtype=np.int)

    def __call__(
        self,
        betas,
        global_orient,
        hand_pose,
        transl,
        return_verts=True,
        return_full_pose=False,
        **kwargs,
    ):
        N = betas.shape[0]
        output = MANOOutput(
            vertices=torch.rand(N, 778, 3) if return_verts else None,
            joints=torch.rand(N, 16, 3) if return_verts else None,
            betas=betas,
            global_orient=global_orient,
            hand_pose=hand_pose,
            full_pose=None,
        )
        return output


class MockMANOHandModel(MANOHandModel):
    def __init__(self):
        super().__init__()
        self.mano_layer_left = MockMANOLayer()
        self.mano_layer_right = MockMANOLayer()


class TestForwardKinematics(unittest.TestCase):
    def setUp(self):
        self.mano_model = MockMANOHandModel()
        test_data_dir = str(
            importlib.resources.files(__package__).joinpath(
                "oss_hand_tracking_toolkit_test_data/test_data/oss_hand_tracking_toolkit",
            )
        )
        with open(
            os.path.join(test_data_dir, "sample_umetrack_hand_model_data.json"), "r"
        ) as fp:
            j = json.load(fp)
        self.umetrack_hand_model_data = from_umetrack_hand_model_json(j)

    def test_mano_forward_kinematics(self):
        N = 3
        pose = MANOHandPose(
            hand_side=torch.zeros(N, dtype=torch.int),
            mano_theta=torch.zeros(
                (N, 15),
                dtype=torch.float32,
            ),
            wrist_xform=torch.zeros((N, 6), dtype=torch.float32),
        )
        mano_beta = torch.rand((N, 10), dtype=torch.float32)

        landmarks, verts, faces = mano_forward_kinematics(
            pose,
            mano_beta,
            self.mano_model,
        )

        self.assertEqual(len(landmarks.shape), 3)
        self.assertEqual(len(verts.shape), 3)
        self.assertEqual(len(faces.shape), 3)

        self.assertEqual(landmarks.shape[0], N)
        self.assertEqual(verts.shape[0], N)
        self.assertEqual(faces.shape[0], N)

    def test_mano_forward_kinematics_single_instance(self):
        pose = MANOHandPose(
            hand_side=torch.tensor(0),
            mano_theta=torch.zeros(
                15,
                dtype=torch.float32,
            ),
            wrist_xform=torch.zeros((6), dtype=torch.float32),
        )
        mano_beta = torch.rand((10), dtype=torch.float32)

        landmarks, verts, faces = mano_forward_kinematics(
            pose,
            mano_beta,
            self.mano_model,
        )

        self.assertEqual(len(landmarks.shape), 2)
        self.assertEqual(len(verts.shape), 2)
        self.assertEqual(len(faces.shape), 2)

    def test_umetrack_forward_kinematics(self):
        N = 3
        pose = UmeTrackHandPose(
            hand_side=torch.zeros(N, dtype=torch.int),
            joint_angles=torch.zeros((N, 22), dtype=torch.float32),
            wrist_xform=torch.stack([torch.eye(4, dtype=torch.float32)] * N),
        )

        # make a batch of hand model data
        umetrack_hand_model_data = UmeTrackHandModelData(
            **{
                k: torch.stack([v] * N)
                for k, v in dataclasses.asdict(self.umetrack_hand_model_data).items()
                if v is not None
            }
        )

        landmarks, verts, faces = umetrack_forward_kinematics(
            pose, umetrack_hand_model_data, requires_mesh=True
        )

        self.assertTrue(landmarks.isnan().sum().item() == 0)
        self.assertEqual(len(landmarks.shape), 3)
        self.assertEqual(len(verts.shape), 3)
        self.assertEqual(len(faces.shape), 3)

        self.assertEqual(landmarks.shape[0], N)
        self.assertEqual(verts.shape[0], N)
        self.assertEqual(faces.shape[0], N)

    def test_umetrack_forward_kinematics_single_instance(self):
        pose = UmeTrackHandPose(
            hand_side=torch.tensor(0),
            joint_angles=torch.zeros((22), dtype=torch.float32),
            wrist_xform=torch.eye(4, dtype=torch.float32),
        )
        landmarks, verts, faces = umetrack_forward_kinematics(
            pose, self.umetrack_hand_model_data, requires_mesh=True
        )

        self.assertEqual(len(landmarks.shape), 2)
        self.assertEqual(len(verts.shape), 2)
        self.assertEqual(len(faces.shape), 2)
