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

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nimble.opensource.hand_tracking_toolkit.hand_tracking_toolkit.dataset import (
    HandSide,
)

from nimble.opensource.hand_tracking_toolkit.hand_tracking_toolkit.submissions import (
    LandmarksSample,
    MANOPoseSample,
    MANOShapeSample,
    write_landmarks_file,
    write_mano_poses_file,
    write_mano_shapes_file,
)


class WriteSubmissionFilesTest(unittest.TestCase):
    def test_write_landmarks_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir, "poses.json")
            write_landmarks_file(
                file_path,
                [
                    LandmarksSample(
                        sequence_name="0000",
                        frame_id=0,
                        landmarks=np.zeros((20, 3)),
                    )
                ],
            )

            assert file_path.exists()

            data = json.loads(file_path.read_text())
            assert len(data) == 1
            data = data[0]
            assert data["sequence_name"] == "0000"
            assert data["frame_id"] == 0

    def test_write_mano_poses_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir, "poses.json")
            write_mano_poses_file(
                file_path,
                [
                    MANOPoseSample(
                        sequence_name="0000",
                        frame_id=0,
                        mano_theta=np.zeros(15),
                        wrist_xform=np.zeros(6),
                        hand_side=HandSide.LEFT,
                    )
                ],
            )

            assert file_path.exists()

            data = json.loads(file_path.read_text())
            assert len(data) == 1
            data = data[0]
            assert data["sequence_name"] == "0000"
            assert data["frame_id"] == 0
            assert data["hand_side"] == 0

    def test_write_mano_shapes_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir, "shapes.json")
            write_mano_shapes_file(
                file_path,
                [
                    MANOShapeSample(
                        sequence_name="0000",
                        mano_beta=np.zeros(10),
                        hand_side=HandSide.LEFT,
                    )
                ],
            )

            assert file_path.exists()
            data = json.loads(file_path.read_text())
            assert len(data) == 1
            data = data[0]
            assert data["sequence_name"] == "0000"
            assert data["hand_side"] == 0
