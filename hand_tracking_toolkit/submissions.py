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
import json
from pathlib import Path
from typing import List

import numpy as np

from .dataset import HandSide


@dataclasses.dataclass
class LandmarksSample:
    sequence_name: str
    frame_id: int
    landmarks: np.ndarray


@dataclasses.dataclass
class MANOPoseSample:
    sequence_name: str
    frame_id: int
    mano_theta: np.ndarray
    wrist_xform: np.ndarray
    hand_side: HandSide


@dataclasses.dataclass
class MANOShapeSample:
    sequence_name: str
    mano_beta: np.ndarray
    hand_side: HandSide


def write_landmarks_file(
    path: Path,
    samples: List[LandmarksSample],
) -> None:
    samples_to_serialize = []
    for sample in samples:
        assert sample.landmarks.shape == (20, 3)
        samples_to_serialize.append(
            {
                "sequence_name": sample.sequence_name,
                "frame_id": sample.frame_id,
                "landmarks": sample.landmarks.tolist(),
            }
        )
    path.write_text(json.dumps(samples_to_serialize, indent=4))


def write_mano_poses_file(
    path: Path,
    samples: List[MANOPoseSample],
) -> None:
    samples_to_serialize = []
    for sample in samples:
        assert len(sample.mano_theta.shape) == 1
        assert sample.mano_theta.shape[0] == 15
        assert len(sample.wrist_xform.shape) == 1
        assert sample.wrist_xform.shape[0] == 6
        assert isinstance(sample.frame_id, int)
        samples_to_serialize.append(
            {
                "sequence_name": sample.sequence_name,
                "frame_id": sample.frame_id,
                "mano_theta": sample.mano_theta.tolist(),
                "wrist_xform": sample.wrist_xform.tolist(),
                "hand_side": int(sample.hand_side == HandSide.RIGHT),
            }
        )
    path.write_text(json.dumps(samples_to_serialize, indent=4))


def write_mano_shapes_file(
    path: Path,
    samples: List[MANOShapeSample],
) -> None:
    samples_to_serialize = []
    for sample in samples:
        assert len(sample.mano_beta.shape) == 1
        assert sample.mano_beta.shape[0] == 10
        samples_to_serialize.append(
            {
                "sequence_name": sample.sequence_name,
                "mano_beta": sample.mano_beta.tolist(),
                "hand_side": int(sample.hand_side == HandSide.RIGHT),
            }
        )
    path.write_text(json.dumps(samples_to_serialize, indent=4))
