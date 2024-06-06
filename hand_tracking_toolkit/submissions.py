# Copyright (c) Meta Platforms, Inc. and affiliates.

import dataclasses
import json
from pathlib import Path
from typing import List

import numpy as np
from hand_tracking_toolkit.dataset import HandSide


@dataclasses.dataclass
class PoseEstimationSubmissionSample:
    sequence_name: str
    frame_id: int
    mano_theta: np.ndarray
    wrist_xform: np.ndarray
    hand_side: HandSide


@dataclasses.dataclass
class ShapeEstimationSubmissionSample:
    sequence_name: str
    mano_beta: np.ndarray
    hand_side: HandSide


HAND_SIDE_TO_STR = {
    HandSide.LEFT: "left",
    HandSide.RIGHT: "right",
}


def write_pose_estimation_submission_file(
    path: Path,
    samples: List[PoseEstimationSubmissionSample],
):
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
                "hand_side": HAND_SIDE_TO_STR[sample.hand_side],
            }
        )
    path.write_text(json.dumps(samples_to_serialize, indent=4))
    return


def write_shape_estimation_submission_file(
    path: Path,
    samples: List[ShapeEstimationSubmissionSample],
):
    samples_to_serialize = []
    for sample in samples:
        assert len(sample.mano_beta.shape) == 1
        assert sample.mano_beta.shape[0] == 10
        samples_to_serialize.append(
            {
                "sequence_name": sample.sequence_name,
                "mano_beta": sample.mano_beta.tolist(),
                "hand_side": HAND_SIDE_TO_STR[sample.hand_side],
            }
        )
    path.write_text(json.dumps(samples_to_serialize, indent=4))
    return
