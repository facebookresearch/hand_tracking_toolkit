import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hand_tracking_toolkit.dataset import HandSide

from hand_tracking_toolkit.submissions import (
    PoseEstimationSubmissionSample,
    ShapeEstimationSubmissionSample,
    write_pose_estimation_submission_file,
    write_shape_estimation_submission_file,
)


class WriteSubmissionFilesTest(unittest.TestCase):
    def test_write_pose_estimation_submission_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir, "poses.json")
            write_pose_estimation_submission_file(
                file_path,
                [
                    PoseEstimationSubmissionSample(
                        sequence_name="0000",
                        frame_id=0,
                        pose=np.zeros(15),
                        global_xform=np.zeros(6),
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
            assert data["hand_side"] == "left"

    def test_write_shape_estimation_submission_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir, "shapes.json")
            write_shape_estimation_submission_file(
                file_path,
                [
                    ShapeEstimationSubmissionSample(
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
            assert data["hand_side"] == "left"
