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

import argparse
import logging
from pathlib import Path

from hand_tracking_toolkit.dataset import build_hand_dataset, HandSide
from hand_tracking_toolkit.hand_models.umetrack_hand_model import (
    forward_kinematics as umetrack_forward_kinematics,
)
from hand_tracking_toolkit.submissions import (
    LandmarksSample,
    MANOPoseSample,
    MANOShapeSample,
    write_landmarks_file,
    write_mano_poses_file,
    write_mano_shapes_file,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        """
        Extracts ground truth annotations from sequences written in webdataset format.
        The ground truth annotations for each track (pose estimation, shape estimation)
        are written in a single json file in order to speed up loading during evaluation.
        """
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    sequence_names = []
    for tar_path in input_dir.glob("*.tar"):
        sequence_names.append(tar_path.name.split(".")[0])
    logger.info(f"Found {len(sequence_names)} sequences in input dir {input_dir}.")

    pose_samples = []
    shape_samples = []
    landmarks_samples = []
    for sequence_name in tqdm(sequence_names):
        dataset = build_hand_dataset(
            str(input_dir), [sequence_name], load_monochrome=False, load_rgb=False
        )

        for i, sample in enumerate(dataset):
            for hand_side in (HandSide.LEFT, HandSide.RIGHT):
                if i == 0:
                    shape_samples.append(
                        MANOShapeSample(
                            sequence_name=sequence_name,
                            mano_beta=sample.hand_shape.mano_beta.numpy(),
                            hand_side=hand_side,
                        )
                    )

                if hand_side not in sample.hand_poses:
                    continue
                hand_pose = sample.hand_poses[hand_side]

                pose_samples.append(
                    MANOPoseSample(
                        sequence_name=sequence_name,
                        frame_id=sample.frame_id,
                        mano_theta=hand_pose.mano.mano_theta,
                        wrist_xform=hand_pose.mano.wrist_xform,
                        hand_side=hand_side,
                    )
                )

                landmarks, _, _ = umetrack_forward_kinematics(
                    hand_pose.umetrack, sample.hand_shape.umetrack, requires_mesh=False
                )

                landmarks_samples.append(
                    LandmarksSample(
                        sequence_name=sequence_name,
                        frame_id=sample.frame_id,
                        landmarks=landmarks.numpy(),
                    )
                )

    write_mano_poses_file(output_dir / "gt_pose.json", pose_samples)
    write_mano_shapes_file(output_dir / "gt_shape.json", shape_samples)
    write_landmarks_file(output_dir / "gt_landmarks.json", landmarks_samples)


if __name__ == "__main__":
    # Do not add code here, it won't be run. Add them to the function called below.
    main()  # pragma: no cover
