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
import json
import logging
from pathlib import Path

import numpy as np
from nimble.opensource.hand_tracking_toolkit.hand_tracking_toolkit.dataset import (
    build_hand_dataset,
    HandSide,
)
from nimble.opensource.hand_tracking_toolkit.hand_tracking_toolkit.hand_models.umetrack_hand_model import (
    forward_kinematics as umetrack_forward_kinematics,
)
from nimble.opensource.hand_tracking_toolkit.hand_tracking_toolkit.submissions import (
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
    parser.add_argument("--hot3d-split", type=str, required=False, default="")
    parser.add_argument("--every-n-frames", type=int, required=False, default=1)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.hot3d_split:
        hot3d_release_dir = input_dir
        with open(str(hot3d_release_dir / "clip_splits.json"), "rb") as fp:
            j = json.load(fp)

        sequence_names = []
        for device in ["Aria", "Quest3"]:
            for clip_id in j[args.hot3d_split][device]:
                sequence_names.append(
                    f"test_{device.lower()}_private/clip-{clip_id:06d}"
                )
    else:
        sequence_names = []
        for tar_path in input_dir.glob("*.tar"):
            sequence_names.append(tar_path.name.split(".")[0])

    logger.info(f"Found {len(sequence_names)} sequences in input dir {input_dir}.")

    pose_samples = []
    shape_samples = []
    landmarks_samples = []

    for sequence_name in tqdm(sequence_names):
        dataset = build_hand_dataset(
            str(input_dir),
            [sequence_name],
            load_monochrome=False,
            load_rgb=False,
            output_crops=True,
        )

        # If sequence_name is a relative path, keep the basename
        sequence_name = sequence_name.split("/")[-1]

        shape_saved = False
        for i, hand_crops in enumerate(dataset):
            if i % args.every_n_frames:
                continue

            for sample in hand_crops:
                if not shape_saved:
                    # For each sequence, save the shape once. The shape parameters
                    # are shared between left/right hands
                    shape_samples.append(
                        MANOShapeSample(
                            sequence_name=sequence_name,
                            mano_beta=sample.hand_shape.mano_beta.numpy(),
                            hand_side=HandSide.LEFT,
                        )
                    )
                    shape_saved = True

                hand_pose = sample.hand_pose
                mano_pose = hand_pose.mano

                pose_samples.append(
                    MANOPoseSample(
                        sequence_name=sequence_name,
                        frame_id=sample.frame_id,
                        mano_theta=(
                            np.zeros(15, dtype=np.float32)
                            if mano_pose is None
                            else mano_pose.mano_theta
                        ),
                        wrist_xform=(
                            np.zeros(6, dtype=np.float32)
                            if mano_pose is None
                            else mano_pose.wrist_xform
                        ),
                        hand_side=hand_pose.hand_side,
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
