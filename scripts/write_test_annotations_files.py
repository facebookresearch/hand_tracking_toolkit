import argparse
import logging
from pathlib import Path

from hand_tracking_toolkit.dataset import HandSide, load_sequence_data
from hand_tracking_toolkit.submissions import (
    PoseEstimationSubmissionSample,
    ShapeEstimationSubmissionSample,
    write_pose_estimation_submission_file,
    write_shape_estimation_submission_file,
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
    for sequence_name in tqdm(sequence_names):
        sequence_data = load_sequence_data(
            tar_path=input_dir / (sequence_name + ".tar"),
            shape_params_path=input_dir / (sequence_name + "_shape_params.json"),
        )
        for hand_side in (HandSide.LEFT, HandSide.RIGHT):
            shape_samples.append(
                ShapeEstimationSubmissionSample(
                    sequence_name=sequence_name,
                    mano_beta=sequence_data.mano_betas[hand_side],
                    hand_side=hand_side,
                )
            )

        for n, _ in enumerate(sequence_data.frame_ids):
            for hand_pose in sequence_data.hand_poses[n]:
                pose_samples.append(
                    PoseEstimationSubmissionSample(
                        sequence_name=sequence_data.sequence_name,
                        frame_id=int(sequence_data.frame_ids[n]),
                        pose=hand_pose.pose,
                        global_xform=hand_pose.global_xform,
                        hand_side=hand_pose.hand_side,
                    )
                )
    write_pose_estimation_submission_file(
        output_dir / "test_pose_annotations.json", pose_samples
    )
    write_shape_estimation_submission_file(
        output_dir / "test_shape_annotations.json", shape_samples
    )


if __name__ == "__main__":
    # Do not add code here, it won't be run. Add them to the function called below.
    main()  # pragma: no cover
