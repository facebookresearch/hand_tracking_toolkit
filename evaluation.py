# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Code to evaluate a submission file
"""

import argparse
import json
import tempfile
import zipfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from hand_tracking_toolkit.metrics import compute_pose_metrics, compute_shape_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-annotation-file", type=str, required=True, help="Path to GT zip"
    )
    parser.add_argument(
        "--user-annotation-file",
        type=str,
        required=True,
        help="Path to submission file",
    )
    parser.add_argument("--phase-codename")
    args = parser.parse_args()
    return args


def extract_zip(zip_file: Path, extract_dir: Path) -> None:
    assert zip_file.exists() and zip_file.suffix == ".zip"

    with zipfile.ZipFile(zip_file) as z:
        z.extractall(extract_dir)
    return


def load_gt_files(gt_dir: Path, dataset_suffix: str) -> Tuple[List[Dict], List[Dict]]:
    pose_file = gt_dir.joinpath(f"test_pose_annotations_{dataset_suffix}.json")
    shape_file = gt_dir.joinpath(f"test_shape_annotations_{dataset_suffix}.json")

    pose_gt = None
    with ExitStack() as stack:
        if pose_file.exists():
            pose_gt = json.load(stack.enter_context(open(pose_file)))
        shape_gt = json.load(stack.enter_context(open(shape_file)))
    return pose_gt, shape_gt


def load_pred_pose_file(
    pred_pose_dir: Path, dataset_suffix: str
) -> Optional[List[Dict]]:
    pred_pose_file = Path(pred_pose_dir, f"result_pose_{dataset_suffix}.json")
    if not pred_pose_file.exists():
        return None
    with open(pred_pose_file) as f:
        pose_data = json.load(f)
    return pose_data


def load_pred_shape_file(
    pred_shape_dir: Path, dataset_suffix: str
) -> Optional[List[Dict]]:
    pred_shape_file = Path(pred_shape_dir, f"result_shape_{dataset_suffix}.json")
    if not pred_shape_file.exists():
        return None
    with open(pred_shape_file) as f:
        shape_data = json.load(f)
    return shape_data


def get_batch_pose_data(pose_data: List[Dict[str, Any]]) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    poses = []
    global_xform = []
    is_right_hand = []
    frame_ids = []

    for pose_entry in pose_data:
        poses.append(torch.tensor(pose_entry["pose"]))
        global_xform.append(torch.tensor(pose_entry["global_xform"]))
        is_right_hand.append(pose_entry["hand_side"] == "right")
        frame_ids.append(pose_entry["frame_id"])

    poses = torch.vstack(poses)
    global_xform = torch.vstack(global_xform)
    is_right_hand = torch.tensor(is_right_hand)
    frame_ids = torch.tensor(frame_ids)

    # Sort by frame id
    sort_order = torch.argsort(frame_ids)
    poses = poses[sort_order]
    global_xform = global_xform[sort_order]
    is_right_hand = is_right_hand[sort_order]
    return poses, global_xform, is_right_hand


def compute_aggregate_metrics(all_metrics, all_num_frames):
    """
    Compute aggregate metrics over all sequences by taking weighted average
    of per sequence metrics.
    """
    assert len(all_metrics) > 0

    metric_dict = {}
    for metric in all_metrics[0]:
        vals = [x[metric] for x in all_metrics]
        metric_dict[metric] = np.average(vals, weights=all_num_frames)

    return metric_dict


def compute_overall_pose_metrics(
    unique_sequences, pose_pred, pose_gt, shape_gt, mano_dir
):
    all_num_frames = []
    all_pose_metrics = []
    for seq in unique_sequences:
        print(f"Processing sequence {seq}")
        pose_gt_seq = [x for x in pose_gt if x["sequence_name"] == seq]
        pose_pred_seq = [x for x in pose_pred if x["sequence_name"] == seq]

        shape_gt_seq = [x for x in shape_gt if x["sequence_name"] == seq]

        # Compute pose metrics
        gt_poses, gt_global_xform, gt_is_right_hand = get_batch_pose_data(pose_gt_seq)
        pred_poses, pred_global_xform, pred_is_right_hand = get_batch_pose_data(
            pose_pred_seq
        )

        assert gt_poses.shape[0] == pred_poses.shape[0]
        all_num_frames.append(gt_poses.shape[0])

        gt_shape_params = torch.tensor(shape_gt_seq[0]["mano_beta"])

        pose_metrics = compute_pose_metrics(
            pred_poses,
            gt_poses,
            pred_global_xform,
            gt_global_xform,
            gt_shape_params,
            gt_shape_params,
            gt_is_right_hand,
            mano_dir,
        )
        all_pose_metrics.append(pose_metrics)

    # Aggregate metrics
    pose_metrics = compute_aggregate_metrics(all_pose_metrics, all_num_frames)
    return pose_metrics


def compute_overall_shape_metrics(unique_sequences, shape_pred, shape_gt, mano_dir):
    all_shape_metrics = []
    for seq in unique_sequences:
        print(f"Processing sequence {seq}")

        shape_gt_seq = [x for x in shape_gt if x["sequence_name"] == seq]
        shape_pred_seq = [x for x in shape_pred if x["sequence_name"] == seq]

        # Compute pose metrics
        # Assuming single hand shape per sequence
        gt_shape_params = torch.tensor(
            np.vstack([x["mano_beta"] for x in shape_gt_seq])
        )
        pred_shape_params = torch.tensor(
            np.vstack([x["mano_beta"] for x in shape_pred_seq])
        )

        assert len(gt_shape_params) == len(pred_shape_params), "Shape params mismatch"

        shape_metrics = compute_shape_metrics(
            pred_shape_params, gt_shape_params, mano_dir
        )
        all_shape_metrics.append(shape_metrics)

    # Aggregate
    shape_metrics = {"MPVPE": np.mean([x["MPVPE"] for x in all_shape_metrics])}
    return shape_metrics


def evaluate_pose_dataset(
    pred_dir: Path, gt_dir: Path, mano_dir: Path, dataset_suffix: str
):
    pose_pred = load_pred_pose_file(pred_dir, dataset_suffix)
    if pose_pred is None:
        return None

    pose_gt, shape_gt = load_gt_files(gt_dir, dataset_suffix)

    unique_sequences = list({x["sequence_name"] for x in pose_gt})
    pose_metrics = compute_overall_pose_metrics(
        unique_sequences, pose_pred, pose_gt, shape_gt, mano_dir
    )
    return pose_metrics


def evaluate_shape_dataset(
    pred_dir: Path, gt_dir: Path, mano_dir: str, dataset_suffix: str
):
    shape_pred = load_pred_shape_file(pred_dir, dataset_suffix)
    if shape_pred is None:
        return None

    _, shape_gt = load_gt_files(gt_dir, dataset_suffix)

    unique_sequences = list({x["sequence_name"] for x in shape_gt})
    shape_metrics = compute_overall_shape_metrics(
        unique_sequences, shape_pred, shape_gt, mano_dir
    )
    return shape_metrics


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:
        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made
        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']
        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    print(f"Evaluating {phase_codename}")
    output = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        gt_dir = Path(tmp_dir, "gt")
        pred_dir = Path(tmp_dir, "pred")
        extract_zip(Path(test_annotation_file), gt_dir)
        extract_zip(Path(user_submission_file), pred_dir)
        mano_dir = gt_dir.joinpath("mano")
        if phase_codename == "pose_estimation":

            output_pose = {}
            for dataset_suffix in ["umetrack", "hot3d"]:
                res = evaluate_pose_dataset(
                    pred_dir, gt_dir, mano_dir, dataset_suffix=f"{dataset_suffix}"
                )
                if res:
                    output_pose[f"{dataset_suffix}_test"] = res
            # # To display the results in the result file
            output["submission_result"] = output_pose
        elif phase_codename == "shape_estimation":
            output_shape = {}
            for dataset_suffix in ["umetrack", "hot3d"]:
                res = evaluate_shape_dataset(
                    pred_dir, gt_dir, mano_dir, dataset_suffix=f"{dataset_suffix}"
                )
                if res:
                    output_shape[f"{dataset_suffix}_test"] = res
            # # To display the results in the result file
            output["submission_result"] = output_shape
        print(f"Completed evaluation for {phase_codename}")
    return output


if __name__ == "__main__":
    args = parse_args()
    output = evaluate(
        args.test_annotation_file, args.user_submission_file, args.phase_codename
    )
    print(output)
