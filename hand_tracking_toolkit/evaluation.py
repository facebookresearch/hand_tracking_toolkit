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

"""
Code to evaluate a submission file
"""

import collections
import json
import tarfile
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import os

from .hand_models.mano_hand_model import MANOHandModel

from .metrics import compute_pose_metrics, compute_shape_metrics


def extract_tar(tar_file: Path, extract_dir: Path) -> None:
    assert tar_file.exists()

    # if tar_file.suffix == ".tar":
    #     mode = "r"
    # elif tar_file.suffix == ".gz":
    #     mode = "r:gz"
    # elif tar_file.suffix == ".tgz":
    #     mode = "r:gz"
    # else:
    #     raise RuntimeError("Incorrect tar file format")
    mode = "r"

    print(f"Untaring {tar_file}: {os.path.getsize(tar_file)}")
    with tarfile.open(tar_file, mode) as tf:
        print(f"Opened tar file")
        tf.extractall(extract_dir)
    print(f"Finished untaring {tar_file}")


def group_by_sequence_names(all_entries):
    if all_entries is None:
        return
    sequence_name_to_entries = collections.defaultdict(list)
    for entry in all_entries:
        sequence_name_to_entries[entry["sequence_name"]].append(entry)

    # sort by frame_id if possible
    for entries in sequence_name_to_entries.values():
        entries.sort(key=lambda x: x.get("frame_id", 0))

    sequence_name_to_collated_data = {}
    for sequence_name in sequence_name_to_entries:
        entries = sequence_name_to_entries[sequence_name]
        entry0 = entries[0]
        collated = {}
        for k, v0 in entry0.items():
            if isinstance(v0, (int, float)) or isinstance(v0, list):
                collated[k] = torch.tensor([entry[k] for entry in entries]).float()

        sequence_name_to_collated_data[sequence_name] = collated

    return sequence_name_to_collated_data


def load_gt_files(gt_dir: Path, dataset_suffix: str):
    landmarks_file = gt_dir.joinpath(f"gt_landmarks_{dataset_suffix}.json")
    shape_file = gt_dir.joinpath(f"gt_shape_{dataset_suffix}.json")

    landmarks_gt = None
    with ExitStack() as stack:
        if landmarks_file.exists():
            landmarks_gt = json.load(stack.enter_context(open(landmarks_file)))
        shape_gt = json.load(stack.enter_context(open(shape_file)))
    return group_by_sequence_names(landmarks_gt), group_by_sequence_names(shape_gt)


def load_pred_pose_file(pred_pose_dir: Path, dataset_suffix: str):
    pred_pose_file = Path(pred_pose_dir, f"result_pose_{dataset_suffix}.json")
    if not pred_pose_file.exists():
        return None
    with open(pred_pose_file) as f:
        pose_data = json.load(f)
    return group_by_sequence_names(pose_data)


def load_pred_shape_file(pred_shape_dir: Path, dataset_suffix: str):
    pred_shape_file = Path(pred_shape_dir, f"result_shape_{dataset_suffix}.json")
    if not pred_shape_file.exists():
        return None
    with open(pred_shape_file) as f:
        shape_data = json.load(f)
    return group_by_sequence_names(shape_data)


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
    *,
    sequence_name_to_pose_pred,
    sequence_name_to_landmarks_gt,
    sequence_name_to_shape_gt,
    mano_model,
):
    all_num_frames = []
    all_pose_metrics = []
    for seq in sequence_name_to_landmarks_gt:
        print(f"Processing sequence {seq}")

        pose_pred = sequence_name_to_pose_pred[seq]
        landmarks_gt = sequence_name_to_landmarks_gt[seq]
        shape_gt = sequence_name_to_shape_gt[seq]

        num_frames = landmarks_gt["landmarks"].shape[0]
        pose_metrics = compute_pose_metrics(
            pred_pose_params=pose_pred["mano_theta"],
            pred_wrist_xform=pose_pred["wrist_xform"],
            pred_shape_params=shape_gt["mano_beta"][0:1].repeat(num_frames, 1),
            gt_landmarks=landmarks_gt["landmarks"],
            hand_side=pose_pred["hand_side"],
            mano_model=mano_model,
        )

        all_pose_metrics.append(pose_metrics)
        all_num_frames.append(num_frames)

    # Aggregate metrics
    pose_metrics = compute_aggregate_metrics(all_pose_metrics, all_num_frames)
    return pose_metrics


def compute_overall_shape_metrics(
    *, sequence_name_to_shape_gt, sequence_name_to_shape_pred, mano_model
):
    all_shape_metrics = []
    for seq in sequence_name_to_shape_gt:
        print(f"Processing sequence {seq}")
        # Compute pose metrics
        # Assuming single hand shape per sequence
        shape_gt = sequence_name_to_shape_gt[seq]["mano_beta"]
        shape_pred = sequence_name_to_shape_pred[seq]["mano_beta"]
        assert len(shape_gt) == len(shape_pred), "Sizes of shape params mismatch"

        shape_metrics = compute_shape_metrics(
            pred_shape_params=shape_pred,
            gt_shape_params=shape_gt,
            mano_model=mano_model,
        )
        all_shape_metrics.append(shape_metrics)

    # Aggregate
    shape_metrics = {"MPVPE": np.mean([x["MPVPE"] for x in all_shape_metrics])}
    return shape_metrics


def evaluate_pose_dataset(
    pred_dir: Path, gt_dir: Path, mano_model, dataset_suffix: str
):
    pose_pred = load_pred_pose_file(pred_dir, dataset_suffix)
    if pose_pred is None:
        return None
    print(f"Done loading pred file from {pred_dir}")

    landmarks_gt, shape_gt = load_gt_files(gt_dir, dataset_suffix)
    print(f"Done loading gt file from {gt_dir}")

    pose_metrics = compute_overall_pose_metrics(
        sequence_name_to_pose_pred=pose_pred,
        sequence_name_to_landmarks_gt=landmarks_gt,
        sequence_name_to_shape_gt=shape_gt,
        mano_model=mano_model,
    )
    return pose_metrics


def evaluate_shape_dataset(
    pred_dir: Path, gt_dir: Path, mano_model, dataset_suffix: str
):
    shape_pred = load_pred_shape_file(pred_dir, dataset_suffix)
    if shape_pred is None:
        return None
    print(f"Done loading pred file from {pred_dir}")

    _, shape_gt = load_gt_files(gt_dir, dataset_suffix)
    print(f"Done loading gt file from {gt_dir}")

    shape_metrics = compute_overall_shape_metrics(
        sequence_name_to_shape_gt=shape_gt,
        sequence_name_to_shape_pred=shape_pred,
        mano_model=mano_model,
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
    print(f"Evaluating {phase_codename} branch:linguang")
    output = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        gt_dir = Path(tmp_dir, "gt")
        pred_dir = Path(tmp_dir, "pred")
        extract_tar(Path(user_submission_file), pred_dir)
        extract_tar(Path(test_annotation_file), gt_dir)
        mano_model = MANOHandModel(str(gt_dir.joinpath("mano")))
        print("Down building mano model")

        if phase_codename == "pose_estimation":
            for dataset_suffix in ["umetrack", "hot3d"]:
                res = evaluate_pose_dataset(
                    pred_dir, gt_dir, mano_model, dataset_suffix=f"{dataset_suffix}"
                )
                if res:
                    output.append({f"{dataset_suffix}_test": res})

        elif phase_codename == "shape_estimation":
            for dataset_suffix in ["umetrack", "hot3d"]:
                res = evaluate_shape_dataset(
                    pred_dir, gt_dir, mano_model, dataset_suffix=f"{dataset_suffix}"
                )
                if res:
                    output.append({f"{dataset_suffix}_test": res})

        print(f"Completed evaluation for {phase_codename}")

    return {"result": output}
