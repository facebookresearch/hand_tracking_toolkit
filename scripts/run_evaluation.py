# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse

from hand_tracking_toolkit.evaluation import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-annotation-file", type=str, required=True, help="Path to GT tar"
    )
    parser.add_argument(
        "--user-submission-file",
        type=str,
        required=True,
        help="Path to submission file",
    )
    parser.add_argument("--phase-codename")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    output = evaluate(
        args.test_annotation_file, args.user_submission_file, args.phase_codename
    )
    print(output)
