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


def main() -> None:
    args = parse_args()
    output = evaluate(
        args.test_annotation_file,
        args.user_submission_file,
        args.phase_codename,
    )
    print(output)


if __name__ == "__main__":
    # Do not add code here, it won't be run. Add them to the function called below.
    main()  # pragma: no cover
