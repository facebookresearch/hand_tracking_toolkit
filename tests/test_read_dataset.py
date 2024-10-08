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

import importlib.resources
import os.path
import shutil
import unittest

from hand_tracking_toolkit.dataset import build_hand_dataset

TMP_FOLDER = "/tmp"


class TestReadDataset(unittest.TestCase):
    def setUp(self):
        self.test_data_dir = str(
            importlib.resources.files(__package__).joinpath(
                "oss_hand_tracking_toolkit_test_data/test_data/oss_hand_tracking_toolkit",
            )
        )
        self.test_sequence_name = "small"

        self.tar_path = os.path.join(
            self.test_data_dir, f"{self.test_sequence_name}.tar"
        )
        shutil.copyfile(self.tar_path, f"{TMP_FOLDER}/{self.test_sequence_name}.tar")

    def test_full_dataset(self):
        iter_dataset = build_hand_dataset(
            TMP_FOLDER,
            [self.test_sequence_name],
            load_monochrome=True,
            load_rgb=True,
            output_crops=False,
        )
        self.assertGreater(len(list(iter_dataset)), 0)

    def test_crop_dataset(self):
        iter_dataset = build_hand_dataset(
            TMP_FOLDER,
            [self.test_sequence_name],
            load_monochrome=True,
            load_rgb=True,
            output_crops=True,
        )
        self.assertGreater(len(list(iter_dataset)), 0)
