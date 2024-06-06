# Copyright (c) Meta Platforms, Inc. and affiliates.

import os.path
import unittest

from hand_tracking_toolkit.dataset import build_hand_dataset
from libfb.py import parutil


class TestReadDataset(unittest.TestCase):
    def setUp(self):
        self.test_data_dir = parutil.get_file_path(
            "oss_hand_tracking_toolkit_test_data/test_data/oss_hand_tracking_toolkit",
            __package__,
        )
        self.test_sequence_name = "small"

        self.tar_path = os.path.join(
            self.test_data_dir, f"{self.test_sequence_name}.tar"
        )

    def test_full_dataset(self):
        iter_dataset = build_hand_dataset(
            self.test_data_dir,
            [self.test_sequence_name],
            load_monochrome=True,
            load_rgb=True,
            output_crops=False,
        )
        self.assertGreater(len(list(iter_dataset)), 0)

    def test_crop_dataset(self):
        iter_dataset = build_hand_dataset(
            self.test_data_dir,
            [self.test_sequence_name],
            load_monochrome=True,
            load_rgb=True,
            output_crops=True,
        )
        self.assertGreater(len(list(iter_dataset)), 0)
