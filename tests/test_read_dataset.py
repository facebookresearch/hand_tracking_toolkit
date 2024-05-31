# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os.path
import unittest

from hand_tracking_toolkit.dataset import build_hand_crop_dataset, load_sequence_data
from libfb.py import parutil


class TestReadDataset(unittest.TestCase):
    def setUp(self):
        self.test_data_dir = parutil.get_file_path(
            "oss_hand_tracking_toolkit_test_data/test_data/oss_hand_tracking_toolkit",
            __package__,
        )
        self.test_sequence_name = "sequence0000"
        self.tar_path = os.path.join(
            self.test_data_dir, f"{self.test_sequence_name}.tar"
        )
        self.shape_params_path = os.path.join(
            self.test_data_dir, f"{self.test_sequence_name}_shape_params.json"
        )

    def test_read_dataset(self):
        sequence_data = load_sequence_data(
            tar_path=self.tar_path,
            shape_params_path=self.shape_params_path,
        )
        self.assertEqual(len(sequence_data.frame_ids), len(sequence_data.images))
        self.assertEqual(len(sequence_data.frame_ids), len(sequence_data.cameras))
        self.assertEqual(len(sequence_data.frame_ids), len(sequence_data.hand_poses))

        sequence_data = load_sequence_data(
            tar_path=self.tar_path,
            shape_params_path=self.shape_params_path,
            load_images=False,
        )
        self.assertEqual(len(list(sequence_data.images[0].keys())), 0)
        self.assertEqual(len(sequence_data.frame_ids), len(sequence_data.images))
        self.assertEqual(len(sequence_data.frame_ids), len(sequence_data.hand_poses))

    def test_hand_crop_dataset(self):
        iter_dataset = build_hand_crop_dataset(
            self.test_data_dir, [self.test_sequence_name]
        )
        self.assertGreater(len(list(iter_dataset)), 0)
