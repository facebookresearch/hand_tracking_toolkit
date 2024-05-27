# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pathlib
import unittest

from hand_tracking_toolkit.dataset import load_sequence_data

from libfb.py import parutil


class TestReadDataset(unittest.TestCase):
    def setUp(self):
        self.test_data_dir = pathlib.Path(
            "oss_hand_tracking_toolkit_test_data/test_data/oss_hand_tracking_toolkit",
        )
        self.test_sequence_name = (
            "synthetic_dpe_small_2021_h2_clap_both_9053_ganon_doom_2021-09-09_111837_0"
        )
        self.tar_path = parutil.get_file_path(
            self.test_data_dir / (self.test_sequence_name + ".tar"),
            __package__,
        )
        self.shape_params_path = parutil.get_file_path(
            self.test_data_dir / (self.test_sequence_name + "_shape_params.json"),
            __package__,
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
