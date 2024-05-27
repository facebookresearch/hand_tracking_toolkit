# pyre-unsafe
import unittest

import numpy as np
from hand_tracking_toolkit.camera import PinholePlaneCameraModel

from hand_tracking_toolkit.visualization import rasterize_mesh


class TestVisualization(unittest.TestCase):
    def test_rasterize_mesh(self):
        verts_np = np.array(
            [
                [-0.52573111, 0.85065081, 0.0],
                [0.52573111, 0.85065081, 0.0],
                [-0.52573111, -0.85065081, 0.0],
                [0.52573111, -0.85065081, 0.0],
                [0.0, -0.52573111, 0.85065081],
                [0.0, 0.52573111, 0.85065081],
                [0.0, -0.52573111, -0.85065081],
                [0.0, 0.52573111, -0.85065081],
                [0.85065081, 0.0, -0.52573111],
                [0.85065081, 0.0, 0.52573111],
                [-0.85065081, 0.0, -0.52573111],
                [-0.85065081, 0.0, 0.52573111],
            ]
        )

        faces_np = np.array(
            [
                [0, 11, 5],
                [0, 5, 1],
                [0, 1, 7],
                [0, 7, 10],
                [0, 10, 11],
                [1, 5, 9],
                [5, 11, 4],
                [11, 10, 2],
                [10, 7, 6],
                [7, 1, 8],
                [3, 9, 4],
                [3, 4, 2],
                [3, 2, 6],
                [3, 6, 8],
                [3, 8, 9],
                [4, 9, 5],
                [2, 4, 11],
                [6, 2, 10],
                [8, 6, 7],
                [9, 8, 1],
            ]
        )

        camera = PinholePlaneCameraModel(
            width=96,
            height=96,
            f=(80, 80),
            c=(47.5, 47.5),
            distort_coeffs=[],
            undistort_coeffs=[],
            T_world_from_eye=np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -3], [0, 0, 0, 1]]
            ),
        )

        img, mask = rasterize_mesh(
            verts_np, faces_np, mesh_color=[0.8, 0.8, 0.8], camera=camera
        )
        self.assertTrue(mask.sum() != 0)
