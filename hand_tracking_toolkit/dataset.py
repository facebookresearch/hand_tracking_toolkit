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

import dataclasses
import io
import json
import os.path
import tarfile
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional

import cv2

import numpy as np
import numpy.typing as npt
import PIL.Image
import torch
import webdataset as wds  # @manual

from . import math_utils
from .camera import CameraModel, from_json as from_camera_json, PinholePlaneCameraModel
from .hand_models.mano_hand_model import MANOHandPose
from .hand_models.umetrack_hand_model import (
    from_json as from_umetrack_hand_model_json,
    UmeTrackHandModelData,
    UmeTrackHandPose,
)


class HandSide(Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclasses.dataclass
class HandPoseCollection:
    hand_side: HandSide
    mano: Optional[MANOHandPose] = None
    umetrack: Optional[UmeTrackHandPose] = None


@dataclasses.dataclass
class HandShapeCollection:
    mano_beta: torch.Tensor
    umetrack: UmeTrackHandModelData


@dataclasses.dataclass
class HandData:
    url: str
    frame_id: int
    images: Dict[str, npt.NDArray]
    cameras: Dict[str, CameraModel]
    hand_poses: Optional[Dict[HandSide, HandPoseCollection]] = None
    hand_shape: Optional[HandShapeCollection] = None


@dataclasses.dataclass
class HandCropData:
    url: str
    frame_id: int
    images: Dict[str, npt.NDArray]
    cameras: Dict[str, PinholePlaneCameraModel]
    hand_pose: Optional[HandPoseCollection] = None
    hand_shape: Optional[HandShapeCollection] = None


def decode_cam_params(j) -> Dict[str, CameraModel]:
    cameras: Dict[str, CameraModel] = {}
    for stream_id, cam_params in j.items():
        cameras[stream_id] = from_camera_json(cam_params)
    return cameras


def decode_hand_crop_params(
    j, crop_size: int
) -> Dict[HandSide, Dict[str, PinholePlaneCameraModel]]:
    crop_cameras: Dict[HandSide, Dict[str, PinholePlaneCameraModel]] = defaultdict(dict)

    for k, hand_side in zip(("left", "right"), (HandSide.LEFT, HandSide.RIGHT)):
        if k not in j or j[k] is None:
            continue

        for stream_id, p in j[k].items():
            fov = p["crop_camera_fov"]
            f = crop_size / 2 / np.tan(fov / 2)
            c = (crop_size - 1) / 2

            T_world_from_eye = p["T_world_from_crop_camera"]

            crop_cameras[hand_side][stream_id] = PinholePlaneCameraModel(
                width=crop_size,
                height=crop_size,
                f=(f, f),
                c=(c, c),
                distort_coeffs=[],
                T_world_from_eye=math_utils.quat_trans_to_matrix(
                    *T_world_from_eye["quaternion_wxyz"],
                    *T_world_from_eye["translation_xyz"],
                ),
            )

    return crop_cameras


def decode_hand_pose(j) -> Dict[HandSide, HandPoseCollection]:
    hand_poses: Dict[HandSide, HandPoseCollection] = {}

    for hand_side in (HandSide.LEFT, HandSide.RIGHT):
        k = hand_side.value
        if k not in j or j[k] is None:
            continue

        umetrack_hand_pose = None
        mano_hand_pose = None

        if "umetrack_pose" in j[k]:
            T_world_from_wrist = j[k]["umetrack_pose"]["T_world_from_wrist"]
            umetrack_hand_pose = UmeTrackHandPose(
                hand_side=torch.tensor(0 if hand_side == HandSide.LEFT else 1),
                joint_angles=torch.tensor(j[k]["umetrack_pose"]["joint_angles"]),
                wrist_xform=torch.from_numpy(
                    math_utils.quat_trans_to_matrix(
                        *T_world_from_wrist["quaternion_wxyz"],
                        *T_world_from_wrist["translation_xyz"],
                    )
                ),
            )
            # only look for mano pose if umetrack pose is available
            if "mano_pose" in j[k]:
                # It's possible that MANO registration could fail at some frames
                mano_hand_pose = MANOHandPose(
                    hand_side=torch.tensor(0 if hand_side == HandSide.LEFT else 1),
                    mano_theta=torch.tensor(j[k]["mano_pose"]["thetas"]),
                    wrist_xform=torch.tensor(j[k]["mano_pose"]["wrist_xform"]),
                )

        hand_poses[hand_side] = HandPoseCollection(
            hand_side=hand_side,
            mano=mano_hand_pose,
            umetrack=umetrack_hand_pose,
        )

    return hand_poses


def warp_image(
    src_camera: CameraModel,
    dst_camera: PinholePlaneCameraModel,
    src_image: npt.NDArray,
    interpolation: int = cv2.INTER_LINEAR,
    depth_check: bool = True,
) -> npt.NDArray:
    """
    Warp an image from the source camera to the destination camera.

    Parameters
    ----------
    src_camera :
        Source camera model
    dst_camera :
        Destination camera model
    src_image :
        Source image
    interpolation :
        Interpolation method
    depth_check :
        If True, mask out points with negative z coordinates
    """

    W, H = dst_camera.width, dst_camera.height
    px, py = np.meshgrid(np.arange(W), np.arange(H))
    dst_win_pts = np.column_stack((px.flatten(), py.flatten()))

    dst_eye_pts = dst_camera.window_to_eye(dst_win_pts)
    world_pts = dst_camera.eye_to_world(dst_eye_pts)
    src_eye_pts = src_camera.world_to_eye(world_pts)
    src_win_pts = src_camera.eye_to_window(src_eye_pts)

    # Mask out points with negative z coordinates
    if depth_check:
        mask = src_eye_pts[:, 2] < 0
        src_win_pts[mask] = -1

    src_win_pts = src_win_pts.astype(np.float32)

    map_x = src_win_pts[:, 0].reshape((H, W))
    map_y = src_win_pts[:, 1].reshape((H, W))

    return cv2.remap(src_image, map_x, map_y, interpolation)


def make_hand_crops(
    sample: HandData,
    all_crop_cameras: Dict[HandSide, Dict[str, PinholePlaneCameraModel]],
) -> List[HandCropData]:
    hand_crop_data = []
    for hand_side, crop_cameras in all_crop_cameras.items():
        stream_ids_sel = [
            stream_id for stream_id in crop_cameras if stream_id in sample.images
        ]

        images = {}
        cameras = {}
        for stream_id in stream_ids_sel:
            images[stream_id] = warp_image(
                sample.cameras[stream_id],
                crop_cameras[stream_id],
                np.array(sample.images[stream_id]),
            )
            cameras[stream_id] = crop_cameras[stream_id]

        hand_crop_data.append(
            HandCropData(
                url=sample.url,
                frame_id=sample.frame_id,
                images=images,
                cameras=cameras,
                hand_pose=(
                    sample.hand_poses[hand_side]
                    if sample.hand_poses is not None
                    else None
                ),
                hand_shape=sample.hand_shape,
            )
        )

    return hand_crop_data


class SampleDecoder:
    def __init__(
        self,
        load_monochrome: bool,
        load_rgb: bool,
        output_crops: bool,
        crop_size: int,
    ):
        self.load_monochrome = load_monochrome
        self.load_rgb = load_rgb
        self.output_crops = output_crops
        self.crop_size = crop_size
        self.shape_params = {}

    def _get_shape_params(self, url: str) -> Dict[HandSide, npt.NDArray]:
        if url not in self.shape_params:
            tar = tarfile.open(url)
            HAND_SHAPE_FILE = "__hand_shapes.json__"
            if HAND_SHAPE_FILE in tar.getnames():
                # pyre-ignore
                shape_params_dict = json.load(tar.extractfile(HAND_SHAPE_FILE))
                self.shape_params[url] = HandShapeCollection(
                    mano_beta=torch.tensor(shape_params_dict["mano"]),
                    umetrack=from_umetrack_hand_model_json(
                        shape_params_dict["umetrack"]
                    ),
                )
            else:
                self.shape_params[url] = None
            tar.close()

        return self.shape_params[url]

    def __call__(self, sample):
        url = sample["__url__"]
        frame_id = int(sample["__key__"])

        cameras = decode_cam_params(json.loads(sample["cameras.json"]))
        if "hands.json" in sample:
            j = json.loads(sample["hands.json"])
            hand_poses = decode_hand_pose(j)
        else:
            hand_poses = None

        images = {}
        for stream_id, camera in cameras.items():
            is_rgb = "rgb" in camera.label
            if is_rgb and not self.load_rgb:
                continue
            if not is_rgb and not self.load_monochrome:
                continue

            img = PIL.Image.open(io.BytesIO(sample[f"image_{stream_id}.jpg"]))
            img.load()
            img = img.convert("RGB" if is_rgb else "L")
            images[stream_id] = np.asarray(img)

        hand_data = HandData(
            frame_id=frame_id,
            url=url,
            images=images,
            cameras=cameras,
            hand_poses=hand_poses,
            hand_shape=self._get_shape_params(url),
        )
        if not self.output_crops:
            return hand_data

        crop_cameras = decode_hand_crop_params(
            json.loads(sample["hand_crops.json"]), self.crop_size
        )
        return make_hand_crops(hand_data, crop_cameras)


def build_hand_dataset(
    root: str,
    sequence_names: List[str],
    load_monochrome: bool = True,
    load_rgb: bool = False,
    output_crops: bool = False,
    crop_size: int = 128,
):
    """Function to build a hand dataset

    Args:
        root (str): Path to the directory that contains the tar files
        sequence_names (List[str]): List of sequence names without the ".tar"
            extension.
        load_monochrome (bool, optional): If true, load monochrome images.
            Defaults to True.
        load_rgb (bool, optional): If true, load RGB images. Defaults to False.
        output_crops (bool, optional): If true, the dataset generates single
            hand crops. Defaults to False.
        crop_size (int, optional): Controls the size of the hand crop when
            output_crops=True. Defaults to 128.

    Returns:
        A webdataset that can be iterated using iter(dataset)
    """
    decoder = SampleDecoder(
        load_monochrome=load_monochrome,
        load_rgb=load_rgb,
        output_crops=output_crops,
        crop_size=crop_size,
    )

    dataset = wds.WebDataset(
        [os.path.join(root, f"{s}.tar") for s in sequence_names],
    ).map(decoder)

    return dataset
