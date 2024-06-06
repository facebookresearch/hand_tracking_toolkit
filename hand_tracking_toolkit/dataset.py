# Copyright (c) Meta Platforms, Inc. and affiliates.

import dataclasses
import enum
import io
import json
import os.path
import tarfile
from collections import defaultdict
from typing import Dict, List, Optional

import cv2

import numpy as np
import PIL.Image
import webdataset as wds  # @manual
from scipy.spatial.transform import Rotation

from .camera import CameraModel, from_json, PinholePlaneCameraModel

TARGET_CROP_SIZE = 128


class HandSide(enum.Enum):
    LEFT = enum.auto()
    RIGHT = enum.auto()


@dataclasses.dataclass
class HandPose:
    hand_side: HandSide
    mano_theta: np.ndarray
    wrist_xform: np.ndarray


@dataclasses.dataclass
class HandData:
    url: str
    frame_id: int
    images: Dict[str, np.ndarray]
    cameras: Dict[str, CameraModel]
    hand_poses: Optional[Dict[HandSide, HandPose]] = None
    mano_betas: Optional[Dict[HandSide, np.ndarray]] = None


@dataclasses.dataclass
class HandCropData:
    url: str
    frame_id: int
    images: Dict[str, np.ndarray]
    cameras: Dict[str, PinholePlaneCameraModel]
    hand_pose: Optional[HandPose] = None
    mano_beta: Optional[np.ndarray] = None


def decode_cam_params(j) -> Dict[str, CameraModel]:
    cameras: Dict[str, CameraModel] = {}
    for stream_id, cam_params in j.items():
        cameras[stream_id] = from_json(cam_params)
    return cameras


def decode_hand_crop_params(
    j, crop_size: int
) -> Dict[HandSide, Dict[str, PinholePlaneCameraModel]]:
    crop_cameras: Dict[HandSide, Dict[str, PinholePlaneCameraModel]] = defaultdict(dict)

    for k, hand_side in zip(("left", "right"), (HandSide.LEFT, HandSide.RIGHT)):
        for stream_id, p in j[k].items():
            fov = p["crop_camera_fov"]
            f = crop_size / 2 / np.tan(fov / 2)
            c = (crop_size - 1) / 2

            T_world_from_camera = np.eye(4, dtype=np.float32)
            T_world_from_camera[:3, :3] = Rotation.from_quat(
                np.array(p["T_world_from_crop_camera"]["quaternion_wxyz"])
            ).as_matrix()
            T_world_from_camera[:3, 3] = np.array(
                p["T_world_from_crop_camera"]["translation_xyz"]
            )

            crop_cameras[hand_side][stream_id] = PinholePlaneCameraModel(
                width=crop_size,
                height=crop_size,
                f=(f, f),
                c=(c, c),
                distort_coeffs=[],
                T_world_from_eye=T_world_from_camera,
            )

    return crop_cameras


def decode_hand_pose(j) -> Optional[Dict[HandSide, HandPose]]:
    if j is None:
        return None

    hand_poses: Dict[HandSide, HandPose] = {}

    for k, hand_side in zip(("left", "right"), (HandSide.LEFT, HandSide.RIGHT)):
        hand_poses[hand_side] = HandPose(
            hand_side=hand_side,
            mano_theta=np.array(j[k]["thetas"]),
            wrist_xform=np.array(j[k]["wrist_xform"]),
        )

    return hand_poses


def warp_image(
    src_camera: CameraModel,
    dst_camera: PinholePlaneCameraModel,
    src_image: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    depth_check: bool = True,
) -> np.ndarray:
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
                mano_beta=(
                    sample.mano_betas[hand_side]
                    if sample.mano_betas is not None
                    else None
                ),
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

    def _get_shape_params(self, url: str) -> Dict[HandSide, np.ndarray]:
        if url not in self.shape_params:
            tar = tarfile.open(url)
            HAND_SHAPE_FILE = "__hand_shapes.json__"
            if HAND_SHAPE_FILE in tar.getnames():
                # pyre-ignore
                shape_params_dict = json.load(tar.extractfile(HAND_SHAPE_FILE))
                self.shape_params[url] = {
                    hand_side: np.array(shape_params_dict[k])
                    for k, hand_side in zip(
                        ("left", "right"), (HandSide.LEFT, HandSide.RIGHT)
                    )
                }
            else:
                self.shape_params[url] = None
            tar.close()

        return self.shape_params[url]

    def __call__(self, sample):
        url = sample["__url__"]
        frame_id = int(sample["__key__"])

        cameras = decode_cam_params(json.loads(sample["cameras.json"]))
        hand_poses = decode_hand_pose(json.loads(sample["hands.json"]))

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
            mano_betas=self._get_shape_params(url),
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
