import dataclasses
import enum
import functools
import json
import os.path
import pathlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

import numpy as np
import webdataset as wds  # @manual

from .camera import CameraModel, from_json, PinholePlaneCameraModel

TARGET_CROP_SIZE = 128


class HandSide(enum.Enum):
    LEFT = enum.auto()
    RIGHT = enum.auto()


@dataclasses.dataclass
class HandPose:
    hand_side: HandSide
    pose: np.ndarray
    global_xform: np.ndarray


@dataclasses.dataclass
class SequenceData:
    sequence_name: str
    frame_ids: np.ndarray
    mano_betas: Dict[HandSide, np.ndarray]
    images: List[Dict[str, np.ndarray]]
    cameras: List[Dict[str, CameraModel]]
    hand_poses: List[List[HandPose]]
    crop_cameras: List[Dict[HandSide, Dict[str, PinholePlaneCameraModel]]]


@dataclasses.dataclass
class SampleData:
    frame_id: int
    sequence_name: str
    images: Dict[str, np.ndarray]
    cameras: Dict[str, CameraModel]
    hand_poses: List[HandPose]
    crop_cameras: Dict[HandSide, Dict[str, PinholePlaneCameraModel]]


@dataclasses.dataclass
class HandCropData:
    sequence_name: str
    image: np.ndarray
    camera: PinholePlaneCameraModel
    hand_pose: HandPose
    mano_beta: np.ndarray


def cam_params_decoder(
    value: Any,
) -> Tuple[Optional[Dict[str, CameraModel]], Dict[HandSide, Dict[str, CameraModel]]]:
    cam_params_dict = json.loads(value)
    cameras: Dict[str, CameraModel] = {}
    for n, cam in enumerate(cam_params_dict["raw_cam_params"]):
        cameras[f"image_{n:04d}"] = from_json(cam)

    crop_cameras: Dict[HandSide, Dict[str, CameraModel]] = defaultdict(dict)
    crop_params = cam_params_dict["crop_params"]
    for k, hand_side in zip(("left", "right"), (HandSide.LEFT, HandSide.RIGHT)):
        for p in crop_params[k]:
            view_idx = p["view_idx"]
            fov = p["fov"]

            f = TARGET_CROP_SIZE / 2 / np.tan(fov / 2)
            c = (TARGET_CROP_SIZE - 1) / 2

            crop_cameras[hand_side][f"image_{view_idx:04d}"] = PinholePlaneCameraModel(
                width=TARGET_CROP_SIZE,
                height=TARGET_CROP_SIZE,
                f=(f, f),
                c=(c, c),
                distort_coeffs=[],
                T_world_from_eye=np.array(p["T_world_from_camera"]).astype(np.float32),
            )

    return cameras, crop_cameras


def pose_params_decoder(value: Any) -> Optional[List[HandPose]]:
    pose_params_dict = json.loads(value)
    hand_poses: List[HandPose] = []
    for k, hand_side in zip(("left", "right"), (HandSide.LEFT, HandSide.RIGHT)):
        hand_pose = HandPose(
            hand_side=hand_side,
            pose=np.array(pose_params_dict[k]["pose"]),
            global_xform=np.array(pose_params_dict[k]["global_xform"]),
        )
        hand_poses.append(hand_pose)
    return hand_poses


def read_sample_dict(
    wds_dict: Dict[str, Any],
    read_images: bool = False,
) -> SampleData:

    frame_id = int(wds_dict["__key__"].split("sample_")[1])
    sequence_name = pathlib.Path(wds_dict["__url__"]).name.split(".tar")[0]
    if read_images:
        image_names = {
            k.split(".")[0] for k in wds_dict.keys() if k.startswith("image")
        }
        images = {k: wds_dict[k + ".png"] for k in image_names}
    else:
        images = {}
    cameras, crop_cameras = wds_dict["cam_params.json"]
    hand_poses = wds_dict["pose_params.json"]

    return SampleData(
        frame_id=frame_id,
        sequence_name=sequence_name,
        images=images,
        cameras=cameras,
        hand_poses=hand_poses,
        crop_cameras=crop_cameras,
    )


def load_sequence_data(
    tar_path: Path,
    shape_params_path: Path,
    load_images: bool = False,
) -> SequenceData:
    decoders = [
        wds.handle_extension("cam_params.json", cam_params_decoder),
        wds.handle_extension("pose_params.json", pose_params_decoder),
    ]
    if load_images:
        decoders += ["pil"]
    _read_sample_dict = functools.partial(read_sample_dict, read_images=load_images)

    dataset = wds.WebDataset(str(tar_path)).decode(*decoders).map(_read_sample_dict)
    frame_ids, images, cameras, hand_poses, crop_cameras = [], [], [], [], []
    sequence_name: Optional[str] = None
    for sample in iter(dataset):
        sequence_name = sample.sequence_name
        frame_ids.append(sample.frame_id)
        cameras.append(sample.cameras)
        images.append(sample.images)
        hand_poses.append(sample.hand_poses)
        crop_cameras.append(sample.crop_cameras)
    assert sequence_name is not None
    mano_betas = json.loads(Path(shape_params_path).read_text())

    return SequenceData(
        sequence_name=sequence_name,
        frame_ids=np.array(frame_ids),
        mano_betas={
            HandSide.LEFT: np.array(mano_betas["left"]),
            HandSide.RIGHT: np.array(mano_betas["right"]),
        },
        images=images,
        cameras=cameras,
        hand_poses=hand_poses,
        crop_cameras=crop_cameras,
    )


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


class HandCropDataset(wds.WebDataset):
    def __init__(self, root: str, sequence_names: List[str], *args, **kwargs):
        super().__init__(
            [os.path.join(root, f"{s}.tar") for s in sequence_names],
            *args,
            **kwargs,
        )

        self.root = root
        self.shape_params = {}

    def _get_shape_params(self, sequence_name: str) -> Dict[HandSide, np.ndarray]:
        if sequence_name not in self.shape_params:
            with open(
                os.path.join(self.root, f"{sequence_name}_shape_params.json"), "rb"
            ) as fp:
                shape_params_dict = json.load(fp)

                self.shape_params[sequence_name] = {
                    hand_side: shape_params_dict[k]
                    for k, hand_side in zip(
                        ("left", "right"), (HandSide.LEFT, HandSide.RIGHT)
                    )
                }

        return self.shape_params[sequence_name]

    def __iter__(self):
        for sample in super().__iter__():
            hand_poses = {p.hand_side: p for p in sample.hand_poses}

            for hand_side, crop_cameras in sample.crop_cameras.items():
                for k in crop_cameras:
                    image = warp_image(
                        sample.cameras[k], crop_cameras[k], np.array(sample.images[k])
                    )

                    yield HandCropData(
                        sequence_name=sample.sequence_name,
                        image=image,
                        camera=crop_cameras[k],
                        hand_pose=hand_poses[hand_side],
                        mano_beta=self._get_shape_params(sample.sequence_name)[
                            hand_side
                        ],
                    )


def build_hand_crop_dataset(root: str, sequence_names: List[str]) -> HandCropDataset:
    decoders = [
        wds.handle_extension("cam_params.json", cam_params_decoder),
        wds.handle_extension("pose_params.json", pose_params_decoder),
        "pil",
    ]
    _read_sample_dict = functools.partial(read_sample_dict, read_images=True)

    # pyre-ignore
    return (
        HandCropDataset(root, sequence_names).decode(*decoders).map(_read_sample_dict)
    )
