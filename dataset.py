import dataclasses
import enum
import functools
import json
import pathlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import webdataset as wds  # @manual

from .camera import CameraModel, from_nimble_json


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


@dataclasses.dataclass
class SampleData:
    frame_id: int
    sequence_name: str
    images: Dict[str, np.ndarray]
    cameras: Dict[str, CameraModel]
    hand_poses: List[HandPose]


def cam_params_decoder(value: Any) -> Optional[Dict[str, CameraModel]]:
    cam_params_dict = json.loads(value)
    cameras: Dict[str, CameraModel] = {}
    for n, cam in enumerate(cam_params_dict["raw_cam_params"]):
        cameras[f"image_{n:04d}"] = from_nimble_json(cam)
    return cameras


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
    cameras = wds_dict["cam_params.json"]
    hand_poses = wds_dict["pose_params.json"]

    return SampleData(
        frame_id=frame_id,
        sequence_name=sequence_name,
        images=images,
        cameras=cameras,
        hand_poses=hand_poses,
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
    frame_ids, images, cameras, hand_poses = [], [], [], []
    sequence_name: Optional[str] = None
    for sample in iter(dataset):
        sequence_name = sample.sequence_name
        frame_ids.append(sample.frame_id)
        cameras.append(sample.cameras)
        images.append(sample.images)
        hand_poses.append(sample.hand_poses)
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
    )
