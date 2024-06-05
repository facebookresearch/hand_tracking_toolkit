# A toolkit for egocentric hand tracking research

This repository contains the following tools for hand tracking research:

- APIs for loading data from the UmeTrack and HOT3D datasets
- Computation of metrics used in the Hand Tracking Challenge organised at ECCV
  2024
- Data visualization helpers

## Datasets

The datasets are provided using the
[WebDataset](https://huggingface.co/docs/hub/en/datasets-webdataset) format. The
file structure of each sequence is as follows:

```
├─ train
│  ├─ subject_000_separate_hand_000000.tar
│  ├─ subject_000_hand_hand_000001.tar
│  ├─ ...
│  ├─ subject_049_hand_hand_000099.tar
├─ pose_test
│  ├─ subject_050_separate_hand_000100.tar
│  ├─ subject_050_hand_hand_000101.tar
│  ├─ ...
│  ├─ subject_079_hand_hand_000199.tar
├─ shape_test
│  ├─ subject_080_separate_hand_000300.tar
│  ├─ subject_080_hand_hand_000301.tar
│  ├─ ...
│  ├─ subject_099_hand_hand_000399.tar
```

Each tar file is assumed to contain 2~4 synchronized monochrome streams plus an
optional RGB stream. The image files are suffixed with the image stream ID which
can be used to look up the camera parameters from `*.cameras.json`. For example,
the file could have the following structure:

```
├─ subject_000_separate_hand_000000.tar
│  ├─ 000000.image_1201-1.jpg
│  ├─ 000000.image_1201-2.jpg
│  ├─ 000000.cameras.json
│  ├─ 000000.hands.json
│  ├─ 000000.hand_crops.json
│  ├─ ...
│  ├─ __hand_shapes.json__
```

Each sequence contains only one subject thus the MANO shape parameters are
shared by all frames in the same sequence and are saved in
`__hand_shapes.json__`. Hand pose annotations are provided as MANO pose
parameters for each frame in `*.hands.json`.

<img src="assets/perspective_crop.png" alt="drawing" width="400"/>

Following the
[UmeTrack](https://research.facebook.com/publications/umetrack-unified-multi-view-end-to-end-hand-tracking-for-vr/)
paper, we also provide the perspective crop camera parameters which can be used
to produce hand crop images.The figure above illustrates the usage. The
perspective crop camera parameters can be found in `*.hand_crops.json`,

### Data splits

Each dataset has three splits:

1. Training: all annotations are available
2. Pose estimation test: `*.hands.json` files are removed
3. Shape estimation test: `*.hands.json` files and `__hand_shapes.json__` are
   removed.

## Getting Started

### Downloading the datasts

**Coming soon!!**

### Installing the tooklit

Run the following command to install the toolkit:

```
pip install git+https://github.com/facebookresearch/hand_tracking_toolkit
```

### Downloading MANO assets

Go to [https://smpl-x.is.tue.mpg.de/](https://smpl-x.is.tue.mpg.de/) to download
the MANO pickle files (`MANO_LEFT.pkl` and `MANO_RIGHT.pkl`).

### Building a simple hand crop dataset and visualizing the sample

You can build a hand crop image dataset using the following code snippet:

```python
from hand_tracking_toolkit.dataset import build_hand_crop_dataset
from hand_tracking_toolkit.visualization import visualize_hand_crop_data
from hand_tracking_tooklit.mano_layer import MANOHandModel

mano_layer = MANOHandModel("/path/to/mano/pkl/files")

root = (
    "/path/to/dataset/root"
)
sequence_names = [
    "sequence0000",
    "sequence0001"
]
dataset = build_hand_dataset(
    root,
    sequence_names,
    load_monochrome=True,
    load_rgb=True,
    output_crops=True,
    crop_size=128,
)

for i, sample in enumerate(dataset):
    img = visualize_hand_crop_data(
        sample,
        mano_layer,
        visualize_mesh=True,
        visualize_keypoints=False
    )
    # use your favorite library to visualize the image
```

## Evaluation

Evaluation is performed using the
[Multiview Egocentric Hand Tracking Challenge]() website. Following the data
splits, the challenge has two tracks: 1) pose estimation track and 2) shape
estimation track. For each track, a submission tar file is expected.
`submissions.py` provides utility functions to generate the submission files.
The evaluation server stores the test annotation files which are compared with
the submission files to calculate the metrics.

### Submission formats

The submission file for the **pose estimation track** looks like this:

```
[
    {
        "sequence_name": "sequence0000",
        "frame_id": 0,
        "pose": [...], // mano pose parameters
        "wrist_xform": [...], // global wrist transformation
        "hand_side": "left"
    },
    {
        "sequence_name": "sequence0000",
        "frame_id": 1,
        "pose": [...],
        "wrist_xform": [...],
        "hand_side": "right"
    },
]
```

The submission file for the **shape estimation track** looks like this:

```
[
    {
        "sequence_name": "sequence0000",
        "mano_beta": [...], // mano shape parameters
        "hand_side": "left"
    },
    {
        "sequence_name": "sequence0001",
        "mano_beta": [...], // mano shape parameters
        "hand_side": "right"
    },
]
```

### How to submit

The evaluation server expects a single submission tar file for each track. The
submission tar file should contain the results of all datasets you want to
evaluate on.

Example for the pose estimation track:

```
├─ pose_submission.tar
│  ├─ result_pose_umetrack.json
│  ├─ result_pose_hot3d.json
```

Example for the shape estimation track:

```
├─ shape_submission.tar
│  ├─ result_shape_umetrack.json
│  ├─ result_shape_hot3d.json
```

**NOTE**: It's okay to only include the result for UmeTrack or HOT3D. The
evaluation server will automatically skip the missing files.

### Local validation

We do not provide a validation set. You can create a validation set by selecting
a subset from the training set. To prepare the test annotation files (same
format as the submission files):

```sh
$OUTPUT_DIR = /path/to/output/dir
python3 scripts/write_test_annotations_files.py --input-dir /path/to/your/umetrack/subset --output-dir $OUTPUT_DIR

# Rename the files by adding the dataset name as the suffix
cd $OUTPUT_DIR
mv test_pose_annotations.json test_pose_annotations_umetrack.json
mv test_shape_annotations.json test_shape_annotations_umetrack.json

# Pack everything into a tar file
tar -cf gt.tar /path/to/mano_pkl_files test_pose_annotations_umetrack.json test_shape_annotations_umetrack.json
```

After obtaining the submission files, pack the files similarly:

```sh
tar -cf pose_submission.tar result_pose_umetrack.json
tar -cf shape_submission.tar result_shape_umetrack.json
```

the metrics can be obtained by running the evaluation script (the same script
that runs on the challenge server):

```sh
# pose estimation evaluation
python3 scripts/run_evaluation.par --test-annotation-file ~/eval_files/gt.tar --user-submission-file pose_submission.tar --phase-codename pose_estimation
# shape estimation evaluation
python3 scripts/run_evaluation.par --test-annotation-file ~/eval_files/gt.tar --user-submission-file shape_submission.tar --phase-codename shape_estimation
```

## How to contribute

We welcome contributions! Go to [CONTRIBUTING](CONTRIBUTING.md) and our
[CODE OF CONDUCT](CODE_OF_CONDUCT.md) for how to get started.

## Citation

If you use this dataset for publications, please cite this work:

```
@inproceedings{han2022umetrack,
  title={UmeTrack: Unified multi-view end-to-end hand tracking for VR},
  author={Han, Shangchen and Wu, Po-chen and Zhang, Yubo and Liu, Beibei and Zhang, Linguang and Wang, Zheng and Si, Weiguang and Zhang, Peizhao and Cai, Yujun and Hodan, Tomas and others},
  booktitle={SIGGRAPH Asia 2022 Conference Papers},
  pages={1--9},
  year={2022}
}
```

## License

This research toolkit is released under the Apache 2.0 license, as found in the
[LICENSE](LICENSE.md) file.
