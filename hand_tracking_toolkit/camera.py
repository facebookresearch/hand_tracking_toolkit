# pyre-unsafe
"""
Camera Distortion Models

A camera model (as used here) defines a mapping between 3D coordinates
in the world and 2D coordinates in some sensor image, for use in
computer vision.

This generally consists of three steps:

1. Project the 3D coordinates `v` down to 2D coordinates `p` via some
   fixed projection function.

2. Distort the 2D coordinates using fitted coefficients of some general
   function (e.g. a polynomial), producing distorted coordinates `q`.

3. Scale the distorted coordinates by the focal length and offset to the
   focal center, to get window coordinates `w`.

## Coordinate convention

Right-handed coords following the OpenCV convention:
`+X` and `+U` are to the right, `+Y` and `+V` are down. `+Z` is in
front of camera plane, `-Z` is behind.

## Projections

There are many possible projection functions to choose from. Included
here are only those that are already used in ovrsource, plus
stereographic.

All of the camera models are defined so they produce the same results in
neighborhood of the image center: near the point [0,0,1], `u ~= x` and
`v ~= y`.

The interface to projection functions is in :class:`Projection`.

Implementations are :class:`PerspectiveProjection`,
:class:`ArctanProjection`, and :class:`SteregraphicProjection`.


## Distortion functions

The distortion function applies a fitted polynomial to adjust the
`(u,v)` coordinates so they closely match some actual physical sensor.

Distortion coefficients are usually broken into "radial" and "tangential"
terms, named "k1, k2, ..." and "p1, p2, ..." respectively.


## Window coordinates

Finally, the window coordinates are found by scaling the distorted `u,v`
by the focal length `f` (or `fx, fy` for non-square pixels), and adding
the window center `cx, cy`.

Note: There is an important subtlety here in how the continuous 2D range
of coordinates then maps to 2D array indices. Two conventions are
possible:

1. Coordinate (0.0, 0.0) maps to the *corner* of the pixel at
   `image[0,0]`.

2. Coordinate (0.0, 0.0) maps to the *center* of the pixel at
   `image[0,0]`.

   This means that the full range coordinates within a window goes from
   `(-0.5, -0.5)` to `(width-0.5, height-0.5)`. If an image is scaled by
   a factor `s`, coordinates are *not* simply scaled by the same factor:
   `[cx,cy]` maps to `[(cx + 0.5) * s - 0.5, (cy + 0.5) * s - 0.5]`.

The former convention tends to produce simpler code that is easier to
get correct, and therefore is dominant in computer graphics.

Unfortunately the latter convention is dominant in computer vision, and
is what OpenCV does (though this isn't clearly documented), and in the
ovrsource camera models assume, so that is what this library assumes.

Note that this issue remains whether pixels are modeled as "little
squares" or as something more sophisticated like a gaussian blob
centered on the pixel. Since an image is still a rectangular array of
values, it still necessary to decide whether pixel `[i,j]` is centered
at `u=i+0.5, v=j+0.5` or `u=i, v=j`.


## Depth vs. Distance

So far we've only talked about 2D window coordinates. 3D eye coordinates
lose information when projected down, and unprojecting can only yield
a direction vector rather than a 3D point.

But in practice we often need to retain this information, so we also
need a notion of 3D window coordinates. For narrow field of view, the
third component can simply be the raw Z coordinate, unchanged; for
wider field of view it should be the distance to the point instead.
"""

import abc
import math

from typing import Tuple, Type

import numpy as np
from hand_tracking_toolkit import affine
from scipy.spatial.transform import Rotation

from . import camera_distortion as dis


# ---------------------------------------------------------------------
# API Conventions and naming
#
# Points have the xyz or uv components in the last axis, and may have
# arbitrary batch shapes. ([...,2] for 2d and [...,3] for 3d).
#
# v
#    3D xyz position in eye space, usually unit-length.
# p
#    projected uv coordinates: `p = project(v)`
# q
#    distorted uv coordinates: `q = distort(p)`
# w
#    window coordinates: `q = q * f + [cx, cy]`
#
# A trailing underscore (e.g. `p_`, `q_`) should be read as "hat", and
# generally indicates an approximation to another value.
# ---------------------------------------------------------------------


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Transform an array of 3D points with an SE3 transform (rotation and translation).

    *WARNING* this function does not support arbitrary affine transforms that also scale
    the coordinates (i.e., if a 4x4 matrix is provided as input, the last row of the
    matrix must be `[0, 0, 0, 1]`).

    Matrix or points can be batched as long as the batch shapes are broadcastable.

    Args:
        matrix: SE3 transform(s)  [..., 3, 4] or [..., 4, 4]
        points: Array of 3d points [..., 3]

    Returns:
        Transformed points [..., 3]
    """
    return rotate_points(matrix, points) + matrix[..., :3, 3]


def rotate_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Rotates an array of 3D points with an affine transform,
    which is equivalent to transforming an array of 3D rays.

    *WARNING* This ignores the translation in `m`; to transform 3D *points*, use
    `transform_points()` instead.

    Note that we specifically optimize for ndim=2, which is a frequent
    use case, for better performance. See n388920 for the comparison.

    Matrix or points can be batched as long as the batch shapes are broadcastable.

    Args:
        matrix: SE3 transform(s)  [..., 3, 4] or [..., 4, 4]
        points: Array of 3d points or 3d direction vectors [..., 3]

    Returns:
        Rotated points / direction vectors [..., 3]
    """
    if matrix.ndim == 2:
        return (points.reshape(-1, 3) @ matrix[:3, :3].T).reshape(points.shape)
    else:
        return (matrix[..., :3, :3] @ points[..., None]).squeeze(-1)


class CameraModel(dis.CameraProjection, abc.ABC):
    """
    Parameters
    ----------
    width, height : int
        Size of the sensor window

    f : float or tuple(float, float)
        Focal length

    c : tuple(float, float)
        Optical center in window coordinates

    distort_coeffs
        Forward distortion coefficients (eye -> window).

        If this is an instance of DistortionModel, it will be used as-is
        (even if it's a different polynomial than this camera model
        would normally use.) If it's a simple tuple or array, it will
        used as coefficients for `self.distortion_model`.

    T_world_from_eye : np.ndarray
        Camera's position and orientation in world space, represented as
        a 3x4 or 4x4 matrix.

        The matrix be a rigid transform (only rotation and translation).

        You can change a camera's extrinsics after construction by
        assigning to or modifying this matrix.

    serial : string
        Arbitrary string identifying the specific camera.

    Attributes
    ----------
    Most attributes are the same as constructor parameters.

    distortion_model
        Class attribute giving the distortion model for new instances.

    model_fov_limit
        Class attribute giving fov limit as angle from +Z axis

        This isn't a hard limit, but rather a limit beyond which the
        model will probably be unable to do a good job fitting camera
        parameters. Only points inside the limit are used when fitting
        new distortion coefficients.

        It is entirely possible to have max_angle > model_fov_limit,
        but this indicates that the parameters will probably be a
        poor fit to the actual physical camera.
    """

    width: int
    height: int

    f: Tuple[float, float]
    c: Tuple[float, float]

    T_world_from_eye: np.ndarray

    distortion_model: Type[dis.DistortionModel]
    distort: dis.DistortionModel

    def __init__(
        self,
        width,
        height,
        f,
        c,
        distort_coeffs,
        T_world_from_eye=None,
        serial="",
    ):  # pylint: disable=super-init-not-called (see issue 4790 on pylint github)
        self.width = width
        self.height = height
        self.serial = serial

        # f can be either a scalar or (fx,fy) pair. We only fit scalars,
        # but may load (fx, fy) from a stored file.
        self.f = tuple(np.broadcast_to(f, 2))
        self.c = tuple(c)

        if T_world_from_eye is None:
            self.T_world_from_eye = np.eye(4)
        else:
            self.T_world_from_eye = affine.as_4x4(T_world_from_eye, copy=True)
            if (
                np.abs(
                    (self.T_world_from_eye.T @ self.T_world_from_eye)[:3, :3]
                    - np.eye(3)
                ).max()
                >= 1.0e-5
            ):
                info_str = "camera T_world_from_eye must be a rigid transform\n"
                info_str = info_str + "T\n{}\n".format(self.T_world_from_eye.T)
                info_str = info_str + "(T*T_t - I).max()\n{}\n".format(
                    np.abs(
                        (self.T_world_from_eye.T @ self.T_world_from_eye)[:3, :3]
                        - np.eye(3)
                    ).max()
                )
                raise ValueError(info_str)

        if isinstance(distort_coeffs, dis.DistortionModel):
            self.distort = distort_coeffs
        else:
            self.distort = self.distortion_model(*distort_coeffs)

    def __repr__(self):
        return (
            f"{type(self).__name__}({self.width}x{self.height}, f={self.f} c={self.c}"
        )

    def is_fisheye(self):
        """
        Test if the camera model is fisheye, i.e., uses ArctanProjection.
        """
        return isinstance(self, dis.ArctanProjection)

    def copy(
        self,
        T_world_from_eye=None,
        serial=None,
    ):
        """Return a copy of this camera

        Arguments
        ---------
        T_world_from_eye : 4x4 np.ndarray
            Optional new extrinsics for the new camera model.
            Default is to copy this camera's extrinsics.

        serial : str
            Optional replacement serial number.
            Default is to copy this camera's serial number.
        """
        return self.crop(
            0,
            0,
            self.width,
            self.height,
            T_world_from_eye=T_world_from_eye,
            serial=serial,
        )

    def max_angle(self):
        return self._max_angle

    def world_to_window(self, v):
        """Project world space points to 2D window coordinates"""
        return self.eye_to_window(self.world_to_eye(v))

    def world_to_window3(self, v):
        """Project world space points to 3D window coordinates (uv + depth)"""
        return self.eye_to_window3(self.world_to_eye(v))

    def pos(self):
        """Return world position of camera"""
        return self.T_world_from_eye[:3, 3]

    def orient(self):
        """Return world orientation of camera as 3x3 matrix"""
        return self.T_world_from_eye[:3, :3]

    def world_to_eye(self, v):
        """
        Apply camera inverse extrinsics to points `v` to get eye coords
        """
        return rotate_points(self.T_world_from_eye.T, v - self.T_world_from_eye[:3, 3])

    def eye_to_world(self, v):
        """
        Apply camera extrinsics to eye points `v` to get world coords
        """
        return transform_points(self.T_world_from_eye, v)

    def eye_to_window(self, v):
        """Project eye coordinates to 2d window coordinates"""
        p = self.project(v)
        q = self.distort.evaluate(p)
        return q * self.f + self.c

    def eye_to_window3(self, v):
        """Project eye coordinates to 3d window coordinates (uv + depth)"""
        p = self.project3(v)
        q = self.distort.evaluate(p[..., :2])
        p[..., :2] = q * self.f + self.c
        return p

    def w_visible(self, w, *, margin=0):
        """
        Return True if the 2d window coordinate `w` is inside the window

        Can be called with an array, returning a bool array.
        """
        x, y = np.moveaxis(w, -1, 0)
        x0 = -margin - 0.5
        y0 = -margin - 0.5
        x1 = self.width + margin - 0.5
        y1 = self.height + margin - 0.5
        return (x > x0) & (x < x1) & (y >= y0) & (y < y1)

    def crop(
        self,
        src_x,
        src_y,
        target_width,
        target_height,
        scale=1,
        T_world_from_eye=None,
        serial=None,
    ):
        """
        Return intrinsics for a crop of the sensor image.

        No scaling is applied; this just returns the model for a sub-
        array of image data. (Or for a larger array, if (x,y)<=0 and
        (width, height) > (self.width, self.height).

        To do both cropping and scaling, use :meth:`subrect`

        Parameters
        ----------
        x, y, width, height
            Location and size in this camera's window coordinates
        """
        return type(self)(
            target_width,
            target_height,
            np.asarray(self.f) * scale,
            (np.array(self.c) - (src_x, src_y) + 0.5) * scale - 0.5,
            self.distort,
            self.T_world_from_eye if T_world_from_eye is None else T_world_from_eye,
            self.serial if serial is None else serial,
        )

    def subrect(self, transform, width, height):
        """
        Return intrinsics for a scaled crop of the sensor image.

        Parameters
        ----------
        Transform
            a 2x3 affine transform matrix that takes coordinates in the
            old image rect to coordinates in the new image rect, as for
            `cv.WarpAffine`.

            The transform is given in continuous coords, so it must
            follow the "pixel center on integer grid coordinates"
            convention. E.g. resizing an image by 1/N is not just
            scaling by 1/N, but scaling by `1/N` and translating by
            `(1-N)/(2N)`

            Yes, this is confusing. Blame the CV community for failing
            to learn anything from the graphics community.

        width, height : int
            size of target image
        """
        # Currently only support scale and translation.
        #
        # We could support 90 degree rotation by careful manipulation of polynomial
        # coefficients, or arbitrary rotation by applying an 2D affine transform
        # instead of just (f, c) to convert distorted coords to window coords.
        f = np.diag(transform[:-2, :-2])
        c = transform[:-2, 2]
        offdiag = np.diag(np.flip(transform[..., :2, :2], -1))
        if not np.all(offdiag == 0.0):
            raise NotImplementedError("transforms with rotation not yet supported")
        cam = type(self)(
            width,
            height,
            self.f * f,
            self.c * f + c,
            self.distort_coeffs,
            self.T_world_from_eye,
            self.serial,
        )
        return cam


def from_json(js):
    calib = js["calibration"]

    width = calib["image_width"]
    height = calib["image_height"]
    model = calib["projection_model_type"]

    T_world_from_camera = np.eye(4, dtype=np.float32)
    T_world_from_camera[:3, :3] = Rotation.from_quat(
        np.array(js["T_world_from_camera"]["quaternion_wxyz"])
    ).as_matrix()
    T_world_from_camera[:3, 3] = np.array(js["T_world_from_camera"]["translation_xyz"])

    fx, fy, cx, cy = calib["projection_params"][:4]
    coeffs = calib["projection_params"][4:]

    cls = model_by_name[model]

    return cls(
        width,
        height,
        (fx, fy),
        (cx, cy),
        coeffs,
        T_world_from_camera,
    )


# Camera models
# =============


class PinholePlaneCameraModel(dis.PerspectiveProjection, CameraModel):
    distortion_model = dis.NoDistortion
    model_fov_limit = 50 * (math.pi / 180)

    def uv_to_window_matrix(self):
        """Return the 3x3 intrinsics matrix"""
        return np.array(
            [[self.f[0], 0, self.c[0]], [0, self.f[1], self.c[1]], [0, 0, 1]]
        )

    def window_to_eye(self, w):
        p = (np.asarray(w) - self.c) / self.f
        return self.unproject(p)


class OpenCVCameraModel(dis.PerspectiveProjection, CameraModel):
    distortion_model = dis.OpenCVDistortion
    model_fov_limit = 50 * (math.pi / 180)


class OpenCVFullCameraModel(dis.PerspectiveProjection, CameraModel):
    distortion_model = dis.OpenCVFullDistortion
    model_fov_limit = 50 * (math.pi / 180)


class OVR44CameraModel(dis.ArctanProjection, CameraModel):
    distortion_model = dis.OVR44Distortion
    model_fov_limit = math.pi / 2


class OVR62CameraModel(dis.ArctanProjection, CameraModel):
    distortion_model = dis.OVR62Distortion
    model_fov_limit = math.pi / 2


class OVR624CameraModel(dis.ArctanProjection, CameraModel):
    distortion_model = dis.OVR624Distortion
    model_fov_limit = math.pi / 2


model_by_name = {
    "PinholePlane": PinholePlaneCameraModel,
    "OpenCV": OpenCVCameraModel,
    "OculusVisionFishEye": OVR44CameraModel,
    "OculusVisionFishEye62": OVR62CameraModel,
    "OculusVisionFishEye624": OVR624CameraModel,
    "CameraModelType.FISHEYE624": OVR624CameraModel,
    "Rational": OpenCVFullCameraModel,
}

all_models = set(model_by_name.values())
