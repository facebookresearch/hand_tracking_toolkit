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

Going from 2D window coordinates reverses these steps, producing a
direction vector:

1. Scale the window coordinates 'w' by the inverse focal length and
   offset from the focal center, to get distorted coordinates `q`.

2. Undistort the 2D coordinates using fitted coefficients of some
   general function (e.g. a polynomial), producing undistorted
   coordinates `p`.

   Rather than exactly inverting the distortion polynomial, we can just
   fit new parameters for the same the polynomial in the inverse
   direction, and store both sets of parameters with the model.

3. Unproject the 2D coordinates `p` to a unit 3D direction `v` via some
   fixed unprojection function.


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


## File formats

Nimble file format example::

    {
        'DistortionModel': 'OculusVisionFishEye',
        'ImageSizeX': 640,
        'ImageSizeY': 480,
        'ModelViewMatrix': [
            [-0.7446, 0.4606, -0.4830, -14.3485],
            [0.1369, 0.8137, 0.5648, 23.6725],
            [0.6532, 0.3543, -0.6690, -130.5179],
            [0.0, 0.0, 0.0, 1.0]
        ],
        'SerialNo': '0072510f171e0702010000031b0a00010:mono',
        'cx': 311.2256,
        'cy': 234.9109,
        'fx': 189.5322,
        'fy': 189.5141,
        'k1': 0.3050,
        'k2': -0.05414,
        'k3': -0.04977,
        'k4': 0.01936,
        'k5': 0.0,
        'k6': 0.0,
        'p1': -0.0001268,
        'p2': -0.0001938,
        'p3': 0.7929,
        'p4': -0.2831
    }
"""

import abc
import json
import math
import warnings
from typing import Optional, Tuple, Type

import numpy as np
from nimble.pylib.functional import affine
from nimble.pylib.math import transform3d

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

    undistort_coeffs=None
        Inverse distortion coefficients (window -> eye)

        Like `distort_coeffs`, except fitted to go in the opposite
        direction.

        If not given, it will be approximately fitted using some
        generated sample points.

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

    undistortion_coeffs : tuple of float
        Fitted parameters for inverse distortion function. Used in the
        same polynomial as distortion_coeffs, but fitted to go in the
        opposite direction.

        Undistortion is not an exact inverse of distortion in general,
        since a polynomial can only represent an approximate self-
        inverse.

    zmin
        Smallest z coordinate of a visible unit-length eye vector.
        (Unit-length) eye rays with z < zmin are known not to be visible
        without doing any extra work.

        This check is needed because for points far outside the window,
        as the distortion polynomial explodes and incorrectly maps some
        distant points back to coordinates inside the window.

        `zmin = cos(max_angle)`

    max_angle
        Maximum angle from +Z axis of a visible eye vector.

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
    undistort: dis.DistortionModel

    _zmin: Optional[float]
    _max_angle: Optional[float]

    def __init__(
        self,
        width,
        height,
        f,
        c,
        distort_coeffs,
        undistort_coeffs=None,
        T_world_from_eye=None,
        serial="",
        bypass_fit_undistort_coeffs=False,
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

        if isinstance(undistort_coeffs, dis.DistortionModel):
            self.undistort = undistort_coeffs
        elif bypass_fit_undistort_coeffs:
            self.undistort = None
        else:
            if self.distortion_model == dis.NoDistortion:
                # Special case: we can skip fit_undistort_coeffs() which is expensive
                assert (
                    undistort_coeffs is None or undistort_coeffs == []
                ), "Cannot supply undistort_coeffs for NoDistortion"
                self.undistort = dis.NoDistortion()
            else:
                if undistort_coeffs is None:
                    undistort_coeffs = self._fit_undistort_coeffs()
                self.undistort = self.distortion_model(*undistort_coeffs)

        # These are computed only when needed, use the getters zmin() and max_angle()
        self._zmin = None
        self._max_angle = None

    def __repr__(self):
        return (
            f"{type(self).__name__}({self.width}x{self.height}, f={self.f} c={self.c}"
        )

    def is_fisheye(self):
        """
        Test if the camera model is fisheye, i.e., uses ArctanProjection.
        """
        return isinstance(self, dis.ArctanProjection)

    def to_nimble_json(self):
        js = {}
        js["ImageSizeX"] = self.width
        js["ImageSizeY"] = self.height
        js["T_WorldFromCamera"] = self.T_world_from_eye.tolist()

        # ModelViewMatrix kept for backward compatibility TODO(pigi): we should be able to remove this
        js["ModelViewMatrix"] = np.linalg.inv(self.T_world_from_eye).tolist()

        if self.__class__ != FisheyeCubicPlusRD4:
            js["fx"], js["fy"] = np.asarray(self.f).tolist()
            js["cx"], js["cy"] = np.asarray(self.c).tolist()

        dist_model = None
        if self.__class__ == PinholePlaneCameraModel:
            dist_model = "PinholePlane"
        else:
            for k, v in model_by_name.items():
                if v == self.__class__:
                    dist_model = k
                    break
            if dist_model is None:
                raise RuntimeError(
                    f"Camera model {self.__class__.__name__} is not supported"
                )
        js["DistortionModel"] = dist_model

        if self.__class__ != FisheyeCubicPlusRD4:
            for name, coeff in zip(self.distortion_model._fields, self.distort):
                js[name] = coeff
        else:
            js["perceptionIntrinsicParameters"] = [*self.f, *self.c, *self.distort[1:]]

        # We don't save inverse distortion parameters in camera json
        return js

    def copy(
        self, T_world_from_eye=None, serial=None, bypass_fit_undistort_coeffs=False
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
            bypass_fit_undistort_coeffs=bypass_fit_undistort_coeffs,
        )

    def compute_zmin(self):
        corners = (
            np.array(
                [[0, 0], [self.width, 0], [0, self.height], [self.width, self.height]]
            )
            - 0.5
        )
        self._zmin = self.window_to_eye(corners)[:, 2].min()
        self._max_angle = np.arccos(self._zmin)

    def zmin(self):
        if self._zmin is None:
            self.compute_zmin()
        return self._zmin

    def max_angle(self):
        if self._max_angle is None:
            self.compute_zmin()
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

    def window_to_world_ray(self, w):
        """
        Unproject 2D window coordinates to world rays.

        Returns a tuple of (origin, direction)
        """
        v = transform3d.rotate_points(self.T_world_from_eye, self.window_to_eye(w))
        o = np.broadcast_to(self.pos(), v.shape)
        return (o, v)

    def window_to_world3(self, w):
        """Unproject 3D window coordinates (uv + depth) to world points"""
        return self.eye_to_world(self.window_to_eye3(w))

    def world_visible(self, v):
        """
        Returns true if the given world-space points are visible in this camera
        """
        return self.eye_visible(self.world_to_eye(v))

    def world_to_eye(self, v):
        """
        Apply camera inverse extrinsics to points `v` to get eye coords
        """
        return transform3d.rotate_points(
            self.T_world_from_eye.T, v - self.T_world_from_eye[:3, 3]
        )

    def eye_to_world(self, v):
        """
        Apply camera extrinsics to eye points `v` to get world coords
        """
        return transform3d.transform_points(self.T_world_from_eye, v)

    def eye_to_window(self, v):
        """Project eye coordinates to 2d window coordinates"""
        p = self.project(v)
        q = self.distort.evaluate(p)
        return q * self.f + self.c

    def window_to_eye(self, w):
        """Unproject 2d window coordinates to unit-length 3D eye coordinates"""
        assert self.undistort is not None
        q = (np.asarray(w) - self.c) / self.f
        if not isinstance(self.undistort, dis.NoDistortion):
            warnings.warn(
                "TODO: solved undistortion parameters cause large errors near the image border"
            )
        p = self.undistort.evaluate(q)
        return self.unproject(p)

    def eye_to_window3(self, v):
        """Project eye coordinates to 3d window coordinates (uv + depth)"""
        p = self.project3(v)
        q = self.distort.evaluate(p[..., :2])
        p[..., :2] = q * self.f + self.c
        return p

    def window_to_eye3(self, w):
        """Unproject 3d window coordinates (uv + depth) to eye coordinates"""
        assert self.undistort is not None
        temp = np.array(w, dtype=np.float64)
        temp[..., :2] -= self.c
        temp[..., :2] /= self.f
        temp[..., :2] = self.undistort.evaluate(temp[..., :2])
        return self.unproject3(temp)

    def visible(self, v):
        """
        Returns true if the given world-space points are visible in this camera
        """
        return self.eye_visible(self.world_to_eye(v))

    def eye_visible(self, v):
        """
        Returns true if the given eye points are visible in this camera
        """
        v = affine.normalized(v)
        w = self.eye_to_window(v)
        return (v[..., 2] >= self.zmin()) & self.w_visible(w)

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
        bypass_fit_undistort_coeffs=False,
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
            self.undistort,
            self.T_world_from_eye if T_world_from_eye is None else T_world_from_eye,
            self.serial if serial is None else serial,
            bypass_fit_undistort_coeffs,
        )

    def subrect(self, transform, width, height, bypass_fit_undistort_coeffs=False):
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
            self.undistort_coeffs,
            self.T_world_from_eye,
            self.serial,
            bypass_fit_undistort_coeffs,
        )
        return cam

    def to_model(self, cls):
        """Convert to a different distortion model.

        Since this relies on the fitted coefficients of the first model,
        it cannot be as accurate as directly fitting the desired model from
        measurements, but it's better than nothing.
        """
        w, h = self.width, self.height
        rays, _ = _gen_rays_in_window(self, spacing=0.05)
        # forward project again, because the unproject coefficients may
        # have been fitted from the forward projection and so have extra
        # error.
        w_pts = self.eye_to_window(rays)
        return cls.fit_from_points(
            rays, w_pts, w, h, self.T_world_from_eye, self.serial
        )

    @classmethod
    def fit_from_points(
        cls, eye_pts, w_pts, width, height, T_world_from_eye=None, serial=""
    ):
        """Fit intrinsics from points with corresponding eye vectors"""
        p_pts = cls.project(eye_pts)

        # to make the solve faster and more stable:
        # first solve for just approximate (f, cx, cy), ignoring distortion
        fc_x0 = (width**2 + height**2) ** 0.5 / 4, width / 2, height / 2
        f, cx, cy = dis.fit_coeffs(
            lambda coeffs, p: p * coeffs[0] + coeffs[1:], p_pts, w_pts, x0=fc_x0
        )

        # ...then solve for full distortion coefficients
        coeffs = dis.fit_coeffs(
            dis.add_f_c_coeffs(cls.distortion_model.evaluate),
            p_pts,
            w_pts,
            x0=(0,) * len(cls.distortion_model._fields) + (f, cx, cy),
        )
        f, cx, cy = coeffs[-3:]
        coeffs = coeffs[:-3]

        q_pts = (w_pts - [cx, cy]) / f
        uncoeffs = dis.fit_coeffs(cls.distortion_model, q_pts, p_pts)

        return cls(
            width, height, f, (cx, cy), coeffs, uncoeffs, T_world_from_eye, serial
        )

    def _fit_undistort_coeffs(self):
        eye_pts, weights = _gen_rays_in_window(self, spacing=0.05)
        p = self.project(eye_pts)
        q = self.distort.evaluate(p)
        return dis.fit_coeffs(self.distortion_model, q, p, w=weights)


def from_nimble_json(js):
    if isinstance(js, str):
        js = json.loads(js)
    js = js.get("Camera", js)

    width = js["ImageSizeX"]
    height = js["ImageSizeY"]
    model = js["DistortionModel"]

    T_World_Camera = (
        np.array(js["T_WorldFromCamera"])
        if "T_WorldFromCamera" in js
        else np.linalg.inv(np.array(js["ModelViewMatrix"]))
    )  # fallback to ModelViewMatrix = T_CameraFromWorld

    params = (
        js["perceptionIntrinsicParameters"] if model == "FisheyeCubicPlusRD4" else []
    )
    (fx, fy, cx, cy) = (
        (js["fx"], js["fy"], js["cx"], js["cy"])
        if model != "FisheyeCubicPlusRD4"
        else params[0:4]
    )

    cls = model_by_name[model]
    distort_params = cls.distortion_model._fields
    coeffs = (
        [js[name] for name in distort_params]
        if model != "FisheyeCubicPlusRD4"
        else [params[0], *params[4:]]
    )

    js_inverse = js.get("inverse", None)
    bypass_fit_undistort_coeffs = js.get("bypass_fit_undistort_coeffs", False)
    if js_inverse is not None:
        uncoeffs = [js_inverse[name] for name in distort_params]
    else:
        uncoeffs = None

    return cls(
        width,
        height,
        (fx, fy),
        (cx, cy),
        coeffs,
        uncoeffs,
        T_World_Camera,
        bypass_fit_undistort_coeffs=bypass_fit_undistort_coeffs,
    )


# Return an array of 3d unit direction vectors, spaced approximately
# 'spacing' radians apart, that all project to within a camera's
# window
def _gen_rays_in_window(cam, *, spacing, margin=0, fov=None):
    rays = []
    weights = []
    if fov is None:
        fov = cam.model_fov_limit
    for strip in _iter_rings_on_sphere(spacing, fov=fov):
        w = cam.eye_to_window(strip)
        valid = cam.w_visible(w, margin=margin)
        if rays and not any(valid):
            break
        vs = strip[valid]
        rays.append(vs)
        weights.append(np.ones(len(vs)) * len(strip) / len(vs))
    return np.concatenate(rays, axis=0), np.concatenate(weights, axis=0)


def _iter_rings_on_sphere(spacing, fov=np.pi):
    # offset phi for each ring to even out spacing
    phi_offset = 0.0
    aspacing = np.pi * (5**0.5 - 1)
    for i in range(int(fov // spacing) + 1, -1, -1):
        theta = fov - i * spacing
        z = np.cos(theta)
        r = np.sin(theta)
        n = np.maximum(1, np.round(r * 2 * np.pi / spacing).astype(int))
        phi_offset = (phi_offset + aspacing) % (2 * np.pi)
        phi = np.linspace(0, 2 * np.pi, n, endpoint=False) - phi_offset
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        yield np.stack([x, y, np.repeat(z, n)], axis=-1)


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


class OpenCVCameraModel(dis.PerspectiveProjection, CameraModel):
    distortion_model = dis.OpenCVDistortion
    model_fov_limit = 50 * (math.pi / 180)


class OpenCVFullCameraModel(dis.PerspectiveProjection, CameraModel):
    distortion_model = dis.OpenCVFullDistortion
    model_fov_limit = 50 * (math.pi / 180)


class FisheyeCubicPlusRD4(dis.PerspectiveProjection, CameraModel):
    distortion_model = dis.FisheyeCubicPlusRD4
    model_fov_limit = 50 * (math.pi / 180)


class OVR44CameraModel(dis.ArctanProjection, CameraModel):
    distortion_model = dis.OVR44Distortion
    model_fov_limit = math.pi / 2


class OVR62CameraModel(dis.ArctanProjection, CameraModel):
    distortion_model = dis.OVR62Distortion
    model_fov_limit = math.pi / 2


class Stereographic44CameraModel(dis.StereographicProjection, CameraModel):
    distortion_model = dis.OVR44Distortion
    model_fov_limit = 135 * (math.pi / 180)


class Stereographic62CameraModel(dis.StereographicProjection, CameraModel):
    distortion_model = dis.OVR62Distortion
    model_fov_limit = 190 * (math.pi / 180)


model_by_name = {
    "PinholePlane": PinholePlaneCameraModel,
    "OpenCV": OpenCVCameraModel,
    "OculusVisionFishEye": OVR44CameraModel,
    "OculusVisionFishEye62": OVR62CameraModel,
    "Rational": OpenCVFullCameraModel,
    "Stereographic62": Stereographic62CameraModel,
    "FisheyeCubicPlusRD4": FisheyeCubicPlusRD4,
}

all_models = set(model_by_name.values())
