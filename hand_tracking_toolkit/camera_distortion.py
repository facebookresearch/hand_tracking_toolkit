# pyre-unsafe
"""
Various implementations of Distortion and CameraProjection types.
"""

import abc
from typing import NamedTuple, Sequence

import numpy as np
import scipy.optimize
from typing_extensions import Protocol, runtime_checkable

from . import affine


class CameraProjection(Protocol):
    """
    Defines a projection from a 3D `xyz` direction or point to 2D.
    """

    @classmethod
    @abc.abstractmethod
    def project(cls, v):
        """
        Project a 3d vector in eye space down to 2d.
        """
        ...

    @classmethod
    @abc.abstractmethod
    def unproject(cls, p):
        """
        Unproject a 2d point to a unit-length vector in eye space.

        `project(unproject(p)) == p`
        `unproject(project(v)) == v / |v|`
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def project3(v):
        """
        Same as project(), but return a 3d vector.

        The third component represents the distance to the point (either
        from the eye or Z plane, depending on the projection).
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def unproject3(p):
        """
        Inverse of project3()

        `project3(unproject3(p)) == p`
        `unproject3(project3(v)) == v`
        """
        ...


@runtime_checkable
class DistortionModel(Protocol):
    """
    Defines the polynomial used for distortion.

    Implementations are NamedTuples, so coefficients are accessible as
    either named values or a single vector. The order of the
    coefficients in the vector is chosen for compatibility with C++
    implementations.
    """

    @abc.abstractmethod
    def evaluate(self: Sequence[float], p: np.ndarray) -> np.ndarray:
        """
        Evaluate this polynomial at the given points.

        This can also be called as a standalone function with a tuple or
        array of coefficients, without creating a NamedTuple instance.

        Arguments
        ---------
        p: ndarray[..., 2]
            Array of 2D points, of arbitrary batch shape.

        Returns
        -------
        q: ndarray[..., 2]
            Distorted points with same shape as input
        """
        ...


class PerspectiveProjection(CameraProjection):
    """
    Perspective projection: `uv = xy / z`

    This has some nice properties:

    - straight lines map to straight lines
    - math is very nice, especially when working with homogeneous
      coordinates (4x4 matrices).
    - graphics rasterisers generally produce this by default

    However it produces extreme distortion at wide fields of view, and
    cannot represent FOV >= 180°.

    `project_perspective3()` returns the Z component unchanged.
    """

    @staticmethod
    def project(v):
        # map to [x/z, y/z]
        assert v.shape[-1] == 3
        return v[..., :2] / v[..., 2, None]

    @staticmethod
    def unproject(p):
        # map to [u,v,1] and renormalize
        assert p.shape[-1] == 2
        x, y = np.moveaxis(p, -1, 0)
        v = np.stack((x, y, np.ones(shape=x.shape, dtype=x.dtype)), axis=-1)
        v = affine.normalized(v, axis=-1)
        return v

    @staticmethod
    def project3(v):
        # map to [x/z, y/z, z]
        x, y, z = np.moveaxis(v, -1, 0)
        return np.stack([x / z, y / z, z], axis=-1)

    @staticmethod
    def unproject3(p):
        # map to [p*z, v*z, z]
        x, y, z = np.moveaxis(p, -1, 0)
        return np.stack((x * z, y * z, z), axis=-1)


class ArctanProjection(CameraProjection):
    """
    Arctan projection: uv = xy / |xy| * atan2(|xy|, z)

    This is what the Oculus fisheye models use: the length of the 2D
    u vector is the angle in radians between +Z and the target ray
    direction.

    Unlike perspective projection, this can represent a FOV over 180
    degrees. However, distortion increases as the FOV gets wider. Near
    180 degrees, the aspect ratio of a pixel is about 1:1.57, which
    must then be corrected for in the distortion function. (This is why
    the OVRFisheye model needed to move to 6 radial terms: it has more
    distortion to correct for.)

    It's also relatively expensive to compute on since there is a trig
    function involved.
    """

    @staticmethod
    # pyre-fixme[14]: `project` overrides method defined in `CameraProjection`
    #  inconsistently.
    # pyre-fixme[40]: Static method `project` cannot override a non-static method
    #  defined in `CameraProjection`.
    def project(p, eps: float = 2.0**-128):
        assert p.shape[-1] == 3
        x, y, z = np.moveaxis(p, -1, 0)
        r = np.sqrt(x * x + y * y)
        s = np.arctan2(r, z) / np.maximum(r, eps)
        return np.stack((x * s, y * s), axis=-1)

    @staticmethod
    def unproject(uv):
        assert uv.shape[-1] == 2
        u, v = np.moveaxis(uv, -1, 0)
        r = np.sqrt(u * u + v * v)
        c = np.cos(r)
        s = np.sinc(r / np.pi)
        return np.stack([u * s, v * s, c], axis=-1)

    @staticmethod
    # pyre-fixme[14]: `project3` overrides method defined in `CameraProjection`
    #  inconsistently.
    def project3(p, eps: float = 2.0**-128):
        assert p.shape[-1] == 3
        xy = p[..., :2]
        z = p[..., 2]
        r2 = np.linalg.norm(xy, axis=-1)
        r3 = np.linalg.norm(p, axis=-1)
        s = np.arctan2(r2, z) / np.maximum(r2, eps)
        return np.stack([xy[..., 0] * s, xy[..., 1] * s, r3], axis=-1)

    @staticmethod
    def unproject3(uvd):
        assert uvd.shape[-1] == 3
        u, v, d = np.moveaxis(uvd, -1, 0)
        r = np.sqrt(u * u + v * v)
        c = np.cos(r)
        s = np.sinc(r / np.pi)
        return np.moveaxis(np.stack([u * s, v * s, c]) * d, 0, -1)


class StereographicProjection(CameraProjection):
    """
    Stereographic projection: uv = xy / (z + 1)

    This is a conformal mapping, which means it preserves angles
    everywhere, i.e. pixels are square everywhere in the image (though
    possibly of different sizes).

    The entire 360° FOV maps to the full `uv` plane. The 180° forward
    hemisphere maps to the unit disk.

    Since minimial distortion is one of the design goals of most
    physical sensor designs, this generally ends up being fairly close
    fit even before the distortion function is applied. It is also very
    cheap to compute, comparable with simple perspective projection.
    """

    @staticmethod
    # pyre-fixme[14]: `project` overrides method defined in `CameraProjection`
    #  inconsistently.
    # pyre-fixme[40]: Static method `project` cannot override a non-static method
    #  defined in `CameraProjection`.
    def project(p) -> float:
        assert p.shape[-1] == 3
        p = affine.normalized(p)
        return 2.0 * p[..., :2] / (1.0 + p[..., 2, None])

    @staticmethod
    def unproject(uv):
        u, v = np.moveaxis(uv, -1, 0)
        t = (u * u + v * v) * 0.25
        s = t + 1
        return np.stack([u, v, (1 - t)] / s, axis=-1)

    @staticmethod
    # pyre-fixme[14]: `project3` overrides method defined in `CameraProjection`
    #  inconsistently.
    def project3(p, eps: float = 1.0e-9):
        assert p.shape[-1] == 3
        d = np.sqrt((p * p).sum(axis=-1))
        x, y, z = np.moveaxis(p, -1, 0) / np.maximum(d, eps)
        return np.stack([2.0 * x / (1.0 + z), 2.0 * y / (1.0 + z), d], axis=-1)

    @staticmethod
    def unproject3(uvd):
        u, v, d = np.moveaxis(uvd, -1, 0)
        t = (u * u + v * v) * 0.25
        s = d / (t + 1)
        return np.stack([u, v, (1 - t)], axis=-1) * s[..., None]


class NoDistortion(NamedTuple):
    """
    A trivial distortion model that does not distort the incoming rays.
    """

    def evaluate(self, p: np.ndarray) -> np.ndarray:
        return p


class OpenCVDistortion(NamedTuple):
    """
    Default OpenCV distortion model, also cf.known as Brown-Conrady.

    http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

    This model only uses radial distortion up to k3, and tangential
    distortion up to p2.
    """

    k1: float
    k2: float
    k3: float
    p1: float
    p2: float

    def evaluate(self: Sequence[float], p: np.ndarray) -> np.ndarray:
        assert p.shape[-1] == 2
        k1, k2, k3, p1, p2 = self  # pylint: disable=unpacking-non-sequence
        x, y = p[..., 0], p[..., 1]
        x2 = x * x
        y2 = y * y
        xy = x * y
        r2 = x2 + y2
        r4 = r2 * r2
        r6 = r2 * r4
        radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
        x_out = x * radial + 2 * p1 * xy + p2 * (r2 + 2 * x2)
        y_out = y * radial + 2 * p2 * xy + p1 * (r2 + 2 * y2)
        return np.stack((x_out, y_out), axis=-1)


class OpenCVFullDistortion(NamedTuple):
    """
    OpenCV distortion with all 6 parameters
    """

    k1: float
    k2: float
    k3: float
    p1: float
    p2: float
    k4: float
    k5: float
    k6: float

    def evaluate(self: Sequence[float], p: np.ndarray) -> np.ndarray:
        assert p.shape[-1] == 2
        k1, k2, k3, p1, p2, k4, k5, k6 = self  # pylint: disable=unpacking-non-sequence
        x, y = p[..., 0], p[..., 1]
        x2 = x * x
        y2 = y * y
        xy = x * y
        r2 = x2 + y2
        r4 = r2 * r2
        r6 = r2 * r4
        radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
        x_out = x * radial + 2 * p1 * xy + p2 * (r2 + 2 * x2)
        y_out = y * radial + 2 * p2 * xy + p1 * (r2 + 2 * y2)
        return np.stack((x_out, y_out), axis=-1)


class OVR44Distortion(NamedTuple):
    """
    OVRFisheye44 model, with 4 radial and 4 tangential coeffs.
    """

    k1: float
    k2: float
    k3: float
    k4: float
    p1: float
    p2: float
    p3: float
    p4: float

    def evaluate(self: Sequence[float], p: np.ndarray) -> np.ndarray:
        # note: this code looks a little different than some of the C++
        # implementations, because here we're taking `uv` that is
        # already scaled to be in radians, while the C++ code mixes the
        # projection and distortion functions together.
        k1, k2, k3, k4, p1, p2, p3, p4 = self  # pylint: disable=unpacking-non-sequence

        # radial component
        r2 = (p * p).sum(axis=-1, keepdims=True)
        # n388920 shows why we compute r^n using previous terms
        # instead of r^n = r ** n
        r4 = r2 * r2
        r6 = r2 * r4
        r8 = r4 * r4
        radial = 1 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8
        uv = p * radial

        # tangential component
        x, y = uv[..., 0], uv[..., 1]
        x2 = x * x
        y2 = y * y
        xy = x * y
        r2 = x2 + y2
        r4 = r2 * r2
        p3p4term = 1 + p3 * r2 + p4 * r4
        x += (2 * p1 * xy + p2 * (r2 + 2 * x2)) * p3p4term
        y += (2 * p2 * xy + p1 * (r2 + 2 * y2)) * p3p4term
        return np.stack((x, y), axis=-1)


class OVR62Distortion(NamedTuple):
    """
    OVRFisheye62 model, with 6 radial and 2 tangential coeffs.
    """

    k1: float
    k2: float
    k3: float
    k4: float
    p1: float
    p2: float
    k5: float
    k6: float

    def evaluate(self: Sequence[float], p: np.ndarray) -> np.ndarray:
        k1, k2, k3, k4, p1, p2, k5, k6 = self  # pylint: disable=unpacking-non-sequence
        # radial component
        r2 = (p * p).sum(axis=-1, keepdims=True)
        r2 = np.clip(r2, -np.pi**2, np.pi**2)
        # n388920 shows why we compute r^n using previous terms
        # instead of r^n = r ** n
        r4 = r2 * r2
        r6 = r2 * r4
        r8 = r4 * r4
        r10 = r4 * r6
        r12 = r6 * r6
        radial = 1 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8 + k5 * r10 + k6 * r12
        uv = p * radial

        # tangential component
        x, y = uv[..., 0], uv[..., 1]
        x2 = x * x
        y2 = y * y
        xy = x * y
        r2 = x2 + y2
        x += 2 * p2 * xy + p1 * (r2 + 2 * x2)
        y += 2 * p1 * xy + p2 * (r2 + 2 * y2)
        return np.stack((x, y), axis=-1)


class FisheyeCubicPlusRD4(NamedTuple):
    fx: float
    s: float
    k0: float
    k1: float
    k2: float
    k3: float
    k4: float
    k5: float
    p0: float
    p1: float
    p2: float
    p3: float
    p4: float
    p5: float
    p6: float
    p7: float
    p8: float
    p9: float
    p10: float
    p11: float
    p12: float
    s0: float
    s1: float

    def evaluate(self: Sequence[float], p: np.ndarray) -> np.ndarray:
        (
            fx,
            s,
            k0,
            k1,
            k2,
            k3,
            k4,
            k5,
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            p8,
            p9,
            p10,
            p11,
            p12,
            s0,
            s1,
        ) = self

        # p == ab
        abSq = np.power(p, 2)
        rSq = np.sum(abSq, axis=-1)
        r = np.sqrt(rSq)
        th = np.arctan(r)
        thSq = np.power(th, 2)
        k_terms = (
            np.ones(thSq.shape)
            + k0 * thSq
            + k1 * np.power(thSq, 2)
            + k2 * np.power(thSq, 3)
            + k3 * np.power(thSq, 4)
            + k4 * np.power(thSq, 5)
            + k5 * np.power(thSq, 6)
        )
        thR = np.array(th / r)
        thR[np.isnan(thR)] = 1.0  # avoid nan values
        xr, yr = np.moveaxis(p, -1, 0) * k_terms * thR

        cubic_x = (
            p0 * np.power(xr, 2)
            + p2 * np.power(yr, 2)
            + p4 * xr * yr
            + p6 * np.power(xr, 3)
            + p8 * np.power(yr, 3)
            + p10 * np.power(yr, 2) * xr
            + p12 * yr * np.power(xr, 2)
        )
        cubic_y = (
            p1 * np.power(xr, 2)
            + p3 * np.power(yr, 2)
            + p5 * xr * yr
            + p7 * np.power(xr, 3)
            + p9 * np.power(yr, 3)
            + p11 * np.power(yr, 2) * xr
        )
        cubicDistortion = np.stack(np.array((cubic_x, cubic_y)), axis=-1)

        rd4 = np.power((np.power(xr, 2) + np.power(yr, 2)), 2)
        rd4thDistortion = np.stack(np.array((s0 * rd4, s1 * rd4)), axis=-1)

        uv_distorted = np.stack((xr, yr), axis=-1) + cubicDistortion + rd4thDistortion
        # This is a trick to apply the s parameter inside the distortion model instead of
        # in the projection. When fx != 0, uDistorted * fx == uDistorted * fx + vDistorted * s
        if fx != 0:
            uv_distorted[..., 0] += uv_distorted[..., 1] * s / fx
        return uv_distorted


# wrap a distortion poly so it also takes additional "f", "cx", "cy"
# parameters, so we can solve for those in fit_coeffs().
def add_f_c_coeffs(fn):
    fn = getattr(fn, "evaluate", fn)

    def wrapper(coeffs, p):
        f, cx, cy = coeffs[-3:]
        return fn(coeffs[:-3], p) * f + [cx, cy]

    wrapper.__qualname__ = wrapper.__name__ = fn.__name__ + "_fc"
    return wrapper


def fit_coeffs(poly, ps, qs, x0=None, w: int = 1):
    """
    Fit coefficients `k` so `poly(k, ps) == qs`

    Arguments
    ---------
    poly
        Distortional polynomial, callable as `poly(coeffs, ps)` or
        `poly.evaluate(coeffs, ps)`.

    ps
        input points

    qs
        corresponding target points

    x0
        Initial value of coefficients to start the solver. Required
        if `poly._fields` does not exist. Default is all zeros.

    w
        optional vector of per-point weights

    Returns
    -------
    vector of coefficients.
    """
    if x0 is None:
        x0 = [0.0] * len(poly._fields)
    if not x0:
        return x0

    poly_fn = getattr(poly, "evaluate", poly)
    x0 = np.asarray(x0)
    # residual is squared, so need sqrt of weight
    w = np.asarray(w**0.5)

    def residual(coeffs):
        dxy = poly_fn(coeffs, ps) - qs
        # minimize d^4 instead of d^2, to more heavily penalize outliers.
        # We actually want d^inf here but d^4 is the best numpy's solvers
        # can do.
        # d^4 = (x^2 + y^2)^2
        #     = x^4 + y^4 + 2*x^2*y^2
        #     = (x^2)^2 + (y^2)^2 + (sqrt(2)*x*y)^2
        # so r = [x^2, y^2, sqrt(2)*x*y]
        n = len(ps)
        r = np.zeros(n * 3)
        r[: n * 2] = (dxy**2 * w[..., None]).reshape(-1)
        r[n * 2 :] = dxy[:, 0] * dxy[:, 1] * np.sqrt(2.0) * w
        return r

    soln = scipy.optimize.least_squares(
        residual, x0=x0, ftol=1.0e-6, method="trf"
    )  # , xtol=3.0e-7)
    if not soln.success:
        raise ValueError(f"solver failed: {soln.message}")
    return soln.x
