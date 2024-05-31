# pyre-unsafe
"""
Helpers for affine transformation matrix operations.
"""

from typing import TypeVar, Union

import numpy as np


# torch or numpy tensor, for defining function signatures
# that work with both.
# It's a TypeVar, not just a Union, so it must be one or the other
# consistently across the whole signature.
AnyTensor = TypeVar("AnyTensor", np.ndarray, "torch.Tensor")  # noqa: F821 (avoid import torch only to define a type hint)


def mtranspose(m: AnyTensor) -> AnyTensor:
    """
    Matrix transpose that works correctly for batches.

    Matrix mul in torch and numpy assumes the last two dims are the
    matrix shape and batches across other dims, but their transpose
    functions assume differently and interpret arguments differently.
    """
    return np.swapaxes(m, -1, -2)


def rotate2(a: Union[float, np.ndarray], dtype=None) -> np.ndarray:
    """
    return 2D rotation as 3x3 homogeneous matrix or matrices

    Input can be an angle or array of angles. Output will then be a same-shaped
    array of matrices.
    """
    c, s = np.cos(a), np.sin(a)
    out = np.zeros(c.shape + (3, 3), dtype=dtype)
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 1] = c
    out[..., 1, 0] = s
    out[..., 2, 2] = 1
    return out


def scale2(s: Union[float, np.ndarray]) -> np.ndarray:
    """
    Return a 2D uniform scale as a 3x3 homogeneous matrix

    `s` may be scalar for uniform scale or [sx,sy] for nonuniform.
    """
    s = np.broadcast_to(s, 2)
    return np.array([[s[0], 0, 0], [0, s[1], 0], [0, 0, 1]], dtype=s.dtype)


def scale3(s: Union[float, np.ndarray]) -> np.ndarray:
    """return a 3D scale as a 4x4 homogeneous matrix

    `s` may be scalar for uniform scale or [sx,sy,sz] for nonuniform.
    """
    s = np.broadcast_to(s, (3,))
    return np.array(
        [[s[0], 0, 0, 0], [0, s[1], 0, 0], [0, 0, s[2], 0], [0, 0, 0, 1]], dtype=s.dtype
    )


def translate2(t: np.ndarray) -> np.ndarray:
    """return a 2D translation as a 3x3 homogeneous matrix"""
    t = np.asarray(t)
    out = np.eye(3, dtype=t.dtype)
    out[:2, 2] = t
    return out


def translate3(t):
    """return a 3D translation as a 4x4 homogeneous matrix"""
    out = np.eye(4)
    out[:3, 3] = t
    return out


def transform2(m: AnyTensor, v: AnyTensor) -> AnyTensor:
    """
    Transform an array of 2D points with an affine transform.

    Note that we specifically optimize for ndim=2, which is a frequent
    use case, for better performance. See n388920 for the comparison.

    Parameters
    ----------
    m
        affine transform(s) as 2x3 or 3x3 matrices
    v
        Array of 2D points

    m or v can be batched as long as the batch shapes are broadcastable.
    """
    if m.ndim == 2:
        # use non-batched matrix mul if possible, as it's much faster
        return (v.reshape(-1, 2) @ m[:2, :2].T).reshape(v.shape) + m[:2, 2]
    else:
        return (m[..., :2, :2] @ v[..., None] + m[..., :2, 2, None]).squeeze(-1)


def as_3x3(a: np.ndarray, *, copy: bool = False) -> np.ndarray:
    """
    Append [0,0,1] to convert a 2x3 matrices to a 3x3 homogeneous matrices

    If the matrices are already 3x3 they will be returned unchanged.
    """
    if a.shape[-2:] == (3, 3):
        if copy:
            a = np.array(a)
        return a
    if a.shape[-2:] == (2, 3):
        return np.concatenate(
            (
                a,
                np.broadcast_to(
                    np.array([0, 0, 1], dtype=a.dtype), a.shape[:-2] + (1, 3)
                ),
            ),
            axis=-2,
        )
    raise ValueError("expected 2x3 or 3x3 affine transform")


def as_4x4(a: np.ndarray, *, copy: bool = False) -> np.ndarray:
    """
    Append [0,0,0,1] to convert 3x4 matrices to a 4x4 homogeneous matrices

    If the matrices are already 4x4 they will be returned unchanged.
    """
    if a.shape[-2:] == (4, 4):
        if copy:
            a = np.array(a)
        return a
    if a.shape[-2:] == (3, 4):
        return np.concatenate(
            (
                a,
                np.broadcast_to(
                    np.array([0, 0, 0, 1], dtype=a.dtype), a.shape[:-2] + (1, 4)
                ),
            ),
            axis=-2,
        )
    raise ValueError("expected 3x4 or 4x4 affine transform")


def normalized(v: AnyTensor, axis: int = -1, eps: float = 5.43e-20) -> AnyTensor:
    """
    Return a unit-length copy of vector(s) v

    Parameters
    ----------
    axis : int = -1
        Which axis to normalize on

    eps
        Epsilon to avoid division by zero. Vectors with length below
        eps will not be normalized. The default is 2^-64, which is
        where squared single-precision floats will start to lose
        precision.
    """
    d = np.maximum(eps, (v * v).sum(axis=axis, keepdims=True) ** 0.5)
    return v / d
