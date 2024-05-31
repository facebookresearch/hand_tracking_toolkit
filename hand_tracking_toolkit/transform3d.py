#!/usr/bin/env python3

# pyre-unsafe

"""Helper functions related to 3d transformations"""
import numpy as np


# that work with both.
# It's a TypeVar, not just a Union, so it must be one or the other
# consistently across the whole signature.
AnyTensor = np.ndarray


def transform_points(matrix: AnyTensor, points: AnyTensor) -> AnyTensor:
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


def rotate_points(matrix: AnyTensor, points: AnyTensor) -> AnyTensor:
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
