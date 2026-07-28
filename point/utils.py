"""
utils.py
Common utility functions for gait analysis
"""

import math


def calculate_angle(a, b, c):
    """
    Calculate angle ABC in degrees.
    B is the vertex.
    """

    if a is None or b is None or c is None:
        return None

    ax, ay = a
    bx, by = b
    cx, cy = c

    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)

    dot = ba[0] * bc[0] + ba[1] * bc[1]

    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if mag_ba == 0 or mag_bc == 0:
        return None

    cos_angle = dot / (mag_ba * mag_bc)

    cos_angle = max(-1.0, min(1.0, cos_angle))

    angle = math.degrees(math.acos(cos_angle))

    return round(angle, 2)


def midpoint(p1, p2):
    """
    Return midpoint between two points.
    """

    if p1 is None or p2 is None:
        return None

    return (
        int((p1[0] + p2[0]) / 2),
        int((p1[1] + p2[1]) / 2)
    )


def euclidean_distance(p1, p2):
    """
    Distance between two points.
    """

    if p1 is None or p2 is None:
        return 0

    return math.sqrt(
        (p2[0] - p1[0]) ** 2 +
        (p2[1] - p1[1]) ** 2
    )


def pixel_to_normalized(x, y, width, height):
    """
    Convert pixel coordinates to normalized coordinates.
    """

    return (
        round(x / width, 6),
        round(y / height, 6)
    )


def normalized_to_pixel(x, y, width, height):
    """
    Convert normalized coordinates to pixel coordinates.
    """

    return (
        int(x * width),
        int(y * height)
    )


def clamp(value, minimum, maximum):
    """
    Limit value between minimum and maximum.
    """

    return max(minimum, min(value, maximum))


def vector(p1, p2):
    """
    Vector from p1 to p2.
    """

    return (
        p2[0] - p1[0],
        p2[1] - p1[1]
    )


def vector_length(v):
    """
    Length of vector.
    """

    return math.sqrt(v[0] ** 2 + v[1] ** 2)


def normalize_vector(v):
    """
    Normalize vector.
    """

    length = vector_length(v)

    if length == 0:
        return (0, 0)

    return (
        v[0] / length,
        v[1] / length
    )


def calculate_slope(p1, p2):
    """
    Calculate slope between two points.
    """

    if p1 is None or p2 is None:
        return 0

    if p2[0] == p1[0]:
        return float("inf")

    return (p2[1] - p1[1]) / (p2[0] - p1[0])


def calculate_line_angle(p1, p2):
    """
    Angle of line with horizontal axis.
    """

    if p1 is None or p2 is None:
        return 0

    angle = math.degrees(
        math.atan2(
            p2[1] - p1[1],
            p2[0] - p1[0]
        )
    )

    return round(angle, 2)