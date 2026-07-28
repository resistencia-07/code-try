"""
kinematics.py
Biomechanics calculations for gait analysis
"""
from .utils import calculate_angle, midpoint, calculate_line_angle

# -----------------------------
# MediaPipe Landmark IDs
# -----------------------------

NOSE = 0

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_ELBOW = 13
RIGHT_ELBOW = 14

LEFT_WRIST = 15
RIGHT_WRIST = 16

LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_KNEE = 25
RIGHT_KNEE = 26

LEFT_ANKLE = 27
RIGHT_ANKLE = 28

LEFT_HEEL = 29
RIGHT_HEEL = 30

LEFT_FOOT = 31
RIGHT_FOOT = 32


# ---------------------------------------
# Get Point
# ---------------------------------------

def get_point(points, index):

    if index >= len(points):
        return None

    return points[index]


# ---------------------------------------
# Main Kinematics
# ---------------------------------------

def analyze_pose(points):

    data = {}

    head = get_point(points, NOSE)

    l_shoulder = get_point(points, LEFT_SHOULDER)
    r_shoulder = get_point(points, RIGHT_SHOULDER)

    l_elbow = get_point(points, LEFT_ELBOW)
    r_elbow = get_point(points, RIGHT_ELBOW)

    l_wrist = get_point(points, LEFT_WRIST)
    r_wrist = get_point(points, RIGHT_WRIST)

    l_hip = get_point(points, LEFT_HIP)
    r_hip = get_point(points, RIGHT_HIP)

    l_knee = get_point(points, LEFT_KNEE)
    r_knee = get_point(points, RIGHT_KNEE)

    l_ankle = get_point(points, LEFT_ANKLE)
    r_ankle = get_point(points, RIGHT_ANKLE)

    l_heel = get_point(points, LEFT_HEEL)
    r_heel = get_point(points, RIGHT_HEEL)

    l_foot = get_point(points, LEFT_FOOT)
    r_foot = get_point(points, RIGHT_FOOT)

    shoulder_center = midpoint(l_shoulder, r_shoulder)
    hip_center = midpoint(l_hip, r_hip)

    # -----------------------------
    # Angles
    # -----------------------------

    data["left_elbow"] = calculate_angle(
        l_shoulder,
        l_elbow,
        l_wrist
    )

    data["right_elbow"] = calculate_angle(
        r_shoulder,
        r_elbow,
        r_wrist
    )

    data["left_knee"] = calculate_angle(
        l_hip,
        l_knee,
        l_ankle
    )

    data["right_knee"] = calculate_angle(
        r_hip,
        r_knee,
        r_ankle
    )

    data["left_hip"] = calculate_angle(
        l_shoulder,
        l_hip,
        l_knee
    )

    data["right_hip"] = calculate_angle(
        r_shoulder,
        r_hip,
        r_knee
    )

    data["left_ankle"] = calculate_angle(
        l_knee,
        l_ankle,
        l_foot
    )

    data["right_ankle"] = calculate_angle(
        r_knee,
        r_ankle,
        r_foot
    )

    # -----------------------------
    # Body Alignment
    # -----------------------------

    data["shoulder_alignment"] = (
        calculate_line_angle(
            l_shoulder,
            r_shoulder
        )
        if l_shoulder and r_shoulder
        else None
    )

    data["hip_alignment"] = (
        calculate_line_angle(
            l_hip,
            r_hip
        )
        if l_hip and r_hip
        else None
    )

    data["spine_inclination"] = (
        calculate_line_angle(
            shoulder_center,
            hip_center
        )
        if shoulder_center and hip_center
        else None
    )

    data["neck_angle"] = (
        calculate_line_angle(
            head,
            shoulder_center
        )
        if head and shoulder_center
        else None
    )

    # -----------------------------
    # XY Coordinates
    # -----------------------------

    data["coordinates"] = {

        "head": head,

        "left_shoulder": l_shoulder,
        "right_shoulder": r_shoulder,

        "left_elbow": l_elbow,
        "right_elbow": r_elbow,

        "left_wrist": l_wrist,
        "right_wrist": r_wrist,

        "left_hip": l_hip,
        "right_hip": r_hip,

        "left_knee": l_knee,
        "right_knee": r_knee,

        "left_ankle": l_ankle,
        "right_ankle": r_ankle,

        "left_heel": l_heel,
        "right_heel": r_heel,

        "left_foot": l_foot,
        "right_foot": r_foot

    }

    return data