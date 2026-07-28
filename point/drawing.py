"""
drawing.py
Draw joints, labels, coordinates and biomechanics values
"""

import cv2

# -----------------------------------
# Colors (BGR)
# -----------------------------------

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
ORANGE = (0, 165, 255)


# -----------------------------------
# Landmark Names
# -----------------------------------

LANDMARKS = {

    0: "Head",

    11: "L Shoulder",
    12: "R Shoulder",

    13: "L Elbow",
    14: "R Elbow",

    15: "L Wrist",
    16: "R Wrist",

    23: "L Hip",
    24: "R Hip",

    25: "L Knee",
    26: "R Knee",

    27: "L Ankle",
    28: "R Ankle",

    29: "L Heel",
    30: "R Heel",

    31: "L Foot",
    32: "R Foot"
}


# -----------------------------------
# Draw Joint Labels
# -----------------------------------

def draw_joint_labels(frame, points):

    for idx, name in LANDMARKS.items():

        if idx >= len(points):
            continue

        pt = points[idx]

        if pt is None:
            continue

        x, y = pt

        cv2.circle(frame, pt, 9, RED, -1,cv2.LINE_AA)

        cv2.putText(
            frame,
            name,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            CYAN,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"({x},{y})",
            (x + 8, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            WHITE,
            1,
            cv2.LINE_AA
        )


# -----------------------------------
# Draw Angles
# -----------------------------------

def draw_angles(frame, analysis):

    x = 20
    y = 30

    display_order = [

        ("neck_angle", "Neck"),

        ("shoulder_alignment", "Shoulder"),

        ("hip_alignment", "Hip Align"),

        ("spine_inclination", "Spine"),

        ("left_hip", "L Hip"),
        ("right_hip", "R Hip"),

        ("left_knee", "L Knee"),
        ("right_knee", "R Knee"),

        ("left_ankle", "L Ankle"),
        ("right_ankle", "R Ankle"),

        ("left_elbow", "L Elbow"),
        ("right_elbow", "R Elbow")
    ]

    for key, label in display_order:

        value = analysis.get(key)

        if value is None:
            continue

        cv2.putText(
            frame,
            f"{label}: {value:.1f} deg",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            YELLOW,
            2,
            cv2.LINE_AA
        )

        y += 28


# -----------------------------------
# Draw Center Points
# -----------------------------------

def draw_body_centers(frame, analysis):

    coords = analysis["coordinates"]

    if coords["left_shoulder"] and coords["right_shoulder"]:

        sx = int((coords["left_shoulder"][0] + coords["right_shoulder"][0]) / 2)
        sy = int((coords["left_shoulder"][1] + coords["right_shoulder"][1]) / 2)

        cv2.circle(frame, (sx, sy), 9, ORANGE, -1, cv2.LINE_AA)

        cv2.putText(
            frame,
            "Shoulder Center",
            (sx + 8, sy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            ORANGE,
            1
        )

    if coords["left_hip"] and coords["right_hip"]:

        hx = int((coords["left_hip"][0] + coords["right_hip"][0]) / 2)
        hy = int((coords["left_hip"][1] + coords["right_hip"][1]) / 2)

        cv2.circle(frame, (hx, hy), 6, BLUE, -1)

        cv2.putText(
            frame,
            "Hip Center",
            (hx + 8, hy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            BLUE,
            1
        )
        


# -----------------------------------
# Draw Information Panel
# -----------------------------------

def draw_information_panel(frame):
    # No background panel
    cv2.putText(
        frame,
        "StrideAI Gait Analysis",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )