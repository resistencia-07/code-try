import os
import cv2
import numpy as np

from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
import importlib
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")

def create_hand_landmarker(model_path: str) -> HandLandmarker:
    base_opts = base_options.BaseOptions(model_asset_path=model_path)
    options = HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=VisionTaskRunningMode.IMAGE,
        num_hands=2,
    )
    return HandLandmarker.create_from_options(options)


from typing import List, Optional


def _get_handedness_label(handedness_list) -> Optional[str]:
    """Safely extract the handedness label string from MediaPipe category object."""
    try:
        # handedness_list is a list of Category lists per hand; take first classification
        cat = handedness_list[0][0]
    except Exception:
        return None
    for attr in ("category_name", "label", "display_name"):
        if hasattr(cat, attr):
            val = getattr(cat, attr)
            if isinstance(val, str):
                return val
    try:
        return str(cat)
    except Exception:
        return None


def count_raised_fingers(landmarks: List, handedness_label: Optional[str]) -> int:
    
    if not landmarks or len(landmarks) < 21:
        return 0

    def lx(i):
        return landmarks[i].x

    def ly(i):
        return landmarks[i].y

    fingers_up = 0

    # Index, Middle, Ring, Pinky: tip y < pip y -> finger is up (image y grows downward)
    tips_pips = ((8, 6), (12, 10), (16, 14), (20, 18))
    for tip, pip in tips_pips:
        try:
            if ly(tip) < ly(pip):
                fingers_up += 1
        except Exception:
            continue

    # Thumb: tip = 4, ip = 3
    thumb_up = False
    try:
        tip_x = lx(4)
        ip_x = lx(3)
        if handedness_label:
            label = handedness_label.lower()
            if "right" in label:
                thumb_up = tip_x > ip_x
            elif "left" in label:
                thumb_up = tip_x < ip_x
            else:
                thumb_up = tip_x > ip_x
        else:
            wrist_x = lx(0)
            thumb_up = abs(tip_x - wrist_x) > abs(ip_x - wrist_x)
    except Exception:
        thumb_up = False

    if thumb_up:
        fingers_up += 1

    return max(0, min(5, fingers_up))


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    cap = cv2.VideoCapture(0)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Download a compatible hand-landmarker model into models/ and retry.")

    landmarker = create_hand_landmarker(MODEL_PATH)

    running_total = 0
    last_count: Optional[int] = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_img = mp_image.Image(mp_image.ImageFormat.SRGB, rgb)

        result = landmarker.detect(mp_img)

        current_count = 0
        if result and result.hand_landmarks:
            hl = importlib.import_module(HandLandmarker.__module__)
            connections = hl.HandLandmarksConnections.HAND_CONNECTIONS
            # use first detected hand
            hand_landmarks = result.hand_landmarks[0]
            drawing_utils.draw_landmarks(frame, hand_landmarks, connections)

            # determine handedness label if available
            label = _get_handedness_label(result.handedness)

            current_count = count_raised_fingers(hand_landmarks, label)

            # update running total only when count changes
            if last_count is None:
                running_total += current_count
                last_count = current_count
            elif current_count != last_count:
                running_total += current_count
                last_count = current_count

        # Display current count and running total
        cv2.putText(frame, f"Count: {current_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Total: {running_total}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow("Hand Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break
        if key in (ord('r'), ord('R')):
            running_total = 0
            last_count = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()