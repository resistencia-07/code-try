import os
import cv2
import importlib

from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
)
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
        num_hands=1,
    )

    return HandLandmarker.create_from_options(options)


def count_fingers(hand_landmarks):
    """
    Count raised fingers (0-5)
    """

    tips = [4, 8, 12, 16, 20]
    count = 0

    # Thumb
    if hand_landmarks[tips[0]].x < hand_landmarks[tips[0] - 1].x:
        count += 1

    # Index, Middle, Ring, Pinky
    for tip in tips[1:]:
        if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
            count += 1

    return count


def configure_camera(cap):
    """Stabilize the webcam feed by disabling auto-exposure and white balance."""

    # Use the Linux V4L2 backend for finer camera-property control.
    cap.set(cv2.CAP_PROP_FPS, 30)  # use 25 for 50 Hz mains regions

    # Disable auto exposure and lock a fixed exposure value.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)

    # Disable auto white balance if flicker persists.
    cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)

    # Print the values so you can confirm the camera accepted them.
    print("Camera settings:")
    print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS):.0f}")
    print(f"  Auto exposure: {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
    print(f"  Exposure: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    print(f"  Auto white balance: {cap.get(cv2.CAP_PROP_AUTO_WB)}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Download hand_landmarker.task first."
        )

    landmarker = create_hand_landmarker(MODEL_PATH)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam with OpenCV V4L2 backend")

    configure_camera(cap)

    total = 0
    last_count = -1

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_img = mp_image.Image(
            image_format=mp_image.ImageFormat.SRGB,
            data=rgb,
        )

        result = landmarker.detect(mp_img)

        if result.hand_landmarks:

            hl = importlib.import_module(HandLandmarker.__module__)
            connections = hl.HandLandmarksConnections.HAND_CONNECTIONS

            for hand_landmarks in result.hand_landmarks:

                drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    connections,
                )

                finger_count = count_fingers(hand_landmarks)

                if finger_count != last_count:
                    total += finger_count
                    last_count = finger_count

                cv2.putText(
                    frame,
                    f"Finger Count : {finger_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Total : {total}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )

        else:
            last_count = -1

        cv2.imshow("Hand Tracking + Finger Counter", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            total = 0
            last_count = -1

        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()