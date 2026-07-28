import os
import sys
from pathlib import Path
import cv2
import mediapipe as mp
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------
# Import Custom Modules
# -----------------------------
from point.kinematics import analyze_pose
from point.drawing import (
    draw_joint_labels,
    draw_angles,
    draw_body_centers,
    draw_information_panel,
)

# -----------------------------
# Camera Configuration
# -----------------------------

def configure_camera(cap):
    """
    Configure webcam for smoother video.
    """

    cap.set(cv2.CAP_PROP_FPS, 30)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)

    cap.set(cv2.CAP_PROP_AUTO_WB, 0)

    print("\n========== Camera ==========")
    print("FPS :", cap.get(cv2.CAP_PROP_FPS))
    print("Resolution :",
          int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
          "x",
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("============================\n")


# -----------------------------
# Open Camera
# -----------------------------

def open_camera(camera):

    try:
        idx = int(camera)
        return cv2.VideoCapture(idx, cv2.CAP_V4L2)

    except Exception:

        return cv2.VideoCapture(camera, cv2.CAP_V4L2)


# -----------------------------
# Main
# -----------------------------

def main():

    pose = None
    landmarker = None

    use_solutions = (
        hasattr(mp, "solutions")
        and hasattr(mp.solutions, "pose")
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--camera",
        "-c",
        help="USB Camera Index"
    )

    parser.add_argument(
        "--input",
        "-i",
        help="Video File"
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output Video"
    )

    parser.add_argument(
        "--no-flip",
        action="store_true"
    )

    args = parser.parse_args()

    cap = None

    processing_file = False

    # -------------------------
    # Video File
    # -------------------------

    if args.input:

        if not os.path.exists(args.input):

            print("Video not found")

            return

        cap = cv2.VideoCapture(args.input)

        processing_file = True

    else:

        if args.camera:

            cap = open_camera(args.camera)

        else:

            for idx in [1, 2, 3, 4, 5, 0]:

                temp = cv2.VideoCapture(idx, cv2.CAP_V4L2)

                if temp.isOpened():

                    cap = temp

                    print(f"Using Camera {idx}")

                    break

                temp.release()

    if cap is None or not cap.isOpened():

        print("Unable to open camera.")

        return

    configure_camera(cap)

    print("Press ESC to Exit")

    # -------------------------
    # MediaPipe Solutions
    # -------------------------

    if use_solutions:

        mp_pose = mp.solutions.pose

        pose = mp_pose.Pose(

            static_image_mode=False,

            model_complexity=1,

            smooth_landmarks=True,

            min_detection_confidence=0.5,

            min_tracking_confidence=0.5

        )

    else:

        from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker

        model_path = os.path.join(
            "models",
            "pose_landmarker.task"
        )

        if not os.path.exists(model_path):

            print("pose_landmarker.task missing")

            return

        landmarker = PoseLandmarker.create_from_model_path(
            model_path
        )

    # -------------------------
    # Video Writer
    # -------------------------

    writer = None

    if processing_file:

        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not args.output:

            base = os.path.splitext(args.input)[0]

            args.output = base + "_output.mp4"

        writer = cv2.VideoWriter(

            args.output,

            cv2.VideoWriter_fourcc(*"mp4v"),

            fps,

            (width, height)

        )

    # -------------------------
    # Smooth State
    # -------------------------

    smooth_state = {

        "points": None,

        "frames": 0

    }

    alpha = 0.35
    try:

        while True:

            success, frame = cap.read()

            if not success:
                print("Unable to read frame.")
                break

            # Mirror webcam
            if not processing_file and not args.no_flip:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            height, width = frame.shape[:2]

            landmarks = None

            # -------------------------
            # MediaPipe Detection
            # -------------------------

            if use_solutions:

                rgb.flags.writeable = False

                results = pose.process(rgb)

                rgb.flags.writeable = True

                if results.pose_landmarks:

                    landmarks = results.pose_landmarks.landmark

                    connections = mp_pose.POSE_CONNECTIONS

            else:

                from mediapipe import Image
                from mediapipe import ImageFormat

                mp_image = Image(
                    image_format=ImageFormat.SRGB,
                    data=rgb
                )

                result = landmarker.detect(mp_image)

                if result.pose_landmarks:

                    landmarks = result.pose_landmarks[0]

                    from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections

                    connections = PoseLandmarksConnections.POSE_LANDMARKS

            # -------------------------
            # Convert to Pixel Points
            # -------------------------

            current_points = None

            if landmarks:

                current_points = []

                for lm in landmarks:

                    try:

                        x = int(lm.x * width)
                        y = int(lm.y * height)

                        current_points.append((x, y))

                    except Exception:

                        current_points.append(None)

            # -------------------------
            # Landmark Smoothing
            # -------------------------

            if current_points:

                smooth_state["frames"] = 0

                if smooth_state["points"] is None:

                    smooth_state["points"] = current_points

                else:

                    new_points = []

                    for old, new in zip(
                        smooth_state["points"],
                        current_points
                    ):

                        if old is None:

                            new_points.append(new)

                            continue

                        if new is None:

                            new_points.append(old)

                            continue

                        nx = int(
                            alpha * new[0] +
                            (1 - alpha) * old[0]
                        )

                        ny = int(
                            alpha * new[1] +
                            (1 - alpha) * old[1]
                        )

                        new_points.append((nx, ny))

                    smooth_state["points"] = new_points

            else:

                smooth_state["frames"] += 1

                if smooth_state["frames"] > 60:

                    smooth_state["points"] = None

            # -------------------------
            # Draw Skeleton
            # -------------------------

            if smooth_state["points"]:

                if use_solutions:

                    for connection in connections:

                        start, end = connection

                        p1 = smooth_state["points"][start]
                        p2 = smooth_state["points"][end]

                        if p1 and p2:

                            cv2.line(
                                frame,
                                p1,
                                p2,
                                (0,255,0),
                                2
                            )

                else:

                    for connection in connections:

                        p1 = smooth_state["points"][connection.start]
                        p2 = smooth_state["points"][connection.end]

                        if p1 and p2:

                            cv2.line(
                                frame,
                                p1,
                                p2,
                                (0,255,0),
                                2
                            )

                # -------------------------
                # Biomechanics Analysis
                # -------------------------

                analysis = analyze_pose(
                    smooth_state["points"]
                )

                draw_information_panel(frame)

                draw_joint_labels(
                    frame,
                    smooth_state["points"]
                )

                draw_body_centers(
                    frame,
                    analysis
                )

                draw_angles(
                    frame,
                    analysis
                )
                            # -------------------------
            # FPS Display
            # -------------------------

            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 1:
                fps = 30

            cv2.putText(
                frame,
                f"FPS : {fps:.1f}",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # -------------------------
            # Save Output Video
            # -------------------------

            if writer is not None:
                writer.write(frame)

            # -------------------------
            # Show Window
            # -------------------------

            cv2.imshow(
                "StrideAI - Real Time Gait Analysis",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord("q"):
                break

    except KeyboardInterrupt:

        print("\nInterrupted by user.")

    finally:

        print("Closing application...")

        if pose is not None:
            pose.close()

        if landmarker is not None:
            landmarker.close()

        if writer is not None:
            writer.release()

        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()


# ----------------------------------------
# Program Entry
# ----------------------------------------

if __name__ == "__main__":
    main()