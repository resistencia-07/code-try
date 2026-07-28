# StrideAI Pose and Hand Tracking Project

This project uses OpenCV and MediaPipe to run real-time pose analysis and hand tracking experiments from a webcam.

## Features

- Real-time pose estimation with MediaPipe
- Biomechanics analysis for joints and body alignment
- Visual overlays for landmarks, angles, and body centers
- Hand tracking with finger counting
- Optional input from a video file or webcam

## Project Structure

- [point/pose.py](point/pose.py) - Main pose analysis application
- [point/kinematics.py](point/kinematics.py) - Joint angle and posture calculations
- [point/drawing.py](point/drawing.py) - Visual drawing helpers for overlays
- [hand.py](hand.py) - Hand tracking and finger-count demo
- [hand_tracking.py](hand_tracking.py) - Additional hand tracking experiment
- [models/](models/) - MediaPipe model assets

## Requirements

Install the dependencies in your virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install opencv-python mediapipe
```

## Run the Pose Application

From the project root:

```bash
source .venv/bin/activate
python point/pose.py
```

Optional arguments:

```bash
python point/pose.py --camera 0
python point/pose.py --input video.mp4 --output output.mp4
python point/pose.py --no-flip
```

## Run the Hand Tracking Demo

```bash
source .venv/bin/activate
python hand.py
```

## Notes

- A webcam or supported camera device is required for live demos.
- The app uses the MediaPipe model files stored in [models/](models/).
- If you run the pose script directly and see an import error, the project root is now handled automatically in [point/pose.py](point/pose.py).
