import mediapipe as mp

print("MediaPipe version:", mp.__version__)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

print("Tasks API is working!")