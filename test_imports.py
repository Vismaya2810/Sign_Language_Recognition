try:
    import mediapipe as mp
    import cv2
    print("MediaPipe and OpenCV imported successfully!")
except Exception as e:
    print(f"Error importing mediapipe or cv2: {e}")
input("Press Enter to exit...")