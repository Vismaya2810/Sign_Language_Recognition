import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

# Extract hand landmarks from a video file
# Returns a list of (frame_idx, landmarks) for each frame where a hand is detected
def extract_hand_landmarks(video_path, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    frame_landmarks = []
    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Flatten the landmark coordinates
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                frame_landmarks.append((frame_idx, landmarks))
        frame_idx += 1
    cap.release()
    hands.close()
    return frame_landmarks

# Example usage (for quick test):
# landmarks = extract_hand_landmarks('path_to_video.mp4')
# print(landmarks) 