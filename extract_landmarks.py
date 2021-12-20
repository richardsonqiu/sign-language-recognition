import cv2
import mediapipe as mp
import os
import numpy as np
from functools import reduce
from datetime import datetime

locations = (
    (33, 4),
    (468, 3),
    (21, 3),
    (21, 3)
)
keypoints_len = reduce(lambda r, loc: r + loc[0] * loc[1], locations, 0)

VIDEO_PATH = 'videos'

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for word in os.listdir(VIDEO_PATH):
        print(f'{datetime.now().time()} | Start processing {word}')
        word_path = os.path.join(VIDEO_PATH, word)

        for i, name in enumerate(os.listdir(word_path)):
            word_video_path = os.path.join(word_path, name)

            cap = cv2.VideoCapture(word_video_path)
            frame_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            track = np.empty([frame_len, keypoints_len])

            for i in range(frame_len):
                ret, frame = cap.read()

                landmarks = holistic.process(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )

                track[i] = extract_keypoints(landmarks)
            
            save_path = os.path.join('tracks_binary', word)
            os.makedirs(save_path, exist_ok=True)
            track.astype('float32').tofile(f'tracks_binary/{word}/{name}')

    cap.release()
