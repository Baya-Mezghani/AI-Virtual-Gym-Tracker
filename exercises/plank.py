import cv2
import mediapipe as mp
import numpy as np
import time
from angle_utils import calculate_angle

mp_pose = mp.solutions.pose

# Global variables
plank_start = None
plank_time = 0
stable_start = None

def reset():
    global plank_start, plank_time, stable_start
    plank_start = None
    plank_time = 0
    stable_start = None

def update(image, results):
    global plank_start, plank_time, stable_start

    if not results.pose_landmarks:
        return image

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # LEFT
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

        l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # RIGHT
        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

        r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

        r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Angles
        left_angle = calculate_angle(l_shoulder, l_hip, l_ankle)
        right_angle = calculate_angle(r_shoulder, r_hip, r_ankle)

        body_angle = (left_angle + right_angle) / 2

        # Logic
        if 160 < body_angle < 200:

            if stable_start is None:
                stable_start = time.time()

            if time.time() - stable_start > 1:

                if plank_start is None:
                    plank_start = time.time()

                plank_time = int(time.time() - plank_start)

        else:
            stable_start = None
            plank_start = None
            plank_time = 0

        # UI
        cv2.rectangle(image, (0,0), (260,80), (245,117,16), -1)

        cv2.putText(image, "PLANK TIME", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

        cv2.putText(image, str(plank_time) + " s", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

    return image