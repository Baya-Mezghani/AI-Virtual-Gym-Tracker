import cv2
import mediapipe as mp
import numpy as np
import time
from angle_utils import calculate_angle

mp_pose = mp.solutions.pose

counter = 0
stage = None

def reset():
    global counter, stage
    counter = 0
    stage = None

def update(image, results):
    global counter, stage

    if not results.pose_landmarks:
        return image

    if results.pose_landmarks:
        h, w, _ = image.shape
        landmarks = results.pose_landmarks.landmark

        # LEFT SIDE
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

        # RIGHT SIDE
        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

        r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

        r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

        # Angles
        left_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        right_angle = calculate_angle(r_shoulder, r_hip, r_knee)

        torso_angle = (left_angle + right_angle) / 2

        # Draw angle
        coords = tuple(np.multiply(l_hip, [w, h]).astype(int))

        cv2.putText(image, str(round(torso_angle,2)),
                        (coords[0]+10, coords[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,255), 2)

        # SIT-UP LOGIC
        if torso_angle > 150:
            stage = "down"

        elif torso_angle < 90 and stage == "down":
            counter += 1
            stage = "up"

        # UI
        cv2.rectangle(image, (0,0), (260,80), (245,117,16), -1)

        cv2.putText(image, "SIT UPS", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

        cv2.putText(image, str(counter), (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

        cv2.putText(image, "STAGE", (130,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

        cv2.putText(image, str(stage if stage else ""),
                    (130,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

    return image