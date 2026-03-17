import cv2
import mediapipe as mp
import numpy as np
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

        # landmarks
        HIP = mp_pose.PoseLandmark.LEFT_HIP.value
        KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
        ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE.value

        hip = [landmarks[HIP].x, landmarks[HIP].y]
        knee = [landmarks[KNEE].x, landmarks[KNEE].y]
        ankle = [landmarks[ANKLE].x, landmarks[ANKLE].y]

        # angle
        angle = calculate_angle(hip, knee, ankle)

        # draw angle
        coords = tuple(np.multiply(knee, [w, h]).astype(int))

        cv2.putText(image, str(round(angle,2)),
                        (coords[0]+10, coords[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,255), 2, cv2.LINE_AA)

        # squat logic
        if angle > 160:
                stage = "up"

        elif angle < 90 and stage == "up":
                stage = "down"
                counter += 1

        # UI
        cv2.rectangle(image, (0,0), (250,80), (245,117,16), -1)

        cv2.putText(image, "SQUATS", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

        cv2.putText(image, str(counter), (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

        cv2.putText(image, "STAGE", (120,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

        cv2.putText(image, str(stage or ""), (120,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

    return image