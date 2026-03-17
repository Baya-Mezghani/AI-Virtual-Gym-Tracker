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

        # LEFT ARM
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

        l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # RIGHT ARM
        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

        r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

        r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # Calculate angles
        left_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        right_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

        arm_angle = (left_angle + right_angle) / 2

        # Draw angle
        coords = tuple(np.multiply(l_elbow, [w, h]).astype(int))

        cv2.putText(image, str(round(arm_angle,2)),
                        (coords[0]+10, coords[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,255), 2)

        # SHOULDER PRESS LOGIC
        if arm_angle < 90:
            stage = "down"

        elif arm_angle > 160 and stage == "down":
            stage = "up"
            counter += 1

        # UI
        cv2.rectangle(image, (0,0), (260,80), (245,117,16), -1)

        cv2.putText(image, "SHOULDER PRESS", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

        cv2.putText(image, str(counter), (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

        cv2.putText(image, "STAGE", (150,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

        cv2.putText(image, str(stage if stage else ""),
                    (150,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

    return image