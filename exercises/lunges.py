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

        # LEFT LEG
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

        l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # RIGHT LEG
        r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

        r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

        r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Angles
        left_angle = calculate_angle(l_hip, l_knee, l_ankle)
        right_angle = calculate_angle(r_hip, r_knee, r_ankle)

        knee_angle = min(left_angle, right_angle)

        # Draw angle
        coords = tuple(np.multiply(
            l_knee if left_angle < right_angle else r_knee,
            [w, h]
        ).astype(int))

        cv2.putText(image, str(round(knee_angle, 2)),
                    (coords[0]+10, coords[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,255,255), 2)

        # Logic
        if knee_angle < 120:
            stage = "down"

        elif knee_angle > 160 and stage == "down":
            stage = "up"
            counter += 1

        # UI
        cv2.rectangle(image, (0,0), (260,80), (245,117,16), -1)

        cv2.putText(image, "LUNGES", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

        cv2.putText(image, str(counter), (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

        cv2.putText(image, "STAGE", (130,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

        cv2.putText(image, str(stage or ""),
                    (130,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

    return image