import cv2
import mediapipe as mp
import numpy as np
from angle_utils import calculate_angle

mp_pose = mp.solutions.pose

# Curl counters
left_counter = 0
left_stage = None

right_counter = 0
right_stage = None

def reset():
    global left_counter, left_stage, right_counter, right_stage
    left_counter = 0
    left_stage = None

    right_counter = 0
    right_stage = None

def update(image, results):
        global left_counter, left_stage, right_counter, right_stage

        if not results.pose_landmarks:
            return image

        if results.pose_landmarks:
            h, w, _ = image.shape
            landmarks = results.pose_landmarks.landmark

            # LEFT ARM

            LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            LEFT_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW.value
            LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value

            left_shoulder = [landmarks[LEFT_SHOULDER].x, landmarks[LEFT_SHOULDER].y]
            left_elbow = [landmarks[LEFT_ELBOW].x, landmarks[LEFT_ELBOW].y]
            left_wrist = [landmarks[LEFT_WRIST].x, landmarks[LEFT_WRIST].y]

            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

            left_coords = tuple(np.multiply(left_elbow, [w, h]).astype(int))

            cv2.putText(image, str(round(left_angle,2)),
                        (left_coords[0]+10, left_coords[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,255), 2, cv2.LINE_AA)

            # Curl logic
            if left_angle > 160:
                left_stage = "down"

            elif left_angle < 30 and left_stage == "down":
                left_stage = "up"
                left_counter += 1


            # RIGHT ARM

            RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            RIGHT_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW.value
            RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value

            right_shoulder = [landmarks[RIGHT_SHOULDER].x, landmarks[RIGHT_SHOULDER].y]
            right_elbow = [landmarks[RIGHT_ELBOW].x, landmarks[RIGHT_ELBOW].y]
            right_wrist = [landmarks[RIGHT_WRIST].x, landmarks[RIGHT_WRIST].y]

            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            right_coords = tuple(np.multiply(right_elbow, [w, h]).astype(int))

            cv2.putText(image, str(round(right_angle,2)),
                        (right_coords[0]+10, right_coords[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,255), 2, cv2.LINE_AA)

            # Curl logic
            if right_angle > 160:
                right_stage = "down"

            elif right_angle < 30 and right_stage == "down":
                right_stage = "up"
                right_counter += 1


            # UI PANEL

            cv2.rectangle(image, (0,0), (300,80), (245,117,16), -1)

            cv2.putText(image, "LEFT REPS", (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            cv2.putText(image, str(left_counter), (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

            cv2.putText(image, "RIGHT REPS", (150,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            cv2.putText(image, str(right_counter), (150,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

        return image