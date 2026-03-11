import cv2
import mediapipe as mp
import numpy as np
import time

pTime = 0

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


cap = cv2.VideoCapture(0)

# Curl counters
left_counter = 0
left_stage = None

right_counter = 0
right_stage = None


with mp_pose.Pose(min_detection_confidence=0.7,
                  min_tracking_confidence=0.7) as pose:

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR → RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Pose detection
        results = pose.process(image)

        # Convert RGB → BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w, _ = image.shape

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            # ---------------- LEFT ARM ----------------

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


            # ---------------- RIGHT ARM ----------------

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


        # ---------------- UI PANEL ----------------

        cv2.rectangle(image, (0,0), (300,80), (245,117,16), -1)

        cv2.putText(image, "LEFT REPS", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        cv2.putText(image, str(left_counter), (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

        cv2.putText(image, "RIGHT REPS", (150,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        cv2.putText(image, str(right_counter), (150,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)


        # ---------------- FPS ----------------

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(image, f'FPS: {int(fps)}', (450,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


        # Draw pose skeleton
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,240), thickness=2, circle_radius=2)
        )


        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()