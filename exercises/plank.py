import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


cap = cv2.VideoCapture(0)

plank_start = None
plank_time = 0
stable_start = None


with mp_pose.Pose(min_detection_confidence=0.7,
                  min_tracking_confidence=0.7) as pose:

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            # LEFT BODY
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # RIGHT BODY
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

            # Correct plank position
            if 160 < body_angle < 200:

                if stable_start is None:
                    stable_start = time.time()

                # wait 1 second before starting plank timer
                if time.time() - stable_start > 1:

                    if plank_start is None:
                        plank_start = time.time()

                    plank_time = int(time.time() - plank_start)

            else:
                stable_start = None
                plank_start = None
                plank_time = 0

        cv2.rectangle(image, (0,0), (260,80), (245,117,16), -1)

        cv2.putText(image, "PLANK TIME", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

        cv2.putText(image, str(plank_time)+" s", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Plank Tracker", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()