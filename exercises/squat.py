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
    angle = np.abs(radians * 180 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


cap = cv2.VideoCapture(0)

counter = 0
stage = None


with mp_pose.Pose(min_detection_confidence=0.7,
                  min_tracking_confidence=0.7) as pose:

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w, _ = image.shape

        if results.pose_landmarks:

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

        cv2.putText(image, str(stage), (120,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

        # FPS
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(image, f'FPS: {int(fps)}', (450,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # draw skeleton
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        cv2.imshow("Squat Tracker", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()