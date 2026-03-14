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

        # Convert BGR → RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        # Convert RGB → BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w, _ = image.shape

        if results.pose_landmarks:

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

            # BODY ALIGNMENT
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculate angles
            left_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            right_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

            arm_angle = (left_angle + right_angle) / 2

            body_angle = calculate_angle(l_shoulder, hip, ankle)

            # Draw arm angle
            coords = tuple(np.multiply(l_elbow, [w, h]).astype(int))

            cv2.putText(image, str(round(arm_angle,2)),
                        (coords[0]+10, coords[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,255), 2)

            # PUSH-UP LOGIC
            if body_angle > 150:  # Ensure plank position

                if arm_angle < 110:
                    stage = "down"

                if arm_angle > 150 and stage == "down":
                    stage = "up"
                    counter += 1

        # UI
        cv2.rectangle(image, (0,0), (260,80), (245,117,16), -1)

        cv2.putText(image, "PUSH UPS", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

        cv2.putText(image, str(counter), (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

        cv2.putText(image, "STAGE", (130,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

        cv2.putText(image, str(stage if stage else ""),
                    (130,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

        # FPS
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(image, f'FPS: {int(fps)}', (450,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # Draw pose
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        cv2.imshow("Push-Up Tracker", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()