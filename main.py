import cv2
import mediapipe as mp
import time

from exercises import bicep_curl
from exercises import squat
from exercises import push_up
from exercises import lunges
from exercises import shoulder_press
from exercises import sit_ups
from exercises import plank

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def run_exercise(exercise_name):

    cap = cv2.VideoCapture(0)

    if exercise_name == "bicep":
        bicep_curl.reset()
    elif exercise_name == "squat":
        squat.reset()
    elif exercise_name == "pushup":
        push_up.reset()
    elif exercise_name == "lunge":
        lunges.reset()
    elif exercise_name == "shoulder":
        shoulder_press.reset()
    elif exercise_name == "situps":
        sit_ups.reset()
    elif exercise_name == "plank":
        plank.reset()

    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert color
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Run selected exercise
            if exercise_name == "bicep":
                image = bicep_curl.update(image, results)

            elif exercise_name == "squat":
                image = squat.update(image, results)

            elif exercise_name == "pushup":
                image = push_up.update(image, results)

            elif exercise_name == "lunge":
                image = lunges.update(image, results)

            elif exercise_name == "shoulder":
                image = shoulder_press.update(image, results)

            elif exercise_name == "situps":
                image = sit_ups.update(image, results)

            elif exercise_name == "plank":
                image = plank.update(image, results)

            # Draw landmarks safely
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            cv2.imshow("AI Gym Trainer", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()