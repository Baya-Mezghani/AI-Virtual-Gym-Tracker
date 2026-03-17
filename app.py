import streamlit as st
from main import run_exercise

st.title("🏋️ AI Fitness Trainer")

# Body part selection
body_part = st.selectbox(
    "Select Body Part",
    ["Arms", "Chest", "Legs", "Core"]
)

# Exercise options depending on body part
exercise_options = {
    "Arms": ["Bicep Curl", "Shoulder Press"],
    "Chest": ["Push-Ups"],
    "Legs": ["Squats", "Lunges"],
    "Core": ["Sit-Ups", "Plank"]
}

exercise = st.selectbox(
    "Select Exercise",
    exercise_options[body_part]
)

# Map exercise names to engine names
exercise_map = {
    "Bicep Curl": "bicep",
    "Shoulder Press": "shoulder",
    "Push-Ups": "pushup",
    "Squats": "squat",
    "Lunges": "lunge",
    "Sit-Ups": "situps",
    "Plank": "plank"
}

if st.button("Start Workout"):
    selected_exercise = exercise_map.get(exercise)

    if selected_exercise:
        st.write(f"Starting {exercise} tracker...")
        run_exercise(selected_exercise)