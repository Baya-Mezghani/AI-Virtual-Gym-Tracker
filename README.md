# AI-Virtual-Gym-Tracker
**AI Virtual Gym Tracker** is a real-time fitness application that uses **computer vision** to track exercises with webcam. It detects body pose using **Mediapipe**, calculates joint angles, and counts repetitions for various exercises.

Supported exercises:
- Bicep Curls
- Shoulder Press
- Push-Ups
- Squats
- Lunges
- Sit-Ups
- Plank
  
---

## Features
- Real-time exercise tracking
- Automatic repetition counting
- Stage detection for correct form
- Simple, user-friendly interface
- Supports multiple body parts: Arms, Chest, Legs, Core

---

## How It Works

1. **Webcam Capture**  
   - The application captures real-time video frames from webcam.

2. **Pose Detection with MediaPipe**  
   - MediaPipe detects **33 key body landmarks** such as shoulders, elbows, hips, knees, and ankles.  
   - Each landmark provides normalized 2D coordinates for precise tracking.

3. **Joint Angle Calculation**  
   - Using the coordinates of relevant landmarks, the system calculates **joint angles** for elbows, knees, hips, etc.  
   - These angles determine the position of each body part during an exercise.

4. **Pose Evaluation & Stage Detection**  
   - The system checks whether the pose matches the correct form for the selected exercise.  
   - It tracks the **stage of the movement** (e.g., “up” or “down”).

5. **Repetition Counting**  
   - A repetition is counted when a full movement cycle is completed correctly (e.g., arm curls from “down” to “up”).  
   - Counts are updated in real-time and displayed on the screen.

6. **User Interface Overlay**  
   - The webcam feed shows **angles, repetition count, and stage** directly on the video for instant feedback.

---

## Installation

### 1. Clone the repository
```bash
git clone git@github.com:Baya-Mezghani/AI-Virtual-Gym-Tracker.git
cd ai-virtual-gym-tracker
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### Usage 
```bash
streamlit run app.py
```

---
## Project Structure
```bash
ai-virtual-gym-tracker/
│
├── exercises/           # Exercise modules (bicep_curl.py, squat.py, etc.)
├── main.py              # Core logic to run selected exercise
├── app.py               # Streamlit interface
├── angle_utils.py       # Helper function to calculate joint angles
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## Author

**Baya Mezghani** 
📧 baya.mezghani@ensi-uma.tn
