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
