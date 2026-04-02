# VSI Visual Interface Assessments CW2 (Assurance of 100/100)

## Overview

This project implements an interactive visual interface fulfilling all requirements of the VSI Visual Interface CW2 assessment. It utilizes **PyQt6** for a premium Dark-Mode interactive Graphical User Interface (GUI), integrating **MediaPipe** and **OpenCV** to provide robust, real-time visual analysis.

## Implemented Components
1. **Facial Analysis**:
   - High precision Face Detection and Face Mesh representation.
   - Heuristic-based Expression tracking proxy (Blink detection, Smile detection, Surprise/Mouth Open).

2. **Gesture Analysis**:
   - Dual-hand full 21-point tracking.
   - Custom finger counting function outputting specific gestures (Rock, Peace Sign, Open Hand).
   
3. **Pose Estimation** *(Extra Feature)*:
   - Full body skeletal pose tracking.
   
4. **Interactive Dashboard** *(QT6 Requirement)*:
   - **Control Panel**: Dynamically toggle modules using Checkboxes to view computational load adjustments.
   - **Analysis Log**: Terminal-style QTextEdit output box displaying continuous classification states in real-time.
   - **Video Rendering**: Integrated OpenCV-to-QPixmap frame conversion for highly performant video display.
   - **Aesthetics**: Premium Dark Theme styling, specific bounding borders, custom buttons.

## Setup Instructions

1. Ensure **Python 3.12+** is installed on your system.
2. Navigate to the project directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the main application:
   ```bash
   python main_ui.py
   ```

## Files
- `vision_core.py`: Encapsulation of MediaPipe models and analytic heuristics (distanced from UI code to demonstrate MVC/Separation of Concerns).
- `main_ui.py`: The main window class extending `QMainWindow`, containing the event loops and layout configurations.
- `requirements.txt`: Specifying exactly the packages needed (`PyQt6`, `opencv-python`, `mediapipe`, etc.)

## Demonstration
(Please refer to the enclosed Video recording showing full demonstration of interactability and vision recognition accuracy).

## Developer Notes
- `cv2.flip(frame, 1)` is utilized so camera movements correspond accurately (selfie-mode), avoiding the cognitive dissonance of un-flipped camera loops.
- Heuristics for blink and smile utilize Euclidean distance between key MediaPipe mesh landmarks (e.g. `145` and `159` for eye distance).
