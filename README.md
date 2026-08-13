# Vision Hill Climb

Vision Hill Climb is a computer vision-based gesture controller designed for games like *Hill Climb Racing*. By leveraging your webcam, this project allows you to play racing games using hand positions, hand tilt angles, and even facial expressions (blinking) instead of a traditional keyboard.

## Features
* **Hand Tracking for Movement**: Move your hand left or right to simulate pressing the Brake (A / Left Arrow) or Accelerate (D / Right Arrow) keys.
* **Hand Angle Detection**: Tilt your hand to apply nuanced controls.
* **Facial Gesture Support**: The advanced `real_game_controller.py` uses eye-aspect-ratio (EAR) tracking to detect when you blink, automatically triggering the 'Enter' key.
* **Variable Intensity**: Calculates smooth, variable acceleration and braking based on how far your hand moves from the center dead-zone.

## Project Structure
The repository includes several scripts that show the progression of the controller from basic tests to a full-fledged game controller:
* `camera_test.py` / `gesture_test.py` / `hand_tracking.py` / `keyboard_test.py`: Foundational testing scripts for camera feed, keyboard simulation, and MediaPipe tracking.
* `phase2_controller.py` & `vision_controller.py`: Basic controllers utilizing simple left/right hand zones to steer.
* `phase2_variable.py`: Upgraded controller with a dead-zone and intensity smoothing.
* `phase2_combined.py`: Controller that adds hand tilt angle detection.
* `game_controller.py`: An advanced script that specifically focuses on the *Hill Climb Racing* game window, mapping gestures to the Left/Right arrow keys.
* `real_game_controller.py`: The most advanced version utilizing local MediaPipe `.task` models to track both hands and faces simultaneously.

## Requirements
* Python 3.x
* A working webcam
* [OpenCV](https://pypi.org/project/opencv-python/)
* [MediaPipe](https://pypi.org/project/mediapipe/)
* [PyAutoGUI](https://pypi.org/project/PyAutoGUI/)
* [PyGetWindow](https://pypi.org/project/PyGetWindow/)

## Installation
1. Clone this repository to your local machine.
2. Create and activate a virtual environment (optional but recommended).
3. Install the required dependencies:
   ```bash
   pip install opencv-python mediapipe pyautogui pygetwindow
   ```

## Usage
1. Open *Hill Climb Racing* (or your target game) and ensure it's ready to play.
2. Run your preferred controller script from the terminal. For the most advanced experience, use:
   ```bash
   python real_game_controller.py
   ```
3. A window will pop up showing your webcam feed and tracked landmarks. 
4. Move your hand left/right to control the vehicle, and blink to press Enter!

## Disclaimer
When using PyAutoGUI, ensure you have an emergency way to stop the script (you can usually drag your mouse to a corner of the screen to trigger PyAutoGUI's failsafe, or press `ESC` while focused on the webcam window).
