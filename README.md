# Facial Recognition Login System

## Overview
This project is a facial recognition login system implemented in Python, utilizing OpenCV, Mediapipe, and SciPy libraries. It uses a webcam to capture a live video feed and detects and recognizes faces based on a set of known face images stored in a designated folder. If a recognized face is detected, access is granted; otherwise, access is denied.

## Features
- **Real-time face detection and recognition** using Mediapipe's face mesh landmarks.
- **Confidence-based recognition** to evaluate the similarity between known and detected face landmarks.
- **Visual feedback** on recognized faces with marked facial landmarks and names displayed on the video feed.
- **Configurable parameters** for confidence thresholds and distance calculations.

## Requirements
Ensure you have the following dependencies installed:
- Python 3.x
- OpenCV (`cv2`)
- Mediapipe
- NumPy
- SciPy
- A webcam connected to your system

## Installation
1. Clone this repository or download the script file.
2. Ensure Python is installed on your system.
3. Install the required libraries:
   ```bash
   pip install opencv-python mediapipe numpy scipy
   ```

4. Create a folder named `knownfaces` in the same directory as the script. Add images of known faces in `.jpg` or `.png` format. Ensure the image filenames are descriptive (e.g., `john_doe.jpg`).

## Usage
1. Run the script:
   ```bash
   python facial_recognition_login.py
   ```

2. The system will access the webcam and start capturing video frames.
3. If a face matches any of the known faces, access will be granted, and the name of the recognized individual will be displayed.
4. Press the 'q' key to exit the video feed at any time.

## Code Structure
- **`load_known_face_landmarks`**: Loads and processes known face images to extract face mesh landmarks.
- **`compute_confidence`**: Computes a confidence score based on the distance between detected and known landmarks.
- **`detect_and_recognize_faces_with_mesh`**: Detects faces in a video frame and compares them with known faces to recognize individuals.
- **`detect_and_draw_face_mesh`**: Draws key facial landmarks on the detected face for visual feedback.
- **`facial_recognition_login`**: Captures video feed, performs recognition, and manages login logic.

## Customization
- **Confidence Threshold**: Adjust the `confidence_threshold` parameter in the `detect_and_recognize_faces_with_mesh` function to change the strictness of face recognition.
- **Landmark Points**: Modify the `important_indices` list in the `detect_and_draw_face_mesh` function to change which facial landmarks are highlighted.

## Troubleshooting
- **Camera Access Issues**: Ensure that your webcam is connected and accessible. Test using a simple OpenCV script if necessary.
- **Face Not Detected**: Check the lighting conditions and ensure the face is well-positioned within the frame.
- **Recognition Errors**: Ensure known face images have good resolution and proper face orientation.

## Limitations
- This system uses a basic Euclidean distance-based approach and may not perform as well as more advanced deep learning models.
- Works best for front-facing, well-lit images.

## Future Improvements
- Implement more advanced facial recognition algorithms using deep learning for better accuracy.
- Add multi-face support and improve the comparison logic for better performance in complex environments.
- Integrate face preprocessing steps for better face alignment and normalization.

## License
This project is open-source and free to use under the MIT License.

