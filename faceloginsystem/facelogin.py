import cv2
import numpy as np
import os
import mediapipe as mp
from scipy.spatial import distance
from pathlib import Path

script_dir = Path(__file__).parent

known_faces_folder = script_dir / "knownfaces"

def load_known_face_landmarks(face_mesh):
    known_faces_landmarks = {}
    for filename in os.listdir(known_faces_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(known_faces_folder, filename)
            name = os.path.splitext(filename)[0]
            img = cv2.imread(img_path)
            #detect face mesh landmarks for the known faces
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            if results.multi_face_landmarks:
                #save the landmarks of the known face
                known_faces_landmarks[name] = results.multi_face_landmarks[0]
    return known_faces_landmarks

def compute_confidence(min_distance, min_possible_distance=100, max_possible_distance=270):
    confidence = (max_possible_distance - min_distance) / (max_possible_distance - min_possible_distance)
    confidence = max(0.0, min(confidence, 1.0))  #confidence must be between 0 and 1
    return confidence

#detect faces and compare using face mesh landmarks
def detect_and_recognize_faces_with_mesh(image, face_mesh, known_faces_landmarks, confidence_threshold=0.4):
    recognized_faces = []
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            best_match_name = "Unknown"
            min_distance = float('inf')

            for name, known_face_landmarks in known_faces_landmarks.items():
                #compare between detected and known face
                distances = []
                for i in range(len(face_landmarks.landmark)):
                    #get the current coordinates for both faces
                    known_x = known_face_landmarks.landmark[i].x * w
                    known_y = known_face_landmarks.landmark[i].y * h
                    detected_x = face_landmarks.landmark[i].x * w
                    detected_y = face_landmarks.landmark[i].y * h
                    #calculate distances between the landmarks
                    distances.append(distance.euclidean((known_x, known_y), (detected_x, detected_y)))

                avg_distance = np.mean(distances)
                if avg_distance < min_distance:
                    min_distance = avg_distance
                    best_match_name = name
                    
            confidence = compute_confidence(min_distance)

            if confidence < confidence_threshold:
                best_match_name = "Unknown"

            recognized_faces.append((face_landmarks, best_match_name))

    return recognized_faces

def detect_and_draw_face_mesh(image, face_landmarks):
    h, w, _ = image.shape
    important_indices = [
        234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 379, 365, 397, 288, 361, 323, 454, 356, 389, 
        251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 361, 389, 356, 454, 323, 361, 368,
        10, 152, 234, 454, 323, 93, 132, 58, 172, 152, 
        70, 63, 105, 107, 52, 53, 55, 66, 64, 
        336, 285, 295, 282, 52, 53, 55, 296, 293, 
        33, 133, 160, 159, 158, 144, 153, 154, 
        263, 362, 387, 386, 385, 373, 380, 374, 
        6, 168, 195, 5, 4, 
        94, 129, 2, 4, 
        279, 358, 421, 430,
        78, 308, 61, 291, 13, 14, 17, 18, 80, 82, 312, 324, 37, 267, 269, 270, 273, 277, 402, 14,
        185, 191, 202, 204, 209, 211, 217, 219, 222, 227, 233, 240, 243, 246, 249, 251,
    ]

    #draw mesh points
    for idx in important_indices:
        x = int(face_landmarks.landmark[idx].x * w)
        y = int(face_landmarks.landmark[idx].y * h)
        cv2.circle(image, (x, y), 5, (238, 130, 238), -1)  

    return image

def facial_recognition_login():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not accessible or not found.")
        return False

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    known_faces_landmarks = load_known_face_landmarks(face_mesh)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        #face detection and recognition using face mesh 
        recognized_faces = detect_and_recognize_faces_with_mesh(frame, face_mesh, known_faces_landmarks)

        for face_landmarks, name in recognized_faces:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            frame = detect_and_draw_face_mesh(frame, face_landmarks)
            cv2.putText(frame, name, (int(face_landmarks.landmark[0].x * frame.shape[1]), int(face_landmarks.landmark[0].y * frame.shape[0]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if name != "Unknown":
                print(f"Face recognized as {name}. Access granted.")
                cap.release()
                cv2.destroyAllWindows()
                return True

        #display the results
        cv2.imshow('Facial Recognition Login', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Face not recognized. Access denied.")
    return False

#run the facial recognition login function
if facial_recognition_login():
    print("Login successful!")
else:
    print("Login failed.")
