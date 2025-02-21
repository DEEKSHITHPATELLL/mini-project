from flask import Flask, render_template, Response, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import os
import mediapipe as mp
import json
import pyttsx3  
import threading  

app = Flask(__name__)

img_height, img_width = 128, 128
model_path = "gesture_model.keras"
labels_path = "class_labels.txt"
gesture_matrices_path = "gesture_matrices.json"
model = load_model(model_path)
with open(labels_path, "r") as f:
    class_labels = [line.strip() for line in f if line.strip()]
with open(gesture_matrices_path, "r") as f:
    gesture_matrices = json.load(f)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

current_prediction = ""
tts_engine = pyttsx3.init()
def calculate_distance(landmarks1, landmarks2):
    try:
        if len(landmarks1) != len(landmarks2):
            return float('inf')  
        landmarks1 = np.array(landmarks1)
        landmarks2 = np.array(landmarks2)

        return np.sqrt(np.sum((landmarks1 - landmarks2) ** 2, axis=1)).mean()
    except Exception as e:
        print(f"Error in calculate_distance: {e}")
        return float('inf')
def match_gesture(landmarks):
    try:
        normalized_landmarks = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
        min_distance = float('inf')
        matched_label = None

        for label, reference_landmarks in gesture_matrices.items():
            distance = calculate_distance(normalized_landmarks, reference_landmarks)
            if distance < min_distance:
                min_distance = distance
                matched_label = label

        return matched_label, min_distance
    except Exception as e:
        print(f"Error in match_gesture: {e}")
        return None, float('inf')
def speak(text):
    def run_tts():
        tts_engine.say(text)
        tts_engine.runAndWait()
    threading.Thread(target=run_tts, daemon=True).start()
def generate_frames():
    global current_prediction
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            detected = False

            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                    matched_label, distance = match_gesture(landmarks)

                    if matched_label and distance < 0.1:
                        current_prediction = f"{matched_label} (Confidence: {(1-distance)*100:.1f}%)"
                        speak(current_prediction)
                    else:
                        current_prediction = "Gesture not recognized. Please try again."
                    detected = True
            
            if not detected:
                current_prediction = "No hand detected. Please show your hand."
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                print("Error encoding frame.")
                continue

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        print(f"Error in generate_frames: {e}")

    finally:
        cap.release()
        print("Video capture released.")
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    return jsonify({'prediction': current_prediction})
if __name__ == '__main__':
    app.run(debug=True)
