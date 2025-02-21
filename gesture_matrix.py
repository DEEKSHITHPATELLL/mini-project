import mediapipe as mp
import cv2
import json
import os

def load_class_labels(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Class labels file '{file_path}' not found.")
        return []

    with open(file_path, "r") as file:
        return [line.strip() for line in file if line.strip()]

def save_gesture_matrix(landmarks, gesture_name, class_labels, hand_type, file_path="gesture_matrices.json"):
    if gesture_name not in class_labels:
        print(f"Error: Gesture name '{gesture_name}' not found in class labels.")
        return

    normalized_landmarks = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]

    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
    else:
        data = {}
    data[f"{hand_type}_{gesture_name}"] = normalized_landmarks

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Gesture '{gesture_name}' for {hand_type} hand saved successfully!")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

class_labels_path = r"E:/miniproject/class_labels.txt"
class_labels = load_class_labels(class_labels_path)
if not class_labels:
    print("No class labels found. Exiting.")
    exit()

print("Loaded class labels:", class_labels)

cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_type = handedness.classification[0].label
                print(f"Detected {hand_type} hand.")

                print("Enter the gesture name (or 'skip' to continue):")
                gesture_name = input().strip()

                if gesture_name.lower() == "skip":
                    continue

                save_gesture_matrix(hand_landmarks, gesture_name, class_labels, hand_type)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Program interrupted")

finally:
    cap.release()
    cv2.destroyAllWindows()