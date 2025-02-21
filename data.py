import cv2
import os
import mediapipe as mp
gesture_name = input("Enter the gesture name: ").strip()
save_dir = f"E:/miniproject/dataset/dataset/dataset/{gesture_name}"  
os.makedirs(save_dir, exist_ok=True)  
capture_delay = 5 
num_images = 100  
resize_dim = (1280,1280)  
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()
print(f"Prepare to show the '{gesture_name}' gesture.")
for i in range(capture_delay, 0, -1):
    print(f"Starting in {i} seconds...")
    cv2.waitKey(1000)

print("Capturing images...")
count = 0
while count < num_images:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min = int(min([landmark.x for landmark in landmarks.landmark]) * w)
            x_max = int(max([landmark.x for landmark in landmarks.landmark]) * w)
            y_min = int(min([landmark.y for landmark in landmarks.landmark]) * h)
            y_max = int(max([landmark.y for landmark in landmarks.landmark]) * h)
            cropped_hand = frame[y_min:y_max, x_min:x_max]
            cropped_hand_resized = cv2.resize(cropped_hand, resize_dim)
            cropped_hand_gray = cv2.cvtColor(cropped_hand_resized, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Cropped Hand (Gray)", cropped_hand_gray)
            image_path = os.path.join(save_dir, f"{gesture_name}_{count:03d}.jpg")
            cv2.imwrite(image_path, cropped_hand_gray)
            count += 1
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
    if count >= num_images:
        break

print(f"Captured {count} grayscale images for gesture '{gesture_name}'.")
cap.release()
cv2.destroyAllWindows()
