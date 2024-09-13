# main.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import speech_recognition as sr
from mediapipe import solutions

# Load the trained model for handicap detection
model = load_model('model/model_weights.h5')

# Initialize MediaPipe Hands for sign language detection
mp_hands = solutions.hands
hands = mp_hands.Hands()

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

# Toggle for detection
detection_enabled = True

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Function to draw bounding boxes, labels, and keypoints
def draw_detection(frame, label, confidence, keypoints=None):
    if keypoints:
        for point in keypoints:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)

    # Display label and confidence
    text = f'{label}: {confidence:.2f}%'
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Function to detect keypoints for sign language
def detect_sign_language(frame):
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    keypoints = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                keypoints.append((x, y))
    
    return keypoints

# Prediction function using the trained model
def predict_handicap(frame, model):
    # Preprocess the frame: resize and normalize
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0) / 255.0

    # Predict using the trained model
    prediction = model.predict(img)[0][0]
    confidence = round(prediction * 100, 2)

    # Threshold adjustment for prediction
    if prediction > 0.7:
        return 'Handicapped', confidence
    else:
        return 'Not Handicapped', 100 - confidence

# Voice detection function to check if the user speaks
def detect_voice():
    with sr.Microphone() as source:
        print("Listening for voice response...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=3)
            text = recognizer.recognize_google(audio)
            print(f"Detected speech: {text}")
            return True
        except sr.UnknownValueError:
            print("Could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        except sr.WaitTimeoutError:
            print("Listening timed out.")
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from the camera. Exiting.")
        break

    if detection_enabled:
        # Predict handicap status
        label, confidence = predict_handicap(frame, model)

        # Detect keypoints for sign language
        keypoints = detect_sign_language(frame)

        # If no keypoints and no voice detected, possibly a deaf user
        deaf_detected = not keypoints and not detect_voice()
        if deaf_detected:
            label = 'Deaf Detected'
            confidence = 100  # Adjust as per model output

        # Draw detection results on the frame
        draw_detection(frame, label, confidence, keypoints=keypoints)

    # Display instructions on frame
    cv2.putText(frame, "Press 'd' to Toggle Detection, 'q' to Quit", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Show the frame with detections
    cv2.imshow('Handicap & Deaf Detection', frame)

    # Handle keypress for quit or toggle detection
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        detection_enabled = not detection_enabled

cap.release()
cv2.destroyAllWindows()
