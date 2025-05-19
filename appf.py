import time
import cv2
import os
import streamlit as st
from datetime import datetime
from ultralytics import YOLO

st.title("Driver Monitoring System")
st.text("Detects Drowsiness, Mobile Usage, and Smoking in Real Time")

model = YOLO('bestts.pt')

# seconds
drowsy_threshold = 2
mobile_threshold = 5
smoking_threshold = 5

# alarm sound
import pygame
pygame.mixer.init()

def play_alarm(): # Function to play the alarm 
    pygame.mixer.music.load('C:/Users/zafar/Downloads/download/generic-alarm-clock-86759.mp3')
    pygame.mixer.music.play()

def log_activity(activity, log_file): #log detected activities
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now()}: {activity}\n")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')# Load the pre-trained face detection model from OpenCV


def save_face(frame, log_folder, activity): # save the face image when an activity is detected
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale for face detection
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            face_file_path = os.path.join(log_folder, f'{activity}_face_{timestamp}.jpg')
            cv2.imwrite(face_file_path, face_img)
            st.text(f"Saved face image for {activity} as {face_file_path}")

#name input
driver_name = st.text_input("Enter Driver's Name", "Driver1")

# Dir and log file each driver
log_folder = f'C:/Users/zafar/Downloads/driver_alert_model3/driverdata/{driver_name}'
os.makedirs(log_folder, exist_ok=True)
log_file = os.path.join(log_folder, f'{driver_name}_activity_log.txt')

run_app = st.button("Start Monitoring")

if run_app:
    # Initialize timers for activities
    drowsy_timer = None
    mobile_timer = None
    smoking_timer = None

    # Start video capture (webcam)
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Unable to access webcam")
            break

        results = model(frame)

        # Flags to check if a specific class is detected
        drowsy_detected = False
        mobile_detected = False
        smoking_detected = False

        # Check detected objects in the frame
        for result in results[0].boxes:
            xmin, ymin, xmax, ymax = result.xyxy[0]  # Bounding box coordinates
            conf = result.conf[0]  # Confidence score
            class_id = int(result.cls[0])  # Class ID
            label = model.names[class_id]  # Class label

            if conf > 0.5: 
                # Draw the bounding box and label
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Handle drowsiness detection
                if label == "Closed Eye":
                    drowsy_detected = True
                    if drowsy_timer is None:
                        drowsy_timer = time.time()
                    elif time.time() - drowsy_timer >= drowsy_threshold:
                        play_alarm()
                        log_activity("Drowsiness detected", log_file)
                        save_face(frame, log_folder, "Drowsiness")
                        drowsy_timer = None  # Reset timer after alert

                # Handle mobile phone usage detection
                elif label == "Phone":
                    mobile_detected = True
                    if mobile_timer is None:
                        mobile_timer = time.time()
                    elif time.time() - mobile_timer >= mobile_threshold:
                        play_alarm()
                        log_activity("Mobile phone usage detected", log_file)
                        save_face(frame, log_folder, "MobilePhone")
                        mobile_timer = None  # Reset timer after alert

                # Handle smoking detection
                elif label == "Cigarette":
                    smoking_detected = True
                    if smoking_timer is None:
                        smoking_timer = time.time()
                    elif time.time() - smoking_timer >= smoking_threshold:
                        play_alarm()
                        log_activity("Smoking detected", log_file)
                        save_face(frame, log_folder, "Smoking")
                        smoking_timer = None  # Reset timer after alert

        # Reset timers if no detection
        if not drowsy_detected:
            drowsy_timer = None
        if not mobile_detected:
            mobile_timer = None
        if not smoking_detected:
            smoking_timer = None

        # Display the video stream in Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

    cap.release()
else:
    st.info("Click the button to start monitoring.")
