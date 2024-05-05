import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import mysql.connector
from datetime import datetime
import time

# Load the emotion detection model
emotion_classifier = load_model('model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the Haar cascade classifier for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the object detection model
classNames = []
classFile = 'coco.names'

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightpath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightpath, configPath)
net.setInputSize(320, 230)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 740)
cap.set(4, 580)

# Establish connection to MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="face-object-detection"
)
cursor = db.cursor()

# Variables to store previous detection results
prev_emotion = None
prev_object = None

# Timestamp to keep track of the last time data was stored
last_store_time = time.time()

# Main loop for real-time processing
while True:
    # Read frame from the webcam
    ret, frame = cap.read()

    # Detect faces for emotion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    # Perform object detection
    classIds, confs, bbox = net.detect(frame, confThreshold=0.5)

    # Process each detected face for emotion detection
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Predict emotion
            prediction = emotion_classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Store emotion detection result if it's different from previous result
            if label != prev_emotion and label != 'Neutral' and time.time() - last_store_time >= 3:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                sql = "INSERT INTO detections (timestamp, emotion, object) VALUES (%s, %s, %s)"
                val = (timestamp, label, prev_object)
                cursor.execute(sql, val)
                db.commit()
                prev_emotion = label
                last_store_time = time.time()

    # Process each detected object for object detection
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, classNames[classId - 1], (box[0] + 10, box[1] + 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 255, 0), thickness=2)

        # Store object detection result if it's different from previous result
        if classNames[classId - 1] != prev_object and time.time() - last_store_time >= 3:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sql = "INSERT INTO detections (timestamp, emotion, object) VALUES (%s, %s, %s)"
            val = (timestamp, prev_emotion, classNames[classId - 1])
            cursor.execute(sql, val)
            db.commit()
            prev_object = classNames[classId - 1]
            last_store_time = time.time()

    # Display the combined output
    cv2.imshow('Output', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
