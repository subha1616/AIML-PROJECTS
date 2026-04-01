#STEP 1
import os
import cv2
import numpy as np
# Try TensorFlow Keras first (best for most Windows setups)
try:
    from tensorflow.keras.models import load_model
except ImportError:
    from keras.models import load_model

# STEP 2
# File paths
#--------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default (1).xml")
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.hdf5")

# Load face cascade
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError(f"Could not load face cascade from {CASCADE_PATH}")

# Load emotion model
emotion_model = load_model(MODEL_PATH, compile=False)
print(f"Model loaded from {MODEL_PATH}")

#STEP 5
# Emotion labels
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

#STEP 6
# Start webcam (Windows-friendly)
# Try default camera first
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    # Backup attempt for some Windows laptops
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Please check camera permissions.")
print("Emotion Detection System started.")
print("Press 'q' to quit.")

# STEP 7
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from webcam.")
        break
    # Optional mirror view for students
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        try:
            roi_gray = cv2.resize(roi_gray, (64, 64))
        except Exception:
            continue
        roi_gray = roi_gray.astype("float32") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        #(64, 64, 1)
        roi_gray = np.expand_dims(roi_gray, axis=0)
        #(1, 64, 64, 1)
        prediction = emotion_model.predict(roi_gray, verbose=0)
        max_index = int(np.argmax(prediction))
        emotion = emotion_labels[max_index]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            emotion,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
    cv2.imshow("Emotion Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()