import cv2
import dlib
import time
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from inference import get_model
import winsound


# Model configuration
model = get_model(model_id="eyes-classification/1", api_key="...") #Contact to get API key

# Paths for face detection and landmark model
landmark_model_path = "D:/AIProject/shape_predictor_68_face_landmarks.dat"

# Create a directory to save captured images if it doesn't exist
output_folder = "D:/AIProject/captured_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load dlib's facial landmark predictor
predictor = dlib.shape_predictor(landmark_model_path)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access the webcam.")
    exit()

frame_count = 0
capture_interval = 0.01  # Interval between captures in seconds
img_size = 100  # Expected input size for the eye detection model
alpha = 0.7  # Smoothing factor

def crop_eyes(img, landmarks):
    """
    Extracts the eye regions using the detected facial landmarks.
    """
    left_eye_points = landmarks[36:42]
    right_eye_points = landmarks[42:48]

    # Find bounding box around both eyes
    all_points = np.concatenate((left_eye_points, right_eye_points), axis=0)
    x, y, w, h = cv2.boundingRect(all_points)

    # Expand the bounding box slightly
    margin = 20
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img.shape[1], x + w + margin)
    y2 = min(img.shape[0], y + h + margin)

    eyes_region = img[y1:y2, x1:x2]
    return eyes_region
prev_time = 0
fps = 0
count = 0
prev_label = "Non-Drowsy"
thresh_hold = 5
while True:
    # Capture frame-by-frame
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Cannot capture frame from webcam.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Get the coordinates of the first detected face
        x, y, w, h = faces[0]
        padding = 20

        # Smooth the bounding box coordinates
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        face_only = frame[y1:y2, x1:x2]

        # Convert the cropped face to grayscale for dlib processing
        gray_face = cv2.cvtColor(face_only, cv2.COLOR_BGR2GRAY)

        # Use dlib to detect face and landmarks within the cropped region
        dlib_faces = dlib.rectangle(0, 0, face_only.shape[1], face_only.shape[0])
        landmarks = predictor(gray_face, dlib_faces)
        landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Extract eye regions using the landmarks
        eyes_image = crop_eyes(face_only, landmarks_np)

        # Preprocess the eyes region for the model prediction
        if eyes_image is not None and eyes_image.size > 0:
            eyes_resized = cv2.resize(eyes_image, (img_size, img_size))
            eyes_resized = eyes_resized.astype("float32") / 255.0
            eyes_array = img_to_array(eyes_resized)
            eyes_array = np.expand_dims(eyes_array, axis=0)

            # Make a prediction using your model
            prediction = model.infer(eyes_image)[0]
            label = prediction.top

            # Display the prediction on the frame
            color = (0, 255, 0) if label == "non-drowsy" else (0, 0, 255)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Save the eyes region if necessary
            frame_path = os.path.join(output_folder, f"eyes_{frame_count}.jpg")
            cv2.imwrite(frame_path, eyes_image)
            # print(f"Saved eyes region: {frame_path}")
            frame_count += 1

            if label == "drowsy":
                count += 1
                if prev_label == "drowsy" and count >= thresh_hold:
                    winsound.Beep(1000, 500)
            else:
                count = 0

            prev_label = label
    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Drowsiness Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Delay for the specified interval
    time.sleep(capture_interval)

# Release resources
cap.release()
cv2.destroyAllWindows()
