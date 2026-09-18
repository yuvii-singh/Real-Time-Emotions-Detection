
'''
Emotion Detection Using AI
Multiple Face Emotion Detection
'''

from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Load face detector
face_classifier = cv2.CascadeClassifier(
    './haarcascade_frontalface_default.xml'
)

# Load emotion model
classifier = load_model('./Emotion_Detection.h5')

# Emotion labels
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera")
        break

    # Convert camera frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ALL faces
    faces = face_classifier.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(50, 50)
)

    # Process every detected face
    for person_number, (x, y, w, h) in enumerate(faces, start=1):

        # Draw rectangle around face
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        # Extract face
        roi_gray = gray[y:y+h, x:x+w]

        # Resize face
        roi_gray = cv2.resize(
            roi_gray,
            (48, 48),
            interpolation=cv2.INTER_AREA
        )

        if np.sum(roi_gray) != 0:

            # Normalize
            roi = roi_gray.astype('float') / 255.0

            # Convert to array
            roi = img_to_array(roi)

            # Add batch dimension
            roi = np.expand_dims(roi, axis=0)

            # Predict emotion
            preds = classifier.predict(
                roi,
                verbose=0
            )[0]

            # Get highest probability emotion
            emotion_index = preds.argmax()
            label = class_labels[emotion_index]

            # Display Person number
            cv2.putText(
                frame,
                f"Person {person_number}",
                (x, y - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            # Display emotion
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "No Face Found",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    # Display total number of faces
    cv2.putText(
        frame,
        f"People Detected: {len(faces)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # Show camera
    cv2.imshow(
        'Multiple Face Emotion Detector - Press Q to Exit',
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera
cap.release()
cv2.destroyAllWindows()

