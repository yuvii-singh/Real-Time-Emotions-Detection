import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import time
from collections import Counter
from threading import Lock

import csv
from datetime import datetime
import os


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Real-Time Emotion Detection",
    page_icon="😊",
    layout="wide"
)

st.title("😊 Real-Time Emotion Detection")

st.write(
    "Start the camera to detect Real-time emotions."
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_emotion_model():

    return load_model(
        "./Emotion_Detection.h5"
    )


@st.cache_resource
def load_face_detector():

    return cv2.CascadeClassifier(
        "./haarcascade_frontalface_default.xml"
    )


classifier = load_emotion_model()

face_classifier = load_face_detector()


# =========================================================
# EMOTION LABELS
# =========================================================

class_labels = [
    "Angry",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


# =========================================================
# VIDEO PROCESSOR
# =========================================================

class EmotionDetector(VideoProcessorBase):

    def __init__(self):

        self.lock = Lock()

        # Current people and emotions
        self.current_people = {}

        # Maximum simultaneous people
        self.max_faces = 0

        # Total predictions
        self.total_predictions = 0

        # Overall emotion counts
        self.emotion_counts = Counter()

        # Start time
        self.start_time = time.time()

        # Time of last report
        self.last_report_time = time.time()


    def recv(self, frame):

        # Convert WebRTC frame
        img = frame.to_ndarray(
            format="bgr24"
        )

        # Grayscale
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        # Detect faces
        faces = face_classifier.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        # Update maximum people
        with self.lock:

            if len(faces) > self.max_faces:

                self.max_faces = len(faces)

            self.current_people = {}


        # =================================================
        # PROCESS EVERY PERSON
        # =================================================

        for person_number, (x, y, w, h) in enumerate(
            faces,
            start=1
        ):

            # Draw rectangle
            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            # Extract face
            roi_gray = gray[
                y:y+h,
                x:x+w
            ]

            # Resize
            roi_gray = cv2.resize(
                roi_gray,
                (48, 48),
                interpolation=cv2.INTER_AREA
            )

            if np.sum(roi_gray) != 0:

                # Normalize
                roi = (
                    roi_gray.astype("float32")
                    / 255.0
                )

                # Convert to array
                roi = img_to_array(
                    roi
                )

                # Add batch dimension
                roi = np.expand_dims(
                    roi,
                    axis=0
                )

                # Predict
                preds = classifier.predict(
                    roi,
                    verbose=0
                )[0]

                # Get emotion
                emotion_index = preds.argmax()

                label = class_labels[
                    emotion_index
                ]

                # Save current person's emotion
                with self.lock:

                    self.current_people[
                        person_number
                    ] = label

                    self.emotion_counts[
                        label
                    ] += 1

                    self.total_predictions += 1


                # Person number
                cv2.putText(
                    img,
                    f"Person {person_number}",
                    (x, y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                # Emotion
                cv2.putText(
                    img,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )


        # =================================================
        # NUMBER OF PEOPLE
        # =================================================

        cv2.putText(
            img,
            f"People Detected: {len(faces)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )


        # Return frame
        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# =========================================================
# CAMERA
# =========================================================

ctx = webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionDetector,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
    video_html_attrs={
        "style": {
            "width": "550px",
            "height": "400px",
            "margin": "0 auto"
        },
        "controls": False,
        "autoPlay": True,
    }
)


# =========================================================
# REPORT BUTTON
# =========================================================

st.divider()

st.subheader("📊 Generate Report")


if "report" not in st.session_state:

    st.session_state.report = None


# =========================================================
# GENERATE CURRENT REPORT
# =========================================================

if st.button(
    "📋 Generate Current Report",
    use_container_width=True
):

    if ctx.video_processor is not None:

        processor = ctx.video_processor

        # Safely copy current data
        with processor.lock:

            current_people = dict(
                processor.current_people
            )

            max_faces = processor.max_faces

            total_predictions = (
                processor.total_predictions
            )

            emotion_counts = dict(
                processor.emotion_counts
            )

            last_report_time = (
                processor.last_report_time
            )

            current_time = time.time()

            # Reset counters for the next report
            processor.emotion_counts.clear()

            processor.total_predictions = 0

            processor.max_faces = 0

            processor.last_report_time = current_time


        # Save snapshot
        st.session_state.report = {

            "people": current_people,

            "max_faces": max_faces,

            "total_predictions":
                total_predictions,

            "emotion_counts":
                emotion_counts,

            "start_time":
                last_report_time,

            "report_time":
                current_time
        }

        st.success(
            "✅ Report captured successfully!"
        )

    else:

        st.warning(
            "⚠️ Please click START first."
        )


# =========================================================
# DISPLAY REPORT
# =========================================================

if st.session_state.report is not None:

    report = st.session_state.report

    st.divider()

    st.header(
        "📊 Emotion Detection Report"
    )


    # =====================================================
    # CURRENT PEOPLE
    # =====================================================

    st.subheader(
        "👥 People Detected"
    )

    people = report["people"]


    if len(people) > 0:

        for person, emotion in people.items():

            st.write(
                f"👤 **Person {person}:** "
                f"😊 **{emotion}**"
            )

    else:

        st.warning(
            "No face was detected when the report was generated."
        )


    # =====================================================
    # MAXIMUM PEOPLE
    # =====================================================

    st.subheader(
        "👥 Maximum People Detected"
    )

    st.metric(
        "Maximum Simultaneous Faces",
        report["max_faces"]
    )


    # =====================================================
    # TOTAL PREDICTIONS
    # =====================================================

    st.metric(
        "Total Emotion Predictions",
        report["total_predictions"]
    )


    # =====================================================
    # EMOTION ANALYSIS
    # =====================================================

    st.subheader(
        "😊 Overall Emotion Analysis"
    )

    emotion_counts = report[
        "emotion_counts"
    ]

    total = report[
        "total_predictions"
    ]


    if total > 0:

        for emotion in class_labels:

            count = emotion_counts.get(
                emotion,
                0
            )

            percentage = (
                count / total
            ) * 100

            st.write(
                f"**{emotion}:** "
                f"{count} predictions "
                f"({percentage:.2f}%)"
            )

            st.progress(
                min(
                    int(percentage),
                    100
                )
            )


        # =================================================
        # MAXIMUM / MOST DETECTED EMOTION
        # =================================================

        most_common = sorted(
            emotion_counts.items(),
            key=lambda item: item[1],
            reverse=True
        )


        if most_common:

            max_emotion = (
                most_common[0][0]
            )

            max_count = (
                most_common[0][1]
            )

            max_percentage = (
                max_count / total
            ) * 100


            st.success(
                f"🏆 Most Detected Emotion: "
                f"**{max_emotion}** "
                f"({max_percentage:.2f}%)"
            )


    else:

        st.warning(
            "No emotion predictions available."
        )


    # =====================================================
    # DURATION
    # =====================================================

    duration = (
        report["report_time"]
        - report["start_time"]
    )

    minutes = int(
        duration // 60
    )

    seconds = int(
        duration % 60
    )


    st.metric(
        "⏱️ Session Duration",
        f"{minutes} min {seconds} sec"
    )


    # =====================================================
    # SAVE REPORT TO CSV
    # =====================================================

    st.divider()

    st.subheader(
        "💾 Save Report"
    )


    if st.button(
        "💾 Save Report to CSV",
        use_container_width=True
    ):

        file_name = "emotion_reports.csv"

        file_exists = os.path.exists(
            file_name
        )


        emotion_counts = report[
            "emotion_counts"
        ]

        total = report[
            "total_predictions"
        ]


        # Find most detected emotion
        most_common = sorted(
            emotion_counts.items(),
            key=lambda item: item[1],
            reverse=True
        )


        if most_common and total > 0:

            most_detected_emotion = (
                most_common[0][0]
            )

            most_detected_percentage = (
                most_common[0][1]
                / total
            ) * 100

        else:

            most_detected_emotion = "None"

            most_detected_percentage = 0


        # Session duration
        duration = (
            report["report_time"]
            - report["start_time"]
        )

        minutes = int(
            duration // 60
        )

        seconds = int(
            duration % 60
        )


        # Current date and time
        current_datetime = datetime.now()


        # Open CSV file
        with open(
            file_name,
            mode="a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)


            # Create header if file does not exist
            if not file_exists:

                writer.writerow([
                    "Date",
                    "Time",
                    "Maximum People",
                    "Total Predictions",
                    "Angry",
                    "Happy",
                    "Neutral",
                    "Sad",
                    "Surprise",
                    "Most Detected Emotion",
                    "Most Detected Percentage",
                    "Session Duration"
                ])


            # Save report data
            writer.writerow([

                current_datetime.strftime(
                    "%Y-%m-%d"
                ),

                current_datetime.strftime(
                    "%H:%M:%S"
                ),

                report["max_faces"],

                report["total_predictions"],

                emotion_counts.get(
                    "Angry",
                    0
                ),

                emotion_counts.get(
                    "Happy",
                    0
                ),

                emotion_counts.get(
                    "Neutral",
                    0
                ),

                emotion_counts.get(
                    "Sad",
                    0
                ),

                emotion_counts.get(
                    "Surprise",
                    0
                ),

                most_detected_emotion,

                f"{most_detected_percentage:.2f}%",

                f"{minutes} min {seconds} sec"
            ])


        st.success(
            "✅ Report saved successfully!"
        )

        st.info(
            f"📁 Saved as: {file_name}"
        )