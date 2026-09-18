# Real-Time Emotion Detection

A real-time facial emotion detection application built using **Python, OpenCV, CNN, Streamlit, and WebRTC**. The application uses a webcam to detect multiple faces and classify their facial expressions in real time.

## Features

* 🎥 Real-time webcam-based emotion detection
* 👥 Multiple face detection
* 😊 Emotion classification for each detected person
* 📊 Real-time emotion analysis and session reports
* 📈 Emotion count and percentage calculation
* 🏆 Most detected emotion during the report interval
* 👤 Maximum number of people detected simultaneously
* ⏱️ Session/report duration tracking
* 💾 Save emotion analysis reports to a CSV file
* 🌐 Interactive Streamlit web application

## Detected Emotions

The model classifies facial expressions into five categories:

* Angry
* Happy
* Neutral
* Sad
* Surprise

## Technologies Used

* **Python**
* **OpenCV** – Face detection and image processing
* **TensorFlow / Keras** – CNN-based emotion classification
* **NumPy** – Numerical processing
* **Streamlit** – Web application interface
* **Streamlit-WebRTC** – Real-time webcam streaming
* **CSV** – Storing session reports

## Dataset

The project uses the **FER (Facial Expression Recognition) dataset** from Kaggle:

[FER Dataset on Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

## Project Structure

```text
Real-Time-Emotion-Detection/
│
├── app.py
├── Emotion_Detection.h5
├── haarcascade_frontalface_default.xml
├── requirements.txt
├── README.md
└── LICENSE
```

## How It Works

```text
Webcam
   ↓
Video Stream
   ↓
Face Detection using OpenCV
   ↓
Face Extraction
   ↓
48 × 48 Image Preprocessing
   ↓
CNN Emotion Model
   ↓
Emotion Prediction
   ↓
Real-Time Display
   ↓
Session Report
   ↓
CSV Storage
```

## Installation

Clone the repository:

```bash
git clone https://github.com/yuvii-singh/Real-Time-Emotions-Detection.git
```

Move into the project directory:

```bash
cd Real-Time-Emotions-Detection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal and allow webcam access.

## Report Generation

During the session, the application can generate a report containing:

* Current detected people and their emotions
* Maximum simultaneous faces
* Total emotion predictions
* Emotion-wise counts
* Emotion percentages
* Most detected emotion
* Report duration

## CSV Report

The generated report can be saved as:

```text
emotion_reports.csv
```

The CSV contains information such as:

```text
Date
Time
Maximum People
Total Predictions
Angry
Happy
Neutral
Sad
Surprise
Most Detected Emotion
Most Detected Percentage
Session Duration
```

## Important Note

Facial-expression recognition provides an estimate based on visible facial features and should not be treated as a definitive measurement of a person's actual internal emotional state.

## License

This project is licensed under the **MIT License**.

## Author

**Anup Kumar Singh**

GitHub: [yuvii-singh](https://github.com/yuvii-singh)
