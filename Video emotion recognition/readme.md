# Video Emotion Recognition (1D-CNN-LSTM)

This directory contains the implementation for the Video Modality branch of our Multimodal Emotion Recognition project. It focuses on detecting emotions from facial landmark sequences using a **1D-CNN-LSTM** architecture, independent of audio signals.

## Files
* `cnn_lstm_ravdess_video_detection.ipynb`: The main Jupyter Notebook containing data loading, preprocessing, model training, evaluation, and inference pipelines.

## Model Architecture
We treat facial expressions as time-series data. The model processes sequences of 68 facial landmarks (x, y coordinates) over 90 frames (~3 seconds).

1.  **Input:** (90, 136) — Sequence of normalized landmark coordinates.
2.  **Feature Extraction (1D-CNN):** Two layers of 1D-Convolutions (64 & 128 filters) extract local temporal micro-expressions.
3.  **Temporal Dynamics (Bi-LSTM):** A Bidirectional LSTM layer (128 units) captures the evolution of emotion over time.
4.  **Classification:** Dense layer with Softmax for 7 emotion classes.

**Techniques Used:**
* **Label Smoothing:** Mitigates overconfidence on ambiguous expressions.
* **Gaussian Noise Augmentation:** Prevents overfitting to specific actor poses.
* **Geometric Normalization:** Centers and scales landmarks to be invariant to camera distance.

## Dataset
We utilize the **RAVDESS Facial Landmark Tracking** dataset.
* **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-facial-landmark-tracking)
* **Data:** 68 2D facial landmarks extracted via OpenFace 2.1.0.
* **Classes:** Neutral, Happiness, Sadness, Anger, Fear, Disgust, Surprise.

### 1. Requirements
* Python 3.10+
* TensorFlow / Keras
* MediaPipe (for inference on raw video)
* OpenCV, NumPy, Pandas, Scikit-learn

### 2. Training
Run the training cells in the notebook. Ensure the RAVDESS CSV dataset is located in the input directory defined in the `CONFIG` section.

### 3. Inference on New Videos
The notebook includes a full inference pipeline using **MediaPipe Face Mesh**.
1.  Place your video files (`.mp4`) in the test directory.
2.  The script extracts landmarks, normalizes them to match the training format, and generates a second-by-second emotion probability JSON.

## Results
* **Validation Accuracy:** 66% (Speaker-Independent Split)
* **Best Performance:** Happiness (F1: 0.84) and Neutral (F1: 0.76).