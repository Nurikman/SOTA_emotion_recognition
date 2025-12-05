# SOTA_emotion_recognition
Implementation of State-of-the-Art Multimodal Emotion Recognition for Long-Duration Video and Audio

# Setup settings


## Image emotion recognition


## Video emotion recognition


## Audio emotion recognition


## Late Fusion emotion recognition
The inference file inference_late_fusion.ipynb contains the code to run. All the required files are inside the folder. To change the video for inference you may change the first line of 3rd block:
```
filename = "/content/emilia.mp4"
```

## Intermediate Fusion emotion recognition

This project implements a lightweight **CNN-based multimodal emotion recognition system** using **video frames + audio spectrograms**. The method uses **feature-level fusion** (concatenating face and audio features) and does **not** rely on Transformers, making it computationally efficient. Based on https://github.com/katerynaCh/multimodal-emotion-recognition

# 1. Overview

The system extracts:

* **Video features** using a 2D CNN applied to sequences of face frames
* **Audio features** using a 1D CNN applied to log-mel spectrograms
* **Concatenates** them into a single feature vector
* Passes the fused vector through a small MLP to predict emotion probabilities

Training and inference both rely on an options file (**opts.py**) to set parameters.

---

# 2. Project Structure

```
project/
│── main.py                 # Entry point for inference
│── opts.py                 # CLI options
│── training.py             # Training procedure
│── validation.py           # Validation procedure
│── inference.py            # TrainingAlignedInference class
│── models/                 # 1D and 2D CNN encoders
│── results/                # Saved model weights
```
and others
---


## 3.3 Running Training

Example:

```
python python main.py
```

---

## 3.4 Important Training Options
--pretrain_path = path to EfficientFace pre-trained model.
--result_path = path where the results and trained models will be saved
--annotation_path = path to annotation file generated at previous step
# 4. Inference

## 4.0 Model Path Options
To toggle only inference mode write:
```
--only_inference
```
You can override the default model paths directly from the command line:

```
--model_path path/to/model.pth
--efficientface_path path/to/EfficientFace.pth.tar
```

These values are passed into `main.py` → `TrainingAlignedInference` so they correctly update the model loading inside **inference.py**.

## 4.1 What Inference Does

The inference system:

1. Reads a **full video or audio file**
2. Extracts frames and audio
3. Breaks them into sliding windows (e.g., 1-second)
4. Runs the model on each window
5. Produces emotion probabilities for each timestamp

---

## 4.2 Sliding Window Inference

Each input video is divided into time windows:

```
[0–1s], [1–2s], [2–3s], ...
```

For each window, the model extracts:

* The last 15 face frames
* A 1-second audio spectrogram

Outputs (probabilities + dominant emotion) are stored.

---

# 4.3 Running Inference

Basic example:

```
python main.py --media_path path/to/video.mp4
```


Disable plotting:

```
python main.py --media_path video.mp4 --no_plot
```

Save a plot:

```
python main.py --media_path video.mp4 --save_plot results/plot.png
```

---

