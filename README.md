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

---


## 3.3 Running Training

Example:

```
python training.py \
    --train_dir /data/train \
    --val_dir /data/val \
    --epochs 40 \
    --batch_size 8 \
    --lr 1e-4
```

---

## 3.4 Important Training Options

### **Dataset options**

* `--train_dir`: Path to training dataset
* `--val_dir`: Path to validation dataset
* `--labels`: Path to label mapping

### **Training hyperparameters**

* `--epochs`: Number of epochs
* `--batch_size`: Batch size
* `--lr`: Learning rate
* `--weight_decay`: L2 regularization

### **Model + hardware**

* `--device {cuda,cpu}`: Force device
* `--save_every`: Model checkpoint frequency
* `--resume`: Continue from checkpoint

### **Video/audio preprocessing**

* `--video_len`: Number of frames used per training sample
* `--audio_win`: Audio window size (in spectrogram bins)

---

# 4. Inference

## 4.0 Model Path Options

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

Specify the sliding step:

```
python main.py --media_path video.mp4 --step_sec 0.5
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

## 4.4 Important Inference Options

### **Input + processing**

* `--media_path`: Video or audio file (required)
* `--step_sec`: Sliding window step size in seconds

### **Device**

* `--device {cuda,cpu}`: Select GPU or CPU

### **Plotting**

* `--no_plot`: Disable visualization
* `--save_plot <path>`: Save figure to file

---

