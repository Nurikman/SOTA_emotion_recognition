# SOTA_emotion_recognition
Implementation of State-of-the-Art Multimodal Emotion Recognition for Long-Duration Video and Audio

# Setup settings


## Image emotion recognition

## IER Structure

```
├── train_emotion_model_improved.py   # Training script with all improvements
├── video_inference_improved.py       # Video analysis with visualizations
├── test_on_ravdess_fixed.py          # Evaluation on RAVDESS dataset
├── best_model_improved.pth           # Trained model weights/download from Google Drive link
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Model Weights

The pretrained model weights are hosted on Google Drive due to file size (~427MB).

**Download:** [best_model_improved.pth](https://drive.google.com/drive/folders/1NKlADd_xK4hjNa8lxW8gi7eZuLfQyi86?usp=drive_link)

After downloading, place the file in the project root directory:
```
emotion-recognition/
├── best_model_improved.pth  ← place here
├── train_emotion_model_improved.py
├── video_inference_improved.py
└── ...
```

### Hardware
- NVIDIA GPU with CUDA support (recommended)
- Minimum 4GB VRAM for inference
- 8GB+ VRAM for training

### Software
- Python 3.8+
- CUDA 11.x or 12.x (for GPU acceleration)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/emotion-recognition.git
   cd emotion-recognition
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install facenet-pytorch opencv-python numpy pandas matplotlib seaborn scikit-learn tqdm Pillow
   ```

## Usage

### Video Inference

Analyze emotions in a video file:

```bash
python video_inference_improved.py --model best_model_improved.pth --video path/to/video.mp4
```

With visualizations:
```bash
python video_inference_improved.py --model best_model_improved.pth --video path/to/video.mp4 --visualize
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Path to trained model weights | Required |
| `--video` | Path to input video file | Required |
| `--sample-rate` | Process every Nth frame | 10 |
| `--output` | Output CSV filename | Auto-generated |
| `--device` | Device to use (cuda/cpu) | cuda |
| `--visualize` | Generate visualization plots | False |
| `--smooth` | Smoothing window for plots | 3 |

**Output files:**
- `emotions_<video_name>_<timestamp>.csv` - Frame-by-frame emotion predictions
- `emotion_timeline_<video_name>.png` - Temporal emotion plot (with `--visualize`)
- `emotion_distribution_<video_name>.png` - Emotion distribution chart (with `--visualize`)
- `emotion_heatmap_<video_name>.png` - Emotion heatmap (with `--visualize`)

### Evaluation on RAVDESS

Test the model on RAVDESS video dataset:

```bash
python test_on_ravdess_fixed.py --model best_model_improved.pth --ravdess-dir path/to/ravdess/videos
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Path to trained model | Required |
| `--ravdess-dir` | Path to RAVDESS video folder | Required |
| `--max-per-emotion` | Max videos per emotion class | All |
| `--batch-size` | Batch size for evaluation | 16 |
| `--device` | Device to use (cuda/cpu) | cuda |
| `--num-frames` | Frames to sample per video | 1 |

**Output files:**
- `ravdess_confusion_matrix.png` - Confusion matrix visualization
- `ravdess_predictions.csv` - Detailed predictions with probabilities

### Training

Train the model from scratch on FER2013:

1. **Download FER2013** from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

2. **Organize the data:**
   ```
   fer2013_data/
   ├── train/
   │   ├── angry/
   │   ├── disgust/
   │   ├── fear/
   │   ├── happy/
   │   ├── sad/
   │   ├── surprise/
   │   └── neutral/
   └── test/
       └── (same structure)
   ```

3. **Configure and run training:**
   ```bash
   python train_emotion_model_improved.py
   ```

   Edit the `Config` class in the script to adjust hyperparameters.

**Training Features:**
- Weighted Random Sampler for balanced class sampling
- Focal Loss (γ=2.0) for hard example mining
- MixUp augmentation (α=0.2)
- Partial backbone unfreezing (last 2 ResNet blocks)
- Differential learning rates (backbone: 5e-6, head: 5e-5)
- Mixed precision training (FP16)
- ReduceLROnPlateau scheduler
- Early stopping (patience=15)

**Output files:**
- `best_model_improved.pth` - Best model checkpoint
- `training_history.png` - Loss and accuracy curves
- `confusion_matrix.png` - Test set confusion matrix

## Model Architecture

```
ResNetViTHybrid
├── ResNetFeatureExtractor (ResNet-50)
│   ├── conv1, bn1, relu, maxpool
│   ├── layer1 (frozen)
│   ├── layer2 (frozen)
│   ├── layer3 (trainable)
│   └── layer4 (trainable)
│   └── Output: 2048-dim feature maps (7×7)
│
├── SpatialTransformerEncoder
│   ├── Feature Projection: Linear(2048 → 512)
│   ├── Positional Embedding (49 positions)
│   ├── Learnable [CLS] token
│   └── Transformer Encoder (4 layers)
│       ├── Multi-Head Self-Attention (8 heads)
│       ├── Pre-LayerNorm
│       ├── GELU activation
│       └── FFN (512 → 2048 → 512)
│   └── Output: 512-dim [CLS] token embedding
│
└── Classifier
    ├── LayerNorm(512)
    ├── Dropout(0.4)
    ├── Linear(512 → 256)
    ├── GELU
    ├── Dropout(0.3)
    └── Linear(256 → 7)
```

**Total parameters:** ~28M (trainable: ~16M with partial unfreezing)

## Emotion Labels

| Index | Emotion  | FER2013 Distribution |
|-------|----------|---------------------|
| 0     | Angry    | ~12% |
| 1     | Disgust  | ~1.5% (minority) |
| 2     | Fear     | ~12% |
| 3     | Happy    | ~25% (majority) |
| 4     | Sad      | ~15% |
| 5     | Surprise | ~10% |
| 6     | Neutral  | ~15% |

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in training config or use `--batch-size 8`
- Use CPU for inference: `--device cpu`

### No Face Detected in Video
- Ensure good lighting conditions
- Face should be clearly visible and frontal
- MTCNN requires faces to be at least ~20×20 pixels
- Try reducing `--sample-rate` for more frames

### Model Loading Errors
- Ensure the model architecture in inference scripts matches training
- The checkpoint contains: `model_state_dict`, `optimizer_state_dict`, `epoch`, `val_f1`, `val_acc`, `config`

### Slow Video Processing
- Use GPU: `--device cuda`
- Increase sample rate: `--sample-rate 30`
- Process shorter video clips

## RAVDESS Emotion Mapping

RAVDESS uses different emotion codes that are mapped to FER2013 labels:

| RAVDESS Code | RAVDESS Emotion | FER2013 Label |
|--------------|-----------------|---------------|
| 01 | Neutral | Neutral |
| 02 | Calm | Neutral |
| 03 | Happy | Happy |
| 04 | Sad | Sad |
| 05 | Angry | Angry |
| 06 | Fearful | Fear |
| 07 | Disgust | Disgust |
| 08 | Surprised | Surprise |


## Acknowledgments

- FER2013 dataset: Goodfellow et al., 2013
- RAVDESS dataset: Livingstone & Russo, 2018
- PyTorch and torchvision teams
- facenet-pytorch for MTCNN implementation

## Video emotion recognition


## Audio emotion recognition


## Late Fusion emotion recognition
The inference file inference_late_fusion.ipynb contains the code to run. All the required files are inside the folder. To change the video for inference you may change the first line of 3rd block:
```
filename = "/content/emilia.mp4"
```

## Intermediate Fusion emotion recognition

This project implements a lightweight **CNN-based multimodal emotion recognition system** using **video frames + audio spectrograms**. The method uses **feature-level fusion** (concatenating face and audio features) and does **not** rely on Transformers, making it computationally efficient. Based on https://github.com/katerynaCh/multimodal-emotion-recognition

### 1. Overview

The system extracts:

* **Video features** using a 2D CNN applied to sequences of face frames
* **Audio features** using a 1D CNN applied to log-mel spectrograms
* **Concatenates** them into a single feature vector
* Passes the fused vector through a small MLP to predict emotion probabilities

Training and inference both rely on an options file (**opts.py**) to set parameters.

---

### 2. Project Structure

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


#### 3.3 Running Training

Example:

```
python python main.py
```

---

#### 3.4 Important Training Options
--pretrain_path = path to EfficientFace pre-trained model.
--result_path = path where the results and trained models will be saved
--annotation_path = path to annotation file generated at previous step
### 4. Inference

#### 4.0 Model Path Options
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

#### 4.1 What Inference Does

The inference system:

1. Reads a **full video or audio file**
2. Extracts frames and audio
3. Breaks them into sliding windows (e.g., 1-second)
4. Runs the model on each window
5. Produces emotion probabilities for each timestamp

---

#### 4.2 Sliding Window Inference

Each input video is divided into time windows:

```
[0–1s], [1–2s], [2–3s], ...
```

For each window, the model extracts:

* The last 15 face frames
* A 1-second audio spectrogram

Outputs (probabilities + dominant emotion) are stored.

---

### 4.3 Running Inference

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

