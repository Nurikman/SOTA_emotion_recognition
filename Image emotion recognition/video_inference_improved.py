#!/usr/bin/env python3
"""
Video Emotion Inference - Updated for Improved Model
Run this script to analyze emotions in videos using your trained model.

Usage:
    python video_inference_improved.py --model best_model_improved.pth --video test_video.mp4
    python video_inference_improved.py --model best_model_improved.pth --video test_video.mp4 --visualize
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import pandas as pd
from tqdm import tqdm
import argparse
import os
from datetime import datetime
import matplotlib.pyplot as plt


# ============================================================================
# MODEL ARCHITECTURE (Updated to match train_emotion_model_improved.py)
# ============================================================================

class ResNetFeatureExtractor(nn.Module):
    """
    ResNet feature extractor - MUST match training architecture!
    Uses separate layer attributes instead of nn.Sequential
    """
    def __init__(self, backbone='resnet50', pretrained=True, 
                 freeze_backbone=True, unfreeze_layers=2):
        super().__init__()
        
        # Load backbone
        if backbone == 'resnet50':
            resnet = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        elif backbone == 'resnet101':
            resnet = models.resnet101(weights='IMAGENET1K_V2' if pretrained else None)
        else:
            resnet = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        
        # Extract feature layers as SEPARATE attributes (not Sequential!)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.feature_dim = 2048
        
        if freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class SpatialTransformerEncoder(nn.Module):
    """Vision Transformer encoder for spatial attention over CNN features"""
    def __init__(self, feature_dim=2048, embed_dim=512, num_heads=8,
                 num_layers=4, dropout=0.3):
        super().__init__()
        
        self.feature_projection = nn.Linear(feature_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 49, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.feature_projection.weight)
        nn.init.zeros_(self.feature_projection.bias)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, HW, C
        x = self.feature_projection(x)     # B, HW, embed_dim
        x = x + self.pos_embedding[:, :H*W, :]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.dropout(x)
        x = self.transformer(x)
        
        return x[:, 0]  # CLS token


class ResNetViTHybrid(nn.Module):
    """Complete hybrid model - matches train_emotion_model_improved.py"""
    def __init__(self, num_classes=7, pretrained_resnet=True, freeze_resnet=True,
                 embed_dim=512, num_heads=8, num_layers=4, dropout=0.3):
        super().__init__()
        
        self.resnet_extractor = ResNetFeatureExtractor(
            backbone='resnet50',
            pretrained=pretrained_resnet,
            freeze_backbone=freeze_resnet,
            unfreeze_layers=2
        )
        
        self.transformer_encoder = SpatialTransformerEncoder(
            feature_dim=2048,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Classification head with more regularization
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout + 0.1),  # Extra dropout before classifier
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        features = self.resnet_extractor(x)
        encoded = self.transformer_encoder(features)
        logits = self.classifier(encoded)
        return logits


# ============================================================================
# EMOTION VISUALIZATION
# ============================================================================

# Custom color scheme for emotions
EMOTION_COLORS = {
    "happy": "#FFFF00",      # Vibrant Yellow
    "sad": "#0000FF",        # Blue
    "angry": "#FF0000",      # Bright Red
    "surprise": "#00FFFF",   # Light Cyan
    "disgust": "#008000",    # Classic Green
    "fear": "#BB00FF",       # Purple
    "neutral": "#808080"     # Gray
}

# Display names for legend
EMOTION_DISPLAY_NAMES = {
    "happy": "Happy",
    "sad": "Sad", 
    "angry": "Angry",
    "surprise": "Surprised",
    "disgust": "Disgust",
    "fear": "Fearful",
    "neutral": "Neutral"
}


def plot_emotions_over_time(df, output_path='emotion_timeline.png', title='Emotion Analysis Over Time'):
    """
    Plot emotion probabilities over time with custom colors.
    
    Args:
        df: DataFrame with columns ['timestamp', 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        output_path: Path to save the plot
        title: Plot title
    """
    if len(df) == 0:
        print("⚠️  No data to plot!")
        return
    
    # Create figure
    plt.figure(figsize=(14, 6))
    
    # Emotion columns in order
    emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
    
    # Plot each emotion
    for emotion in emotions:
        if emotion in df.columns:
            plt.plot(
                df['timestamp'], 
                df[emotion],
                color=EMOTION_COLORS[emotion],
                linewidth=2.5,
                label=EMOTION_DISPLAY_NAMES[emotion]
            )
    
    # Styling
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Emotion Probability', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.xlim(df['timestamp'].min(), df['timestamp'].max())
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Emotion timeline saved to: {output_path}")


def plot_emotion_distribution(df, output_path='emotion_distribution.png'):
    """
    Plot pie chart of dominant emotions throughout the video.
    """
    if len(df) == 0 or 'dominant_emotion' not in df.columns:
        print("⚠️  No data to plot!")
        return
    
    # Count dominant emotions
    emotion_counts = df['dominant_emotion'].value_counts()
    
    # Prepare colors in same order as counts
    colors = [EMOTION_COLORS[emotion] for emotion in emotion_counts.index]
    labels = [EMOTION_DISPLAY_NAMES[emotion] for emotion in emotion_counts.index]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    wedges, texts, autotexts = ax.pie(
        emotion_counts.values,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.85
    )
    
    # Style the percentage text
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    ax.set_title('Emotion Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Emotion distribution saved to: {output_path}")


def plot_emotion_heatmap(df, output_path='emotion_heatmap.png', window_size=5):
    """
    Plot emotion probabilities as a heatmap over time.
    """
    if len(df) == 0:
        print("⚠️  No data to plot!")
        return
    
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    display_names = [EMOTION_DISPLAY_NAMES[e] for e in emotions]
    
    # Create data matrix
    data = df[emotions].values.T  # Shape: (7, num_frames)
    
    # Smooth with rolling average if enough data
    if len(df) > window_size:
        smoothed = pd.DataFrame(data.T, columns=emotions).rolling(window=window_size, center=True).mean().bfill().ffill()
        data = smoothed.values.T
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 4))
    
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    
    # Labels
    ax.set_yticks(range(len(emotions)))
    ax.set_yticklabels(display_names)
    
    # X-axis: time
    num_ticks = min(10, len(df))
    tick_positions = np.linspace(0, len(df)-1, num_ticks, dtype=int)
    tick_labels = [f"{df['timestamp'].iloc[i]:.1f}s" for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_title('Emotion Intensity Over Time', fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Emotion heatmap saved to: {output_path}")


# ============================================================================
# VIDEO EMOTION DETECTOR
# ============================================================================

class VideoEmotionDetector:
    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def __init__(self, model_path, device='cuda', image_size=224):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = ResNetViTHybrid(
            num_classes=7, 
            freeze_resnet=True,
            embed_dim=512, 
            num_heads=8, 
            num_layers=4,
            dropout=0.3
        )
        
        # Load checkpoint (handle both formats)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Model loaded (from epoch {checkpoint.get('epoch', 'unknown')})")
            if 'val_acc' in checkpoint:
                print(f"  Training val accuracy: {checkpoint['val_acc']:.2%}")
        else:
            self.model.load_state_dict(checkpoint)
            print("✓ Model loaded")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Face detector
        print("Initializing face detector...")
        self.face_detector = MTCNN(
            image_size=image_size, 
            margin=20, 
            keep_all=False,
            device=self.device, 
            post_process=False
        )
        print("✓ Face detector ready!\n")
        
        # Preprocessing (must match training)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def detect_face(self, frame):
        """Detect and crop face from frame"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # CRITICAL: Convert to grayscale then back to 3-channel
            # This matches FER2013 training data which is grayscale
            frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            frame_rgb = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)
            
            boxes, probs = self.face_detector.detect(frame_rgb)
            
            if boxes is not None and len(boxes) > 0:
                box = boxes[0].astype(int)
                h, w = frame_rgb.shape[:2]
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
                face = frame_rgb[y1:y2, x1:x2]
                
                if face.size > 0:
                    return Image.fromarray(face)
            return None
        except:
            return None
    
    def predict_emotion(self, face_image):
        """Predict emotion probabilities from face image"""
        if face_image is None:
            return None
        
        try:
            img_tensor = self.transform(face_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()[0]
        except Exception as e:
            return None
    
    def process_video(self, video_path, sample_rate=10, output_csv=None):
        """Process video and extract frame-by-frame emotions"""
        import time
        start_time = time.time()
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print("=" * 70)
        print("VIDEO INFORMATION")
        print("=" * 70)
        print(f"FPS: {fps:.2f}")
        print(f"Total Frames: {total_frames:,}")
        print(f"Duration: {duration:.2f}s ({duration/60:.2f} min)")
        print(f"Sample Rate: Every {sample_rate} frame(s)")
        print(f"Frames to Process: {total_frames//sample_rate:,}")
        print("=" * 70 + "\n")
        
        results = []
        frame_idx = 0
        faces_detected = 0
        
        pbar = tqdm(total=total_frames//sample_rate, desc="Processing", unit="frame")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                timestamp = frame_idx / fps
                face = self.detect_face(frame)
                
                if face is not None:
                    probs = self.predict_emotion(face)
                    
                    if probs is not None:
                        result = {'timestamp': timestamp, 'frame_number': frame_idx}
                        for i, label in enumerate(self.EMOTION_LABELS):
                            result[label] = float(probs[i])
                        results.append(result)
                        faces_detected += 1
                
                pbar.update(1)
            frame_idx += 1
        
        pbar.close()
        cap.release()
        
        # Calculate timing
        elapsed_time = time.time() - start_time
        frames_processed = total_frames // sample_rate
        
        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE")
        print("=" * 70)
        print(f"Frames processed: {frames_processed:,}")
        print(f"Faces detected: {faces_detected:,}")
        print(f"Detection rate: {(faces_detected/frames_processed*100):.1f}%")
        print(f"Total time: {elapsed_time:.1f}s")
        print(f"Speed: {frames_processed/elapsed_time:.1f} frames/sec")
        print("=" * 70 + "\n")
        
        if len(results) == 0:
            print("⚠️  No faces detected in the video!")
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df['dominant_emotion'] = df[self.EMOTION_LABELS].idxmax(axis=1)
        df['confidence'] = df[self.EMOTION_LABELS].max(axis=1)
        
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"✓ Results saved to: {output_csv}\n")
        
        return df
    
    def print_summary(self, df):
        """Print emotion summary statistics"""
        if len(df) == 0:
            return
        
        emotion_pct = df['dominant_emotion'].value_counts(normalize=True) * 100
        avg_conf = df['confidence'].mean()
        most_common = df['dominant_emotion'].mode()[0]
        
        print("=" * 70)
        print("EMOTION SUMMARY")
        print("=" * 70)
        print(f"\nTotal detections: {len(df)}")
        print(f"Average confidence: {avg_conf:.1%}")
        print(f"Most common emotion: {EMOTION_DISPLAY_NAMES[most_common]}")
        
        print("\nEmotion Distribution:")
        for emotion in self.EMOTION_LABELS:
            if emotion in emotion_pct.index:
                percentage = emotion_pct[emotion]
                bar = '█' * int(percentage / 2)
                print(f"  {EMOTION_DISPLAY_NAMES[emotion]:10s}: {percentage:5.1f}%  {bar}")
        
        print("=" * 70 + "\n")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze emotions in video')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--sample-rate', type=int, default=10, help='Process every Nth frame (default: 10)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file (optional)')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu (default: cuda)')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization plots')
    parser.add_argument('--smooth', type=int, default=3, help='Smoothing window for plots (default: 3)')
    
    args = parser.parse_args()
    
    # Verify files exist
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file not found: {args.video}")
        return
    
    # Auto-generate output filename if not provided
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.output is None:
        args.output = f'emotions_{video_name}_{timestamp}.csv'
    
    print("\n" + "=" * 70)
    print("VIDEO EMOTION ANALYZER (Improved Model)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Video: {args.video}")
    print(f"Sample rate: Every {args.sample_rate} frame(s)")
    print(f"Output: {args.output}")
    print(f"Visualize: {args.visualize}")
    print("=" * 70 + "\n")
    
    # Initialize detector
    detector = VideoEmotionDetector(args.model, device=args.device)
    
    # Process video
    results = detector.process_video(args.video, sample_rate=args.sample_rate, output_csv=args.output)
    
    # Print summary
    detector.print_summary(results)
    
    # Generate visualizations if requested
    if args.visualize and len(results) > 0:
        print("Generating visualizations...")
        
        # Smooth the data for cleaner plots
        if args.smooth > 1 and len(results) > args.smooth:
            smoothed_results = results.copy()
            for emotion in detector.EMOTION_LABELS:
                smoothed_results[emotion] = results[emotion].rolling(
                    window=args.smooth, center=True
                ).mean().bfill().ffill()
        else:
            smoothed_results = results
        
        # Generate plots
        plot_emotions_over_time(
            smoothed_results, 
            output_path=f'emotion_timeline_{video_name}.png',
            title=f'Emotion Analysis: {video_name}'
        )
        
        plot_emotion_distribution(
            results,
            output_path=f'emotion_distribution_{video_name}.png'
        )
        
        plot_emotion_heatmap(
            results,
            output_path=f'emotion_heatmap_{video_name}.png'
        )
    
    if len(results) > 0:
        print(f"\n✓ Analysis complete!")
        print(f"✓ Results saved to: {args.output}")
        print(f"\nFirst 5 predictions:")
        print(results[['timestamp', 'dominant_emotion', 'confidence']].head())


if __name__ == '__main__':
    main()
