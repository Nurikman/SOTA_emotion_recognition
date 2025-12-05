#!/usr/bin/env python3
"""
Test Your Current Model on RAVDESS Subset
Simple script to evaluate performance on RAVDESS videos

FIXED: Updated model architecture to match train_emotion_model_improved.py
       (ResNetFeatureExtractor now uses separate layer attributes instead of Sequential)
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# MODEL ARCHITECTURE (Same as your training)
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
    def __init__(self, feature_dim=2048, embed_dim=512, num_heads=8, 
                 num_layers=4, dropout=0.3):
        super().__init__()
        
        self.feature_projection = nn.Linear(feature_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 49, embed_dim) * 0.02)  # Match training init
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            activation='gelu', batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)  # Match training init
        self.dropout = nn.Dropout(dropout)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.feature_projection.weight)
        nn.init.zeros_(self.feature_projection.bias)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.feature_projection(x)
        x = x + self.pos_embedding[:, :H*W, :]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.dropout(x)
        x = self.transformer(x)
        
        return x[:, 0]


class ResNetViTHybrid(nn.Module):
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
            feature_dim=2048, embed_dim=embed_dim, num_heads=num_heads,
            num_layers=num_layers, dropout=dropout  # Changed from hardcoded 0.2
        )
        
        # UPDATED: Extra dropout before classifier to match training
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout + 0.1),  # UPDATED: Extra dropout (0.4)
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),  # 0.3
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
# RAVDESS VIDEO DATASET
# ============================================================================

class RAVDESSVideoDataset(Dataset):
    """
    RAVDESS Dataset Loader
    
    RAVDESS naming format: XX-YY-ZZ-AA-BB-CC-DD.mp4
    Where ZZ is emotion code:
        01 = neutral
        02 = calm (treat as neutral)
        03 = happy
        04 = sad
        05 = angry
        06 = fearful (fear)
        07 = disgust
        08 = surprised (surprise)
    """
    def __init__(self, video_dir, transform=None, max_videos_per_emotion=None, use_face_detection=True, num_frames=1):
        self.video_dir = Path(video_dir)
        self.transform = transform
        self.max_videos_per_emotion = max_videos_per_emotion
        self.use_face_detection = use_face_detection
        self.num_frames = num_frames  # Number of frames to sample from middle
        
        # Initialize face detector if needed
        if self.use_face_detection:
            try:
                from facenet_pytorch import MTCNN
                self.face_detector = MTCNN(keep_all=False, device='cpu', post_process=False)
                print("✓ Face detector initialized")
            except ImportError:
                print("⚠️  facenet-pytorch not found, disabling face detection")
                print("   Install with: pip install facenet-pytorch")
                self.use_face_detection = False
                self.face_detector = None
        else:
            self.face_detector = None
        
        # RAVDESS emotion mapping
        self.ravdess_to_fer = {
            '01': 6,  # neutral
            '02': 6,  # calm → neutral
            '03': 3,  # happy
            '04': 4,  # sad
            '05': 0,  # angry
            '06': 2,  # fearful → fear
            '07': 1,  # disgust
            '08': 5,  # surprised → surprise
        }
        
        self.emotion_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Load video paths
        self.samples = []
        self._load_videos()
        
        print(f"✓ Loaded {len(self.samples)} RAVDESS videos")
        self._print_distribution()
    
    def _load_videos(self):
        """Find all RAVDESS video files"""
        video_files = list(self.video_dir.glob("**/*.mp4")) + \
                     list(self.video_dir.glob("**/*.avi"))
        
        # Group by emotion
        emotion_groups = {i: [] for i in range(7)}
        
        for video_path in video_files:
            # Parse RAVDESS filename: XX-YY-ZZ-AA-BB-CC-DD.mp4
            filename = video_path.stem
            parts = filename.split('-')
            
            if len(parts) >= 3:
                emotion_code = parts[2]  # ZZ part
                
                if emotion_code in self.ravdess_to_fer:
                    label = self.ravdess_to_fer[emotion_code]
                    emotion_groups[label].append(str(video_path))
        
        # Limit videos per emotion if specified
        for emotion_idx, videos in emotion_groups.items():
            if self.max_videos_per_emotion:
                videos = videos[:self.max_videos_per_emotion]
            
            for video_path in videos:
                self.samples.append((video_path, emotion_idx))
    
    def _print_distribution(self):
        """Print emotion distribution"""
        counts = [0] * 7
        for _, label in self.samples:
            counts[label] += 1
        
        print("\nEmotion distribution:")
        for emotion, count in zip(self.emotion_names, counts):
            print(f"  {emotion:10s}: {count:3d}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        # Extract frame(s) from middle section
        if self.num_frames == 1:
            # Single frame (original approach)
            frame = self.extract_middle_frame(video_path)
            if self.transform:
                frame = self.transform(frame)
            return frame, label
        else:
            # Multiple frames from middle section
            frames = self.extract_middle_frames(video_path, self.num_frames)
            if self.transform:
                frames = [self.transform(f) for f in frames]
            # Stack frames: (num_frames, C, H, W)
            frames = torch.stack(frames) if len(frames) > 0 else torch.zeros(self.num_frames, 3, 224, 224)
            return frames, label
    
    def extract_middle_frame(self, video_path):
        """Extract the middle frame from video with proper preprocessing"""
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_idx = total_frames // 2
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # Fallback: return blank image
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            frame = self.preprocess_frame(frame)
        
        return Image.fromarray(frame)
    
    def extract_middle_frames(self, video_path, num_frames=5):
        """
        Extract multiple frames from MIDDLE 60% of video (peak expression)
        Avoids beginning and end where emotion transitions happen
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample from middle 60% of video (skip first 20% and last 20%)
        start_frame = int(total_frames * 0.2)  # Skip first 20%
        end_frame = int(total_frames * 0.8)     # Skip last 20%
        
        # Sample frames uniformly from this middle section
        frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                processed_frame = self.preprocess_frame(frame)
                frames.append(Image.fromarray(processed_frame))
        
        cap.release()
        
        # If no frames extracted, return blank
        if len(frames) == 0:
            blank = np.zeros((224, 224, 3), dtype=np.uint8)
            frames = [Image.fromarray(blank)] * num_frames
        
        return frames
    
    def preprocess_frame(self, frame):
        """Preprocess frame with face detection and grayscale conversion"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect and crop face (if face detector available)
        if self.use_face_detection and self.face_detector is not None:
            try:
                boxes, probs = self.face_detector.detect(frame)
                if boxes is not None and len(boxes) > 0:
                    # Crop to face bounding box
                    box = boxes[0].astype(int)
                    h, w = frame.shape[:2]
                    x1 = max(0, box[0])
                    y1 = max(0, box[1])
                    x2 = min(w, box[2])
                    y2 = min(h, box[3])
                    
                    if x2 > x1 and y2 > y1:
                        frame = frame[y1:y2, x1:x2]
            except Exception as e:
                # If face detection fails, continue with full frame
                pass
        
        # CRITICAL: Match FER2013 training preprocessing
        # FER2013 is grayscale → convert to grayscale then back to 3-channel
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)
        
        return frame


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_on_ravdess(model, test_loader, device, num_frames=1):
    """Evaluate model on RAVDESS dataset"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nEvaluating on RAVDESS...")
    with torch.no_grad():
        for batch_data, labels in tqdm(test_loader, desc="Testing"):
            labels = labels.to(device)
            
            if num_frames == 1:
                # Single frame per video
                images = batch_data.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
            else:
                # Multiple frames per video - aggregate predictions
                batch_size = batch_data.shape[0]
                # batch_data shape: (batch_size, num_frames, C, H, W)
                
                batch_predictions = []
                batch_probs = []
                
                for i in range(batch_size):
                    video_frames = batch_data[i].to(device)  # (num_frames, C, H, W)
                    
                    # Predict on each frame
                    frame_outputs = model(video_frames)
                    frame_probs = torch.softmax(frame_outputs, dim=1)
                    
                    # Aggregate: Average probabilities across frames
                    avg_probs = frame_probs.mean(dim=0)  # (7,)
                    
                    # Final prediction
                    predicted_class = avg_probs.argmax()
                    
                    batch_predictions.append(predicted_class)
                    batch_probs.append(avg_probs)
                
                predicted = torch.tensor(batch_predictions, device=device)
                probs = torch.stack(batch_probs)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    return all_preds, all_labels, all_probs, accuracy


def plot_results(y_true, y_pred, save_path='ravdess_results.png'):
    """Plot confusion matrix and per-class results"""
    emotion_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_names, yticklabels=emotion_names,
                ax=ax1)
    ax1.set_title('Confusion Matrix - RAVDESS', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Per-class accuracy
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, target_names=emotion_names, 
                                   output_dict=True, zero_division=0)
    
    emotions = []
    accuracies = []
    for emotion in emotion_names:
        if emotion in report:
            emotions.append(emotion)
            accuracies.append(report[emotion]['recall'] * 100)
    
    colors = ['red', 'green', 'purple', 'gold', 'blue', 'orange', 'gray']
    ax2.barh(emotions, accuracies, color=colors)
    ax2.set_xlabel('Recall (%)', fontsize=12)
    ax2.set_title('Per-Class Performance', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 100])
    
    for i, v in enumerate(accuracies):
        ax2.text(v + 2, i, f'{v:.1f}%', va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Results plot saved to: {save_path}")
    plt.close()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test model on RAVDESS subset')
    parser.add_argument('--model', type=str, required=True, help='Path to model (.pth file)')
    parser.add_argument('--ravdess-dir', type=str, required=True, help='Path to RAVDESS video folder')
    parser.add_argument('--max-per-emotion', type=int, default=None, 
                       help='Max videos per emotion (None = use all)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--num-frames', type=int, default=1, 
                       help='Number of frames to sample from middle of each video (1=single frame, 3-7=multi-frame)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("TESTING ON RAVDESS DATASET")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"RAVDESS dir: {args.ravdess_dir}")
    print(f"Max per emotion: {args.max_per_emotion if args.max_per_emotion else 'All'}")
    print(f"Frames per video: {args.num_frames} {'(single frame)' if args.num_frames == 1 else '(multi-frame from middle 60%)'}")
    print(f"Device: {device}")
    print("="*70 + "\n")
    
    # Load model
    print("Loading model...")
    model = ResNetViTHybrid(num_classes=7, freeze_resnet=True, dropout=0.3)
    
    # Load checkpoint (handle both old and new formats)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    # Check if it's a full checkpoint or just state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format: full checkpoint with metadata
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded (checkpoint from epoch {checkpoint.get('epoch', 'unknown')})")
        if 'val_acc' in checkpoint:
            print(f"  Training val accuracy: {checkpoint['val_acc']:.2%}")
        if 'val_f1' in checkpoint:
            print(f"  Training val F1: {checkpoint['val_f1']:.4f}")
    else:
        # Old format: just state_dict
        model.load_state_dict(checkpoint)
        print("✓ Model loaded")
    
    model = model.to(device)
    model.eval()
    print()
    
    # Create dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = RAVDESSVideoDataset(
        args.ravdess_dir,
        transform=transform,
        max_videos_per_emotion=args.max_per_emotion,
        use_face_detection=True,  # Enable face detection for better accuracy
        num_frames=args.num_frames  # Number of frames to sample
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Evaluate
    preds, labels, probs, accuracy = evaluate_on_ravdess(model, test_loader, device, num_frames=args.num_frames)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    print(f"Total samples: {len(labels)}")
    
    # Classification report
    emotion_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    print("\nDetailed Classification Report:")
    print(classification_report(labels, preds, target_names=emotion_names, zero_division=0))
    
    # Plot results
    plot_results(labels, preds, 'ravdess_confusion_matrix.png')
    
    # Save predictions
    import pandas as pd
    results_df = pd.DataFrame({
        'true_label': [emotion_names[l] for l in labels],
        'predicted_label': [emotion_names[p] for p in preds],
        'correct': [l == p for l, p in zip(labels, preds)]
    })
    
    # Add probabilities
    for i, emotion in enumerate(emotion_names):
        results_df[f'prob_{emotion}'] = [p[i] for p in probs]
    
    results_df.to_csv('ravdess_predictions.csv', index=False)
    print("\n✓ Predictions saved to: ravdess_predictions.csv")
    
    print("\n" + "="*70)
    print("TESTING COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
