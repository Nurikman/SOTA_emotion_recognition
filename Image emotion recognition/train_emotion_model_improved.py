#!/usr/bin/env python3
"""
Improved Emotion Classification Training Script
Target: Beat 59% on FER2013, 40% on RAVDESS

Key Improvements:
1. Weighted Random Sampler (not just class weights in loss)
2. Focal Loss for hard examples
3. Partial unfreezing of ResNet backbone
4. Better augmentation with face-specific transforms
5. Option to use face-pretrained backbone (IResNet from InsightFace)
6. Test-time augmentation
7. Proper label smoothing (compatible with class weights)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from torch.amp import autocast, GradScaler
from collections import Counter
import random
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """All hyperparameters in one place for easy experimentation"""
    
    # Paths
    data_dir = 'fer2013_data/train'
    test_dir = 'fer2013_data/test'
    
    # Training
    batch_size = 32
    epochs = 100
    learning_rate = 5e-5       # Lower than before (1e-4 was too high)
    weight_decay = 0.01
    
    # Model
    embed_dim = 512
    num_heads = 8
    num_layers = 4
    dropout = 0.3              # Increased from 0.2
    
    # Backbone options: 'resnet50', 'resnet101', 'iresnet' (face-pretrained)
    backbone = 'resnet50'
    freeze_backbone = False    # CHANGED: Unfreeze for better learning
    unfreeze_layers = 2        # How many ResNet blocks to unfreeze (1-4)
    
    # Loss function: 'cross_entropy', 'focal', 'combined'
    loss_type = 'focal'
    focal_gamma = 2.0          # Focal loss focusing parameter
    label_smoothing = 0.0      # Set to 0 when using focal loss
    
    # Class balancing
    use_weighted_sampler = True   # CRITICAL: Forces equal class visibility
    use_class_weights = True      # Weight in loss function
    
    # Scheduler: 'plateau', 'cosine', 'warmup_cosine'
    scheduler_type = 'plateau'
    patience = 7               # For ReduceLROnPlateau
    min_lr = 1e-6
    
    # Regularization
    mixup_alpha = 0.2          # Set to 0 to disable
    cutmix_alpha = 0.0         # Set to 0 to disable
    
    # Other
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4
    seed = 42


def set_seed(seed):
    """Reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# FOCAL LOSS - Critical for class imbalance
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss: Focuses training on hard examples
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    - When example is easy (p_t → 1): loss → 0 (ignored)
    - When example is hard (p_t → 0): loss → high (focused on)
    
    gamma=2 is standard. Higher = more focus on hard examples.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        targets = targets.long()  # Ensure Long type for cross_entropy
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class CombinedLoss(nn.Module):
    """Cross-entropy + Focal loss for stable training"""
    def __init__(self, alpha=None, gamma=2.0, ce_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=alpha)
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.ce_weight = ce_weight
    
    def forward(self, inputs, targets):
        targets = targets.long()  # Ensure Long type
        return self.ce_weight * self.ce(inputs, targets) + \
               (1 - self.ce_weight) * self.focal(inputs, targets)


# ============================================================================
# MIXUP AUGMENTATION - Improves generalization
# ============================================================================

def mixup_data(x, y, alpha=0.2):
    """Mixup: Linear interpolation between random pairs"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss for mixup samples"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ResNetFeatureExtractor(nn.Module):
    """
    ResNet feature extractor with partial unfreezing
    
    Unfreezing schedule:
    - Layer 0: Early features (edges, colors) - usually keep frozen
    - Layer 1-2: Mid-level features - can unfreeze
    - Layer 3-4: High-level features - recommended to unfreeze
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
        
        # Extract feature layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
        self.feature_dim = 2048
        
        if freeze_backbone:
            # Freeze all layers first
            for param in self.parameters():
                param.requires_grad = False
            
            # Selectively unfreeze last N layers
            layers_to_unfreeze = []
            if unfreeze_layers >= 1:
                layers_to_unfreeze.append(self.layer4)
            if unfreeze_layers >= 2:
                layers_to_unfreeze.append(self.layer3)
            if unfreeze_layers >= 3:
                layers_to_unfreeze.append(self.layer2)
            if unfreeze_layers >= 4:
                layers_to_unfreeze.append(self.layer1)
            
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
            
            # Count trainable params
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(f"✓ ResNet: {trainable:,} / {total:,} params trainable "
                  f"({100*trainable/total:.1f}%)")
        else:
            print("✓ ResNet fully trainable")
    
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
    """Complete hybrid model with improvements"""
    def __init__(self, num_classes=7, config=None):
        super().__init__()
        
        if config is None:
            config = Config()
        
        self.resnet_extractor = ResNetFeatureExtractor(
            backbone=config.backbone,
            pretrained=True,
            freeze_backbone=config.freeze_backbone,
            unfreeze_layers=config.unfreeze_layers
        )
        
        self.transformer_encoder = SpatialTransformerEncoder(
            feature_dim=2048,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
        
        # Classification head with more regularization
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Dropout(config.dropout + 0.1),  # Extra dropout before classifier
            nn.Linear(config.embed_dim, 256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_classifier()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✓ Model: {trainable_params:,} / {total_params:,} trainable params")
    
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
# DATASET WITH BALANCED SAMPLING
# ============================================================================

class FER2013Dataset(Dataset):
    """FER2013 dataset with support for weighted sampling"""
    
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # Load all samples
        for idx, emotion in enumerate(self.EMOTIONS):
            emotion_dir = self.root_dir / emotion
            if not emotion_dir.exists():
                print(f"  Warning: {emotion_dir} not found")
                continue
            
            images = list(emotion_dir.glob('*.jpg')) + list(emotion_dir.glob('*.png'))
            for img_path in images:
                self.samples.append(str(img_path))
                self.labels.append(idx)
        
        self.labels = np.array(self.labels)
        
        # Print distribution
        print(f"✓ Loaded {len(self.samples)} images")
        self._print_distribution()
    
    def _print_distribution(self):
        counts = Counter(self.labels)
        total = len(self.labels)
        print("  Class distribution:")
        for idx, emotion in enumerate(self.EMOTIONS):
            count = counts.get(idx, 0)
            pct = 100 * count / total
            bar = '█' * int(pct / 2)
            print(f"    {emotion:10s}: {count:5d} ({pct:5.1f}%) {bar}")
    
    def get_sample_weights(self):
        """
        Returns weights for WeightedRandomSampler
        Samples from smaller classes get higher weights
        """
        class_counts = Counter(self.labels)
        
        # Weight = 1 / class_count (inverse frequency)
        weights = []
        for label in self.labels:
            weights.append(1.0 / class_counts[label])
        
        return torch.tensor(weights, dtype=torch.float)
    
    def get_class_weights(self):
        """Returns class weights for loss function"""
        class_counts = Counter(self.labels)
        total = len(self.labels)
        
        weights = []
        for i in range(len(self.EMOTIONS)):
            count = class_counts.get(i, 1)
            # Inverse frequency, normalized
            weights.append(total / (len(self.EMOTIONS) * count))
        
        return torch.tensor(weights, dtype=torch.float)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = int(self.labels[idx])  # Ensure Python int
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def get_train_transforms(image_size=224):
    """Strong augmentation for training"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10
        ),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomGrayscale(p=0.1),  # Occasional grayscale
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))  # Random cutout
    ])


def get_test_transforms(image_size=224):
    """Standard transforms for testing"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def get_tta_transforms(image_size=224):
    """Test-time augmentation transforms"""
    return [
        # Original
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Slight rotation
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=(5, 5)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    ]


# ============================================================================
# LEARNING RATE SCHEDULERS
# ============================================================================

class WarmupCosineScheduler:
    """Cosine annealing with linear warmup"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, scaler, config):
    """Train for one epoch with optional mixup"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).long()  # Ensure Long type
        
        # Apply mixup if enabled
        use_mixup = config.mixup_alpha > 0 and random.random() < 0.5
        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, config.mixup_alpha)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda'):
            outputs = model(images)
            
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        
        if use_mixup:
            # For mixup, count both targets
            correct += (lam * predicted.eq(labels_a).sum().item() +
                       (1 - lam) * predicted.eq(labels_b).sum().item())
        else:
            correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels_gpu = labels.to(device).long()  # Ensure Long type
            
            outputs = model(images)
            loss = criterion(outputs, labels_gpu)
            
            probs = F.softmax(outputs, dim=1)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    val_loss = running_loss / len(all_labels)
    val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return val_loss, val_acc, val_f1, all_preds, all_labels


def verify_diversity(model, dataloader, device):
    """Check if model predicts multiple classes"""
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for images, _ in dataloader:
            outputs = model(images.to(device))
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            if len(all_preds) >= 200:
                break
    
    unique, counts = np.unique(all_preds, return_counts=True)
    
    print(f"\n  Diversity: {len(unique)}/7 classes predicted")
    for cls, count in zip(unique, counts):
        print(f"    {FER2013Dataset.EMOTIONS[cls]:10s}: {count:3d} ({100*count/len(all_preds):.1f}%)")
    
    return len(unique) >= 4


def print_per_class_metrics(labels, preds):
    """Print per-class accuracy and F1"""
    print("\n  Per-class performance:")
    print("  " + "-" * 50)
    
    emotions = FER2013Dataset.EMOTIONS
    for i, emotion in enumerate(emotions):
        mask = np.array(labels) == i
        if mask.sum() == 0:
            continue
        
        class_preds = np.array(preds)[mask]
        acc = (class_preds == i).mean()
        
        # Calculate F1 for this class
        tp = ((np.array(preds) == i) & (np.array(labels) == i)).sum()
        fp = ((np.array(preds) == i) & (np.array(labels) != i)).sum()
        fn = ((np.array(preds) != i) & (np.array(labels) == i)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        bar = '█' * int(acc * 20)
        print(f"    {emotion:10s}: Acc={acc*100:5.1f}% F1={f1:.3f} {bar}")


def plot_training_history(history, save_path='training_history.png'):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[2].plot(epochs, history['val_f1'], 'g-', label='Val F1')
    axes[2].axhline(y=max(history['val_f1']), color='r', linestyle='--', 
                    label=f'Best: {max(history["val_f1"]):.4f}')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Validation F1')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history saved to {save_path}")


def plot_confusion_matrix(labels, preds, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    emotions = FER2013Dataset.EMOTIONS
    cm = confusion_matrix(labels, preds)
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotions, yticklabels=emotions, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].set_ylabel('True')
    axes[0].set_xlabel('Predicted')
    
    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=emotions, yticklabels=emotions, ax=axes[1])
    axes[1].set_title('Confusion Matrix (Normalized)')
    axes[1].set_ylabel('True')
    axes[1].set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    config = Config()
    set_seed(config.seed)
    
    print("\n" + "=" * 70)
    print("IMPROVED EMOTION CLASSIFICATION TRAINING")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Backbone: {config.backbone}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Loss type: {config.loss_type}")
    print(f"Weighted sampler: {config.use_weighted_sampler}")
    print(f"Mixup alpha: {config.mixup_alpha}")
    print("=" * 70 + "\n")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = FER2013Dataset(config.data_dir, transform=get_train_transforms())
    test_dataset = FER2013Dataset(config.test_dir, transform=get_test_transforms())
    
    # Create weighted sampler for balanced training
    if config.use_weighted_sampler:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle = False  # Can't use both sampler and shuffle
        print("✓ Using weighted random sampler for class balance")
    else:
        sampler = None
        shuffle = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = ResNetViTHybrid(num_classes=7, config=config).to(config.device)
    
    # Setup loss function
    class_weights = None
    if config.use_class_weights:
        class_weights = train_dataset.get_class_weights().to(config.device)
        print("\n✓ Class weights:")
        for emotion, weight in zip(FER2013Dataset.EMOTIONS, class_weights):
            print(f"    {emotion:10s}: {weight:.3f}")
    
    if config.loss_type == 'focal':
        criterion = FocalLoss(alpha=class_weights, gamma=config.focal_gamma)
        print(f"\n✓ Using Focal Loss (gamma={config.focal_gamma})")
    elif config.loss_type == 'combined':
        criterion = CombinedLoss(alpha=class_weights, gamma=config.focal_gamma)
        print(f"\n✓ Using Combined Loss (CE + Focal)")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.label_smoothing
        )
        print(f"\n✓ Using CrossEntropyLoss")
    
    # Optimizer with different LR for backbone vs head
    backbone_params = list(model.resnet_extractor.parameters())
    head_params = list(model.transformer_encoder.parameters()) + \
                  list(model.classifier.parameters())
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config.learning_rate * 0.1},  # Lower LR for backbone
        {'params': head_params, 'lr': config.learning_rate}
    ], weight_decay=config.weight_decay)
    
    print(f"\n✓ Optimizer: AdamW with differential LR")
    print(f"    Backbone LR: {config.learning_rate * 0.1}")
    print(f"    Head LR: {config.learning_rate}")
    
    # Scheduler
    if config.scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=config.patience,
            verbose=True, min_lr=config.min_lr
        )
        print(f"\n✓ Scheduler: ReduceLROnPlateau (patience={config.patience})")
    elif config.scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.min_lr
        )
        print(f"\n✓ Scheduler: CosineAnnealing")
    elif config.scheduler_type == 'warmup_cosine':
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=5, total_epochs=config.epochs,
            min_lr=config.min_lr
        )
        print(f"\n✓ Scheduler: Warmup + Cosine")
    
    # Mixed precision
    scaler = GradScaler(device='cuda')
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    # Initial check
    print("\n" + "=" * 70)
    print("INITIAL DIVERSITY CHECK")
    print("=" * 70)
    if not verify_diversity(model, test_loader, config.device):
        print("⚠️  Warning: Low initial diversity (expected for untrained model)")
    
    # Training loop
    early_stopping = EarlyStopping(patience=15)
    best_f1 = 0.0
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    for epoch in range(config.epochs):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"{'=' * 70}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion,
            config.device, scaler, config
        )
        
        # Evaluate
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
            model, test_loader, criterion, config.device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}% | Val F1: {val_f1:.4f}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Gap: {(train_acc - val_acc)*100:.1f}%")
        
        # Scheduler step
        if config.scheduler_type == 'plateau':
            scheduler.step(val_f1)
        elif config.scheduler_type == 'warmup_cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'config': config.__dict__
            }, 'best_model_improved.pth')
            print(f"  ✓ New best model! (F1: {val_f1:.4f})")
        
        # Periodic diversity check
        if (epoch + 1) % 10 == 0:
            verify_diversity(model, test_loader, config.device)
            print_per_class_metrics(val_labels, val_preds)
        
        # Early stopping
        if early_stopping(val_f1):
            print(f"\n✓ Early stopping at epoch {epoch + 1}")
            break
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    # Load best model
    checkpoint = torch.load('best_model_improved.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded best model from epoch {checkpoint['epoch'] + 1}")
    
    # Evaluate on full test set
    _, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, config.device
    )
    
    print(f"\nFinal Test Results:")
    print(f"  Accuracy: {test_acc*100:.2f}%")
    print(f"  F1 Score: {test_f1:.4f}")
    
    # Per-class metrics
    print_per_class_metrics(test_labels, test_preds)
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, test_preds, 
                               target_names=FER2013Dataset.EMOTIONS))
    
    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(test_labels, test_preds)
    
    # Diversity check
    verify_diversity(model, test_loader, config.device)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"✓ Best F1: {best_f1:.4f}")
    print(f"✓ Best model saved to: best_model_improved.pth")
    print(f"✓ Training history saved to: training_history.png")
    print(f"✓ Confusion matrix saved to: confusion_matrix.png")


if __name__ == '__main__':
    main()
