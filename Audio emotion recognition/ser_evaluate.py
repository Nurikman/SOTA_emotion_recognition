"""
Evaluation and Inference Script
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import joblib
import logging
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score
)
import time

from feature_extraction import AudioFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def plot_training_history(history_path: str, output_dir: str):
    """
    Plot and save training history.
    
    Args:
        history_path: Path to training history .npz file
        output_dir: Directory to save plots
    """
    data = np.load(history_path)
    
    epochs = range(1, len(data['loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Loss plot
    axes[0].plot(epochs, data['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, data['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, data['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, data['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Training history plot saved to {save_path}")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    output_dir: str
):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=class_names,
        cmap='Blues',
        ax=ax,
        colorbar=True
    )
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Confusion matrix saved to {save_path}")
    plt.close()


def evaluate_model(
    model_path: str,
    config_path: str = 'configs/config.yaml',
    data_path: str = None,
    save_plots: bool = True
):
    """
    Evaluate trained model on test set.
    
    Args:
        model_path: Path to trained model
        config_path: Path to configuration file
        data_path: Path to test data CSV (if None, loads from config)
        save_plots: Whether to save evaluation plots
    """
    # Load configuration
    config = load_config(config_path)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Load encoder and scaler
    encoder = joblib.load(config['output']['encoder_path'])
    scaler = joblib.load(config['output']['scaler_path'])
    class_names = encoder.categories_[0]
    
    # Load test data
    if data_path is None:
        data_path = os.path.join(
            config['data']['processed_data_path'],
            config['data']['csv_file']
        )
    
    logger.info(f"Loading test data from {data_path}")
    data_df = pd.read_csv(data_path)
    
    # Filter test set
    test_df = data_df[data_df['Split'] == 'test']
    
    X_test = test_df.iloc[:, :-2].values
    Y_test = test_df['Emotions'].values
    
    # Scale and encode
    X_test_scaled = scaler.transform(X_test)
    Y_test_encoded = encoder.transform(Y_test.reshape(-1, 1))
    
    # Reshape for Conv1D
    X_test_cnn = np.expand_dims(X_test_scaled, axis=2)
    
    logger.info(f"Test set shape: {X_test_cnn.shape}")
    
    # Evaluate
    logger.info("\n" + "="*70)
    logger.info("MODEL EVALUATION")
    logger.info("="*70)
    
    test_loss, test_acc = model.evaluate(X_test_cnn, Y_test_encoded, verbose=0)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc*100:.2f}%")
    
    # Predictions
    logger.info("\nGenerating predictions...")
    start_time = time.time()
    y_pred_probs = model.predict(X_test_cnn, verbose=0)
    inference_time = time.time() - start_time
    
    logger.info(f"Inference time: {inference_time:.3f}s")
    logger.info(f"Time per sample: {inference_time/len(X_test_cnn)*1000:.2f}ms")
    
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(Y_test_encoded, axis=1)
    
    # Convert to class names
    y_pred_names = [class_names[i] for i in y_pred]
    y_true_names = [class_names[i] for i in y_true]
    
    # Classification report
    logger.info("\n" + "="*70)
    logger.info("CLASSIFICATION REPORT")
    logger.info("="*70)
    
    report = classification_report(y_true_names, y_pred_names, target_names=class_names)
    logger.info("\n" + report)
    
    # Save report
    metrics_dir = config['output']['metrics_dir']
    os.makedirs(metrics_dir, exist_ok=True)
    report_path = os.path.join(metrics_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
        f.write(f"Inference time: {inference_time:.3f}s\n")
        f.write(f"Time per sample: {inference_time/len(X_test_cnn)*1000:.2f}ms\n\n")
        f.write(report)
    logger.info(f"Classification report saved to {report_path}")
    
    # Save plots
    if save_plots:
        plots_dir = config['output']['plots_dir']
        
        # Plot confusion matrix
        plot_confusion_matrix(y_true_names, y_pred_names, class_names, plots_dir)
        
        # Plot training history if available
        history_path = os.path.join(config['output']['results_dir'], 'training_history.npz')
        if os.path.exists(history_path):
            plot_training_history(history_path, plots_dir)
    
    logger.info("\n" + "="*70)
    logger.info("EVALUATION COMPLETE!")
    logger.info("="*70)
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'y_true': y_true_names,
        'y_pred': y_pred_names,
        'inference_time': inference_time
    }


def predict_emotion(
    audio_path: str,
    model_path: str,
    config_path: str = 'configs/config.yaml'
) -> tuple:
    """
    Predict emotion from a single audio file.
    
    Args:
        audio_path: Path to audio file
        model_path: Path to trained model
        config_path: Path to configuration
        
    Returns:
        Tuple of (emotion, confidence)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load encoder and scaler
    encoder = joblib.load(config['output']['encoder_path'])
    scaler = joblib.load(config['output']['scaler_path'])
    class_names = encoder.categories_[0]
    
    # Extract features
    extractor = AudioFeatureExtractor()
    features = extractor.get_augmented_features(
        audio_path,
        duration=config['features']['duration'],
        offset=config['features']['offset'],
        apply_noise=False,
        apply_pitch=False
    )
    
    # Use only original features (first row)
    features = features[0].reshape(1, -1)
    
    # Scale
    features_scaled = scaler.transform(features)
    
    # Reshape for Conv1D
    features_cnn = np.expand_dims(features_scaled, axis=2)
    
    # Predict
    probs = model.predict(features_cnn, verbose=0)[0]
    emotion_idx = np.argmax(probs)
    emotion = class_names[emotion_idx]
    confidence = probs[emotion_idx]
    
    return emotion, confidence, dict(zip(class_names, probs))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Evaluate Speech Emotion Recognition Model')
    parser.add_argument(
        '--model',
        type=str,
        default='models/checkpoints/best_model.keras',
        help='Path to trained model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--audio',
        type=str,
        default=None,
        help='Path to single audio file for prediction'
    )
    
    args = parser.parse_args()
    
    try:
        if args.audio:
            # Single file prediction
            emotion, confidence, all_probs = predict_emotion(
                args.audio,
                args.model,
                args.config
            )
            logger.info(f"\nPredicted Emotion: {emotion}")
            logger.info(f"Confidence: {confidence*100:.2f}%")
            logger.info("\nAll probabilities:")
            for emotion_name, prob in all_probs.items():
                logger.info(f"  {emotion_name}: {prob*100:.2f}%")
        else:
            # Full evaluation
            results = evaluate_model(
                model_path=args.model,
                config_path=args.config
            )
        
        logger.info("Evaluation completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
