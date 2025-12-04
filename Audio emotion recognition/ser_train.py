"""
Training Script for Speech Emotion Recognition Model
"""

import os
import sys
import argparse
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
import logging
from datetime import datetime

from data_preprocessing import load_and_preprocess_data
from model import create_and_compile_model, print_model_summary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seeds set to {seed}")


def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config


def create_callbacks(config: dict) -> list:
    """Create training callbacks."""
    callbacks = []
    
    callback_config = config['training']['callbacks']
    output_config = config['output']
    
    # Model Checkpoint
    os.makedirs(output_config['model_dir'], exist_ok=True)
    checkpoint_path = callback_config['model_checkpoint']['filepath']
    callbacks.append(ModelCheckpoint(
        checkpoint_path,
        monitor=callback_config['model_checkpoint']['monitor'],
        save_best_only=callback_config['model_checkpoint']['save_best_only'],
        mode=callback_config['model_checkpoint']['mode'],
        verbose=1
    ))
    logger.info(f"ModelCheckpoint: {checkpoint_path}")
    
    # Early Stopping
    callbacks.append(EarlyStopping(
        monitor=callback_config['early_stopping']['monitor'],
        patience=callback_config['early_stopping']['patience'],
        restore_best_weights=callback_config['early_stopping']['restore_best_weights'],
        mode=callback_config['early_stopping']['mode'],
        verbose=1
    ))
    logger.info("EarlyStopping enabled")
    
    # Reduce LR on Plateau
    callbacks.append(ReduceLROnPlateau(
        monitor=callback_config['reduce_lr']['monitor'],
        factor=callback_config['reduce_lr']['factor'],
        patience=callback_config['reduce_lr']['patience'],
        min_lr=callback_config['reduce_lr']['min_lr'],
        mode=callback_config['reduce_lr']['mode'],
        verbose=callback_config['reduce_lr']['verbose']
    ))
    logger.info("ReduceLROnPlateau enabled")
    
    # TensorBoard
    log_dir = os.path.join(
        callback_config['tensorboard']['log_dir'],
        datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    os.makedirs(log_dir, exist_ok=True)
    callbacks.append(TensorBoard(
        log_dir=log_dir,
        histogram_freq=callback_config['tensorboard']['histogram_freq']
    ))
    logger.info(f"TensorBoard logs: {log_dir}")
    
    return callbacks


def train_model(config_path: str = 'configs/config.yaml', resume: bool = False):
    """
    Main training function.
    
    Args:
        config_path: Path to configuration file
        resume: Whether to resume from checkpoint
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set random seeds
    set_random_seeds(config['random_seed'])
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"GPUs available: {len(gpus)}")
    if gpus:
        for gpu in gpus:
            logger.info(f"  - {gpu}")
    
    # Load and preprocess data
    logger.info("="*70)
    logger.info("LOADING AND PREPROCESSING DATA")
    logger.info("="*70)
    
    data_path = config['data']['raw_data_path']
    csv_path = os.path.join(
        config['data']['processed_data_path'],
        config['data']['csv_file']
    )
    
    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_and_preprocess_data(
        data_path=data_path,
        duration=config['features']['duration'],
        offset=config['features']['offset'],
        apply_augmentation=config['augmentation']['apply_noise'],
        save_csv=True,
        csv_path=csv_path
    )
    
    # Reshape for Conv1D
    X_train = np.expand_dims(X_train, axis=2)
    X_val = np.expand_dims(X_val, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    
    logger.info(f"\nData shapes:")
    logger.info(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    logger.info(f"  X_val: {X_val.shape}, Y_val: {Y_val.shape}")
    logger.info(f"  X_test: {X_test.shape}, Y_test: {Y_test.shape}")
    
    # Create model
    logger.info("\n" + "="*70)
    logger.info("CREATING MODEL")
    logger.info("="*70)
    
    input_shape = (X_train.shape[1], 1)
    
    if resume and os.path.exists(config['training']['callbacks']['model_checkpoint']['filepath']):
        logger.info("Loading model from checkpoint...")
        model = tf.keras.models.load_model(
            config['training']['callbacks']['model_checkpoint']['filepath']
        )
    else:
        model = create_and_compile_model(
            input_shape=input_shape,
            num_classes=config['model']['output_classes'],
            learning_rate=config['training']['optimizer']['learning_rate']
        )
    
    print_model_summary(model)
    
    # Create callbacks
    callbacks = create_callbacks(config)
    
    # Train model
    logger.info("\n" + "="*70)
    logger.info("TRAINING MODEL")
    logger.info("="*70)
    
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(
        config['output']['model_dir'],
        'final_model.keras'
    )
    model.save(final_model_path)
    logger.info(f"\nFinal model saved to {final_model_path}")
    
    # Evaluate on test set
    logger.info("\n" + "="*70)
    logger.info("EVALUATING ON TEST SET")
    logger.info("="*70)
    
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc*100:.2f}%")
    
    # Save training history
    history_path = os.path.join(config['output']['results_dir'], 'training_history.npz')
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    np.savez(
        history_path,
        loss=history.history['loss'],
        val_loss=history.history['val_loss'],
        accuracy=history.history['accuracy'],
        val_accuracy=history.history['val_accuracy']
    )
    logger.info(f"Training history saved to {history_path}")
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*70)
    
    return model, history


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train Speech Emotion Recognition Model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from checkpoint'
    )
    
    args = parser.parse_args()
    
    try:
        model, history = train_model(
            config_path=args.config,
            resume=args.resume
        )
        logger.info("Training completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
