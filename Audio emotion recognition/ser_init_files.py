# src/__init__.py
"""
Speech Emotion Recognition Package
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .feature_extraction import AudioFeatureExtractor, extract_features_from_file
from .data_preprocessing import RAVDESSDataProcessor, load_and_preprocess_data
from .model import create_cnn_model, create_and_compile_model
from .train import train_model
from .evaluate import evaluate_model, predict_emotion

__all__ = [
    'AudioFeatureExtractor',
    'extract_features_from_file',
    'RAVDESSDataProcessor',
    'load_and_preprocess_data',
    'create_cnn_model',
    'create_and_compile_model',
    'train_model',
    'evaluate_model',
    'predict_emotion',
]


# tests/__init__.py
"""
Test suite for Speech Emotion Recognition
"""
