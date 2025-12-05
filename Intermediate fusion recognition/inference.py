# from torchview import draw_graph
import sys
import os
import cv2
import torch
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from PIL import Image
from facenet_pytorch import MTCNN
import tqdm
from typing import Tuple, Optional, List, Dict
import warnings
import subprocess
from torch.autograd import Variable
from models import multimodalcnn
from opts import parse_opts
from scipy.ndimage import gaussian_filter1d
opt = parse_opts()
class TrainingConfig:
    MODEL_PATH = opt.model_path
    EFFICIENTFACE_PATH = opt.pretrain_path
    WINDOW_SEC = 3.6
    FRAME_COUNT = 15
    INPUT_FPS = 30
    SR = 22050
    N_MFCC = 10
    NUM_EMOTIONS = 8
    MTCNN_IMAGE_SIZE = (720, 1280)
    FACE_OUTPUT_SIZE = (224, 224)
    EMOTIONS = ["calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def select_distributed(m: int, n: int) -> List[int]:
    return [i*n//m + n//(2*m) for i in range(m)]


class TrainingAlignedInference:
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mtcnn = MTCNN(
            image_size=self.config.MTCNN_IMAGE_SIZE,
            device=self.device
        )

        self._load_model()

    def _load_model(self):
        self.model = multimodalcnn.MultiModalCNN(
            self.config.NUM_EMOTIONS,
            fusion='ia',
            seq_length=self.config.FRAME_COUNT,
            pretr_ef=self.config.EFFICIENTFACE_PATH,
            num_heads=1
        )

        checkpoint = torch.load(self.config.MODEL_PATH, map_location=self.device, weights_only=False)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def preprocess_audio_like_training(self, media_path: str, start_sec: float = 0.0) -> np.ndarray:
        """
        Process audio EXACTLY like in training (extract_audios.py).
        Handles both audio files (.wav) and video files (.mp4, .avi).
        
        Args:
            media_path: Path to audio or video file
            start_sec: Start time in seconds
            
        Returns:
            Preprocessed audio array
        """
        # Check if input is video or audio file
        file_ext = os.path.splitext(media_path)[1].lower()
        is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        
        if is_video:
            # For video files, librosa can extract audio directly
            print(f"Extracting audio from video: {os.path.basename(media_path)}")
            try:
                y, sr = librosa.core.load(media_path, sr=self.config.SR, 
                                    offset=start_sec, 
                                    duration=self.config.WINDOW_SEC)
            except Exception as e:
                print(f"Error extracting audio from video: {e}")
                print("Make sure ffmpeg is installed: pip install ffmpeg-python")
                raise
        else:
            # For audio files, load directly
            y, sr = librosa.load(media_path, sr=self.config.SR, 
                                offset=start_sec, 
                                duration=self.config.WINDOW_SEC)
        target_length = int(self.config.SR * self.config.WINDOW_SEC)
        if len(y) == 0:
            print(f"Error: Could not extract any audio from {media_path}")
            # Handle empty audio (skip or fill with zeros)
            y = np.zeros(target_length)
        
        elif len(y) < target_length:
            # Pad with zeros at the end (efficiently)
            print(len(y))
            y = librosa.util.fix_length(y, size=target_length)
            
        elif len(y) > target_length:
            # Center Crop
            remain = len(y) - target_length
            start = remain // 2
            y = y[start : start + target_length]
        return y
    def extract_mfcc_features(self, y: np.ndarray) -> np.ndarray:
            """
            Extract MFCC features exactly as in training.
            
            Args:
                y: Audio signal
                
            Returns:
                MFCC features
            """
            mfcc = librosa.feature.mfcc(y=y, sr=self.config.SR, n_mfcc=self.config.N_MFCC)
            return mfcc
        
    def preprocess_video_like_training(self, video_path: str, start_sec: float = 0.0) -> np.ndarray:
        """
        Args:
            video_path: Path to video file
            start_sec: Start time in seconds
            
        Returns:
            Array of face crops [15, 224, 224, 3]
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Warning if FPS doesn't match training
        if abs(fps - self.config.INPUT_FPS) > 1:
            warnings.warn(f"Video FPS ({fps}) differs from training FPS ({self.config.INPUT_FPS}). "
                         f"Results may be unreliable.")
        
        # Calculate frame range for 3.6 seconds
        start_frame = int(start_sec * fps)
        total_frames_needed = int(self.config.WINDOW_SEC * fps)
        
        # Get frames to select using EXACT training distribution
        frames_to_select = select_distributed(self.config.FRAME_COUNT, total_frames_needed)
        frames_to_select = [f + start_frame for f in frames_to_select]  # Adjust for start offset
        
        # Set video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        numpy_video = []
        current_frame = start_frame
        frames_to_select_set = set(frames_to_select)
        
        print(f"Extracting frames: {frames_to_select[:3]}...{frames_to_select[-3:]}")
        
        for _ in range(total_frames_needed):
            ret, im = cap.read()
            
            if not ret:
                # If video ends, pad with black frames
                if current_frame in frames_to_select_set:
                    numpy_video.append(np.zeros((224, 224, 3), dtype=np.uint8))
                current_frame += 1
                continue
            
            if current_frame in frames_to_select_set:
                
                # Convert to RGB for MTCNN (training uses RGB)
                im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im_rgb_tensor = torch.tensor(im_rgb).to(self.device)
                
                # Detect face with same MTCNN configuration
                bbox, _ = self.mtcnn.detect(im_rgb_tensor)
                
                if bbox is not None and len(bbox) > 0:
                    # Use first detected face
                    bbox = bbox[0]
                    bbox = [round(x) for x in bbox]
                    x1, y1, x2, y2 = bbox
                    
                    # Ensure valid crop coordinates
                    h, w = im.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        # Crop face from BGR image
                        face_crop = im[y1:y2, x1:x2, :]
                    else:
                        face_crop = im  # Use full frame if crop invalid
                else:
                    # No face detected, use full frame
                    face_crop = im
                
                # Resize to expected size and convert to RGB for model
                face_crop = cv2.resize(face_crop, self.config.FACE_OUTPUT_SIZE)
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                numpy_video.append(face_crop)
            
            current_frame += 1
        
        cap.release()
        
        # Pad with black frames if needed (same as training)
        while len(numpy_video) < self.config.FRAME_COUNT:
            numpy_video.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        # Ensure we have exactly 15 frames
        numpy_video = numpy_video[:self.config.FRAME_COUNT]
        
        return np.array(numpy_video)

    def inference_single_window(self, media_path: str, start_sec: float = 0.0) -> Dict:
        if not os.path.exists(media_path):
            raise FileNotFoundError(media_path)

        audio = self.preprocess_audio_like_training(media_path, start_sec)
        mfcc = self.extract_mfcc_features(audio)
        video = self.preprocess_video_like_training(media_path, start_sec)

        audio_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(self.device)
        video_tensor = torch.tensor(video, dtype=torch.float32).to(self.device) / 255.0
        video_tensor = video_tensor.permute(0, 3, 1, 2)

        with torch.no_grad():
            output = self.model(audio_tensor, video_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]

        neutral = probs[0]
        calm = probs[1]
        merged = neutral + calm

        probs = np.array([
            merged,
            probs[2],
            probs[3],
            probs[4],
            probs[5],
            probs[6],
            probs[7]
        ])

        probs /= probs.sum()

        idx = np.argmax(probs)
        return {
            "probabilities": probs,
            "emotion_labels": self.config.EMOTIONS,
            "dominant_emotion": self.config.EMOTIONS[idx],
            "confidence": probs[idx],
            "start_time": start_sec,
            "duration": self.config.WINDOW_SEC,
            "media_file": os.path.basename(media_path)
        }

    
    def inference_sliding_window(self, media_path: str, step_sec: float = 1.0,
                               plot_results: bool = True) -> Dict:
        """
        Run sliding window inference across entire video.
        
        Args:
            media_path: Path to video or audio+video file
            step_sec: Step size in seconds
            plot_results: Whether to plot results
            
        Returns:
            Dictionary with temporal emotion analysis
        """
        # Verify file exists
        if not os.path.exists(media_path):
            raise FileNotFoundError(f"Media file not found: {media_path}")
        
        # Get video duration
        cap = cv2.VideoCapture(media_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps
        cap.release()
        
        print(f"\nMedia file info:")
        print(f"  File: {os.path.basename(media_path)}")
        print(f"  Duration: {duration_sec:.2f}s")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        
        # Check if video is long enough
        if duration_sec < self.config.WINDOW_SEC:
            raise ValueError(f"Media ({duration_sec:.2f}s) shorter than window ({self.config.WINDOW_SEC}s)")
        audio_path = None
        file_ext = os.path.splitext(media_path)[1].lower()
        is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        # if is_video:
        #     audio_path = extract_audio_without_ffmpeg(media_path)
        # Pre-load audio for efficiency (handles both audio and video files)
        print("\nPre-loading audio track...")
        try:
            # total_duration = librosa.get_duration(path=media_path)
            full_audio, _ = librosa.core.load(media_path, sr=self.config.SR)
            print(f"  Audio loaded: {len(full_audio)/self.config.SR:.2f}s at {self.config.SR}Hz")
        except Exception as e:
            print(f"Error loading audio: {e}")
            print("Ensure ffmpeg is installed for video file audio extraction")
            raise
        
        # Calculate windows
        t_starts = np.arange(0.0, duration_sec - self.config.WINDOW_SEC + 0.001, step_sec)
        
        times = []
        probs_list = []
        dominant_emotions = []
        
        print(f"\nProcessing {len(t_starts)} windows with step={step_sec}s")
        
        for t_start in tqdm.tqdm(t_starts, desc="Processing windows"):
            # Extract audio segment
            start_sample = int(t_start * self.config.SR)
            end_sample = int((t_start + self.config.WINDOW_SEC) * self.config.SR)
            # print("Start:")
            # print(start_sample)
            # print("End:")
            # print(end_sample)
            audio_segment = full_audio[start_sample:end_sample]
            # print("Audio segment before:")
            # print(len(audio_segment))
            # Apply training preprocessing to audio segment
            target_length = int(self.config.SR * self.config.WINDOW_SEC)
            if len(audio_segment) < target_length:
                audio_segment = np.array(list(audio_segment) + [0] * (target_length - len(audio_segment)))
            elif len(audio_segment) > target_length:
                remain = len(audio_segment) - target_length
                audio_segment = audio_segment[remain//2:-(remain - remain//2)]
            # print("Audio segment afbefore:")
            # print(len(audio_segment))
            # Extract MFCC
            mfcc = self.extract_mfcc_features(audio_segment)
            
            # Extract video frames
            video_frames = self.preprocess_video_like_training(media_path, t_start)
            
            # Prepare tensors
            audio_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(self.device)
        
            # 2. Prepare Visual Tensor
            # Input 'video_frames' is numpy: (15, 224, 224, 3) -> [Time, Height, Width, Channel]
            
            # Normalize to 0-1 range (Standard PyTorch practice usually matches Training DataLoaders)
            # If your training data was 0-255, remove the '/ 255.0'
            video_tensor = torch.tensor(video_frames, dtype=torch.float32).to(self.device) / 255.0
            
            # Move Channel to dim 1: 
            # (15, 224, 224, 3) -> (15, 3, 224, 224) -> [Time, Channel, Height, Width]
            video_tensor = video_tensor.permute(0, 3, 1, 2)
            
            # 3. Flatten for Model Input
            # The model expects a flattened batch of images: (Batch * Time, Channel, Height, Width)
            # Since Batch=1, Batch*Time is just Time (15).
            # Current shape is already (15, 3, 224, 224). We just need to enforce it.
            
            # Note: We SKIP the .permute(0,2,1,3,4) seen in training because our 
            # starting tensor is already in the target order (Time, Channel, H, W).
            video_tensor_flat = video_tensor
            
            # Forward pass
            with torch.no_grad():
                output = self.model(audio_tensor, video_tensor_flat)
                # dot = make_dot(output, params = dict(self.model.named_parameters()))
                # dot.render("graph", format="png", view=True)
                prob = torch.softmax(output, dim=1).cpu().numpy()[0]
                # merge neutral (0) + calm (1)
            merged_calm = prob[0] + prob[1]

          
            probabilities = np.array([
                merged_calm,          # calm
                prob[2],
                prob[3],
                prob[4],
                prob[5],
                prob[6],
                prob[7]
            ], dtype=np.float32)

            
            probabilities = probabilities / probabilities.sum()
            probs_list.append(probabilities)
            times.append(t_start)
            dominant_emotions.append(self.config.EMOTIONS[np.argmax(probabilities)])
        
        probs_array = np.stack(probs_list)
        
        # Plot if requested
        if plot_results:
            self.plot_emotions(np.array(times), probs_array)
        
        return {
            'times': np.array(times),
            'probabilities': probs_array,
            'emotion_labels': self.config.EMOTIONS,
            'dominant_emotions': dominant_emotions,
            'window_size': self.config.WINDOW_SEC,
            'step_size': step_sec,
            'video_duration': duration_sec
        }

    def plot_emotions(self, times: np.ndarray, probs: np.ndarray, save_path: str = None):
        # --- 1. Create evenly spaced timeline (0 → duration) ---
        video_duration = times[-1] + self.config.WINDOW_SEC
        timeline = np.linspace(0, video_duration, len(times))
        print(timeline)
        # --- 2. Smooth probabilities for each emotion ---
        smooth_probs = np.zeros_like(probs)
        for i in range(probs.shape[1]):
            smooth_probs[:, i] = gaussian_filter1d(probs[:, i], sigma=1.0)
        """Plot emotion probabilities over time"""
        plt.figure(figsize=(14, 8))
        EMOTION_COLORS = {
            "calm": "#808080",
            "happy": "#FFFF00",
            "sad": "#0000FF",
            "angry": "#FF0000",
            "fearful": "#BB00FF",
            "disgust": "#008000",
            "surprised": "#00FFFF"
        }


        plt.figure(figsize=(14, 8))

        for i, label in enumerate(self.config.EMOTIONS):
            color = EMOTION_COLORS.get(label.lower(), None)
            plt.plot(
                timeline,
                smooth_probs[:, i],
                label=label.capitalize(),
                linewidth=2,
                color=color
            )
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Emotion Probability', fontsize=12)
        plt.title('Emotion Analysis Over Time', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        
        # Add minute markers
        max_time = times[-1]
        for minute in range(1, int(max_time / 60) + 1):
            plt.axvline(x=minute * 60, color='gray', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
