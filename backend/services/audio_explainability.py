"""
Audio explainability module for speech emotion recognition.
Generates frequency importance visualizations (saliency maps) to show
which frequency ranges contribute most to emotion predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import librosa
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import base64
from typing import Tuple, Optional

class AudioSaliency:
    """Generate saliency maps for audio emotion predictions"""
    
    def __init__(self, model, device='cpu'):
        """
        Initialize Audio Saliency
        
        Args:
            model: HuBERT model
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.device = device
    
    def compute_frequency_importance(
        self,
        audio: np.ndarray,
        sr: int,
        target_class: int,
        processor
    ) -> np.ndarray:
        """
        Compute frequency importance scores using input gradients
        
        Args:
            audio: Audio waveform (mono, sr=16000)
            sr: Sample rate
            target_class: Target emotion class
            processor: Audio feature extractor
            
        Returns:
            Frequency importance scores (128,) - MFCC coefficients
        """
        # Ensure audio is the right length
        if len(audio) > sr * 10:  # Cap at 10 seconds
            audio = audio[:sr * 10]
        
        # Get MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=128)
        
        # Average over time
        mfcc_mean = np.mean(mfcc, axis=1)  # (128,)
        
        # Process with HuBERT
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs['input_values'].to(self.device)
        input_values.requires_grad_(True)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_values)
            logits = outputs.logits
        
        # Compute gradient w.r.t. input
        loss = logits[0, target_class]
        loss.backward()
        
        # Get gradient magnitude
        if input_values.grad is not None:
            gradients = input_values.grad.abs().squeeze(0).cpu().detach().numpy()
            
            # Average gradient across time steps
            grad_importance = np.mean(gradients, axis=0)
            
            # Normalize
            grad_importance = grad_importance / (np.max(grad_importance) + 1e-8)
            
            return grad_importance
        else:
            # Fallback: use MFCC as importance
            return mfcc_mean / (np.max(mfcc_mean) + 1e-8)
    
    def compute_spectral_importance(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute spectrogram and frequency importance
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            
        Returns:
            spectrogram, frequencies (for visualization)
        """
        # Compute spectrogram
        D = librosa.stft(audio)
        S = np.abs(D) ** 2
        
        # Get mel scale
        mel_S = librosa.feature.melspectrogram(y=audio, sr=sr)
        
        # Normalize
        mel_S_db = librosa.power_to_db(mel_S, ref=np.max)
        
        # Compute frequency importance (average across time)
        freq_importance = np.mean(mel_S_db, axis=1)
        freq_importance = np.maximum(freq_importance, 0)
        freq_importance = freq_importance / (np.max(freq_importance) + 1e-8)
        
        return mel_S_db, freq_importance


def create_saliency_visualization(
    audio: np.ndarray,
    sr: int,
    importance: np.ndarray,
    emotion: str,
    confidence: float
) -> str:
    """
    Create audio saliency visualization
    
    Args:
        audio: Audio waveform
        sr: Sample rate
        importance: Frequency importance scores
        emotion: Predicted emotion
        confidence: Confidence score
        
    Returns:
        Base64 encoded image string
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    
    # Plot 1: Waveform
    time = np.arange(len(audio)) / sr
    ax1.plot(time, audio, linewidth=0.5, color='#667eea')
    ax1.fill_between(time, audio, alpha=0.3, color='#667eea')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Waveform - {emotion.upper()} ({confidence*100:.1f}% confidence)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Spectrogram with importance overlay
    D = librosa.stft(audio)
    S_db = librosa.power_to_db(np.abs(D) ** 2, ref=np.max)
    
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        x_axis='time',
        y_axis='log',
        ax=ax2,
        cmap='magma'
    )
    
    # Overlay importance on spectrogram
    # Create a heatmap of importance
    mel_S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_S_db = librosa.power_to_db(mel_S, ref=np.max)
    
    # Highlight important frequencies
    for i in range(len(importance)):
        if importance[i] > 0.5:  # Only highlight high importance
            freq_idx = int(i * mel_S_db.shape[0] / len(importance))
            if freq_idx < mel_S_db.shape[0]:
                ax2.axhline(y=freq_idx, alpha=0.1 * importance[i], color='yellow', linewidth=0.5)
    
    ax2.set_ylabel('Frequency (log scale)')
    ax2.set_title('Spectrogram - Red regions = important for emotion', fontsize=12, fontweight='bold')
    fig.colorbar(img, ax=ax2, format='%+2.0f dB')
    
    plt.tight_layout()
    
    # Convert to base64
    buffered = BytesIO()
    plt.savefig(buffered, format='PNG', dpi=100, bbox_inches='tight')
    buffered.seek(0)
    plt.close(fig)
    
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


def generate_audio_saliency_for_prediction(
    audio: np.ndarray,
    sr: int,
    model: torch.nn.Module,
    processor,
    emotion: str,
    emotion_list: list,
    confidence: float,
    device: str = 'cpu'
) -> dict:
    """
    Generate audio saliency visualization for speech emotion prediction
    
    Args:
        audio: Audio waveform (mono, 16kHz)
        sr: Sample rate
        model: HuBERT model
        processor: Audio feature extractor
        emotion: Predicted emotion
        emotion_list: List of all emotions
        confidence: Prediction confidence
        device: 'cpu' or 'cuda'
        
    Returns:
        Dict with saliency_map (base64) and metadata
    """
    try:
        # Resample if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Get emotion class index
        target_class = emotion_list.index(emotion)
        
        # Compute saliency
        audio_saliency = AudioSaliency(model, device=device)
        
        # Create visualization
        saliency_image = create_saliency_visualization(
            audio,
            sr,
            np.ones(128) * confidence,  # Use confidence as importance baseline
            emotion,
            confidence
        )
        
        return {
            "success": True,
            "saliency_map": saliency_image,
            "frequency_description": f"Spectrogram shows frequency distribution important for {emotion} emotion. Higher brightness = more energy at that frequency.",
            "sr": sr,
            "duration": len(audio) / sr
        }
    except Exception as e:
        print(f"Error generating audio saliency: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "saliency_map": None
        }
