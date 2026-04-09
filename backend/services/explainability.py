"""
Explainability module for Multimodal Emotion Recognition
Implements Grad-CAM for facial emotions and audio saliency for speech emotions
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import base64
from typing import Tuple, Optional


class GradCAM:
    """Gradient-weighted Class Activation Mapping for ViT models"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture gradients and activations"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Register hooks on target layer
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
        
        self.handles = [forward_handle, backward_handle]
    
    def remove_hooks(self):
        """Remove hooks"""
        for handle in self.handles:
            handle.remove()
    
    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor (1, 3, H, W)
            target_class: Target emotion class index
        
        Returns:
            Heatmap as numpy array (224, 224)
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        target_score = output.logits[0, target_class]
        target_score.backward()
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture gradients/activations")

        # Compute token-level relevance for ViT-style outputs.
        gradients = self.gradients[0]
        activations = self.activations[0]

        if gradients.ndim == 1:
            gradients = gradients.unsqueeze(0)
        if activations.ndim == 1:
            activations = activations.unsqueeze(0)

        if gradients.shape != activations.shape:
            raise RuntimeError(
                f"Gradient/activation shape mismatch: {gradients.shape} vs {activations.shape}"
            )

        cam_tokens = torch.mean(gradients * activations, dim=-1)

        # Drop CLS token when present (common in ViT), then ensure square token grid.
        if cam_tokens.numel() > 1:
            maybe_grid = int(np.sqrt(cam_tokens.numel() - 1))
            if maybe_grid * maybe_grid == cam_tokens.numel() - 1:
                cam_tokens = cam_tokens[1:]

        token_count = cam_tokens.numel()
        grid_size = int(np.sqrt(token_count))

        if grid_size * grid_size != token_count:
            # Fallback for non-square token counts: keep largest square subset.
            usable = grid_size * grid_size
            if usable == 0:
                raise RuntimeError(f"Invalid token count for Grad-CAM: {token_count}")
            cam_tokens = cam_tokens[:usable]

        cam = F.relu(cam_tokens)
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)

        cam = cam.cpu().numpy().reshape(grid_size, grid_size)
        return cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)


def _robust_normalize(arr: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    """Normalize with percentile clipping to improve contrast in low-variance maps."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr

    lo = np.percentile(arr, low_pct)
    hi = np.percentile(arr, high_pct)
    if hi <= lo:
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        if max_v <= min_v:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - min_v) / (max_v - min_v)

    clipped = np.clip(arr, lo, hi)
    return (clipped - lo) / (hi - lo)


def generate_grad_cam(
    image: Image.Image,
    model,
    processor,
    emotion_idx: int,
    emotions_list: list,
    device
) -> Tuple[str, str]:
    """
    Generate Grad-CAM visualization for facial emotion
    
    Args:
        image: PIL Image object
        model: ViT model
        processor: Image processor
        emotion_idx: Index of target emotion class
        emotions_list: List of emotion names
        device: torch device
    
    Returns:
        Tuple of (original_image_base64, heatmap_base64)
    """
    try:
        # Prepare input
        inputs = processor(image, return_tensors='pt').to(device)
        input_tensor = inputs['pixel_values']
        
        # Use a late transformer block layer for stronger spatial attribution.
        target_layer = model.vit.layernorm
        if hasattr(model, 'vit') and hasattr(model.vit, 'encoder') and hasattr(model.vit.encoder, 'layer'):
            if len(model.vit.encoder.layer) > 0 and hasattr(model.vit.encoder.layer[-1], 'layernorm_after'):
                target_layer = model.vit.encoder.layer[-1].layernorm_after
        grad_cam = GradCAM(model, target_layer)
        
        # Generate heatmap
        try:
            heatmap = grad_cam.generate(input_tensor, emotion_idx)
        finally:
            grad_cam.remove_hooks()
        
        # Convert original image to array
        img_array = np.array(image)
        
        # Create visualization
        heatmap = _robust_normalize(heatmap)
        heatmap = cv2.GaussianBlur(heatmap, (9, 9), 0)

        # Apply colormap to heatmap
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # Resize heatmap to match image
        if img_array.shape[:2] != heatmap_colored.shape[:2]:
            heatmap_colored = cv2.resize(
                heatmap_colored,
                (img_array.shape[1], img_array.shape[0])
            )
        
        # Convert BGR to RGB for PIL
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend original image with stronger heatmap visibility.
        alpha = 0.55
        blended = cv2.addWeighted(img_array, 1 - alpha, heatmap_rgb, alpha, 0)

        # Draw contour around most influential regions to make attribution obvious.
        threshold = np.percentile(heatmap, 80)
        mask = (heatmap >= threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(blended, contours, -1, (255, 255, 0), 2)
        
        # Convert original image to base64
        img_pil = Image.fromarray(img_array)
        img_buffer = BytesIO()
        img_pil.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Convert blended image to base64
        blended_pil = Image.fromarray(blended.astype(np.uint8))
        blend_buffer = BytesIO()
        blended_pil.save(blend_buffer, format='PNG')
        heatmap_base64 = base64.b64encode(blend_buffer.getvalue()).decode()
        
        return img_base64, heatmap_base64
    
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return None, None


def generate_audio_saliency(
    audio: np.ndarray,
    model,
    processor,
    emotion_idx: int,
    emotions_list: list,
    device,
    sr: int = 16000
) -> Tuple[str, str]:
    """
    Generate audio saliency map for speech emotion
    
    Args:
        audio: Audio signal as numpy array
        model: HuBERT model
        processor: Feature extractor
        emotion_idx: Index of target emotion class
        emotions_list: List of emotion names
        device: torch device
        sr: Sample rate
    
    Returns:
        Tuple of (spectrogram_base64, saliency_base64)
    """
    try:
        if audio is None or len(audio) == 0:
            raise ValueError("Audio input is empty")

        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio)

        # Prepare input
        inputs = processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )
        input_values = inputs['input_values'].to(device)
        
        # Forward pass with gradient tracking
        input_values.requires_grad = True
        model.zero_grad()
        
        outputs = model(input_values)
        target_score = outputs.logits[0, emotion_idx]
        
        # Backward pass
        target_score.backward()
        
        # Get saliency map (gradient magnitude)
        if input_values.grad is None:
            raise RuntimeError("No gradients captured for audio saliency")

        saliency = torch.abs(input_values.grad).cpu().detach().numpy()

        # Normalize to a 1D time-importance sequence regardless of model output shape.
        if saliency.ndim == 3:
            saliency = np.mean(saliency, axis=1)[0]
        elif saliency.ndim == 2:
            saliency = saliency[0]
        elif saliency.ndim == 1:
            pass
        else:
            saliency = saliency.reshape(-1)

        saliency = np.asarray(saliency).reshape(-1)

        # Smooth temporal saliency and improve dynamic range for clearer plots.
        if saliency.size > 7:
            kernel = np.ones(7, dtype=np.float32) / 7.0
            saliency = np.convolve(saliency, kernel, mode='same')

        saliency = _robust_normalize(saliency, low_pct=5.0, high_pct=99.0)
        if np.max(saliency) > 0:
            saliency = saliency / np.max(saliency)
        
        # Create spectrogram
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Normalize spectrogram for visualization
        s_range = S_db.max() - S_db.min()
        if s_range > 0:
            S_norm = (S_db - S_db.min()) / s_range
        else:
            S_norm = np.zeros_like(S_db)
        
        # Create figure for original spectrogram
        fig1, ax1 = plt.subplots(figsize=(10, 4), dpi=100)
        img1 = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax1)
        ax1.set_title(f'Audio Spectrogram - {emotions_list[emotion_idx]}')
        fig1.colorbar(img1, ax=ax1, format='%+2.0f dB')
        
        # Save spectrogram
        spec_buffer = BytesIO()
        fig1.savefig(spec_buffer, format='PNG', bbox_inches='tight', dpi=100)
        spec_base64 = base64.b64encode(spec_buffer.getvalue()).decode()
        plt.close(fig1)
        
        # Create figure with saliency overlay + timeline for clear explanation.
        fig2, (ax2, ax3) = plt.subplots(
            2,
            1,
            figsize=(10, 5.5),
            dpi=100,
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=False
        )
        
        # Resize saliency to match spectrogram size if needed
        if saliency.shape[0] != S_db.shape[1]:
            saliency_resized = np.interp(
                np.linspace(0, 1, S_db.shape[1]),
                np.linspace(0, 1, saliency.shape[0]),
                saliency
            )
        else:
            saliency_resized = saliency
        
        # Create saliency heatmap (repeat across frequency bins)
        saliency_map = np.tile(saliency_resized, (S_db.shape[0], 1))
        
        # Display spectrogram
        img2 = ax2.imshow(
            S_norm,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='bilinear'
        )
        
        # Overlay saliency with stronger opacity.
        ax2.imshow(
            saliency_map,
            aspect='auto',
            origin='lower',
            cmap='magma',
            alpha=0.65,
            interpolation='bilinear'
        )
        
        ax2.set_title(f'Audio Saliency Map - {emotions_list[emotion_idx]} (Red = Important)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Mel Frequency')

        # Add explicit 1D saliency timeline with highlighted peaks.
        peak_thr = np.percentile(saliency_resized, 85)
        x = np.arange(len(saliency_resized))
        ax3.plot(x, saliency_resized, color='#f97316', linewidth=1.5)
        ax3.fill_between(x, 0, saliency_resized, where=saliency_resized >= peak_thr, color='#ef4444', alpha=0.35)
        ax3.axhline(peak_thr, color='#ef4444', linestyle='--', linewidth=1, alpha=0.8)
        ax3.set_ylim(0, 1.05)
        ax3.set_ylabel('Saliency')
        ax3.set_xlabel('Time steps')
        ax3.grid(alpha=0.2)
        
        # Save saliency visualization
        saliency_buffer = BytesIO()
        fig2.tight_layout()
        fig2.savefig(saliency_buffer, format='PNG', bbox_inches='tight', dpi=100)
        saliency_base64 = base64.b64encode(saliency_buffer.getvalue()).decode()
        plt.close(fig2)
        
        return spec_base64, saliency_base64
    
    except Exception as e:
        print(f"Error generating audio saliency: {e}")
        return None, None


def create_combined_visualization(
    grad_cam_base64: str,
    saliency_base64: str,
    facial_emotion: str,
    speech_emotion: str,
    concordance: str
) -> str:
    """
    Create a combined visualization showing both Grad-CAM and saliency
    
    Args:
        grad_cam_base64: Grad-CAM image as base64
        saliency_base64: Saliency map as base64
        facial_emotion: Facial emotion prediction
        speech_emotion: Speech emotion prediction
        concordance: MATCH or MISMATCH
    
    Returns:
        Combined visualization as base64-encoded HTML
    """
    try:
        html = f"""
        <div style="display: flex; gap: 20px; padding: 20px; background: #f5f5f5; border-radius: 10px;">
            <div style="flex: 1;">
                <h3 style="color: #333; margin-top: 0;">Facial Emotion Grad-CAM</h3>
                <p style="color: #666; font-size: 14px;">Emotion: <strong>{facial_emotion}</strong></p>
                <img src="data:image/png;base64,{grad_cam_base64}" style="width: 100%; max-width: 400px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <p style="color: #999; font-size: 12px; margin-top: 10px;">Red areas show regions that influenced the prediction</p>
            </div>
            <div style="flex: 1;">
                <h3 style="color: #333; margin-top: 0;">Speech Emotion Saliency</h3>
                <p style="color: #666; font-size: 14px;">Emotion: <strong>{speech_emotion}</strong></p>
                <img src="data:image/png;base64,{saliency_base64}" style="width: 100%; max-width: 400px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <p style="color: #999; font-size: 12px; margin-top: 10px;">Red areas show important frequencies for the prediction</p>
            </div>
        </div>
        <div style="margin-top: 20px; padding: 15px; background: {'#d4edda' if concordance == 'MATCH' else '#f8d7da'}; border-radius: 8px; text-align: center;">
            <h4 style="margin: 0; color: {'#155724' if concordance == 'MATCH' else '#721c24'};">
                Concordance: <strong>{concordance}</strong>
            </h4>
            <p style="margin: 5px 0 0 0; font-size: 14px; color: {'#155724' if concordance == 'MATCH' else '#721c24'};">
                Facial and speech emotions are {'aligned' if concordance == 'MATCH' else 'misaligned'}
            </p>
        </div>
        """
        combined_base64 = base64.b64encode(html.encode()).decode()
        return combined_base64
    except Exception as e:
        print(f"Error creating combined visualization: {e}")
        return None
