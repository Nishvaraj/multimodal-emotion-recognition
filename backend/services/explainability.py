"""
Explainability module for Multimodal Emotion Recognition
Implements Grad-CAM for facial emotions and audio saliency for speech emotions
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import librosa
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
        
        # Compute Grad-CAM
        gradients = self.gradients[0]  # (num_patches, channels)
        activations = self.activations[0]  # (num_patches, channels)
        
        # Weight activations by gradients
        weights = gradients.mean(dim=1)  # (num_patches,)
        cam = torch.sum(weights.unsqueeze(1) * activations, dim=1)  # (num_patches,)
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Reshape to grid (assuming 14x14 patches for 224x224 input)
        cam = cam.cpu().numpy()
        cam = cam.reshape(14, 14)
        
        # Resize to original image size
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        return cam


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
        
        # Initialize Grad-CAM
        target_layer = model.vit.layernorm
        grad_cam = GradCAM(model, target_layer)
        
        # Generate heatmap
        heatmap = grad_cam.generate(input_tensor, emotion_idx)
        grad_cam.remove_hooks()
        
        # Convert original image to array
        img_array = np.array(image)
        
        # Create visualization
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
        
        # Blend original image with heatmap (30% transparency)
        alpha = 0.3
        blended = cv2.addWeighted(img_array, 1 - alpha, heatmap_rgb, alpha, 0)
        
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
        saliency = torch.abs(input_values.grad).cpu().detach().numpy()
        saliency = saliency[0]  # Remove batch dimension
        
        # Normalize saliency
        saliency = np.mean(saliency, axis=0)  # Average across channels
        if np.max(saliency) > 0:
            saliency = saliency / np.max(saliency)
        
        # Create spectrogram
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Normalize spectrogram for visualization
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
        
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
        
        # Create figure with saliency overlay
        fig2, ax2 = plt.subplots(figsize=(10, 4), dpi=100)
        
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
        
        # Overlay saliency with jet colormap
        saliency_cmap = plt.cm.jet(saliency_map)
        ax2.imshow(
            saliency_map,
            aspect='auto',
            origin='lower',
            cmap='jet',
            alpha=0.4,
            interpolation='bilinear'
        )
        
        ax2.set_title(f'Audio Saliency Map - {emotions_list[emotion_idx]} (Red = Important)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Mel Frequency')
        
        # Save saliency visualization
        saliency_buffer = BytesIO()
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
