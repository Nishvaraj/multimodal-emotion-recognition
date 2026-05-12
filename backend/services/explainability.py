"""Explainability utilities for multimodal emotion recognition outputs.

This module generates the visual explanation artefacts used by the MMER backend:

1. Facial explainability:
   - Uses Grad-CAM adapted for ViT.
   - Falls back to EigenCAM when Grad-CAM produces unusable all-zero maps.
   - Returns original image and heatmap overlay as base64 PNG strings.

2. Speech explainability:
   - Uses waveform-gradient saliency for HuBERT.
   - Produces a mel spectrogram and an audio saliency overlay.
   - Returns both visualisations as base64 PNG strings.

3. Combined visualisation:
   - Creates a simple HTML report combining facial and speech explanations.
"""

# =============================================================================
# Imports
# =============================================================================

import os


# =============================================================================
# Runtime Safeguards for Headless Deployment
# =============================================================================

# Hugging Face Spaces / Docker containers usually run without a graphical desktop.
# These settings prevent OpenCV and Qt from trying to use unavailable GUI features.

# Disable OpenEXR support in OpenCV. This avoids codec/security/runtime issues in
# server environments where EXR support is not needed.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"

# Force Qt to use an offscreen backend so Matplotlib/OpenCV do not require a display.
os.environ["QT_QPA_PLATFORM"] = "offscreen"


# Core machine-learning and numerical libraries.
import torch
import torch.nn as nn
import numpy as np
import cv2

# Audio processing and plotting utilities.
import librosa
import librosa.display

# Matplotlib must use a non-interactive backend in a backend server.
# Agg renders plots directly to image buffers instead of opening GUI windows.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Image and encoding utilities.
from PIL import Image
from io import BytesIO
import base64

# Grad-CAM library utilities.
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform


# =============================================================================
# Model Wrapper
# =============================================================================

class ViTLogitsWrapper(nn.Module):
    """Wrap a Hugging Face ViT model so pytorch-grad-cam can call it normally.

    Hugging Face image classification models usually expect named arguments such
    as pixel_values=x and return an object containing .logits. However,
    pytorch-grad-cam expects a standard PyTorch module whose forward(x) returns
    logits directly.

    This wrapper adapts the Hugging Face model interface into the simpler format
    expected by Grad-CAM.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """Return raw class logits for Grad-CAM target computation."""
        return self.model(pixel_values=x).logits


# =============================================================================
# Facial Explainability
# =============================================================================

def generate_grad_cam(image, model, processor, emotion_idx, emotions_list, device):
    """Generate a Grad-CAM or EigenCAM heatmap for a ViT facial prediction.

    Args:
        image: PIL image used for prediction, usually the cropped face.
        model: Fine-tuned ViT emotion classification model.
        processor: Hugging Face image processor for ViT preprocessing.
        emotion_idx: Index of the target emotion class to explain.
        emotions_list: List of emotion label names.
        device: CPU or CUDA device.

    Returns:
        tuple[str | None, str | None]:
            original image as base64 PNG,
            heatmap overlay as base64 PNG.

    Why special handling is needed for ViT:
        CNNs naturally produce spatial feature maps. ViT produces sequential
        patch tokens 1D, so vit_reshape_transform converts the tokens back into a
        2D patch grid before Grad-CAM can operate.
    """
    try:
        # Convert input to RGB NumPy array for OpenCV processing and overlay.
        img_rgb = np.array(image.convert('RGB'))
        h, w = img_rgb.shape[:2]
        img_pil = Image.fromarray(img_rgb)

        # Convert image to ViT input tensor using the same processor as inference.
        inputs = processor(img_pil, return_tensors='pt').to(device)
        input_tensor = inputs['pixel_values']

        # Wrap Hugging Face ViT so Grad-CAM receives logits directly.
        wrapped_model = ViTLogitsWrapper(model)
        wrapped_model.eval()

        # Try several late transformer blocks.
        # The very final block can sometimes produce saturated or weak gradients,
        # so earlier late blocks may provide cleaner spatial maps.
        layers_to_try = [
            model.vit.encoder.layer[-1].layernorm_after,
            model.vit.encoder.layer[-2].layernorm_after,
            model.vit.encoder.layer[-3].layernorm_after,
        ]

        cam_map = None
        method_used = None

        # Attempt gradient-based Grad-CAM first.
        for i, layer in enumerate(layers_to_try):
            try:
                cam = GradCAM(
                    model=wrapped_model,
                    target_layers=[layer],
                    reshape_transform=vit_reshape_transform,
                )

                # Tell Grad-CAM which class logit should be explained.
                targets = [ClassifierOutputTarget(emotion_idx)]

                # grayscale_cam shape is usually [batch, height, width].
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                result = grayscale_cam[0]

                # Reject degenerate maps. An all-zero or near-zero heatmap is not
                # useful to show to the user as an explanation.
                if result.max() > 0.01:
                    cam_map = result
                    method_used = f"GradCAM (encoder block {12 - (i + 1)})"
                    break
                else:
                    print(f"[explainability] layer[-{i+1}] all zeros, trying next")

            except Exception as e:
                print(f"[explainability] GradCAM layer[-{i+1}] failed: {e}")

        # If Grad-CAM fails or produces all-zero maps, use EigenCAM.
        # EigenCAM is gradient-free: it uses principal components of activation
        # maps, making it more stable when gradients vanish or become NaN.
        if cam_map is None:
            print("[explainability] All GradCAM layers zero, using EigenCAM")
            try:
                eigen = EigenCAM(
                    model=wrapped_model,
                    target_layers=[model.vit.encoder.layer[-1].layernorm_after],
                    reshape_transform=vit_reshape_transform,
                )
                grayscale_cam = eigen(input_tensor=input_tensor)
                cam_map = grayscale_cam[0]
                method_used = "EigenCAM"
            except Exception as e:
                print(f"[explainability] EigenCAM failed: {e}")
                return None, None

        print(f"[explainability] {method_used} — min={cam_map.min():.3f}, max={cam_map.max():.3f}")

        # Resize the patch-level CAM to match the original image dimensions.
        cam_resized = cv2.resize(cam_map.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)

        # Smooth the heatmap so it overlays more naturally on the face.
        cam_resized = cv2.GaussianBlur(cam_resized, (13, 13), 0)

        # Normalize CAM to [0, 1] for colour mapping.
        c_min, c_max = cam_resized.min(), cam_resized.max()
        if c_max > c_min:
            cam_resized = (cam_resized - c_min) / (c_max - c_min)

        # Convert normalized CAM to a coloured heatmap.
        cam_uint8 = np.uint8(255 * cam_resized)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Blend only the top 30% most salient regions.
        # This avoids washing the whole image with colour and makes the important
        # facial areas clearer to the user.
        threshold = np.percentile(cam_resized, 70)
        blend_mask = (cam_resized > threshold).astype(np.float32)
        blend_mask = cv2.GaussianBlur(blend_mask, (31, 31), 0)[..., None]

        blended = (
            (1 - blend_mask * 0.65) * img_rgb.astype(np.float32)
            + blend_mask * 0.65 * heatmap_rgb.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

        # Encode original image as base64 PNG for JSON transport.
        orig_buf = BytesIO()
        Image.fromarray(img_rgb).save(orig_buf, format='PNG')
        orig_b64 = base64.b64encode(orig_buf.getvalue()).decode()

        # Encode blended Grad-CAM/EigenCAM overlay as base64 PNG.
        blend_buf = BytesIO()
        Image.fromarray(blended).save(blend_buf, format='PNG')
        blend_b64 = base64.b64encode(blend_buf.getvalue()).decode()

        return orig_b64, blend_b64

    except Exception as e:
        print(f"[explainability] GradCAM generation failed: {e}")
        return None, None


# =============================================================================
# Audio Explainability
# =============================================================================

def generate_audio_saliency(audio, model, processor, emotion_idx, emotions_list, device, sr=16000):
    """Generate audio spectrogram and gradient-based saliency visualisation.

    The method computes gradients of the target emotion logit with respect to
    the input waveform. Larger absolute gradients indicate time regions where
    small changes would most affect the target class score.

    Args:
        audio: Raw waveform as a NumPy array.
        model: Fine-tuned HuBERT audio classification model.
        processor: Hugging Face audio feature extractor.
        emotion_idx: Target emotion class index.
        emotions_list: List of speech emotion labels.
        device: CPU or CUDA device.
        sr: Audio sampling rate. Default is 16000 for HuBERT.

    Returns:
        tuple[str | None, str | None]:
            spectrogram image as base64 PNG,
            saliency overlay image as base64 PNG.
    """
    try:
        if audio is None or len(audio) == 0:
            raise ValueError("Audio input is empty")

        # Convert audio safely to float32 and remove NaN/Inf values.
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio)

        # Convert waveform into HuBERT input tensor.
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs['input_values'].to(device)

        # Enable gradients with respect to the input waveform.
        input_values.requires_grad = True

        # Clear old gradients before backward pass.
        model.zero_grad()

        # Forward pass through HuBERT.
        outputs = model(input_values)

        # Select the logit for the emotion class being explained.
        score = outputs.logits[0, emotion_idx]

        # Backpropagate from selected class score to input waveform.
        score.backward()

        if input_values.grad is None:
            raise RuntimeError("No gradients captured")

        # Absolute gradient magnitude is used as saliency.
        saliency = torch.abs(input_values.grad).cpu().detach().numpy()

        # Flatten possible batch/channel dimensions into one 1D saliency curve.
        if saliency.ndim == 3:
            saliency = np.mean(saliency, axis=1)[0]
        elif saliency.ndim == 2:
            saliency = saliency[0]
        saliency = saliency.reshape(-1).astype(np.float32)

        # Smooth gradient spikes with a moving average so the plot is readable.
        if saliency.size > 11:
            kernel = np.ones(11, dtype=np.float32) / 11.0
            saliency = np.convolve(saliency, kernel, mode='same')

        # Normalize saliency to [0, 1].
        s_min, s_max = saliency.min(), saliency.max()
        if s_max > s_min:
            saliency = (saliency - s_min) / (s_max - s_min)
        else:
            saliency = np.zeros_like(saliency)

        # Create mel spectrogram for interpretable audio display.
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        # ---------------------------------------------------------------------
        # Figure 1: Plain mel spectrogram
        # ---------------------------------------------------------------------
        fig1, ax1 = plt.subplots(figsize=(10, 4), dpi=100)
        img1 = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax1)
        ax1.set_title(f'Audio Spectrogram — {emotions_list[emotion_idx]}')
        fig1.colorbar(img1, ax=ax1, format='%+2.0f dB')

        spec_buf = BytesIO()
        fig1.savefig(spec_buf, format='PNG', bbox_inches='tight', dpi=100)
        spec_b64 = base64.b64encode(spec_buf.getvalue()).decode()
        plt.close(fig1)

        # ---------------------------------------------------------------------
        # Figure 2: Spectrogram with saliency overlay + saliency curve
        # ---------------------------------------------------------------------
        fig2, (ax2, ax3) = plt.subplots(
            2,
            1,
            figsize=(10, 5.5),
            dpi=100,
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=False
        )

        # Normalize spectrogram image to [0, 1] so overlay colours are visible.
        S_norm = (S_db - S_db.min()) / max(S_db.max() - S_db.min(), 1e-8)

        # Resize 1D waveform saliency to match spectrogram time-axis length.
        sal_resized = np.interp(
            np.linspace(0, 1, S_db.shape[1]),
            np.linspace(0, 1, saliency.shape[0]),
            saliency
        )

        # Tile the 1D saliency curve vertically so it can overlay the full
        # spectrogram as a time-based heatmap.
        sal_map = np.tile(sal_resized, (S_db.shape[0], 1))

        # Base spectrogram.
        ax2.imshow(S_norm, aspect='auto', origin='lower', cmap='viridis', interpolation='bilinear')

        # Saliency overlay. Bright areas show the most influential time regions.
        ax2.imshow(sal_map, aspect='auto', origin='lower', cmap='magma', alpha=0.6, interpolation='bilinear')
        ax2.set_title(f'Audio Saliency — {emotions_list[emotion_idx]} (bright = important)')
        ax2.set_ylabel('Mel Frequency')

        # Plot 1D saliency curve and highlight top 15% peaks.
        peak_thr = np.percentile(sal_resized, 85)
        x = np.arange(len(sal_resized))
        ax3.plot(x, sal_resized, color='#f97316', linewidth=1.5)
        ax3.fill_between(x, 0, sal_resized, where=sal_resized >= peak_thr, color='#ef4444', alpha=0.4)
        ax3.axhline(peak_thr, color='#ef4444', linestyle='--', linewidth=1, alpha=0.8)
        ax3.set_ylim(0, 1.05)
        ax3.set_ylabel('Saliency')
        ax3.set_xlabel('Time steps')
        ax3.grid(alpha=0.2)

        sal_buf = BytesIO()
        fig2.tight_layout()
        fig2.savefig(sal_buf, format='PNG', bbox_inches='tight', dpi=100)
        sal_b64 = base64.b64encode(sal_buf.getvalue()).decode()
        plt.close(fig2)

        return spec_b64, sal_b64

    except Exception as e:
        print(f"[explainability] Audio saliency failed: {e}")
        return None, None


# =============================================================================
# Combined Visualisation
# =============================================================================

def create_combined_visualization(grad_cam_base64, saliency_base64, facial_emotion, speech_emotion, concordance):
    """Create a simple HTML-based combined explanation panel.

    This function embeds the facial Grad-CAM and speech saliency image into a
    small HTML layout, then encodes the HTML as base64.

    Note:
        In the current backend flow, the main API usually returns Grad-CAM and
        saliency images separately. This helper is useful for Gradio-style or
        single-block visualisation outputs.
    """
    try:
        # Green tint for match, red tint otherwise.
        bg_color = '#d4edda' if concordance == 'MATCH' else '#f8d7da'

        html = f"""
        <div style="display:flex;gap:20px;padding:20px;background:#f5f5f5;border-radius:10px;">
            <div style="flex:1;">
                <h3>Facial GradCAM — {facial_emotion}</h3>
                <img src="data:image/png;base64,{grad_cam_base64}" style="width:100%;border-radius:8px;">
                <p style="font-size:12px;color:#666;">Red/warm = regions that most influenced the {facial_emotion} prediction.</p>
            </div>
            <div style="flex:1;">
                <h3>Speech Saliency — {speech_emotion}</h3>
                <img src="data:image/png;base64,{saliency_base64}" style="width:100%;border-radius:8px;">
                <p style="font-size:12px;color:#666;">Bright = time-frequency regions with strongest influence.</p>
            </div>
        </div>
        <div style="margin-top:20px;padding:15px;background:{bg_color};border-radius:8px;text-align:center;">
            <h4 style="margin:0;">Concordance: <strong>{concordance}</strong></h4>
        </div>
        """

        # Encode the HTML as base64 so it can be transported as a string.
        return base64.b64encode(html.encode()).decode()
    except Exception as e:
        print(f"[explainability] Combined visualisation failed: {e}")
        return None
