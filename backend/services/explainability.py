"""
Explainability module for Multimodal Emotion Recognition
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


def generate_grad_cam(
    image: Image.Image,
    model,
    processor,
    emotion_idx: int,
    emotions_list: list,
    device
) -> Tuple[str, str]:
    """
    Generate heatmap using ViT attention rollout with contrast enhancement.
    Single forward pass — fast on any hardware.
    Returns (original_base64, heatmap_overlay_base64)
    """
    try:
        img_rgb = np.array(image.convert('RGB'))
        h, w = img_rgb.shape[:2]

        inputs = processor(image, return_tensors='pt').to(device)

        with torch.no_grad():
            # Force attention output
            model.config.output_attentions = True
            outputs = model(**inputs)
            model.config.output_attentions = False  # reset after

        attentions = outputs.attentions  # list of [1, heads, tokens, tokens]

        if not attentions:
            raise RuntimeError("No attention weights returned")

        # Attention rollout across all layers
        num_tokens = attentions[0].shape[-1]
        rollout = torch.eye(num_tokens).to(device)

        for attn in attentions:
            attn_avg = attn[0].mean(dim=0)
            attn_avg = attn_avg + torch.eye(num_tokens).to(device)
            attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
            rollout = torch.matmul(attn_avg, rollout)

        # CLS token attention to all patches
        cls_attn = rollout[0, 1:]
        n = cls_attn.numel()
        grid = int(np.sqrt(n))
        cls_attn = cls_attn[:grid * grid].reshape(grid, grid)
        cam_np = cls_attn.cpu().numpy().astype(np.float32)

        # Resize to image
        cam_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_CUBIC)

        # Smooth
        cam_resized = cv2.GaussianBlur(cam_resized, (15, 15), 0)

        # Use top 40% of values only — makes hotspots pop
        threshold = np.percentile(cam_resized, 60)
        cam_resized = np.where(cam_resized > threshold, cam_resized, threshold)

        # Normalize to full 0-1 range
        cam_min = cam_resized.min()
        cam_max = cam_resized.max()
        if cam_max > cam_min:
            cam_resized = (cam_resized - cam_min) / (cam_max - cam_min)

        # Power transform to increase contrast
        cam_resized = np.power(cam_resized, 0.5)

        # Apply JET colormap
        cam_uint8 = np.uint8(255 * cam_resized)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Blend 50% original + 50% heatmap
        blended = (
            0.5 * img_rgb.astype(np.float32) +
            0.5 * heatmap_rgb.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

        # Encode original
        orig_buf = BytesIO()
        Image.fromarray(img_rgb).save(orig_buf, format='PNG')
        orig_b64 = base64.b64encode(orig_buf.getvalue()).decode()

        # Encode blended
        blend_buf = BytesIO()
        Image.fromarray(blended).save(blend_buf, format='PNG')
        blend_b64 = base64.b64encode(blend_buf.getvalue()).decode()

        return orig_b64, blend_b64

    except Exception as e:
        print(f"Error generating heatmap: {e}")
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
    Generate audio saliency map for speech emotion.
    Returns (spectrogram_base64, saliency_base64)
    """
    try:
        if audio is None or len(audio) == 0:
            raise ValueError("Audio input is empty")

        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio)

        inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs['input_values'].to(device)

        input_values.requires_grad = True
        model.zero_grad()

        outputs = model(input_values)
        score = outputs.logits[0, emotion_idx]
        score.backward()

        if input_values.grad is None:
            raise RuntimeError("No gradients captured")

        saliency = torch.abs(input_values.grad).cpu().detach().numpy()

        if saliency.ndim == 3:
            saliency = np.mean(saliency, axis=1)[0]
        elif saliency.ndim == 2:
            saliency = saliency[0]
        saliency = saliency.reshape(-1).astype(np.float32)

        # Smooth
        if saliency.size > 11:
            kernel = np.ones(11, dtype=np.float32) / 11.0
            saliency = np.convolve(saliency, kernel, mode='same')

        # Normalize
        s_min, s_max = saliency.min(), saliency.max()
        if s_max > s_min:
            saliency = (saliency - s_min) / (s_max - s_min)
        else:
            saliency = np.zeros_like(saliency)

        # Spectrogram
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Plot 1: spectrogram
        fig1, ax1 = plt.subplots(figsize=(10, 4), dpi=100)
        img1 = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax1)
        ax1.set_title(f'Audio Spectrogram — {emotions_list[emotion_idx]}')
        fig1.colorbar(img1, ax=ax1, format='%+2.0f dB')
        spec_buf = BytesIO()
        fig1.savefig(spec_buf, format='PNG', bbox_inches='tight', dpi=100)
        spec_b64 = base64.b64encode(spec_buf.getvalue()).decode()
        plt.close(fig1)

        # Plot 2: saliency overlay + timeline
        fig2, (ax2, ax3) = plt.subplots(
            2, 1, figsize=(10, 5.5), dpi=100,
            gridspec_kw={'height_ratios': [3, 1]}, sharex=False
        )

        S_norm = (S_db - S_db.min()) / max(S_db.max() - S_db.min(), 1e-8)

        # Resize saliency to match spectrogram time axis
        sal_resized = np.interp(
            np.linspace(0, 1, S_db.shape[1]),
            np.linspace(0, 1, saliency.shape[0]),
            saliency
        )
        sal_map = np.tile(sal_resized, (S_db.shape[0], 1))

        ax2.imshow(S_norm, aspect='auto', origin='lower', cmap='viridis', interpolation='bilinear')
        ax2.imshow(sal_map, aspect='auto', origin='lower', cmap='magma', alpha=0.6, interpolation='bilinear')
        ax2.set_title(f'Audio Saliency — {emotions_list[emotion_idx]} (bright = important)')
        ax2.set_ylabel('Mel Frequency')

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
        print(f"Error generating audio saliency: {e}")
        return None, None


def create_combined_visualization(
    grad_cam_base64: str,
    saliency_base64: str,
    facial_emotion: str,
    speech_emotion: str,
    concordance: str
) -> str:
    try:
        html = f"""
        <div style="display:flex;gap:20px;padding:20px;background:#f5f5f5;border-radius:10px;">
            <div style="flex:1;">
                <h3>Facial Grad-CAM — {facial_emotion}</h3>
                <img src="data:image/png;base64,{grad_cam_base64}" style="width:100%;border-radius:8px;">
                <p style="font-size:12px;color:#666;">Red/warm = regions that influenced prediction</p>
            </div>
            <div style="flex:1;">
                <h3>Speech Saliency — {speech_emotion}</h3>
                <img src="data:image/png;base64,{saliency_base64}" style="width:100%;border-radius:8px;">
                <p style="font-size:12px;color:#666;">Bright = important frequencies</p>
            </div>
        </div>
        <div style="margin-top:20px;padding:15px;background:{'#d4edda' if concordance=='MATCH' else '#f8d7da'};border-radius:8px;text-align:center;">
            <h4 style="margin:0;">Concordance: <strong>{concordance}</strong></h4>
        </div>
        """
        return base64.b64encode(html.encode()).decode()
    except Exception as e:
        print(f"Error creating combined visualization: {e}")
        return None