"""Explainability utilities for multimodal emotion recognition outputs."""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import torch
import torch.nn as nn
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

from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform


# ==================== MODEL WRAPPER ====================
class ViTLogitsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Grad-CAM expects a standard forward() that returns logits for the selected class.
        return self.model(pixel_values=x).logits


# ==================== FACIAL EXPLAINABILITY ====================
def generate_grad_cam(image, model, processor, emotion_idx, emotions_list, device):
    try:
        img_rgb = np.array(image.convert('RGB'))
        h, w    = img_rgb.shape[:2]
        img_pil = Image.fromarray(img_rgb)

        inputs       = processor(img_pil, return_tensors='pt').to(device)
        input_tensor = inputs['pixel_values']

        wrapped_model = ViTLogitsWrapper(model)
        wrapped_model.eval()

        # Try multiple layers because the last block can become too saturated for a usable heatmap.
        layers_to_try = [
            model.vit.encoder.layer[-1].layernorm_after,
            model.vit.encoder.layer[-2].layernorm_after,
            model.vit.encoder.layer[-3].layernorm_after,
        ]

        cam_map     = None
        method_used = None

        for i, layer in enumerate(layers_to_try):
            try:
                cam = GradCAM(
                    model=wrapped_model,
                    target_layers=[layer],
                    reshape_transform=vit_reshape_transform,
                )
                targets       = [ClassifierOutputTarget(emotion_idx)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                result        = grayscale_cam[0]

                # Reject degenerate maps so the UI never shows a blank explanation as if it were valid.
                if result.max() > 0.01:
                    cam_map     = result
                    method_used = f"GradCAM (encoder block {12 - (i+1)})"
                    break
                else:
                    print(f"[explainability] layer[-{i+1}] all zeros, trying next")

            except Exception as e:
                print(f"[explainability] GradCAM layer[-{i+1}] failed: {e}")

        # Final fallback: EigenCAM gives a stable PCA-based map when gradients are unhelpful.
        if cam_map is None:
            print("[explainability] All GradCAM layers zero, using EigenCAM")
            try:
                eigen = EigenCAM(
                    model=wrapped_model,
                    target_layers=[model.vit.encoder.layer[-1].layernorm_after],
                    reshape_transform=vit_reshape_transform,
                )
                grayscale_cam = eigen(input_tensor=input_tensor)
                cam_map       = grayscale_cam[0]
                method_used   = "EigenCAM"
            except Exception as e:
                print(f"[explainability] EigenCAM failed: {e}")
                return None, None

        print(f"[explainability] {method_used} — min={cam_map.min():.3f}, max={cam_map.max():.3f}")

        # Upscale and smooth the heatmap so it overlays cleanly on the source image.
        cam_resized = cv2.resize(cam_map.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        cam_resized = cv2.GaussianBlur(cam_resized, (13, 13), 0)

        c_min, c_max = cam_resized.min(), cam_resized.max()
        if c_max > c_min:
            cam_resized = (cam_resized - c_min) / (c_max - c_min)

        # Build the colored overlay and blend only the most salient regions.
        cam_uint8   = np.uint8(255 * cam_resized)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        threshold  = np.percentile(cam_resized, 70)
        blend_mask = (cam_resized > threshold).astype(np.float32)
        blend_mask = cv2.GaussianBlur(blend_mask, (31, 31), 0)[..., None]

        blended = (
            (1 - blend_mask * 0.65) * img_rgb.astype(np.float32)
            + blend_mask * 0.65 * heatmap_rgb.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

        orig_buf = BytesIO()
        Image.fromarray(img_rgb).save(orig_buf, format='PNG')
        orig_b64 = base64.b64encode(orig_buf.getvalue()).decode()

        blend_buf = BytesIO()
        Image.fromarray(blended).save(blend_buf, format='PNG')
        blend_b64 = base64.b64encode(blend_buf.getvalue()).decode()

        return orig_b64, blend_b64

    except Exception as e:
        print(f"[explainability] GradCAM generation failed: {e}")
        return None, None


# ==================== AUDIO EXPLAINABILITY ====================
def generate_audio_saliency(audio, model, processor, emotion_idx, emotions_list, device, sr=16000):
    try:
        if audio is None or len(audio) == 0:
            raise ValueError("Audio input is empty")

        # Sanitize the audio before passing it into the speech backbone.
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio)

        inputs       = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs['input_values'].to(device)

        input_values.requires_grad = True
        model.zero_grad()

        outputs = model(input_values)
        score   = outputs.logits[0, emotion_idx]
        score.backward()

        if input_values.grad is None:
            raise RuntimeError("No gradients captured")

        saliency = torch.abs(input_values.grad).cpu().detach().numpy()

        if saliency.ndim == 3:
            saliency = np.mean(saliency, axis=1)[0]
        elif saliency.ndim == 2:
            saliency = saliency[0]
        saliency = saliency.reshape(-1).astype(np.float32)

        if saliency.size > 11:
            # Smooth the gradient spikes so the curve is readable in the plot.
            kernel   = np.ones(11, dtype=np.float32) / 11.0
            saliency = np.convolve(saliency, kernel, mode='same')

        s_min, s_max = saliency.min(), saliency.max()
        if s_max > s_min:
            saliency = (saliency - s_min) / (s_max - s_min)
        else:
            saliency = np.zeros_like(saliency)

        # Build both the spectrogram and the saliency overlay for a side-by-side explanation.
        S    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        fig1, ax1 = plt.subplots(figsize=(10, 4), dpi=100)
        img1 = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax1)
        ax1.set_title(f'Audio Spectrogram — {emotions_list[emotion_idx]}')
        fig1.colorbar(img1, ax=ax1, format='%+2.0f dB')
        spec_buf = BytesIO()
        fig1.savefig(spec_buf, format='PNG', bbox_inches='tight', dpi=100)
        spec_b64 = base64.b64encode(spec_buf.getvalue()).decode()
        plt.close(fig1)

        fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(10, 5.5), dpi=100,
                                         gridspec_kw={'height_ratios': [3, 1]}, sharex=False)

        # Normalize the spectrogram so the saliency colors stay visible across different recordings.
        S_norm      = (S_db - S_db.min()) / max(S_db.max() - S_db.min(), 1e-8)
        sal_resized = np.interp(np.linspace(0, 1, S_db.shape[1]),
                                np.linspace(0, 1, saliency.shape[0]), saliency)
        sal_map     = np.tile(sal_resized, (S_db.shape[0], 1))

        ax2.imshow(S_norm, aspect='auto', origin='lower', cmap='viridis', interpolation='bilinear')
        ax2.imshow(sal_map, aspect='auto', origin='lower', cmap='magma', alpha=0.6, interpolation='bilinear')
        ax2.set_title(f'Audio Saliency — {emotions_list[emotion_idx]} (bright = important)')
        ax2.set_ylabel('Mel Frequency')

        # Highlight the strongest time steps to give the user a clear peak view.
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


# ==================== COMBINED VISUALIZATION ====================
def create_combined_visualization(grad_cam_base64, saliency_base64, facial_emotion, speech_emotion, concordance):
    try:
        # Use a soft status tint so the combined report communicates agreement at a glance.
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
        return base64.b64encode(html.encode()).decode()
    except Exception as e:
        print(f"[explainability] Combined visualisation failed: {e}")
        return None
