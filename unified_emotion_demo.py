"""
Unified Facial + Voice Emotion Recognition Demo
Tests both modalities simultaneously using Gradio
"""

import os
from pathlib import Path
import subprocess
import tempfile
import warnings

# Suppress all warnings and verbose model loading messages
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_CACHE'] = str(Path(__file__).parent / '.cache' / 'transformers')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
warnings.filterwarnings('ignore')
import sys
# sys.stderr = open(os.devnull, 'w')  # Suppress stderr temporarily during imports

import torch
import numpy as np
import librosa
import librosa.display
import cv2
from PIL import Image
from io import BytesIO
import matplotlib
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:
    HEIF_SUPPORTED = False
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Suppress transformers logging
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)

from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification,
    AutoFeatureExtractor, 
    AutoModelForAudioClassification
)
import gradio as gr
import base64
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

# ==================== Configuration ====================
print("\n" + "="*60)
print("🎭 UNIFIED FACIAL + VOICE EMOTION RECOGNITION DEMO")
print("="*60)

EMOTIONS_FACIAL = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTIONS_SPEECH = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

EMOTION_EMOJIS = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲',
    'calm': '😌',
    'fearful': '😨',
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n📱 Device: {DEVICE}")

PROJECT_ROOT = Path(__file__).parent
FACIAL_MODEL_PATH = PROJECT_ROOT / 'models' / 'vit_emotion_model.pt'
SPEECH_MODEL_PATH = PROJECT_ROOT / 'models' / 'hubert_emotion_model.pt'

# ==================== Model Loading ====================
print("\n🔄 Loading Models...")

# Restore stderr after imports are done
sys.stderr = sys.__stderr__

# Load Facial Emotion Model (ViT)
print("\n📸 Loading Facial Emotion Model (ViT)...")
try:
    facial_processor = AutoImageProcessor.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        trust_remote_code=False,
        use_fast=False
    )
    vit_model = AutoModelForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=len(EMOTIONS_FACIAL),
        ignore_mismatched_sizes=True,
        trust_remote_code=False
    )
    
    if FACIAL_MODEL_PATH.exists():
        checkpoint = torch.load(FACIAL_MODEL_PATH, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            vit_model.load_state_dict(checkpoint['model_state_dict'])
            facial_accuracy = checkpoint.get('accuracy', 'N/A')
            print(f"   ✓ Trained weights loaded (Accuracy: {facial_accuracy:.2%})")
        else:
            vit_model.load_state_dict(checkpoint)
            print(f"   ✓ Weights loaded")
    
    vit_model = vit_model.to(DEVICE)
    vit_model.eval()
    print("   ✅ Facial model ready")
    facial_model_loaded = True
except Exception as e:
    print(f"   ❌ Error: {e}")
    facial_model_loaded = False
    vit_model = None
    facial_processor = None

# Load Speech Emotion Model (HuBERT)
print("\n🎤 Loading Speech Emotion Model (HuBERT)...")
try:
    speech_processor = AutoFeatureExtractor.from_pretrained(
        'facebook/hubert-large-ls960-ft',
        trust_remote_code=False
    )
    speech_model = AutoModelForAudioClassification.from_pretrained(
        'facebook/hubert-large-ls960-ft',
        num_labels=len(EMOTIONS_SPEECH),
        ignore_mismatched_sizes=True,
        trust_remote_code=False
    )
    
    if SPEECH_MODEL_PATH.exists():
        checkpoint = torch.load(SPEECH_MODEL_PATH, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            speech_model.load_state_dict(checkpoint['model_state_dict'])
            speech_accuracy = checkpoint.get('accuracy', 'N/A')
            print(f"   ✓ Trained weights loaded (Accuracy: {speech_accuracy:.2%})")
        else:
            speech_model.load_state_dict(checkpoint)
            print(f"   ✓ Weights loaded")
    
    speech_model = speech_model.to(DEVICE)
    speech_model.eval()
    print("   ✅ Speech model ready")
    speech_model_loaded = True
except Exception as e:
    print(f"   ❌ Error: {e}")
    speech_model_loaded = False
    speech_model = None
    speech_processor = None

print("\n" + "="*60 + "\n")

# ==================== Explainability Functions ====================

def generate_grad_cam(image_array, emotion_idx):
    """Generate Grad-CAM heatmap for facial emotion"""
    try:
        if not facial_model_loaded:
            return None
        
        # Prepare image
        if isinstance(image_array, np.ndarray):
            img = Image.fromarray(image_array.astype('uint8')) if image_array.max() > 1 else Image.fromarray((image_array * 255).astype('uint8'))
        else:
            img = image_array
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Preprocess
        inputs = facial_processor(img, return_tensors='pt').to(DEVICE)
        
        # Forward pass with gradient tracking
        inputs['pixel_values'].requires_grad_(True)
        outputs = vit_model(**inputs)
        logits = outputs.logits
        
        # Compute gradient
        target_logit = logits[0, emotion_idx]
        target_logit.backward()
        
        # Get gradients
        grads = inputs['pixel_values'].grad.data.abs()
        grads = grads.mean(dim=1)[0].cpu().numpy()
        
        # Normalize
        grads = (grads - grads.min()) / (grads.max() - grads.min() + 1e-7)
        grads = np.uint8(255 * grads)
        
        # Resize to original image size
        grads_resized = cv2.resize(grads, (img.width, img.height))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(grads_resized, cv2.COLORMAP_JET)
        
        # Blend with original
        img_array = np.array(img)
        if img_array.shape[2] == 3:
            overlay = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
        else:
            overlay = heatmap
        
        return overlay
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None

def generate_audio_saliency(audio_data, sample_rate, emotion_idx):
    """Generate audio saliency map for speech emotion"""
    try:
        if not speech_model_loaded:
            print("Speech model not loaded for saliency")
            return None
        
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, sr=sample_rate, n_mels=128, n_fft=2048, hop_length=512
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Feature extraction and prediction with gradients
        inputs = speech_processor(audio_data, sampling_rate=sample_rate, 
                                 return_tensors="pt", padding=True)
        
        # Need to enable gradients
        input_values = inputs['input_values'].to(DEVICE)
        input_values.requires_grad = True
        
        # Forward pass
        outputs = speech_model(input_values)
        logits = outputs.logits
        
        # Compute gradient
        target_logit = logits[0, emotion_idx]
        target_logit.backward()
        
        # Get gradients - shape is [1, sequence_length]
        grads = input_values.grad.data.abs()[0].cpu().numpy()
        
        # Normalize gradients
        grads = (grads - grads.min()) / (grads.max() - grads.min() + 1e-7)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        # Plot mel-spectrogram
        img = librosa.display.specshow(mel_spec_db, sr=sample_rate, hop_length=512, 
                                       x_axis='time', y_axis='mel', ax=ax1, cmap='viridis')
        ax1.set_title('Mel-Spectrogram')
        fig.colorbar(img, ax=ax1, format='%+2.0f dB')
        
        # Plot saliency
        ax2.imshow([grads], aspect='auto', cmap='hot', interpolation='bilinear')
        ax2.set_title('Saliency (Frequency Importance)')
        ax2.set_ylabel('Frequency Bins')
        ax2.set_xlabel('Time Frames')
        
        plt.tight_layout()
        
        # Convert to image
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        saliency_img = Image.open(buf)
        plt.close(fig)
        
        print(f"Generated saliency image: {np.array(saliency_img).shape}")
        return np.array(saliency_img)
    except Exception as e:
        print(f"Audio saliency error: {e}")
        return None

# ==================== Prediction Functions ====================


def predict_facial_emotion(image, explain=False):
    """Predict emotion from facial image with optional explainability"""
    if image is None:
        return None, "❌ No image provided", {}, None
    
    try:
        if not facial_model_loaded:
            return image, "❌ Facial model not loaded", {}, None
        
        # Handle file paths (e.g., HEIC images)
        if isinstance(image, str):
            if image.lower().endswith('.heic') or image.lower().endswith('.heif'):
                if not HEIF_SUPPORTED:
                    return None, "❌ HEIC format not supported. Please install pillow-heif.", {}, None
            img = Image.open(image).convert('RGB')
            image_array = np.array(img)
        # Convert to RGB
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    image = image[:,:,:3]
                img = Image.fromarray(image.astype('uint8'), 'RGB')
                image_array = image[:,:,:3] if image.shape[2] > 3 else image
            else:
                img = Image.fromarray(image.astype('uint8'), 'L').convert('RGB')
                image_array = image
        else:
            img = image.convert('RGB')
            image_array = np.array(img)
        
        # Resize for display
        img_display = img.copy()
        
        # Preprocess
        inputs = facial_processor(img, return_tensors='pt').to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = vit_model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]
            probs = np.exp(logits) / np.sum(np.exp(logits))
        
        # Get results
        top_idx = np.argmax(probs)
        top_emotion = EMOTIONS_FACIAL[top_idx]
        top_confidence = float(probs[top_idx])
        
        # Generate Grad-CAM if requested
        grad_cam_img = None
        if explain:
            print(f"Generating Grad-CAM for emotion {top_emotion} (idx={top_idx})")
            grad_cam_img = generate_grad_cam(image_array, top_idx)
            print(f"Grad-CAM generated: {grad_cam_img is not None}")
            if grad_cam_img is not None:
                print(f"Grad-CAM shape: {grad_cam_img.shape}")
        
        # Create annotated image
        img_annotated = np.array(img_display)
        h, w = img_annotated.shape[:2]
        
        # Draw background box
        cv2.rectangle(img_annotated, (10, 10), (w-10, 130), (0, 0, 0), -1)
        
        # Draw emotion text
        emoji = EMOTION_EMOJIS.get(top_emotion, '😐')
        cv2.putText(img_annotated, f"Emotion: {top_emotion.upper()}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 2)
        cv2.putText(img_annotated, f"Confidence: {top_confidence:.1%}", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 255), 2)
        cv2.putText(img_annotated, f"{emoji}", (w-60, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)
        
        # Format results
        result_text = f"### 😊 Facial Emotion: **{top_emotion.upper()}** ({top_confidence:.1%})"
        prob_dict = {f"{EMOTION_EMOJIS.get(e, '😐')} {e}": float(probs[i]) 
                     for i, e in enumerate(EMOTIONS_FACIAL)}
        
        return img_annotated, result_text, prob_dict, grad_cam_img
    
    except Exception as e:
        print(f"Facial prediction error: {e}")
        import traceback
        traceback.print_exc()
        return image, f"❌ Error: {str(e)}", {}, None

def predict_speech_emotion(audio_input, explain=False):
    """Predict emotion from voice"""
    if audio_input is None:
        return "❌ No audio provided", {}, None
    
    try:
        if not speech_model_loaded:
            return "❌ Speech model not loaded", {}, None
        
        # Extract audio data
        sample_rate, audio_data = audio_input
        
        # Handle stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize
        audio_data = audio_data.astype(np.float32)
        if np.max(np.abs(audio_data)) > 1:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Resample to 16kHz
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Feature extraction
        inputs = speech_processor(audio_data, sampling_rate=sample_rate, 
                                 return_tensors="pt", padding=True)
        
        # Predict
        with torch.no_grad():
            outputs = speech_model(inputs['input_values'].to(DEVICE))
            logits = outputs.logits.cpu().numpy()[0]
            probs = np.exp(logits) / np.sum(np.exp(logits))
        
        # Get results
        top_idx = np.argmax(probs)
        top_emotion = EMOTIONS_SPEECH[top_idx]
        top_confidence = float(probs[top_idx])
        
        # Generate saliency if requested
        saliency_img = None
        if explain:
            print(f"Generating saliency for emotion {top_emotion} (idx={top_idx})")
            saliency_img = generate_audio_saliency(audio_data, sample_rate, top_idx)
            print(f"Saliency generated: {saliency_img is not None}")
            if saliency_img is not None:
                print(f"Saliency shape: {saliency_img.shape}")
        
        # Format results
        result_text = f"### 🎤 Voice Emotion: **{top_emotion.upper()}** ({top_confidence:.1%})"
        prob_dict = {f"{EMOTION_EMOJIS.get(e, '😐')} {e}": float(probs[i]) 
                     for i, e in enumerate(EMOTIONS_SPEECH)}
        
        return result_text, prob_dict, saliency_img
    
    except Exception as e:
        print(f"Speech prediction error: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", {}, None

def extract_frame_and_audio_from_video(video_path):
    """Extract a frame and audio from video file for analysis"""
    try:
        import subprocess
        import tempfile
        
        # Read video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None, "❌ Cannot open video file"
        
        # Get middle frame
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_idx = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None, None, "❌ Cannot extract frame from video"
        
        # Convert BGR to RGB for image processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract audio using ffmpeg
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            audio_path = tmp_audio.name
        
        try:
            # Extract audio from video (WAV format with resampling to 16kHz)
            subprocess.run([
                'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1', '-y', audio_path
            ], capture_output=True, check=True, timeout=30)
            
            # Load audio using scipy to preserve wav format
            from scipy.io import wavfile
            sample_rate, audio_data = wavfile.read(audio_path)
            
            # Clean up temp audio file
            os.remove(audio_path)
            
            return frame_rgb, (sample_rate, audio_data), "✅ Video processed successfully"
        
        except Exception as e:
            return frame_rgb, None, f"⚠️ Could not extract audio: {str(e)}"
    
    except Exception as e:
        return None, None, f"❌ Error processing video: {str(e)}"

def predict_combined_from_video(video_input, explain_facial=False, explain_speech=False):
    """Analyze facial and voice from video with optional explainability"""
    if video_input is None:
        return None, "❌ No video provided", None, None
    
    try:
        # Extract frame and audio from video
        frame, audio_data, status_msg = extract_frame_and_audio_from_video(video_input)
        
        if frame is None:
            return None, status_msg, None, None
        
        # Analyze facial emotion with optional explainability
        facial_result = predict_facial_emotion(frame, explain_facial)
        facial_text = facial_result[1] if facial_result else "No facial data"
        facial_output = facial_result[0] if facial_result else None
        facial_gradcam = facial_result[3] if facial_result and len(facial_result) > 3 else None
        
        # Analyze speech emotion with optional explainability
        speech_saliency = None
        if audio_data is not None:
            speech_result = predict_speech_emotion(audio_data, explain_speech)
            speech_text = speech_result[0] if speech_result else "No speech data"
            speech_saliency = speech_result[2] if speech_result and len(speech_result) > 2 else None
        else:
            speech_text = "⚠️ No audio extracted from video"
        
        # Extract emotions for comparison
        facial_emotion = None
        speech_emotion = None
        
        if facial_result and facial_result[1] and "**" in facial_result[1]:
            parts = facial_result[1].split("**")
            if len(parts) >= 2:
                facial_emotion = parts[1].lower()
        
        if audio_data is not None and speech_result and speech_result[0] and "**" in speech_result[0]:
            parts = speech_result[0].split("**")
            if len(parts) >= 2:
                speech_emotion = parts[1].lower()
        
        # Determine combined emotion
        comparison = ""
        if facial_emotion and speech_emotion:
            if facial_emotion == speech_emotion:
                comparison = f"\n\n### ✅ **MATCH!** Both indicate {facial_emotion.upper()}"
            else:
                comparison = f"\n\n### ⚠️ **MISMATCH** - Face: {facial_emotion.upper()} | Voice: {speech_emotion.upper()}"
        
        combined_text = f"""
        {facial_text}
        
        {speech_text}
        {comparison}
        """
        
        return facial_output, combined_text, facial_gradcam, speech_saliency
    
    except Exception as e:
        print(f"Video analysis error: {e}")
        import traceback
        traceback.print_exc()
        return None, f"❌ Error analyzing video: {str(e)}", None, None

def predict_combined(image, audio_input):
    """Combined facial + voice prediction"""
    facial_result = predict_facial_emotion(image)
    speech_result = predict_speech_emotion(audio_input)
    
    facial_text = facial_result[1] if facial_result else "No facial input"
    speech_text = speech_result[0] if speech_result else "No speech input"
    
    # Extract emotions for comparison
    facial_emotion = None
    speech_emotion = None
    
    if facial_result and facial_result[1] and "**" in facial_result[1]:
        parts = facial_result[1].split("**")
        if len(parts) >= 2:
            facial_emotion = parts[1].lower()
    
    if speech_result and speech_result[0] and "**" in speech_result[0]:
        parts = speech_result[0].split("**")
        if len(parts) >= 2:
            speech_emotion = parts[1].lower()
    
    # Determine combined emotion
    comparison = ""
    if facial_emotion and speech_emotion:
        if facial_emotion == speech_emotion:
            comparison = f"✅ **MATCH!** Both indicate {facial_emotion.upper()}"
        else:
            comparison = f"⚠️ **MISMATCH** - Face: {facial_emotion.upper()} | Voice: {speech_emotion.upper()}"
    
    combined_text = f"""
    {facial_text}
    
    {speech_text}
    
    ---
    
    ### 🔗 Multimodal Analysis
    {comparison}
    """
    
    return facial_result[0] if facial_result else None, combined_text

# ==================== Gradio Interface ====================
def create_interface():
    """Create Gradio interface"""
    
    css = """
    .emotion-container {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    """
    
    with gr.Blocks(title="🎭 Facial + Voice Emotion Recognition", css=css, theme=gr.themes.Soft()) as demo:
        
        # Header
        gr.Markdown("""
        # 🎭 Unified Facial + Voice Emotion Recognition
        
        **Test both facial expressions and voice tone simultaneously!**
        
        This demo combines two state-of-the-art emotion recognition models:
        - 📸 **Vision Transformer (ViT)** for facial emotion (71.29% accuracy)
        - 🎤 **HuBERT** for speech emotion (87.50% accuracy)
        
        ---
        """)
        
        with gr.Tabs():
            # Tab 1: Facial Emotion
            with gr.Tab("📸 Facial Emotion"):
                gr.Markdown("### Analyze facial emotions from images\n")
                
                with gr.Row():
                    with gr.Column():
                        facial_image = gr.Image(
                            label="Capture or Upload Image",
                            sources=["webcam", "upload"],
                            type="numpy"
                        )
                        facial_explain = gr.Checkbox(
                            label="🔍 Show Grad-CAM Visualization",
                            value=False,
                            info="See which facial regions influenced the prediction"
                        )
                        facial_predict_btn = gr.Button("🔮 Analyze Face", variant="primary", size="lg")
                    
                    with gr.Column():
                        facial_emotion_text = gr.Markdown("Waiting for input...")
                        facial_probs = gr.Label(label="📊 Confidence Scores")
                        facial_output = gr.Image(label="✨ Annotated Result")
                        facial_gradcam = gr.Image(label="🔥 Grad-CAM Heatmap")
                
                # Callbacks
                def facial_predict_with_explain(image, explain):
                    img_out, text_out, probs_out, gradcam_out = predict_facial_emotion(image, explain)
                    if gradcam_out is not None:
                        return img_out, text_out, probs_out, gradcam_out
                    else:
                        return img_out, text_out, probs_out, None
                
                facial_predict_btn.click(
                    fn=facial_predict_with_explain,
                    inputs=[facial_image, facial_explain],
                    outputs=[facial_output, facial_emotion_text, facial_probs, facial_gradcam]
                )
            
            # Tab 2: Speech Emotion
            with gr.Tab("🎤 Speech Emotion"):
                gr.Markdown("### Analyze speech emotions from audio\n")
                
                with gr.Row():
                    with gr.Column():
                        speech_audio = gr.Audio(
                            label="Record or Upload Audio",
                            sources=["microphone", "upload"],
                            type="numpy"
                        )
                        speech_explain = gr.Checkbox(
                            label="📊 Show Audio Saliency Map",
                            value=False,
                            info="See which frequencies influenced the prediction"
                        )
                        speech_predict_btn = gr.Button("🔮 Analyze Voice", variant="primary", size="lg")
                    
                    with gr.Column():
                        speech_emotion_text = gr.Markdown("Waiting for input...")
                        speech_probs = gr.Label(label="📊 Confidence Scores")
                        speech_saliency = gr.Image(label="📊 Audio Saliency Map")
                
                # Callbacks
                def speech_predict_with_explain(audio, explain):
                    text_out, probs_out, saliency_out = predict_speech_emotion(audio, explain)
                    if saliency_out is not None:
                        return text_out, probs_out, saliency_out
                    else:
                        return text_out, probs_out, None
                
                speech_predict_btn.click(
                    fn=speech_predict_with_explain,
                    inputs=[speech_audio, speech_explain],
                    outputs=[speech_emotion_text, speech_probs, speech_saliency]
                )
            
            # Tab 3: Combined Analysis
            with gr.Tab("🔗 Combined Analysis"):
                gr.Markdown("### Analyze facial and voice together for multimodal comparison\n")
                
                # Mode selection
                mode = gr.Radio(
                    choices=["🎥 Video Upload (MP4)", "📸 Separate Images & Audio"],
                    value="📸 Separate Images & Audio",
                    label="Choose Analysis Mode",
                    info="Select how you want to provide input"
                )
                
                # Video Mode
                with gr.Group(visible=False) as video_mode_group:
                    gr.Markdown("#### 🎥 Upload or Record Video\nThe system will automatically extract facial and voice for analysis\n")
                    
                    with gr.Row():
                        video_input = gr.Video(
                            label="Capture or Upload Video (MP4)",
                            sources=["webcam", "upload"],
                            format="mp4"
                        )
                        with gr.Column():
                            video_explain_facial = gr.Checkbox(
                                label="🔍 Show Facial Grad-CAM",
                                value=False
                            )
                            video_explain_speech = gr.Checkbox(
                                label="📊 Show Audio Saliency",
                                value=False
                            )
                            video_predict_btn = gr.Button("🎬 Analyze Video", variant="primary", size="lg")
                            video_emotion_text = gr.Markdown("Waiting for video...")
                    
                    video_output = gr.Image(label="✨ Extracted Frame")
                    
                    with gr.Row():
                        video_gradcam = gr.Image(label="🔥 Facial Grad-CAM")
                        video_saliency = gr.Image(label="📊 Audio Saliency")
                
                video_predict_btn.click(
                    fn=predict_combined_from_video,
                    inputs=[video_input, video_explain_facial, video_explain_speech],
                    outputs=[video_output, video_emotion_text, video_gradcam, video_saliency]
                )
                
                # Separate Mode
                with gr.Group(visible=True) as separate_mode_group:
                    gr.Markdown("#### 📸 Upload Facial Image and Audio Separately\n")
                    
                    with gr.Row():
                        with gr.Column():
                            combined_image = gr.Image(
                                label="📸 Capture or Upload Image",
                                sources=["webcam", "upload"],
                                type="numpy"
                            )
                        
                        with gr.Column():
                            combined_audio = gr.Audio(
                                label="🎤 Record or Upload Audio",
                                sources=["microphone", "upload"],
                                type="numpy"
                            )
                    
                    with gr.Row():
                        combined_explain_facial = gr.Checkbox(
                            label="🔍 Show Facial Grad-CAM",
                            value=False
                        )
                        combined_explain_speech = gr.Checkbox(
                            label="📊 Show Audio Saliency",
                            value=False
                        )
                    
                    combined_predict_btn = gr.Button(
                        "🚀 Analyze Both",
                        variant="primary",
                        size="lg"
                    )
                    
                    combined_output = gr.Image(label="✨ Annotated Face")
                    combined_text = gr.Markdown("Waiting for inputs...")
                    
                    with gr.Row():
                        combined_gradcam = gr.Image(label="🔥 Facial Grad-CAM")
                        combined_saliency = gr.Image(label="📊 Audio Saliency")
                
                def combined_predict_with_explain(image, audio, explain_facial, explain_speech):
                    # Get facial prediction
                    facial_result = predict_facial_emotion(image, explain_facial)
                    facial_img = facial_result[0] if facial_result else None
                    facial_gradcam = facial_result[3] if facial_result and len(facial_result) > 3 else None
                    
                    # Get speech prediction
                    speech_result = predict_speech_emotion(audio, explain_speech)
                    speech_saliency = speech_result[2] if speech_result and len(speech_result) > 2 else None
                    
                    # Get combined text
                    _, combined_txt = predict_combined(image, audio)
                    
                    return (
                        facial_img,
                        combined_txt,
                        facial_gradcam,
                        speech_saliency
                    )
                
                combined_predict_btn.click(
                    fn=combined_predict_with_explain,
                    inputs=[combined_image, combined_audio, combined_explain_facial, combined_explain_speech],
                    outputs=[combined_output, combined_text, combined_gradcam, combined_saliency]
                )
                
                # Toggle between modes
                def toggle_mode(selected_mode):
                    if "Video" in selected_mode:
                        return gr.Group(visible=True), gr.Group(visible=False)
                    else:
                        return gr.Group(visible=False), gr.Group(visible=True)
                
                mode.change(
                    fn=toggle_mode,
                    inputs=mode,
                    outputs=[video_mode_group, separate_mode_group]
                )
            
            # Tab 4: Model Info
            with gr.Tab("ℹ️ Model Information"):
                info_text = f"""
                ## 📊 Model Details
                
                ### 📸 Facial Emotion Recognition (ViT)
                - **Architecture:** Vision Transformer (google/vit-base-patch16-224-in21k)
                - **Training Data:** FER2013 Dataset (35,887 images)
                - **Emotions:** 7 classes (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
                - **Accuracy:** 71.29%
                - **Model Size:** 327MB
                - **Input:** RGB Images (224×224)
                
                ### 🎤 Speech Emotion Recognition (HuBERT)
                - **Architecture:** HuBERT Large (facebook/hubert-large-ls960-ft)
                - **Training Data:** RAVDESS Dataset (1,440 audio files)
                - **Emotions:** 8 classes (Angry, Calm, Disgust, Fearful, Happy, Neutral, Sad, Surprised)
                - **Accuracy:** 87.50%
                - **Model Size:** ~360MB
                - **Input:** 16kHz Mono Audio
                
                ### 💻 System Info
                - **Device:** {DEVICE}
                - **Facial Model Loaded:** {'✅ Yes' if facial_model_loaded else '❌ No'}
                - **Speech Model Loaded:** {'✅ Yes' if speech_model_loaded else '❌ No'}
                
                ---
                
                ## 🎯 How to Use
                
                **Separate Testing:**
                1. Use the 🔀 tab to test facial or voice separately
                2. Capture/upload image and click "🔮 Analyze Face"
                3. Record/upload audio and click "🔮 Analyze Voice"
                4. View confidence scores for each emotion
                
                **Combined Analysis:**
                1. Use the 🔗 tab for multimodal testing
                2. Capture/upload both image and audio
                3. Click "🚀 Analyze Both"
                4. Compare facial expression with voice tone
                5. Check for emotional concordance (match/mismatch)
                """
                
                gr.Markdown(info_text)
    
    return demo

# ==================== Main ====================
if __name__ == "__main__":
    print("🚀 Launching Unified Emotion Recognition Demo...")
    print(f"   URL: http://localhost:7860\n")
    
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # Local access only
        show_error=True
    )
