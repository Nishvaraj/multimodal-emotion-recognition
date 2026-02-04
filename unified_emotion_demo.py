"""
Unified Facial + Voice Emotion Recognition Demo
Tests both modalities simultaneously using Gradio
"""

import os
from pathlib import Path
import subprocess
import tempfile

# Disable safetensors auto-conversion to prevent background download issues
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_CACHE'] = str(Path(__file__).parent / '.cache' / 'transformers')

import torch
import numpy as np
import librosa
import cv2
from PIL import Image
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification,
    AutoFeatureExtractor, 
    AutoModelForAudioClassification
)
import gradio as gr
import requests
from io import BytesIO
from datetime import datetime
from scipy.io import wavfile

# ==================== Configuration ====================
# Backend API
BACKEND_API = "http://127.0.0.1:8000/api"

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
FACIAL_MODEL_PATH = PROJECT_ROOT / 'models' / 'phase2' / 'vit_emotion_model.pt'
SPEECH_MODEL_PATH = PROJECT_ROOT / 'models' / 'phase3' / 'hubert_emotion_model.pt'

# ==================== Model Loading ====================
print("\n🔄 Loading Models...")

# Load Facial Emotion Model (ViT)
print("\n📸 Loading Facial Emotion Model (ViT)...")
try:
    facial_processor = AutoImageProcessor.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        trust_remote_code=False
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

# ==================== Prediction Functions ====================

def predict_facial_emotion(image):
    """Predict emotion from facial image"""
    if image is None:
        return None, "❌ No image provided", {}
    
    try:
        if not facial_model_loaded:
            return image, "❌ Facial model not loaded", {}
        
        # Convert to RGB
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    image = image[:,:,:3]
                img = Image.fromarray(image.astype('uint8'), 'RGB')
            else:
                img = Image.fromarray(image.astype('uint8'), 'L').convert('RGB')
        else:
            img = image.convert('RGB')
        
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
        
        return img_annotated, result_text, prob_dict
    
    except Exception as e:
        print(f"Facial prediction error: {e}")
        import traceback
        traceback.print_exc()
        return image, f"❌ Error: {str(e)}", {}

def predict_facial_emotion_with_gradcam(image):
    """Predict emotion from facial image WITH Grad-CAM explainability via backend"""
    if image is None:
        return None, "❌ No image provided", {}, None
    
    try:
        # Convert image to PIL
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = image[:,:,:3]
            pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
        else:
            pil_image = image.convert('RGB')
        
        # Send to backend for Grad-CAM
        img_byte_arr = BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        response = requests.post(
            f"{BACKEND_API}/predict/facial",
            files={'file': ('image.png', img_byte_arr, 'image/png')}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                emotion = data.get('emotion', 'unknown')
                confidence = data.get('confidence', 0)
                probabilities = data.get('probabilities', {})
                grad_cam_image = data.get('grad_cam_image')
                
                result_text = f"### 😊 Facial Emotion: **{emotion.upper()}** ({confidence:.1%})\\n\\n🔍 **Grad-CAM:** Heatmap shows facial regions that influenced the prediction. Red = high impact."
                
                # Decode Grad-CAM image
                gradcam_display = None
                if grad_cam_image:
                    if isinstance(grad_cam_image, str) and not grad_cam_image.startswith('data:'):
                        gradcam_display = f"data:image/png;base64,{grad_cam_image}"
                    else:
                        gradcam_display = grad_cam_image
                
                return np.array(pil_image), result_text, probabilities, gradcam_display
        
        return None, f"❌ Backend error: {response.status_code}", {}, None
    
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}", {}, None

def predict_speech_emotion(audio_input):
    """Predict emotion from voice"""
    if audio_input is None:
        return "❌ No audio provided", {}
    
    try:
        if not speech_model_loaded:
            return "❌ Speech model not loaded", {}
        
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
        
        # Format results
        result_text = f"### 🎤 Voice Emotion: **{top_emotion.upper()}** ({top_confidence:.1%})"
        prob_dict = {f"{EMOTION_EMOJIS.get(e, '😐')} {e}": float(probs[i]) 
                     for i, e in enumerate(EMOTIONS_SPEECH)}
        
        return result_text, prob_dict
    
    except Exception as e:
        print(f"Speech prediction error: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", {}

def predict_speech_emotion_with_saliency(audio_input):
    """Predict emotion from voice WITH saliency map via backend"""
    if audio_input is None:
        return "❌ No audio provided", {}, None
    
    try:
        sample_rate, audio_data = audio_input
        
        # Handle stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize
        audio_data = audio_data.astype(np.float32)
        if np.max(np.abs(audio_data)) > 1:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Save to temporary WAV file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            wavfile.write(tmp.name, sample_rate, (audio_data * 32767).astype(np.int16))
            temp_path = tmp.name
        
        try:
            with open(temp_path, 'rb') as audio_file:
                response = requests.post(
                    f"{BACKEND_API}/predict/speech",
                    files={'file': ('audio.wav', audio_file, 'audio/wav')}
                )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    emotion = data.get('emotion', 'unknown')
                    confidence = data.get('confidence', 0)
                    probabilities = data.get('probabilities', {})
                    saliency_map = data.get('saliency_map')
                    
                    result_text = f"### 🎤 Voice Emotion: **{emotion.upper()}** ({confidence:.1%})\\n\\n🔊 **Saliency Map:** Shows audio frequencies most important for prediction."
                    
                    saliency_display = None
                    if saliency_map:
                        if isinstance(saliency_map, str) and not saliency_map.startswith('data:'):
                            saliency_display = f"data:image/png;base64,{saliency_map}"
                        else:
                            saliency_display = saliency_map
                    
                    return result_text, probabilities, saliency_display
            
            return f"❌ Backend error: {response.status_code}", {}, None
        
        finally:
            os.remove(temp_path)
    
    except Exception as e:
        print(f"Saliency error: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", {}, None

def fetch_all_sessions():
    """Fetch all sessions from database"""
    try:
        response = requests.get(f"{BACKEND_API.replace('/api', '')}/api/sessions")
        if response.status_code == 200:
            sessions = response.json().get('sessions', [])
            if sessions:
                table_data = [[s.get('session_id', '?'), s.get('date', '?'), s.get('facial_emotion', 'N/A'), s.get('speech_emotion', 'N/A'), s.get('concordance', 'N/A')] for s in sessions]
                return table_data
    except Exception as e:
        print(f"Error fetching sessions: {e}")
    return [["No sessions", "-", "-", "-", "-"]]

def get_session_details(session_id):
    """Get details for a specific session"""
    try:
        response = requests.get(f"{BACKEND_API.replace('/api', '')}/api/sessions/{session_id}")
        if response.status_code == 200:
            session = response.json()
            details = f"### 📊 Session Details\\n\\n**ID:** `{session.get('session_id', 'N/A')}`\\n**Date:** {session.get('date', 'N/A')}\\n\\n**Facial:** {session.get('facial_emotion', 'N/A')} ({session.get('facial_confidence', 'N/A')})\\n**Speech:** {session.get('speech_emotion', 'N/A')} ({session.get('speech_confidence', 'N/A')})\\n**Concordance:** {session.get('concordance', 'N/A')}"
            return details
    except Exception as e:
        return f"❌ Error: {str(e)}"
    return "❌ Not found"

def delete_session(session_id):
    """Delete a session"""
    try:
        response = requests.delete(f"{BACKEND_API.replace('/api', '')}/api/sessions/{session_id}")
        if response.status_code == 200:
            return f"✅ Deleted: `{session_id}`"
    except Exception as e:
        return f"❌ Error: {str(e)}"
    return "❌ Failed"

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

def predict_combined_from_video(video_input):
    """Analyze facial and voice from video"""
    if video_input is None:
        return None, "❌ No video provided"
    
    try:
        # Extract frame and audio from video
        frame, audio_data, status_msg = extract_frame_and_audio_from_video(video_input)
        
        if frame is None:
            return None, status_msg
        
        # Analyze facial emotion
        facial_result = predict_facial_emotion(frame)
        facial_text = facial_result[1] if facial_result else "No facial data"
        facial_output = facial_result[0] if facial_result else None
        
        # Analyze speech emotion
        if audio_data is not None:
            speech_result = predict_speech_emotion(audio_data)
            speech_text = speech_result[0] if speech_result else "No speech data"
        else:
            speech_text = "⚠️ No audio extracted from video"
        
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
                comparison = f"\n\n### ✅ **MATCH!** Both indicate {facial_emotion.upper()}"
            else:
                comparison = f"\n\n### ⚠️ **MISMATCH** - Face: {facial_emotion.upper()} | Voice: {speech_emotion.upper()}"
        
        combined_text = f"""
        {facial_text}
        
        {speech_text}
        {comparison}
        """
        
        return facial_output, combined_text
    
    except Exception as e:
        print(f"Video analysis error: {e}")
        import traceback
        traceback.print_exc()
        return None, f"❌ Error analyzing video: {str(e)}"

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
    
    with gr.Blocks(title="🎭 Facial + Voice Emotion Recognition") as demo:
        
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
                        facial_predict_btn = gr.Button("🔮 Analyze Face", variant="primary", size="lg")
                    
                    with gr.Column():
                        facial_emotion_text = gr.Markdown("Waiting for input...")
                        facial_probs = gr.Label(label="📊 Confidence Scores")
                        facial_output = gr.Image(label="✨ Annotated Result")
                
                # Callbacks
                facial_predict_btn.click(
                    fn=predict_facial_emotion,
                    inputs=facial_image,
                    outputs=[facial_output, facial_emotion_text, facial_probs]
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
                        speech_predict_btn = gr.Button("🔮 Analyze Voice", variant="primary", size="lg")
                    
                    with gr.Column():
                        speech_emotion_text = gr.Markdown("Waiting for input...")
                        speech_probs = gr.Label(label="📊 Confidence Scores")
                
                # Callbacks
                speech_predict_btn.click(
                    fn=predict_speech_emotion,
                    inputs=speech_audio,
                    outputs=[speech_emotion_text, speech_probs]
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
                            video_predict_btn = gr.Button("🎬 Analyze Video", variant="primary", size="lg")
                            video_emotion_text = gr.Markdown("Waiting for video...")
                            video_output = gr.Image(label="✨ Extracted Frame")
                
                video_predict_btn.click(
                    fn=predict_combined_from_video,
                    inputs=video_input,
                    outputs=[video_output, video_emotion_text]
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
                    
                    combined_predict_btn = gr.Button(
                        "🚀 Analyze Both",
                        variant="primary",
                        size="lg"
                    )
                    
                    combined_output = gr.Image(label="✨ Annotated Face")
                    combined_text = gr.Markdown("Waiting for inputs...")
                
                combined_predict_btn.click(
                    fn=predict_combined,
                    inputs=[combined_image, combined_audio],
                    outputs=[combined_output, combined_text]
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
            
            # Tab 5: Explainability (Grad-CAM & Saliency)
            with gr.Tab("🔍 Explainability"):
                gr.Markdown("### Model Decision-Making with Grad-CAM & Saliency\\n")
                gr.Markdown("Understand which facial regions and audio frequencies influence predictions\\n")
                
                with gr.Tabs():
                    # Grad-CAM Tab
                    with gr.Tab("📸 Facial Grad-CAM"):
                        with gr.Row():
                            with gr.Column():
                                gradcam_image = gr.Image(
                                    label="Capture or Upload Image",
                                    sources=["webcam", "upload"],
                                    type="numpy"
                                )
                                gradcam_btn = gr.Button("🔥 Generate Grad-CAM", variant="primary", size="lg")
                            
                            with gr.Column():
                                gradcam_text = gr.Markdown("Waiting for input...")
                                gradcam_output = gr.Image(label="🔥 Grad-CAM Heatmap")
                                gradcam_probs = gr.Label(label="📊 Confidence Scores")
                        
                        gradcam_btn.click(
                            fn=predict_facial_emotion_with_gradcam,
                            inputs=gradcam_image,
                            outputs=[gradcam_image, gradcam_text, gradcam_probs, gradcam_output]
                        )
                    
                    # Saliency Tab
                    with gr.Tab("🎤 Audio Saliency"):
                        with gr.Row():
                            with gr.Column():
                                saliency_audio = gr.Audio(
                                    label="Record or Upload Audio",
                                    sources=["microphone", "upload"],
                                    type="numpy"
                                )
                                saliency_btn = gr.Button("📊 Generate Saliency", variant="primary", size="lg")
                            
                            with gr.Column():
                                saliency_text = gr.Markdown("Waiting for input...")
                                saliency_output = gr.Image(label="📊 Saliency Heatmap")
                                saliency_probs = gr.Label(label="📊 Confidence Scores")
                        
                        saliency_btn.click(
                            fn=predict_speech_emotion_with_saliency,
                            inputs=saliency_audio,
                            outputs=[saliency_text, saliency_probs, saliency_output]
                        )
            
            # Tab 6: Session History (Database)
            with gr.Tab("💾 Session History"):
                gr.Markdown("### View and Manage Saved Analysis Sessions\\n")
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Sessions", variant="primary", size="lg")
                
                sessions_table = gr.Dataframe(
                    label="📋 Sessions",
                    headers=["Session ID", "Date", "Face Emotion", "Voice Emotion", "Concordance"],
                    interactive=False
                )
                
                def load_sessions():
                    return fetch_all_sessions()
                
                refresh_btn.click(fn=load_sessions, outputs=sessions_table)
                
                gr.Markdown("---")
                gr.Markdown("### Session Details\\n")
                
                with gr.Row():
                    session_id_input = gr.Textbox(
                        label="Session ID",
                        placeholder="Enter session ID to view details"
                    )
                    view_btn = gr.Button("👁️ View Details", variant="primary")
                    delete_btn = gr.Button("🗑️ Delete", variant="stop")
                
                session_details = gr.Markdown("Select a session ID to view details")
                
                view_btn.click(
                    fn=get_session_details,
                    inputs=session_id_input,
                    outputs=session_details
                )
                
                delete_btn.click(
                    fn=delete_session,
                    inputs=session_id_input,
                    outputs=session_details
                )
            
            # Tab 7: Model Info
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
    
    css = """
    .emotion-container {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    """
    
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css=css
    )
