from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import numpy as np
import cv2
import librosa
from PIL import Image
from io import BytesIO
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoFeatureExtractor, AutoModelForAudioClassification
import tempfile
import os
import sys

# Add services to path
sys.path.insert(0, str(Path(__file__).parent))

# Import explainability module
from services.explainability import generate_grad_cam, generate_audio_saliency, create_combined_visualization

app = FastAPI(title="Multi-Modal Emotion Recognition API", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
EMOTIONS_FACIAL = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTIONS_SPEECH = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model storage
vit_model = None
facial_processor = None
speech_model = None
speech_processor = None

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
FACIAL_MODEL_PATH = PROJECT_ROOT / 'models' / 'phase2' / 'vit_emotion_model.pt'
SPEECH_MODEL_PATH = PROJECT_ROOT / 'models' / 'phase3' / 'hubert_emotion_model.pt'

print(f"Device: {DEVICE}")
print(f"Project root: {PROJECT_ROOT}")

# ========== MODEL LOADING ==========

def load_facial_model():
    """Load ViT model for facial emotion"""
    global vit_model, facial_processor
    try:
        print("Loading Facial Emotion Model (ViT)...")
        facial_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        vit_model = AutoModelForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=len(EMOTIONS_FACIAL),
            ignore_mismatched_sizes=True
        )
        
        if FACIAL_MODEL_PATH.exists():
            checkpoint = torch.load(FACIAL_MODEL_PATH, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                vit_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                vit_model.load_state_dict(checkpoint)
            print(f"✓ Loaded ViT checkpoint")
        
        vit_model = vit_model.to(DEVICE)
        vit_model.eval()
        print("✓ Facial model ready")
        return True
    except Exception as e:
        print(f"❌ Error loading facial model: {e}")
        return False

def load_speech_model():
    """Load HuBERT model for speech emotion"""
    global speech_model, speech_processor
    try:
        print("Loading Speech Emotion Model (HuBERT)...")
        speech_processor = AutoFeatureExtractor.from_pretrained('facebook/hubert-large-ls960-ft')
        speech_model = AutoModelForAudioClassification.from_pretrained(
            'facebook/hubert-large-ls960-ft',
            num_labels=len(EMOTIONS_SPEECH),
            ignore_mismatched_sizes=True
        )
        
        if SPEECH_MODEL_PATH.exists():
            checkpoint = torch.load(SPEECH_MODEL_PATH, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                speech_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                speech_model.load_state_dict(checkpoint)
            print(f"✓ Loaded HuBERT checkpoint")
        
        speech_model = speech_model.to(DEVICE)
        speech_model.eval()
        print("✓ Speech model ready")
        return True
    except Exception as e:
        print(f"❌ Error loading speech model: {e}")
        return False

# Load on startup
facial_loaded = load_facial_model()
speech_loaded = load_speech_model()

# ========== VIDEO PROCESSOR ==========

class VideoProcessor:
    @staticmethod
    def extract_frames_and_audio(video_path: str, fps_sample: int = 5):
        """Extract frames and audio from video"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % fps_sample == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            
            frame_count += 1
        
        cap.release()
        
        # Extract audio
        audio, sr = librosa.load(video_path, sr=16000, mono=True)
        
        return frames, audio, sr, fps

# ========== PREDICTION FUNCTIONS ==========

def predict_facial_emotion(image: Image.Image, generate_explainability: bool = False):
    """Predict emotion from image"""
    try:
        if vit_model is None:
            return None
        
        inputs = facial_processor(image, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            outputs = vit_model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]
            probs = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
        
        top_idx = np.argmax(probs)
        result = {
            "emotion": EMOTIONS_FACIAL[top_idx],
            "confidence": float(probs[top_idx]),
            "probabilities": {e: float(p) for e, p in zip(EMOTIONS_FACIAL, probs)}
        }
        
        # Generate Grad-CAM if requested
        if generate_explainability:
            try:
                original_base64, heatmap_base64 = generate_grad_cam(
                    image,
                    vit_model,
                    facial_processor,
                    top_idx,
                    EMOTIONS_FACIAL,
                    DEVICE
                )
                result["original_image"] = original_base64
                result["grad_cam"] = heatmap_base64
            except Exception as e:
                print(f"Warning: Could not generate Grad-CAM: {e}")
        
        return result
    except Exception as e:
        print(f"Error predicting facial emotion: {e}")
        return None

def predict_speech_emotion(audio: np.ndarray, sr: int = 16000, generate_explainability: bool = False):
    """Predict emotion from audio"""
    try:
        if speech_model is None:
            return None
        
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        inputs = speech_processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = speech_model(inputs['input_values'].to(DEVICE))
            logits = outputs.logits.cpu().numpy()[0]
            probs = np.exp(logits) / np.sum(np.exp(logits))
        
        top_idx = np.argmax(probs)
        result = {
            "emotion": EMOTIONS_SPEECH[top_idx],
            "confidence": float(probs[top_idx]),
            "probabilities": {e: float(p) for e, p in zip(EMOTIONS_SPEECH, probs)}
        }
        
        # Generate audio saliency if requested
        if generate_explainability:
            try:
                spec_base64, saliency_base64 = generate_audio_saliency(
                    audio,
                    speech_model,
                    speech_processor,
                    top_idx,
                    EMOTIONS_SPEECH,
                    DEVICE,
                    sr=16000
                )
                result["spectrogram"] = spec_base64
                result["saliency"] = saliency_base64
            except Exception as e:
                print(f"Warning: Could not generate audio saliency: {e}")
        
        return result
    except Exception as e:
        print(f"Error predicting speech emotion: {e}")
        return None

# ========== API ENDPOINTS ==========

@app.get("/")
async def root():
    return {"message": "Multi-Modal Emotion Recognition API v2.0", "status": "active"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "facial_model": facial_loaded,
        "speech_model": speech_loaded,
        "device": str(DEVICE)
    }

@app.post("/api/predict/facial")
async def predict_facial(file: UploadFile = File(...), explain: bool = False):
    """Predict emotion from image"""
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        result = predict_facial_emotion(image, generate_explainability=explain)
        return {"success": True, **result} if result else {"success": False, "error": "Prediction failed"}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/predict/speech")
async def predict_speech(file: UploadFile = File(...), explain: bool = False):
    """Predict emotion from audio"""
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        
        try:
            audio, sr = librosa.load(tmp_path, sr=16000)
            result = predict_speech_emotion(audio, sr, generate_explainability=explain)
            return {"success": True, **result} if result else {"success": False, "error": "Prediction failed"}
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/predict/combined")
async def predict_combined(image_file: UploadFile = File(...), audio_file: UploadFile = File(...), explain: bool = False):
    """Predict emotions from both image and audio, then compare results"""
    try:
        # Process image
        image_contents = await image_file.read()
        image = Image.open(BytesIO(image_contents)).convert('RGB')
        facial_result = predict_facial_emotion(image, generate_explainability=explain)
        
        # Process audio
        audio_contents = await audio_file.read()
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(audio_contents)
            tmp_path = tmp.name
        
        try:
            audio, sr = librosa.load(tmp_path, sr=16000)
            speech_result = predict_speech_emotion(audio, sr, generate_explainability=explain)
        finally:
            os.unlink(tmp_path)
        
        # Extract emotions
        facial_emotion = facial_result["emotion"] if facial_result else None
        facial_confidence = facial_result["confidence"] if facial_result else 0.0
        
        speech_emotion = speech_result["emotion"] if speech_result else None
        speech_confidence = speech_result["confidence"] if speech_result else 0.0
        
        # Compare emotions (concordance)
        concordance = None
        if facial_emotion and speech_emotion:
            if facial_emotion == speech_emotion:
                concordance = "MATCH"
            else:
                concordance = "MISMATCH"
        
        # Determine combined emotion (weighted by confidence)
        combined_emotion = None
        combined_confidence = 0.0
        
        if facial_emotion and speech_emotion:
            # Weight by confidence scores
            if facial_confidence > speech_confidence:
                combined_emotion = facial_emotion
                combined_confidence = facial_confidence
            else:
                combined_emotion = speech_emotion
                combined_confidence = speech_confidence
        elif facial_emotion:
            combined_emotion = facial_emotion
            combined_confidence = facial_confidence
        elif speech_emotion:
            combined_emotion = speech_emotion
            combined_confidence = speech_confidence
        
        response = {
            "success": True,
            "facial_emotion": {
                "emotion": facial_emotion or "unknown",
                "confidence": float(facial_confidence),
                "probabilities": facial_result["probabilities"] if facial_result else {}
            },
            "speech_emotion": {
                "emotion": speech_emotion or "unknown",
                "confidence": float(speech_confidence),
                "probabilities": speech_result["probabilities"] if speech_result else {}
            },
            "combined_emotion": combined_emotion or "unknown",
            "combined_confidence": float(combined_confidence),
            "concordance": concordance or "UNKNOWN",
            "analysis": {
                "match": concordance == "MATCH" if concordance else False,
                "agreement_details": f"Face: {facial_emotion} (conf: {facial_confidence:.2f}) | Voice: {speech_emotion} (conf: {speech_confidence:.2f})"
            }
        }
        
        # Add explainability if requested
        if explain and facial_result and speech_result:
            if "grad_cam" in facial_result and "saliency" in speech_result:
                response["explainability"] = {
                    "grad_cam": facial_result.get("grad_cam"),
                    "saliency": speech_result.get("saliency")
                }
        
        return response
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/predict/video")
async def predict_video_emotion(file: UploadFile = File(...)):
    """Predict emotions from video (facial + speech)"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
        
        try:
            # Extract frames and audio
            processor = VideoProcessor()
            frames, audio, sr, fps = processor.extract_frames_and_audio(tmp_path, fps_sample=5)
            
            # Predict facial emotions from first 10 frames
            facial_results = []
            for frame in frames[:10]:
                result = predict_facial_emotion(frame)
                if result:
                    facial_results.append(result)
            
            # Aggregate facial results
            if facial_results:
                facial_emotions = [r["emotion"] for r in facial_results]
                facial_confidence = np.mean([r["confidence"] for r in facial_results])
                facial_emotion = max(set(facial_emotions), key=facial_emotions.count)
                facial_probs = {}
                for emotion in EMOTIONS_FACIAL:
                    facial_probs[emotion] = float(np.mean([r["probabilities"].get(emotion, 0) for r in facial_results]))
            else:
                facial_emotion = "unknown"
                facial_confidence = 0.0
                facial_probs = {e: 0.0 for e in EMOTIONS_FACIAL}
            
            # Predict speech emotion
            speech_result = predict_speech_emotion(audio, sr)
            
            return {
                "success": True,
                "facial_emotion": {
                    "emotion": facial_emotion,
                    "confidence": float(facial_confidence),
                    "frames_analyzed": len(facial_results),
                    "probabilities": facial_probs
                },
                "speech_emotion": {
                    "emotion": speech_result["emotion"] if speech_result else "unknown",
                    "confidence": float(speech_result["confidence"]) if speech_result else 0.0,
                    "probabilities": speech_result["probabilities"] if speech_result else {e: 0.0 for e in EMOTIONS_SPEECH}
                },
                "combined_emotion": facial_emotion if facial_confidence > 0.5 else (speech_result["emotion"] if speech_result else "unknown"),
                "video_duration": float(len(audio) / sr),
                "frames_processed": len(frames),
                "fps": float(fps)
            }
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/api/emotions/facial")
async def get_facial_emotions():
    return {"emotions": EMOTIONS_FACIAL}

@app.get("/api/emotions/speech")
async def get_speech_emotions():
    return {"emotions": EMOTIONS_SPEECH}

@app.get("/api/models/status")
async def get_models_status():
    return {
        "facial": {"loaded": facial_loaded, "accuracy": 0.7129, "emotions": len(EMOTIONS_FACIAL)},
        "speech": {"loaded": speech_loaded, "accuracy": 0.8750, "emotions": len(EMOTIONS_SPEECH)},
        "device": str(DEVICE)
    }
