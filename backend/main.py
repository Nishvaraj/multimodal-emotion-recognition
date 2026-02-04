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
import json
from services.explainability import generate_grad_cam_for_prediction
from services.audio_explainability import generate_audio_saliency_for_prediction
from services.database import db

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

# Lazy load on startup (only facial, load speech on first use)
facial_loaded = load_facial_model()
speech_loaded = False  # Will be loaded on first use

# Load speech model on first use to avoid long startup times
def ensure_speech_model_loaded():
    global speech_loaded, speech_model
    if not speech_loaded and speech_model is None:
        speech_loaded = load_speech_model()
    return speech_loaded

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

def predict_facial_emotion(image: Image.Image):
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
        return {
            "emotion": EMOTIONS_FACIAL[top_idx],
            "confidence": float(probs[top_idx]),
            "probabilities": {e: float(p) for e, p in zip(EMOTIONS_FACIAL, probs)}
        }
    except Exception as e:
        print(f"Error predicting facial emotion: {e}")
        return None

def predict_speech_emotion(audio: np.ndarray, sr: int = 16000):
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
        return {
            "emotion": EMOTIONS_SPEECH[top_idx],
            "confidence": float(probs[top_idx]),
            "probabilities": {e: float(p) for e, p in zip(EMOTIONS_SPEECH, probs)}
        }
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
async def predict_facial(file: UploadFile = File(...)):
    """Predict emotion from image with Grad-CAM visualization"""
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        result = predict_facial_emotion(image)
        
        if result and vit_model is not None:
            # Generate Grad-CAM visualization
            grad_cam_result = generate_grad_cam_for_prediction(
                image=image,
                model=vit_model,
                processor=facial_processor,
                emotion=result["emotion"],
                emotion_list=EMOTIONS_FACIAL,
                confidence=result["confidence"],
                device=str(DEVICE)
            )
            
            if grad_cam_result.get("success"):
                result["grad_cam_image"] = grad_cam_result["grad_cam_image"]
                result["explainability"] = grad_cam_result["heatmap_description"]
        
        return {"success": True, **result} if result else {"success": False, "error": "Prediction failed"}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/predict/speech")
async def predict_speech(file: UploadFile = File(...)):
    """Predict emotion from audio with saliency visualization"""
    try:
        # Ensure speech model is loaded
        ensure_speech_model_loaded()
        
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        
        try:
            audio, sr = librosa.load(tmp_path, sr=16000)
            result = predict_speech_emotion(audio, sr)
            
            if result and speech_model is not None:
                # Generate audio saliency visualization
                saliency_result = generate_audio_saliency_for_prediction(
                    audio=audio,
                    sr=sr,
                    model=speech_model,
                    processor=speech_processor,
                    emotion=result["emotion"],
                    emotion_list=EMOTIONS_SPEECH,
                    confidence=result["confidence"],
                    device=str(DEVICE)
                )
                
                if saliency_result.get("success"):
                    result["saliency_map"] = saliency_result["saliency_map"]
                    result["explainability"] = saliency_result["frequency_description"]
            
            return {"success": True, **result} if result else {"success": False, "error": "Prediction failed"}
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/predict/combined")
async def predict_combined(image_file: UploadFile = File(...), audio_file: UploadFile = File(...)):
    """Predict emotions from both image and audio, then compare results"""
    try:
        # Ensure speech model is loaded
        ensure_speech_model_loaded()
        # Process image
        image_contents = await image_file.read()
        image = Image.open(BytesIO(image_contents)).convert('RGB')
        facial_result = predict_facial_emotion(image)
        
        # Process audio
        audio_contents = await audio_file.read()
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(audio_contents)
            tmp_path = tmp.name
        
        try:
            audio, sr = librosa.load(tmp_path, sr=16000)
            speech_result = predict_speech_emotion(audio, sr)
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
        
        return {
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
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/predict/video")
async def predict_video_emotion(file: UploadFile = File(...)):
    """Predict emotions from video (facial + speech)"""
    try:
        # Ensure speech model is loaded
        ensure_speech_model_loaded()
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

# ========== SESSION MANAGEMENT ENDPOINTS ==========

@app.post("/api/sessions/create")
async def create_session(user_name: str = None, notes: str = None):
    """Create a new session"""
    try:
        session_id = db.create_session(user_name=user_name, notes=notes)
        session = db.get_session(session_id)
        return {
            "success": True,
            "session_id": session_id,
            "session": session
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/api/sessions")
async def list_sessions(limit: int = 100):
    """Get all sessions"""
    try:
        sessions = db.get_all_sessions(limit=limit)
        return {
            "success": True,
            "total": len(sessions),
            "sessions": sessions
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details and predictions"""
    try:
        session = db.get_session(session_id)
        if not session:
            return JSONResponse(status_code=404, content={"error": "Session not found"})
        
        predictions = db.get_session_predictions(session_id)
        concordance = db.get_session_concordance(session_id)
        stats = db.get_statistics(session_id)
        
        return {
            "success": True,
            "session": session,
            "predictions": predictions,
            "concordance_records": concordance,
            "statistics": stats
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/sessions/{session_id}/save_prediction")
async def save_prediction(
    session_id: str,
    modality: str,
    emotion: str,
    confidence: float,
    probabilities: dict = None,
    grad_cam_image: str = None,
    saliency_map: str = None
):
    """Save a prediction to a session"""
    try:
        session = db.get_session(session_id)
        if not session:
            return JSONResponse(status_code=404, content={"error": "Session not found"})
        
        prediction_id = db.save_prediction(
            session_id=session_id,
            modality=modality,
            emotion=emotion,
            confidence=confidence,
            probabilities=probabilities,
            grad_cam_image=grad_cam_image,
            saliency_map=saliency_map
        )
        
        return {
            "success": True,
            "prediction_id": prediction_id
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/sessions/{session_id}/save_concordance")
async def save_concordance(
    session_id: str,
    facial_emotion: str,
    facial_confidence: float,
    speech_emotion: str,
    speech_confidence: float,
    concordance_status: str
):
    """Save a concordance record"""
    try:
        session = db.get_session(session_id)
        if not session:
            return JSONResponse(status_code=404, content={"error": "Session not found"})
        
        record_id = db.save_concordance(
            session_id=session_id,
            facial_emotion=facial_emotion,
            facial_confidence=facial_confidence,
            speech_emotion=speech_emotion,
            speech_confidence=speech_confidence,
            concordance_status=concordance_status
        )
        
        return {
            "success": True,
            "record_id": record_id
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/api/sessions/{session_id}/export/csv")
async def export_session_csv(session_id: str):
    """Export session as CSV"""
    try:
        session = db.get_session(session_id)
        if not session:
            return JSONResponse(status_code=404, content={"error": "Session not found"})
        
        csv_data = db.export_session_csv(session_id)
        
        return {
            "success": True,
            "format": "csv",
            "data": csv_data,
            "filename": f"emotion_session_{session_id}.csv"
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/api/sessions/{session_id}/export/json")
async def export_session_json(session_id: str):
    """Export session as JSON"""
    try:
        session = db.get_session(session_id)
        if not session:
            return JSONResponse(status_code=404, content={"error": "Session not found"})
        
        json_data = db.export_session_json(session_id)
        
        return {
            "success": True,
            "format": "json",
            "data": json.loads(json_data),
            "filename": f"emotion_session_{session_id}.json"
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its predictions"""
    try:
        session = db.get_session(session_id)
        if not session:
            return JSONResponse(status_code=404, content={"error": "Session not found"})
        
        db.delete_session(session_id)
        
        return {
            "success": True,
            "message": f"Session {session_id} deleted"
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/api/sessions/{session_id}/statistics")
async def get_session_statistics(session_id: str):
    """Get session statistics"""
    try:
        session = db.get_session(session_id)
        if not session:
            return JSONResponse(status_code=404, content={"error": "Session not found"})
        
        stats = db.get_statistics(session_id)
        
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# ========== SERVER STARTUP ==========

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Multi-Modal Emotion Recognition API")
    print("="*60)
    print("📍 Running on: http://127.0.0.1:8000")
    print("📊 Docs: http://127.0.0.1:8000/docs")
    print("✓ Facial Model: Loaded")
    print(f"✓ Speech Model: Will load on first use (lazy loading)")
    print("="*60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
