from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import numpy as np
import cv2
import librosa
import base64
from PIL import Image
from io import BytesIO
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoFeatureExtractor, AutoModelForAudioClassification
from huggingface_hub import hf_hub_download
import tempfile
import os
import sys
import logging
from threading import Lock
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Add services to path
sys.path.insert(0, str(Path(__file__).parent))

# Import explainability module
from services.explainability import generate_grad_cam, generate_audio_saliency, create_combined_visualization

ENV = os.getenv("ENV", "development")
FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    os.getenv("REACT_APP_VERCEL_URL", "http://localhost:3000")
)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "")
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "false").lower() == "true"

app = FastAPI(title="Multi-Modal Emotion Recognition API", version="2.0.0")

# Configure CORS based on environment
if ENV == "production":
    if CORS_ORIGINS.strip():
        allowed_origins = [origin.strip() for origin in CORS_ORIGINS.split(",") if origin.strip()]
    else:
        allowed_origins = [FRONTEND_URL]
else:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(f"CORS enabled for: {allowed_origins}")

# Configuration
EMOTIONS_FACIAL = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTIONS_SPEECH = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
DEVICE = torch.device('cuda' if (torch.cuda.is_available() and USE_GPU) else 'cpu')

# Model storage
vit_model = None
facial_processor = None
speech_model = None
speech_processor = None
facial_loaded = False
speech_loaded = False

_facial_model_lock = Lock()
_speech_model_lock = Lock()

# Paths — download from HuggingFace Hub
logger.info("Resolving model paths from HuggingFace Hub...")
FACIAL_MODEL_PATH = hf_hub_download(
    repo_id="Nishvaraj/emotion-models",
    filename="vit_emotion_model.pt"
)
SPEECH_MODEL_PATH = hf_hub_download(
    repo_id="Nishvaraj/emotion-models",
    filename="hubert_emotion_model.pt"
)
logger.info(f"Facial model path: {FACIAL_MODEL_PATH}")
logger.info(f"Speech model path: {SPEECH_MODEL_PATH}")


def _upload_suffix(filename: str, default_suffix: str) -> str:
    suffix = Path(filename or '').suffix.lower()
    return suffix if suffix else default_suffix


FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def _encode_image_base64(image_array: np.ndarray) -> str:
    image_pil = Image.fromarray(image_array.astype(np.uint8))
    buf = BytesIO()
    image_pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def _decode_image_base64(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64)
    return np.array(Image.open(BytesIO(raw)).convert('RGB'))


def _detect_primary_face(image: Image.Image):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda b: b[2] * b[3])


def _crop_face_with_margin(image_array: np.ndarray, face_box, margin_ratio: float = 0.2):
    x, y, w, h = [int(v) for v in face_box]
    h_img, w_img = image_array.shape[:2]
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w_img, x + w + mx)
    y2 = min(h_img, y + h + my)

    return image_array[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


logger.info(f"Device: {DEVICE}")
logger.info(f"Environment: {ENV}")

# ========== MODEL LOADING ==========

def load_facial_model():
    """Load ViT model for facial emotion"""
    global vit_model, facial_processor, facial_loaded
    if vit_model is not None and facial_processor is not None:
        facial_loaded = True
        return True

    with _facial_model_lock:
        if vit_model is not None and facial_processor is not None:
            facial_loaded = True
            return True

        try:
            logger.info("Loading Facial Emotion Model (ViT)...")
            facial_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
            vit_model = AutoModelForImageClassification.from_pretrained(
                'google/vit-base-patch16-224-in21k',
                num_labels=len(EMOTIONS_FACIAL),
                ignore_mismatched_sizes=True
            )

            checkpoint = torch.load(FACIAL_MODEL_PATH, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                vit_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                vit_model.load_state_dict(checkpoint)
            logger.info("✓ Loaded ViT checkpoint")

            vit_model = vit_model.to(DEVICE)
            vit_model.eval()
            facial_loaded = True
            logger.info("✓ Facial model ready")
            return True
        except Exception as e:
            facial_loaded = False
            logger.error(f"❌ Error loading facial model: {e}")
            return False


def load_speech_model():
    """Load HuBERT model for speech emotion"""
    global speech_model, speech_processor, speech_loaded
    if speech_model is not None and speech_processor is not None:
        speech_loaded = True
        return True

    with _speech_model_lock:
        if speech_model is not None and speech_processor is not None:
            speech_loaded = True
            return True

        try:
            logger.info("Loading Speech Emotion Model (HuBERT)...")
            speech_processor = AutoFeatureExtractor.from_pretrained('facebook/hubert-large-ls960-ft')
            speech_model = AutoModelForAudioClassification.from_pretrained(
                'facebook/hubert-large-ls960-ft',
                num_labels=len(EMOTIONS_SPEECH),
                ignore_mismatched_sizes=True
            )

            checkpoint = torch.load(SPEECH_MODEL_PATH, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                speech_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                speech_model.load_state_dict(checkpoint)
            logger.info("✓ Loaded HuBERT checkpoint")

            speech_model = speech_model.to(DEVICE)
            speech_model.eval()
            speech_loaded = True
            logger.info("✓ Speech model ready")
            return True
        except Exception as e:
            speech_loaded = False
            logger.error(f"❌ Error loading speech model: {e}")
            return False


def ensure_facial_model_loaded() -> bool:
    if vit_model is not None and facial_processor is not None:
        return True
    return load_facial_model()


def ensure_speech_model_loaded() -> bool:
    if speech_model is not None and speech_processor is not None:
        return True
    return load_speech_model()


# Optional eager loading for environments that prefer warm startup.
if PRELOAD_MODELS:
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
        
        audio, sr = librosa.load(video_path, sr=16000, mono=True)
        
        return frames, audio, sr, fps

# ========== PREDICTION FUNCTIONS ==========

def predict_facial_emotion(image: Image.Image, generate_explainability: bool = False):
    """Predict emotion from image"""
    try:
        if not ensure_facial_model_loaded():
            return None
        
        input_array = np.array(image)
        face_box = _detect_primary_face(image)

        model_image = image
        expanded_box = None

        if face_box is not None:
            face_crop, expanded_box = _crop_face_with_margin(input_array, face_box)
            if face_crop.size > 0:
                model_image = Image.fromarray(face_crop)

        annotated = input_array.copy()
        if face_box is not None:
            x, y, w, h = [int(v) for v in face_box]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(
                annotated,
                'Face detected',
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        inputs = facial_processor(model_image, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            outputs = vit_model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]
            probs = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
        
        top_idx = np.argmax(probs)
        result = {
            "emotion": EMOTIONS_FACIAL[top_idx],
            "confidence": float(probs[top_idx]),
            "probabilities": {e: float(p) for e, p in zip(EMOTIONS_FACIAL, probs)},
            "face_detected": face_box is not None,
            "annotated_image": _encode_image_base64(annotated)
        }

        if face_box is not None:
            x, y, w, h = [int(v) for v in face_box]
            result["face_box"] = {"x": x, "y": y, "width": w, "height": h}
        
        if generate_explainability:
            result["explainability_status"] = {
                "requested": True,
                "generated": False,
                "error": None
            }
            try:
                original_base64, heatmap_base64 = generate_grad_cam(
                    model_image,
                    vit_model,
                    facial_processor,
                    top_idx,
                    EMOTIONS_FACIAL,
                    DEVICE
                )
                if original_base64:
                    result["original_image"] = original_base64
                if heatmap_base64:
                    if expanded_box is not None:
                        try:
                            hx, hy, hw, hh = [int(v) for v in expanded_box]
                            heatmap_img = _decode_image_base64(heatmap_base64)
                            heatmap_img = cv2.resize(heatmap_img, (hw, hh), interpolation=cv2.INTER_LINEAR)
                            projected = annotated.copy()
                            projected[hy:hy + hh, hx:hx + hw] = heatmap_img
                            x, y, w, h = [int(v) for v in face_box]
                            cv2.rectangle(projected, (x, y), (x + w, y + h), (0, 255, 255), 2)
                            cv2.putText(
                                projected,
                                'Decision region',
                                (x, min(projected.shape[0] - 10, y + h + 22)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 0),
                                2,
                                cv2.LINE_AA
                            )
                            result["grad_cam"] = _encode_image_base64(projected)
                        except Exception:
                            result["grad_cam"] = heatmap_base64
                    else:
                        result["grad_cam"] = heatmap_base64
                    result["explainability_status"]["generated"] = True
                else:
                    result["explainability_status"]["error"] = "Grad-CAM map returned empty output"
            except Exception as e:
                logger.warning(f"Could not generate Grad-CAM: {e}")
                result["explainability_status"]["error"] = str(e)
        
        return result
    except Exception as e:
        logger.error(f"Error predicting facial emotion: {e}")
        return None

def predict_speech_emotion(audio: np.ndarray, sr: int = 16000, generate_explainability: bool = False):
    """Predict emotion from audio"""
    try:
        if not ensure_speech_model_loaded():
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
        
        if generate_explainability:
            result["explainability_status"] = {
                "requested": True,
                "generated": False,
                "error": None
            }
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
                if spec_base64:
                    result["waveform"] = spec_base64
                if saliency_base64:
                    result["saliency"] = saliency_base64
                    result["explainability_status"]["generated"] = True
                else:
                    result["explainability_status"]["error"] = "Audio saliency map returned empty output"
            except Exception as e:
                logger.warning(f"Could not generate audio saliency: {e}")
                result["explainability_status"]["error"] = str(e)
        
        return result
    except Exception as e:
        logger.error(f"Error predicting speech emotion: {e}")
        return None

# ========== API ENDPOINTS ==========

@app.get("/")
async def root():
    return {"message": "Multi-Modal Emotion Recognition API v2.0", "status": "active"}

@app.get("/health")
async def health():
    facial_ready = vit_model is not None and facial_processor is not None
    speech_ready = speech_model is not None and speech_processor is not None
    return {
        "status": "healthy",
        "facial_model": facial_ready,
        "speech_model": speech_ready,
        "lazy_loading": not PRELOAD_MODELS,
        "device": str(DEVICE)
    }

@app.post("/api/predict/facial")
async def predict_facial(file: UploadFile = File(...), explain: bool = False):
    """Predict emotion from image"""
    try:
        logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")
        if len(contents) == 0:
            return JSONResponse(status_code=400, content={"error": "Empty file received"})
        image = Image.open(BytesIO(contents)).convert('RGB')
        result = predict_facial_emotion(image, generate_explainability=explain)
        return {"success": True, **result} if result else {"success": False, "error": "Prediction failed"}
    except Exception as e:
        logger.error(f"Error in predict_facial: {e}", exc_info=True)
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/predict/speech")
async def predict_speech(file: UploadFile = File(...), explain: bool = False):
    """Predict emotion from audio"""
    try:
        contents = await file.read()
        suffix = _upload_suffix(file.filename, '.wav')
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
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
        image_contents = await image_file.read()
        image = Image.open(BytesIO(image_contents)).convert('RGB')
        facial_result = predict_facial_emotion(image, generate_explainability=explain)
        
        audio_contents = await audio_file.read()
        audio_suffix = _upload_suffix(audio_file.filename, '.wav')
        with tempfile.NamedTemporaryFile(suffix=audio_suffix, delete=False) as tmp:
            tmp.write(audio_contents)
            tmp_path = tmp.name
        
        try:
            audio, sr = librosa.load(tmp_path, sr=16000)
            speech_result = predict_speech_emotion(audio, sr, generate_explainability=explain)
        finally:
            os.unlink(tmp_path)
        
        facial_emotion = facial_result["emotion"] if facial_result else None
        facial_confidence = facial_result["confidence"] if facial_result else 0.0
        
        speech_emotion = speech_result["emotion"] if speech_result else None
        speech_confidence = speech_result["confidence"] if speech_result else 0.0
        
        concordance = None
        if facial_emotion and speech_emotion:
            if facial_emotion == speech_emotion:
                concordance = "MATCH"
            else:
                concordance = "MISMATCH"
        
        combined_emotion = None
        combined_confidence = 0.0
        
        if facial_emotion and speech_emotion:
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
                "probabilities": facial_result["probabilities"] if facial_result else {},
                "face_detected": facial_result.get("face_detected", False) if facial_result else False,
                "face_box": facial_result.get("face_box") if facial_result else None,
                "annotated_image": facial_result.get("annotated_image") if facial_result else None
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
        
        if explain:
            explainability = {}
            errors = []

            facial_status = (facial_result or {}).get("explainability_status") or {
                "requested": True,
                "generated": False,
                "error": "Facial explainability unavailable"
            }
            speech_status = (speech_result or {}).get("explainability_status") or {
                "requested": True,
                "generated": False,
                "error": "Speech explainability unavailable"
            }

            if facial_result and facial_result.get("grad_cam"):
                explainability["grad_cam"] = facial_result.get("grad_cam")
            elif facial_status.get("error"):
                errors.append(f"Facial: {facial_status.get('error')}")

            if speech_result and speech_result.get("saliency"):
                explainability["saliency"] = speech_result.get("saliency")
            elif speech_status.get("error"):
                errors.append(f"Speech: {speech_status.get('error')}")

            if speech_result and speech_result.get("waveform"):
                explainability["waveform"] = speech_result.get("waveform")

            response["explainability_status"] = {
                "requested": True,
                "generated": bool(explainability),
                "facial": facial_status,
                "speech": speech_status,
                "errors": errors
            }

            if explainability:
                response["explainability"] = explainability
        
        return response
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/predict/video")
async def predict_video_emotion(file: UploadFile = File(...), explain: bool = False):
    """Predict emotions from video (facial + speech)"""
    try:
        video_suffix = _upload_suffix(file.filename, '.mp4')
        with tempfile.NamedTemporaryFile(suffix=video_suffix, delete=False) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
        
        try:
            processor = VideoProcessor()
            frames, audio, sr, fps = processor.extract_frames_and_audio(tmp_path, fps_sample=5)
            
            facial_results = []
            for frame in frames[:10]:
                result = predict_facial_emotion(frame)
                if result:
                    facial_results.append(result)
            
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
            
            speech_result = predict_speech_emotion(audio, sr)
            
            response = {
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

            if explain:
                explainability = {}
                errors = []

                facial_exp_status = {"requested": True, "generated": False, "error": None}
                speech_exp_status = {"requested": True, "generated": False, "error": None}

                if frames and facial_emotion != "unknown":
                    try:
                        rep_frame = frames[len(frames) // 2]
                        facial_exp_result = predict_facial_emotion(rep_frame, generate_explainability=True)
                        if facial_exp_result and facial_exp_result.get("grad_cam"):
                            explainability["grad_cam"] = facial_exp_result.get("grad_cam")
                            facial_exp_status["generated"] = True
                        else:
                            facial_exp_status["error"] = (
                                (facial_exp_result or {}).get("explainability_status", {}).get("error")
                                or "Grad-CAM map unavailable for selected video frame"
                            )
                    except Exception as e:
                        facial_exp_status["error"] = str(e)
                else:
                    facial_exp_status["error"] = "No valid face frame found for explainability"

                if speech_result is not None:
                    try:
                        speech_exp_result = predict_speech_emotion(audio, sr, generate_explainability=True)
                        if speech_exp_result and speech_exp_result.get("saliency"):
                            explainability["saliency"] = speech_exp_result.get("saliency")
                            speech_exp_status["generated"] = True
                            if speech_exp_result.get("waveform"):
                                explainability["waveform"] = speech_exp_result.get("waveform")
                        else:
                            speech_exp_status["error"] = (
                                (speech_exp_result or {}).get("explainability_status", {}).get("error")
                                or "Audio saliency map unavailable for this video audio"
                            )
                    except Exception as e:
                        speech_exp_status["error"] = str(e)
                else:
                    speech_exp_status["error"] = "No valid audio prediction found for explainability"

                if facial_exp_status.get("error"):
                    errors.append(f"Facial: {facial_exp_status.get('error')}")
                if speech_exp_status.get("error"):
                    errors.append(f"Speech: {speech_exp_status.get('error')}")

                response["explainability_status"] = {
                    "requested": True,
                    "generated": bool(explainability),
                    "facial": facial_exp_status,
                    "speech": speech_exp_status,
                    "errors": errors
                }

                if explainability:
                    response["explainability"] = explainability

            return response
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
    facial_ready = vit_model is not None and facial_processor is not None
    speech_ready = speech_model is not None and speech_processor is not None
    return {
        "facial": {"loaded": facial_ready, "accuracy": 0.7129, "emotions": len(EMOTIONS_FACIAL)},
        "speech": {"loaded": speech_ready, "accuracy": 0.8750, "emotions": len(EMOTIONS_SPEECH)},
        "lazy_loading": not PRELOAD_MODELS,
        "device": str(DEVICE)
    }
