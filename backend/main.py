from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import numpy as np
import cv2
import librosa
import base64
from PIL import Image, ImageOps
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

try:
    from facenet_pytorch import MTCNN
except Exception:
    MTCNN = None

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
ENABLE_FACE_ROTATION = os.getenv("ENABLE_FACE_ROTATION", "false").lower() == "true"
MAX_FACE_ROTATION_DEGREES = float(os.getenv("MAX_FACE_ROTATION_DEGREES", "8"))
HAAR_MIN_NEIGHBORS = int(os.getenv("HAAR_MIN_NEIGHBORS", "5"))
HAAR_MIN_SIZE = int(os.getenv("HAAR_MIN_SIZE", "40"))

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
logger.info(
    "Face detection config: rotation=%s max_rotation=%.1f haar_min_neighbors=%d haar_min_size=%d",
    ENABLE_FACE_ROTATION,
    MAX_FACE_ROTATION_DEGREES,
    HAAR_MIN_NEIGHBORS,
    HAAR_MIN_SIZE,
)

# Configuration
EMOTIONS_FACIAL = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTIONS_SPEECH = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
DEVICE = torch.device('cuda' if (torch.cuda.is_available() and USE_GPU) else 'cpu')
MAX_SPEECH_INFER_SECONDS = int(os.getenv('MAX_SPEECH_INFER_SECONDS', '15'))
MAX_SPEECH_XAI_SECONDS = int(os.getenv('MAX_SPEECH_XAI_SECONDS', '8'))
CONCORDANCE_SCORE_MAP = {
    'MATCH': 100,
    'PARTIAL': 65,
    'MISMATCH': 30,
    'UNKNOWN': 0,
}

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


def _concordance_score(label: str | None) -> int:
    return CONCORDANCE_SCORE_MAP.get((label or 'UNKNOWN').upper(), 0)


def _calculate_concordance(facial_emotion, speech_emotion, facial_confidence, speech_confidence):
    if facial_emotion == speech_emotion:
        score = (facial_confidence + speech_confidence) / 2
        if score > 0.7:
            concordance = "MATCH"
        elif score >= 0.4:
            concordance = "PARTIAL"
        else:
            concordance = "MISMATCH"
    else:
        # Emotions are different - can NEVER be MATCH
        score = 1 - abs(facial_confidence - speech_confidence)
        if score >= 0.5:
            concordance = "PARTIAL"
        else:
            concordance = "MISMATCH"

    concordance_score = round(score * 100)
    return concordance, concordance_score


FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
MTCNN_DETECTOR = MTCNN(keep_all=False, device=DEVICE) if MTCNN is not None else None


def _encode_image_base64(image_array: np.ndarray) -> str:
    image_pil = Image.fromarray(image_array.astype(np.uint8))
    buf = BytesIO()
    image_pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def _decode_image_base64(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64)
    return np.array(Image.open(BytesIO(raw)).convert('RGB'))


def _detect_primary_face(image: Image.Image):
    if MTCNN_DETECTOR is not None:
        try:
            boxes, probs, points = MTCNN_DETECTOR.detect(image, landmarks=True)
            if boxes is not None and len(boxes) > 0:
                best_idx = int(np.argmax(probs)) if probs is not None else 0
                x1, y1, x2, y2 = boxes[best_idx]
                # Convert from [x1,y1,x2,y2] to [x,y,w,h]
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                return (x, y, w, h), (points[best_idx] if points is not None else None)
        except Exception as e:
            logger.debug(f"MTCNN face detection fallback: {e}")

    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=HAAR_MIN_NEIGHBORS,
        minSize=(HAAR_MIN_SIZE, HAAR_MIN_SIZE)
    )

    if faces is None or len(faces) == 0:
        return None, None
    best_face = max(faces, key=lambda b: b[2] * b[3])
    return tuple(int(v) for v in best_face), None


def _rotate_image_to_level(image: Image.Image, points) -> Image.Image:
    if not ENABLE_FACE_ROTATION:
        return image

    if points is None:
        return image

    try:
        left_eye, right_eye = points[0], points[1]
        angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        if abs(angle) < 1.0:
            return image
        if abs(angle) > MAX_FACE_ROTATION_DEGREES:
            logger.debug("Skipping face rotation due to large angle: %.2f", angle)
            return image
        center_x = image.width / 2
        center_y = image.height / 2
        return image.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True, center=(center_x, center_y), fillcolor=(0, 0, 0))
    except Exception:
        return image


def _crop_face_with_margin(image_array: np.ndarray, face_box, margin_ratio: float = 0.12):
    x, y, w, h = [int(v) for v in face_box]
    h_img, w_img = image_array.shape[:2]
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w_img, x + w + mx)
    y2 = min(h_img, y + h + my)

    return image_array[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def _shrink_box(face_box, shrink_ratio: float = 0.12):
    x, y, w, h = [int(v) for v in face_box]
    dx = int(w * shrink_ratio / 2)
    dy = int(h * shrink_ratio / 2)
    x1 = x + dx
    y1 = y + dy
    width = max(1, w - (dx * 2))
    height = max(1, h - (dy * 2))
    return x1, y1, width, height


def _trim_audio_window(audio: np.ndarray, sr: int, max_seconds: int) -> np.ndarray:
    if audio is None or sr <= 0:
        return audio
    max_len = int(sr * max_seconds)
    if max_len <= 0 or len(audio) <= max_len:
        return audio
    start = (len(audio) - max_len) // 2
    end = start + max_len
    return audio[start:end]


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
                ignore_mismatched_sizes=True,
                attn_implementation='eager'
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

        image = ImageOps.exif_transpose(image).convert('RGB')
        
        detected = _detect_primary_face(image)
        face_box, face_points = detected if isinstance(detected, tuple) else (None, None)

        rotated_image = _rotate_image_to_level(image, face_points)
        if rotated_image is not image:
            rotated_detected = _detect_primary_face(rotated_image)
            if isinstance(rotated_detected, tuple):
                rotated_box, rotated_points = rotated_detected
                if rotated_box is not None:
                    image = rotated_image
                    face_box = rotated_box
                    face_points = rotated_points

        input_array = np.array(image)

        model_image = image

        if face_box is not None:
            face_crop, _ = _crop_face_with_margin(input_array, face_box)
            if face_crop.size > 0:
                model_image = Image.fromarray(face_crop)

        annotated = input_array.copy()
        if face_box is not None:
            x, y, w, h = _shrink_box(face_box, shrink_ratio=0.08)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 128, 0), 2)
            cv2.putText(
                annotated,
                'Face detected',
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 128, 0),
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

        # Keep inference fast and stable for long recordings.
        audio_for_infer = _trim_audio_window(audio, 16000, MAX_SPEECH_INFER_SECONDS)
        
        inputs = speech_processor(audio_for_infer, sampling_rate=16000, return_tensors="pt", padding=True)
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
                # Saliency on a shorter centered chunk avoids multi-minute stalls.
                audio_for_xai = _trim_audio_window(audio_for_infer, 16000, MAX_SPEECH_XAI_SECONDS)
                spec_base64, saliency_base64 = generate_audio_saliency(
                    audio_for_xai,
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
        image = ImageOps.exif_transpose(Image.open(BytesIO(contents))).convert('RGB')
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
        image = ImageOps.exif_transpose(Image.open(BytesIO(image_contents))).convert('RGB')
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
        
        concordance, concordance_score = _calculate_concordance(
            facial_emotion,
            speech_emotion,
            facial_confidence,
            speech_confidence,
        )
        
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
            "concordance": concordance,
            "concordance_score": concordance_score,
            "analysis": {
                "match": concordance == "MATCH",
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
            speech_emotion = speech_result["emotion"] if speech_result else "unknown"
            speech_confidence = float(speech_result["confidence"]) if speech_result else 0.0
            concordance, concordance_score = _calculate_concordance(
                facial_emotion,
                speech_emotion,
                facial_confidence,
                speech_confidence,
            )
            
            response = {
                "success": True,
                "facial_emotion": {
                    "emotion": facial_emotion,
                    "confidence": float(facial_confidence),
                    "frames_analyzed": len(facial_results),
                    "probabilities": facial_probs
                },
                "speech_emotion": {
                    "emotion": speech_emotion,
                    "confidence": speech_confidence,
                    "probabilities": speech_result["probabilities"] if speech_result else {e: 0.0 for e in EMOTIONS_SPEECH}
                },
                "combined_emotion": facial_emotion if facial_confidence > 0.5 else (speech_result["emotion"] if speech_result else "unknown"),
                "concordance": concordance,
                "concordance_score": concordance_score,
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
                    facial_exp_status["error"] = "Grad-CAM disabled for video to prevent timeout"
                    facial_exp_status["generated"] = False
                else:
                    facial_exp_status["error"] = "No valid face frame found for explainability"

                if speech_result is not None:
                    try:
                        audio_short = audio[:5 * sr] if len(audio) > 5 * sr else audio
                        speech_exp_result = predict_speech_emotion(audio_short, sr, generate_explainability=True)
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
