"""FastAPI backend for multimodal emotion inference.

This module is the main backend entry point for MMER. It exposes REST API
endpoints for facial, speech, combined, and video emotion analysis.

Main responsibilities:
1. Configure the FastAPI application and CORS.
2. Load ViT and HuBERT model checkpoints from Hugging Face Hub.
3. Preprocess uploaded images, audio files, and videos.
4. Run model inference for facial and speech emotion recognition.
5. Compute deterministic concordance between facial and vocal predictions.
6. Optionally generate explainability outputs such as Grad-CAM and audio saliency.
7. Return structured JSON responses to the React frontend.
"""

# =============================================================================
# Imports
# =============================================================================

# FastAPI framework imports.
# FastAPI creates the REST API. UploadFile and File are used for file uploads.
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Core ML and media-processing libraries.
import torch                  # PyTorch: model loading and inference
import numpy as np             # Numerical arrays and probability operations
import cv2                     # OpenCV: video processing and Haar face detection
import librosa                 # Audio loading and resampling
import base64                  # Encode images as strings for JSON responses

# Image/file utilities.
from PIL import Image, ImageOps
from io import BytesIO
from pathlib import Path

# Hugging Face Transformers utilities.
# AutoImageProcessor prepares images for ViT.
# AutoModelForImageClassification loads the ViT architecture.
# AutoFeatureExtractor prepares waveform audio for HuBERT.
# AutoModelForAudioClassification loads the HuBERT architecture.
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)

# Downloads model checkpoint files from Hugging Face Hub.
from huggingface_hub import hf_hub_download

# Runtime, logging, and concurrency utilities.
import tempfile                # Temporarily stores uploaded audio/video files
import os                      # Reads environment variables
import logging                 # Server-side logging
from threading import Lock      # Prevents concurrent model-loading race conditions
from dotenv import load_dotenv  # Loads local .env variables during development


# =============================================================================
# Optional Dependencies
# =============================================================================

# MTCNN is preferred for face detection because it is stronger than Haar cascade
# and can also return facial landmarks. However, the app should not crash if
# facenet-pytorch is missing in a deployment environment. In that case, MTCNN is
# set to None and Haar cascade becomes the fallback detector.
try:
    from facenet_pytorch import MTCNN  # type: ignore[import-not-found]
except Exception:
    MTCNN = None


# =============================================================================
# Environment and Logging Setup
# =============================================================================

# Load environment variables from a local .env file when running locally.
# In production, these are usually injected by the hosting platform.
load_dotenv()

# Configure readable logs for deployment debugging.
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Import explainability helper functions.
# generate_grad_cam creates facial Grad-CAM heatmaps.
# generate_audio_saliency creates audio saliency/spectrogram visualisations.
from backend.services.explainability import generate_grad_cam, generate_audio_saliency


# =============================================================================
# Runtime Environment Configuration
# =============================================================================

# ENV controls deployment behaviour. In production, CORS is restricted.
# In development, CORS is usually open to simplify local testing.
ENV = os.getenv("ENV", "development")

# The frontend URL is used for CORS in production.
# The fallback REACT_APP_VERCEL_URL is kept for backward compatibility with
# earlier Vercel environment variable naming.
FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    os.getenv("REACT_APP_VERCEL_URL", "http://localhost:3000")
)

# Comma-separated list of allowed origins, e.g.:
# https://www.mmer.space,https://mmer.vercel.app
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "")

# USE_GPU allows GPU inference when CUDA is available.
# If false, the backend forces CPU even if CUDA exists.
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"

# PRELOAD_MODELS controls whether models load at server startup.
# False = lazy loading on first request; True = eager loading during startup.
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "false").lower() == "true"

# Optional head-tilt correction. Disabled by default because excessive rotation
# can distort facial features and harm emotion classification.
ENABLE_FACE_ROTATION = os.getenv("ENABLE_FACE_ROTATION", "false").lower() == "true"
MAX_FACE_ROTATION_DEGREES = float(os.getenv("MAX_FACE_ROTATION_DEGREES", "8"))

# Haar cascade fallback parameters.
# minNeighbors controls strictness: higher values reduce false positives.
# minSize ignores detections smaller than the configured face size.
HAAR_MIN_NEIGHBORS = int(os.getenv("HAAR_MIN_NEIGHBORS", "5"))
HAAR_MIN_SIZE = int(os.getenv("HAAR_MIN_SIZE", "40"))


# =============================================================================
# API Metadata
# =============================================================================

# Tags organise endpoints in FastAPI's generated Swagger/OpenAPI docs.
API_TAGS = [
    {
        "name": "system",
        "description": "Service metadata, health checks, and model runtime status.",
    },
    {
        "name": "prediction",
        "description": "Facial, speech, combined, and video emotion inference endpoints.",
    },
    {
        "name": "reference",
        "description": "Reference label endpoints consumed by the frontend UI.",
    },
]

APP_DESCRIPTION = """
Production API for multimodal emotion recognition.

- Facial and speech inference using ViT and HuBERT backbones
- Deterministic concordance scoring for multimodal agreement
- Optional explainability outputs (Grad-CAM and audio saliency)
- OpenAPI documentation available at `/docs` and `/redoc`
"""

# Create the FastAPI application instance.
# docs_url and redoc_url enable interactive API documentation.
app = FastAPI(
    title="Multi-Modal Emotion Recognition API",
    version="2.0.0",
    description=APP_DESCRIPTION,
    openapi_tags=API_TAGS,
    docs_url="/docs",
    redoc_url="/redoc",
)


# =============================================================================
# CORS Configuration
# =============================================================================

# CORS is needed because the React frontend and FastAPI backend are hosted on
# different domains. Without CORS, the browser blocks frontend API calls.
if ENV == "production":
    if CORS_ORIGINS.strip():
        # Use explicit comma-separated allowed origins when provided.
        allowed_origins = [origin.strip() for origin in CORS_ORIGINS.split(",") if origin.strip()]
    else:
        # Fallback to one production frontend URL.
        allowed_origins = [FRONTEND_URL]
else:
    # During development, allow all origins for easier local testing.
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


# =============================================================================
# Model and Inference Constants
# =============================================================================

# Facial emotion labels must match the output order used during ViT training.
EMOTIONS_FACIAL = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Speech emotion labels must match the output order used during HuBERT training.
# RAVDESS includes calm and uses fearful/surprised naming.
EMOTIONS_SPEECH = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Choose GPU when available and enabled; otherwise use CPU.
DEVICE = torch.device('cuda' if (torch.cuda.is_available() and USE_GPU) else 'cpu')

# Maximum audio window for normal speech inference.
# Long clips are centre-cropped to bound latency and memory use.
MAX_SPEECH_INFER_SECONDS = int(os.getenv('MAX_SPEECH_INFER_SECONDS', '15'))

# Shorter window for explainability because saliency is more expensive.
MAX_SPEECH_XAI_SECONDS = int(os.getenv('MAX_SPEECH_XAI_SECONDS', '8'))


# =============================================================================
# In-Memory Model State
# =============================================================================

# Models are stored as module-level globals so they are loaded once per server
# process rather than reloaded for every request.
vit_model = None
facial_processor = None
speech_model = None
speech_processor = None

# Status flags used by health/model-status endpoints.
facial_loaded = False
speech_loaded = False

# Locks prevent race conditions where multiple simultaneous requests try to load
# the same large model at the same time.
_facial_model_lock = Lock()
_speech_model_lock = Lock()


# =============================================================================
# Model Checkpoint Download Paths
# =============================================================================

# Model weights are stored separately on Hugging Face Hub rather than baked into
# the Docker image. This keeps the container smaller and allows model updates
# without rebuilding the backend image.
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


# =============================================================================
# Helper Functions
# =============================================================================

def _upload_suffix(filename: str, default_suffix: str) -> str:
    """Return a safe file extension for uploaded temporary files.

    Some libraries such as librosa/OpenCV behave better when a temporary file has
    the correct extension. If the uploaded file has no suffix, we use a safe
    default such as .wav or .mp4.
    """
    suffix = Path(filename or '').suffix.lower()
    return suffix if suffix else default_suffix


def _calculate_concordance(facial_emotion, speech_emotion, facial_confidence, speech_confidence):
    """Compute deterministic face-voice concordance.

    Same-label case:
        If both modalities predict the same label, the score is the mean
        confidence. High mean confidence gives MATCH.

    Different-label case:
        If labels differ, the result can never be a full MATCH. The score is
        based on confidence closeness: 1 - absolute confidence gap. Similar
        confidence with different labels becomes PARTIAL; a large confidence
        gap becomes MISMATCH.

    Returns:
        tuple[str, int]: concordance label and integer score from 0 to 100.
    """
    if facial_emotion == speech_emotion:
        # Same emotion from both modalities: agreement strength depends on
        # average confidence.
        score = (facial_confidence + speech_confidence) / 2
        if score > 0.7:
            concordance = "MATCH"
        elif score >= 0.4:
            concordance = "PARTIAL"
        else:
            concordance = "MISMATCH"
    else:
        # Different labels cannot be a full match. The confidence-gap score
        # captures whether both models are similarly confident or whether one
        # modality strongly dominates the other.
        score = 1 - abs(facial_confidence - speech_confidence)
        if score >= 0.5:
            concordance = "PARTIAL"
        else:
            concordance = "MISMATCH"

    concordance_score = round(score * 100)
    return concordance, concordance_score


# Haar cascade fallback detector.
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Preferred MTCNN detector when facenet-pytorch is available.
# keep_all=False means the system focuses on one primary face.
MTCNN_DETECTOR = MTCNN(keep_all=False, device=DEVICE) if MTCNN is not None else None


def _encode_image_base64(image_array: np.ndarray) -> str:
    """Encode a NumPy image array as a base64 PNG string.

    JSON cannot directly carry binary image data, so generated images such as
    annotated face previews and Grad-CAM overlays are encoded as base64 strings.
    """
    image_pil = Image.fromarray(image_array.astype(np.uint8))
    buf = BytesIO()
    image_pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def _detect_primary_face(image: Image.Image):
    """Detect the main face in an image.

    Detection priority:
    1. MTCNN, if available. This gives stronger detections and landmarks.
    2. Haar cascade fallback. This allows the app to keep working even when
       MTCNN is unavailable or fails on a given image.

    Returns:
        ((x, y, w, h), landmarks) if a face is found.
        (None, None) if no face is detected.
    """
    # Preferred path: MTCNN.
    if MTCNN_DETECTOR is not None:
        try:
            boxes, probs, points = MTCNN_DETECTOR.detect(image, landmarks=True)
            if boxes is not None and len(boxes) > 0:
                # If multiple faces are found, use the highest-probability one.
                best_idx = int(np.argmax(probs)) if probs is not None else 0
                x1, y1, x2, y2 = boxes[best_idx]

                # Convert MTCNN box format [x1, y1, x2, y2] into OpenCV-style
                # [x, y, width, height].
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                return (x, y, w, h), (points[best_idx] if points is not None else None)
        except Exception as e:
            logger.debug(f"MTCNN face detection fallback: {e}")

    # Fallback path: Haar cascade.
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

    # If Haar finds multiple faces, use the largest box as the primary face.
    best_face = max(faces, key=lambda b: b[2] * b[3])
    return tuple(int(v) for v in best_face), None


def _rotate_image_to_level(image: Image.Image, points) -> Image.Image:
    """Optionally rotate a face image to correct small head tilt.

    MTCNN landmarks provide eye coordinates. If enabled, the function estimates
    the angle between both eyes and rotates the image to make the eyes level.
    Rotation is bounded so the system does not over-correct and distort the face.
    """
    if not ENABLE_FACE_ROTATION:
        return image

    if points is None:
        return image

    try:
        left_eye, right_eye = points[0], points[1]
        angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

        # Very small angles do not need correction.
        if abs(angle) < 1.0:
            return image

        # Large angles are skipped to avoid creating distorted crops.
        if abs(angle) > MAX_FACE_ROTATION_DEGREES:
            logger.debug("Skipping face rotation due to large angle: %.2f", angle)
            return image

        center_x = image.width / 2
        center_y = image.height / 2
        return image.rotate(
            -angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            center=(center_x, center_y),
            fillcolor=(0, 0, 0)
        )
    except Exception:
        return image


def _crop_face_with_margin(image_array: np.ndarray, face_box, margin_ratio: float = 0.12):
    """Crop the detected face with a small surrounding margin.

    The 12% margin preserves context around the face, such as eyebrows, cheeks,
    jawline, and nearby periocular regions, which may help emotion recognition.
    """
    x, y, w, h = [int(v) for v in face_box]
    h_img, w_img = image_array.shape[:2]
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)

    # Clamp crop coordinates so they stay inside image boundaries.
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w_img, x + w + mx)
    y2 = min(h_img, y + h + my)

    return image_array[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def _shrink_box(face_box, shrink_ratio: float = 0.12):
    """Shrink a face box for cleaner visual annotation.

    This does not affect inference. It only changes the drawn rectangle shown to
    the user in the annotated preview image.
    """
    x, y, w, h = [int(v) for v in face_box]
    dx = int(w * shrink_ratio / 2)
    dy = int(h * shrink_ratio / 2)
    x1 = x + dx
    y1 = y + dy
    width = max(1, w - (dx * 2))
    height = max(1, h - (dy * 2))
    return x1, y1, width, height


def _trim_audio_window(audio: np.ndarray, sr: int, max_seconds: int) -> np.ndarray:
    """Centre-crop long audio to a fixed maximum duration.

    This keeps inference fast and stable. A centred crop is used rather than
    taking only the beginning because emotional content may occur in the middle
    of the utterance.
    """
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


# =============================================================================
# Model Loading
# =============================================================================

def load_facial_model():
    """Load the ViT facial emotion model and image processor.

    The function is thread-safe and supports both checkpoint formats:
    1. A dictionary containing 'model_state_dict'.
    2. A raw PyTorch state_dict.
    """
    global vit_model, facial_processor, facial_loaded

    # Fast path: model is already loaded.
    if vit_model is not None and facial_processor is not None:
        facial_loaded = True
        return True

    # Lock ensures only one thread loads the model at a time.
    with _facial_model_lock:
        if vit_model is not None and facial_processor is not None:
            facial_loaded = True
            return True

        try:
            logger.info("Loading Facial Emotion Model (ViT)...")

            # Load the preprocessing configuration for the pre-trained ViT.
            facial_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

            # Load the ViT architecture with a 7-class emotion head.
            # ignore_mismatched_sizes=True allows replacing the original ImageNet
            # classification head with the FER2013 emotion head.
            # attn_implementation='eager' is used to keep Grad-CAM gradients stable.
            vit_model = AutoModelForImageClassification.from_pretrained(
                'google/vit-base-patch16-224-in21k',
                num_labels=len(EMOTIONS_FACIAL),
                ignore_mismatched_sizes=True,
                attn_implementation='eager'
            )

            # Load trained checkpoint weights.
            checkpoint = torch.load(FACIAL_MODEL_PATH, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                vit_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                vit_model.load_state_dict(checkpoint)
            logger.info("✓ Loaded ViT checkpoint")

            # Move model to CPU/GPU and switch to inference mode.
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
    """Load the HuBERT speech emotion model and audio feature extractor."""
    global speech_model, speech_processor, speech_loaded

    # Fast path: model is already loaded.
    if speech_model is not None and speech_processor is not None:
        speech_loaded = True
        return True

    # Lock prevents duplicate concurrent loading.
    with _speech_model_lock:
        if speech_model is not None and speech_processor is not None:
            speech_loaded = True
            return True

        try:
            logger.info("Loading Speech Emotion Model (HuBERT)...")

            # Load HuBERT's expected waveform feature extractor.
            speech_processor = AutoFeatureExtractor.from_pretrained('facebook/hubert-large-ls960-ft')

            # Load HuBERT architecture with an 8-class RAVDESS emotion head.
            speech_model = AutoModelForAudioClassification.from_pretrained(
                'facebook/hubert-large-ls960-ft',
                num_labels=len(EMOTIONS_SPEECH),
                ignore_mismatched_sizes=True
            )

            # Load trained checkpoint weights.
            checkpoint = torch.load(SPEECH_MODEL_PATH, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                speech_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                speech_model.load_state_dict(checkpoint)
            logger.info("✓ Loaded HuBERT checkpoint")

            # Move model to CPU/GPU and switch to inference mode.
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
    """Ensure the facial model is available before inference."""
    if vit_model is not None and facial_processor is not None:
        return True
    return load_facial_model()


def ensure_speech_model_loaded() -> bool:
    """Ensure the speech model is available before inference."""
    if speech_model is not None and speech_processor is not None:
        return True
    return load_speech_model()


# Optional eager loading for deployments that prefer warm startup.
# If disabled, models load lazily on first request.
if PRELOAD_MODELS:
    facial_loaded = load_facial_model()
    speech_loaded = load_speech_model()


# =============================================================================
# Video Processing
# =============================================================================

class VideoProcessor:
    """Utility class for extracting sampled frames and audio from a video file."""

    @staticmethod
    def extract_frames_and_audio(video_path: str, fps_sample: int = 5):
        """Extract sampled frames and a mono 16kHz audio waveform from a video.

        Args:
            video_path: Path to uploaded temporary video file.
            fps_sample: Sample every Nth frame. For example, 5 means frames
                0, 5, 10, 15, ... are extracted.

        Returns:
            frames: List of sampled PIL RGB frames.
            audio: Mono audio waveform loaded by librosa.
            sr: Audio sample rate, fixed to 16000.
            fps: Video frames per second, with invalid values replaced by 30.
        """
        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Read video metadata. total_frames is currently not used in later logic,
        # but it can help debugging or future duration calculations.
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Some videos report broken FPS metadata. Use 30 FPS as safe fallback.
        if fps <= 0 or fps > 120:
            fps = 30.0

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Extract every Nth frame to reduce inference cost.
            # Later, only the first 10 sampled frames are actually analysed by ViT.
            if frame_count % fps_sample == 0:
                # OpenCV uses BGR channel order; PIL and transformers expect RGB.
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))

            frame_count += 1

        # Release the video file handle.
        cap.release()

        # Extract audio from the same video file. librosa returns mono waveform.
        # sr=16000 matches HuBERT's expected sampling rate.
        audio, sr = librosa.load(video_path, sr=16000, mono=True)

        return frames, audio, sr, fps


# =============================================================================
# Prediction Functions
# =============================================================================

def predict_facial_emotion(image: Image.Image, generate_explainability: bool = False):
    """Predict facial emotion from a PIL image.

    Pipeline:
    1. Ensure ViT is loaded.
    2. Fix EXIF orientation and convert to RGB.
    3. Detect primary face using MTCNN, falling back to Haar cascade.
    4. Optionally correct small head tilt.
    5. Crop detected face with margin.
    6. Run ViT inference.
    7. Convert logits to probabilities.
    8. Optionally generate Grad-CAM.
    """
    try:
        if not ensure_facial_model_loaded():
            return None

        # Fix rotated smartphone images and ensure 3-channel RGB input.
        image = ImageOps.exif_transpose(image).convert('RGB')

        # Detect face before cropping or optional rotation.
        detected = _detect_primary_face(image)
        face_box, face_points = detected if isinstance(detected, tuple) else (None, None)

        # Optional eye-landmark rotation correction.
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

        # Use a face crop when detection succeeds. If detection fails, the full
        # image is still passed to ViT as graceful fallback.
        if face_box is not None:
            face_crop, _ = _crop_face_with_margin(input_array, face_box)
            if face_crop.size > 0:
                model_image = Image.fromarray(face_crop)

        # Create an annotated preview image for the frontend.
        annotated = input_array.copy()
        if face_box is not None:
            x, y, w, h = _shrink_box(face_box, shrink_ratio=0.08)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 128, 0), 2)
            cv2.putText(annotated, 'Face detected', (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2, cv2.LINE_AA)

        # Convert PIL image into model-ready tensor using the ViT processor.
        inputs = facial_processor(model_image, return_tensors='pt').to(DEVICE)

        # Inference does not require gradients.
        with torch.no_grad():
            outputs = vit_model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]
            probs = torch.softmax(torch.from_numpy(logits), dim=0).numpy()

        # Select highest-probability class.
        top_idx = np.argmax(probs)
        result = {
            "emotion": EMOTIONS_FACIAL[top_idx],
            "confidence": float(probs[top_idx]),
            "probabilities": {e: float(p) for e, p in zip(EMOTIONS_FACIAL, probs)},
            "face_detected": face_box is not None,
            "annotated_image": _encode_image_base64(annotated)
        }

        # Include face box coordinates if a face was found.
        if face_box is not None:
            x, y, w, h = [int(v) for v in face_box]
            result["face_box"] = {"x": x, "y": y, "width": w, "height": h}

        # Optional Grad-CAM explainability.
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
    """Predict speech emotion from an audio waveform.

    Pipeline:
    1. Ensure HuBERT is loaded.
    2. Resample to 16kHz if needed.
    3. Centre-crop long audio to 15 seconds.
    4. Convert waveform into HuBERT input values.
    5. Run HuBERT inference.
    6. Convert logits to probabilities.
    7. Optionally generate audio saliency.
    """
    try:
        if not ensure_speech_model_loaded():
            return None

        # HuBERT expects 16kHz audio because its pre-training used 16kHz speech.
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Bound inference time for long clips.
        audio_for_infer = _trim_audio_window(audio, 16000, MAX_SPEECH_INFER_SECONDS)

        # Convert raw waveform into HuBERT input tensor.
        inputs = speech_processor(
            audio_for_infer,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = speech_model(inputs['input_values'].to(DEVICE))
            logits = outputs.logits.cpu().numpy()[0]
            # Manual softmax: converts raw logits to probability distribution.
            probs = np.exp(logits) / np.sum(np.exp(logits))

        top_idx = np.argmax(probs)
        result = {
            "emotion": EMOTIONS_SPEECH[top_idx],
            "confidence": float(probs[top_idx]),
            "probabilities": {e: float(p) for e, p in zip(EMOTIONS_SPEECH, probs)}
        }

        # Optional audio saliency explainability.
        if generate_explainability:
            result["explainability_status"] = {
                "requested": True,
                "generated": False,
                "error": None
            }
            try:
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


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", tags=["system"], summary="Service Metadata")
async def root():
    """Basic service metadata endpoint for uptime checks."""
    return {"message": "Multi-Modal Emotion Recognition API v2.0", "status": "active"}


@app.get("/health", tags=["system"], summary="Health Check")
async def health():
    """Return backend health and model loading state."""
    facial_ready = vit_model is not None and facial_processor is not None
    speech_ready = speech_model is not None and speech_processor is not None
    return {
        "status": "healthy",
        "facial_model": facial_ready,
        "speech_model": speech_ready,
        "lazy_loading": not PRELOAD_MODELS,
        "device": str(DEVICE)
    }


@app.post("/api/predict/facial", tags=["prediction"], summary="Facial Emotion Prediction")
async def predict_facial(file: UploadFile = File(...), explain: bool = False):
    """API endpoint for image-only facial emotion prediction."""
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


@app.post("/api/predict/speech", tags=["prediction"], summary="Speech Emotion Prediction")
async def predict_speech(file: UploadFile = File(...), explain: bool = False):
    """API endpoint for audio-only speech emotion prediction."""
    try:
        contents = await file.read()
        suffix = _upload_suffix(file.filename, '.wav')

        # Save upload temporarily because librosa expects a file path for many
        # compressed formats. The file is deleted in the finally block.
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


@app.post("/api/predict/combined", tags=["prediction"], summary="Combined Multimodal Prediction")
async def predict_combined(
    image_file: UploadFile = File(...),
    audio_file: UploadFile = File(...),
    explain: bool = False
):
    """API endpoint for separate image + audio multimodal prediction."""
    try:
        # Facial branch.
        image_contents = await image_file.read()
        image = ImageOps.exif_transpose(Image.open(BytesIO(image_contents))).convert('RGB')
        facial_result = predict_facial_emotion(image, generate_explainability=explain)

        # Speech branch.
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

        # Extract emotion labels and confidence values safely.
        facial_emotion = facial_result["emotion"] if facial_result else None
        facial_confidence = facial_result["confidence"] if facial_result else 0.0

        speech_emotion = speech_result["emotion"] if speech_result else None
        speech_confidence = speech_result["confidence"] if speech_result else 0.0

        # Compute deterministic concordance between modalities.
        concordance, concordance_score = _calculate_concordance(
            facial_emotion,
            speech_emotion,
            facial_confidence,
            speech_confidence,
        )

        # Combined emotion is not a learned fusion output. It simply selects the
        # more confident modality when both predictions exist.
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

        # Build stable response shape for the React frontend.
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
                "agreement_details": (
                    f"Face: {facial_emotion} (conf: {facial_confidence:.2f}) | "
                    f"Voice: {speech_emotion} (conf: {speech_confidence:.2f})"
                )
            }
        }

        # Attach optional explainability outputs.
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


@app.post("/api/predict/video", tags=["prediction"], summary="Video Emotion Prediction")
async def predict_video_emotion(file: UploadFile = File(...), explain: bool = False):
    """API endpoint for video-based multimodal prediction.

    Important implementation detail:
    - The system extracts every 5th frame from the video.
    - Only the first 10 sampled frames are analysed by ViT to bound latency.
    - Audio is extracted once and analysed by HuBERT.
    - Facial predictions are aggregated by majority vote.
    """
    try:
        video_suffix = _upload_suffix(file.filename, '.mp4')
        with tempfile.NamedTemporaryFile(suffix=video_suffix, delete=False) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            processor = VideoProcessor()
            frames, audio, sr, fps = processor.extract_frames_and_audio(tmp_path, fps_sample=5)

            # Analyse up to 10 frames evenly across the whole video.
            facial_results = []

            if len(frames) > 10:
                indices = np.linspace(0, len(frames) - 1, 10, dtype=int)
                selected_frames = [frames[i] for i in indices]
            else:
                selected_frames = frames

            for frame in selected_frames:
                result = predict_facial_emotion(frame)
                if result:
                    facial_results.append(result)

            # Aggregate facial predictions across analysed frames.
            if facial_results:
                facial_emotions = [r["emotion"] for r in facial_results]
                facial_confidence = np.mean([r["confidence"] for r in facial_results])
                facial_emotion = max(set(facial_emotions), key=facial_emotions.count)

                # Average the full probability distribution over all analysed frames.
                facial_probs = {}
                for emotion in EMOTIONS_FACIAL:
                    facial_probs[emotion] = float(
                        np.mean([r["probabilities"].get(emotion, 0) for r in facial_results])
                    )
            else:
                facial_emotion = "unknown"
                facial_confidence = 0.0
                facial_probs = {e: 0.0 for e in EMOTIONS_FACIAL}

            # Analyse extracted audio once through HuBERT.
            speech_result = predict_speech_emotion(audio, sr)
            speech_emotion = speech_result["emotion"] if speech_result else "unknown"
            speech_confidence = float(speech_result["confidence"]) if speech_result else 0.0

            # Compute concordance between aggregated face result and speech result.
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
                "combined_emotion": facial_emotion if facial_confidence > 0.5 else (
                    speech_result["emotion"] if speech_result else "unknown"
                ),
                "concordance": concordance,
                "concordance_score": concordance_score,
                "video_duration": float(len(audio) / sr),

                # Note: this reports sampled frames extracted, not frames actually
                # analysed by ViT. The analysed count is facial_emotion.frames_analyzed.
                "frames_processed": len(frames),
                "fps": float(fps)
            }

            # Optional video explainability.
            if explain:
                explainability = {}
                errors = []

                facial_exp_status = {"requested": True, "generated": False, "error": None}
                speech_exp_status = {"requested": True, "generated": False, "error": None}

                # Facial Grad-CAM is generated for one representative frame only.
                if frames and facial_emotion != "unknown":
                    try:
                        best_frame = None
                        best_result = None
                        best_conf = 0

                        # Choose the frame that predicted the aggregated emotion
                        # with the highest confidence.
                        for frame in frames[:10]:
                            r = predict_facial_emotion(frame)
                            if r and r.get("emotion") == facial_emotion and r.get("confidence", 0) > best_conf:
                                best_conf = r["confidence"]
                                best_frame = frame
                                best_result = r

                        # Fallback: if no representative frame is found, use first frame.
                        if best_frame is None and frames:
                            best_frame = frames[0]
                            best_result = predict_facial_emotion(best_frame)

                        if best_frame is not None:
                            top_idx = EMOTIONS_FACIAL.index(facial_emotion) \
                                if facial_emotion in EMOTIONS_FACIAL else 0

                            # Crop face before Grad-CAM so the heatmap focuses on the face.
                            face_box, _ = _detect_primary_face(best_frame)
                            if face_box is not None:
                                frame_array = np.array(best_frame)
                                face_crop_array, _ = _crop_face_with_margin(frame_array, face_box)
                                gradcam_input = Image.fromarray(face_crop_array) if face_crop_array.size > 0 else best_frame
                            else:
                                gradcam_input = best_frame

                            orig_b64, heatmap_b64 = generate_grad_cam(
                                gradcam_input,
                                vit_model,
                                facial_processor,
                                top_idx,
                                EMOTIONS_FACIAL,
                                DEVICE
                            )
                            if heatmap_b64:
                                explainability["grad_cam"] = heatmap_b64
                                facial_exp_status["generated"] = True
                            else:
                                facial_exp_status["error"] = "GradCAM returned empty output"
                    except Exception as e:
                        facial_exp_status["error"] = str(e)
                else:
                    facial_exp_status["error"] = "No valid frame prediction found for facial explainability"

                # Speech saliency is generated on a short centred audio segment.
                if speech_result and speech_emotion != "unknown":
                    try:
                        top_idx = EMOTIONS_SPEECH.index(speech_emotion) \
                            if speech_emotion in EMOTIONS_SPEECH else 0
                        audio_for_xai = _trim_audio_window(audio, sr, max_seconds=MAX_SPEECH_XAI_SECONDS)
                        spec_b64, saliency_b64 = generate_audio_saliency(
                            audio_for_xai,
                            speech_model,
                            speech_processor,
                            top_idx,
                            EMOTIONS_SPEECH,
                            DEVICE,
                            sr=16000
                        )
                        if spec_b64:
                            explainability["waveform"] = spec_b64
                        if saliency_b64:
                            explainability["saliency"] = saliency_b64
                            speech_exp_status["generated"] = True
                        else:
                            speech_exp_status["error"] = "Audio saliency map returned empty output"
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
            # Always remove temporary video file after processing.
            os.unlink(tmp_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/api/emotions/facial", tags=["reference"], summary="List Facial Emotion Labels")
async def get_facial_emotions():
    """Return the 7 facial emotion labels used by the ViT model."""
    return {"emotions": EMOTIONS_FACIAL}


@app.get("/api/emotions/speech", tags=["reference"], summary="List Speech Emotion Labels")
async def get_speech_emotions():
    """Return the 8 speech emotion labels used by the HuBERT model."""
    return {"emotions": EMOTIONS_SPEECH}


@app.get("/api/models/status", tags=["system"], summary="Model Runtime Status")
async def get_models_status():
    """Return model loading status, reported accuracies, and runtime device."""
    facial_ready = vit_model is not None and facial_processor is not None
    speech_ready = speech_model is not None and speech_processor is not None
    return {
        "facial": {"loaded": facial_ready, "accuracy": 0.7129, "emotions": len(EMOTIONS_FACIAL)},
        "speech": {"loaded": speech_ready, "accuracy": 0.8750, "emotions": len(EMOTIONS_SPEECH)},
        "lazy_loading": not PRELOAD_MODELS,
        "device": str(DEVICE)
    }
