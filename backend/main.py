from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForImageClassification

app = FastAPI(title="Emotion Recognition API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vit_model = None
image_processor = None
MODEL_PATH = Path('./models/phase2/vit_emotion_model.pt')

# Load models at startup
def load_models():
    """Load ViT model for facial emotion recognition"""
    global vit_model, image_processor
    
    try:
        print("Loading Vision Transformer model...")
        image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        
        vit_model = AutoModelForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=len(EMOTIONS),
            ignore_mismatched_sizes=True
        )
        
        if MODEL_PATH.exists():
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                vit_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                vit_model.load_state_dict(checkpoint)
            print(f"✓ Loaded trained weights from {MODEL_PATH}")
        
        vit_model = vit_model.to(DEVICE)
        vit_model.eval()
        print("✓ Vision Transformer model loaded successfully")
        
    except Exception as e:
        print(f"Error loading models: {e}")

# Load models on startup
load_models()

@app.get("/")
async def root():
    return {"message": "Emotion Recognition API", "status": "active"}

@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": vit_model is not None}

@app.post("/api/predict/facial")
async def predict_facial_emotion(file: UploadFile = File(...)):
    """Predict facial emotion from uploaded image"""
    try:
        if vit_model is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Model not loaded"}
            )
        
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        
        # Preprocess
        inputs = image_processor(image, return_tensors='pt').to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = vit_model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]
            probs = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
        
        # Get results
        top_idx = np.argmax(probs)
        top_emotion = EMOTIONS[top_idx]
        
        return {
            "success": True,
            "emotion": top_emotion,
            "confidence": float(probs[top_idx]),
            "probabilities": {emotion: float(prob) for emotion, prob in zip(EMOTIONS, probs)}
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

@app.get("/api/predict/facial")
async def predict_facial_emotion_demo():
    """Demo endpoint for facial emotion prediction (GET)"""
    return {
        "message": "Use POST method to predict emotion from an image",
        "usage": "POST /api/predict/facial with file upload",
        "example": "curl -X POST 'http://127.0.0.1:8000/api/predict/facial' -F 'file=@image.jpg'",
        "emotions": EMOTIONS,
        "model_status": {
            "loaded": vit_model is not None,
            "accuracy": 0.7129
        }
    }

@app.get("/api/emotions")
async def get_emotions():
    """Get list of supported emotions"""
    return {"emotions": EMOTIONS, "count": len(EMOTIONS)}

@app.get("/api/models")
async def get_models_status():
    """Get status of loaded models"""
    return {
        "vit": {
            "loaded": vit_model is not None,
            "emotions": len(EMOTIONS),
            "device": str(DEVICE),
            "accuracy": 0.7129
        }
    }
