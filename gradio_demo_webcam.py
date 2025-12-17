"""
Gradio Webcam Emotion Recognition - Simple Version with Manual Predict
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
from transformers import ViTForImageClassification
import gradio as gr

# ==================== Model Setup ====================
print("Loading model...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load model
vit_model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=7,
    ignore_mismatched_sizes=True
)

model_path = 'vit_emotion_model.pt'
if not Path(model_path).exists():
    print(f"Error: {model_path} not found")
    exit(1)

checkpoint = torch.load(model_path, map_location=device)
vit_model.load_state_dict(checkpoint['model_state_dict'])
vit_model = vit_model.to(device)
vit_model.eval()

accuracy = checkpoint.get('accuracy', 'N/A')
print(f"✓ Model loaded (Accuracy: {accuracy:.2%})\n")

# ==================== Configuration ====================
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_colors = {
    'angry': (0, 0, 255),
    'disgust': (0, 165, 255),
    'fear': (255, 0, 0),
    'happy': (0, 255, 0),
    'neutral': (128, 128, 128),
    'sad': (255, 0, 255),
    'surprise': (0, 255, 255)
}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# ==================== Prediction Function ====================
def predict_emotion(image):
    """Predict emotion from image"""
    if image is None:
        return None, "❌ No image captured", {}

    try:
        # Convert to numpy if PIL
        if hasattr(image, 'convert'):
            img_np = np.array(image.convert('RGB'))
        else:
            img_np = np.asarray(image, dtype=np.uint8)

        print(f"📸 Processing image: {img_np.shape}")

        # Ensure RGB
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_rgb = img_np
        else:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

        # Grayscale conversion (like training)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Predict
        img_tensor = transform(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = vit_model(pixel_values=img_tensor)
            logits = outputs.logits.cpu().numpy()[0]
            probs = np.exp(logits) / np.sum(np.exp(logits))

        # Get top emotion
        top_idx = np.argmax(probs)
        top_emotion = emotions[top_idx]
        top_prob = probs[top_idx]

        # Annotate image
        annotated = img_rgb.copy()
        h, w = annotated.shape[:2]

        # Draw background
        cv2.rectangle(annotated, (10, 10), (w-10, 120), (0, 0, 0), -1)

        # Draw prediction
        color = emotion_colors.get(top_emotion, (255, 255, 255))
        cv2.putText(annotated, f"Emotion: {top_emotion.upper()}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(annotated, f"Confidence: {top_prob:.1%}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

        # Format results dict
        results = {emotions[i]: float(probs[i]) for i in range(len(emotions))}

        print(f"✅ {top_emotion.upper()} ({top_prob:.1%})")

        return annotated, f"😊 {top_emotion.upper()} ({top_prob:.1%})", results

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return image, f"❌ Error: {str(e)}", {}

# ==================== Gradio Interface ====================
def launch_app():
    with gr.Blocks(title="😊 Facial Emotion Recognition") as demo:
        gr.Markdown("""
        # 😊 Facial Emotion Recognition - Live Webcam

        **Step 1:** Click the camera button to capture from your webcam
        **Step 2:** Click "Predict Emotion" to analyze

        **Model:** Vision Transformer (ViT-base-patch16-224)
        **Accuracy:** 71.29%
        """)

        with gr.Row():
            webcam_input = gr.Image(
                label="📹 Webcam Capture",
                sources=["webcam"],
                type="pil"
            )

            with gr.Column():
                predict_btn = gr.Button("🔮 Predict Emotion", variant="primary", size="lg")
                emotion_output = gr.Textbox(label="Detected Emotion", interactive=False)
                probs_output = gr.Label(label="📊 Confidence Scores")

        annotated_output = gr.Image(label="✨ Annotated Result")

        # Predict on button click
        predict_btn.click(
            fn=predict_emotion,
            inputs=webcam_input,
            outputs=[annotated_output, emotion_output, probs_output]
        )

    return demo

if __name__ == "__main__":
    print("🎥 Launching webcam emotion recognition app...\n")
    app = launch_app()
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
