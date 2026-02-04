"""
Explainability module for facial emotion recognition.
Generates Grad-CAM visualizations to show which regions of the face
contribute most to the emotion prediction.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import base64
from typing import Tuple, Optional

class GradCAM:
    """Generate Grad-CAM heatmaps for ViT model predictions"""
    
    def __init__(self, model, device='cpu'):
        """
        Initialize Grad-CAM
        
        Args:
            model: Vision Transformer model
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.device = device
        self.gradients = []
        self.activations = []
        
        # Register hooks to capture gradients and activations
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture intermediate values"""
        def forward_hook(module, input, output):
            self.activations.append(output.detach())
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].detach())
        
        # Hook into the last layer before classification
        # For ViT, we hook into the pooler layer
        if hasattr(self.model, 'vit'):
            self.model.vit.layernorm.register_forward_hook(forward_hook)
            self.model.vit.layernorm.register_backward_hook(backward_hook)
        else:
            # Fallback: hook into last hidden layer
            for module in self.model.modules():
                if hasattr(module, 'out_features') and module.out_features == 768:
                    module.register_forward_hook(forward_hook)
                    module.register_backward_hook(backward_hook)
                    break
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor (1, 3, 224, 224)
            target_class: Target class index for which to generate CAM
            
        Returns:
            CAM heatmap (224, 224)
        """
        self.model.eval()
        self.gradients.clear()
        self.activations.clear()
        
        # Forward pass
        output = self.model(input_tensor)
        logits = output.logits
        
        # Create one-hot target
        target = torch.zeros_like(logits)
        target[0, target_class] = 1
        
        # Backward pass
        loss = (logits * target).sum()
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        
        # Get gradients and activations
        if len(self.gradients) == 0 or len(self.activations) == 0:
            # Fallback: use simple feature importance
            return self._simple_importance_map(input_tensor, target_class)
        
        gradients = self.gradients[-1]
        activations = self.activations[-1]
        
        # Average pool the gradients across channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Weight activations by gradients
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        
        # Average activation
        cam = torch.mean(activations, dim=1).squeeze(0).cpu().detach().numpy()
        
        # Normalize CAM
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    def _simple_importance_map(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Fallback method: Generate importance map using input gradient
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class index
            
        Returns:
            Importance heatmap
        """
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        logits = output.logits
        
        # Get gradient w.r.t. input for target class
        loss = logits[0, target_class]
        loss.backward()
        
        # Get absolute gradient values
        gradients = input_tensor.grad.abs().sum(dim=1).squeeze(0)
        
        # Normalize
        importance = gradients.cpu().detach().numpy()
        importance = importance / (importance.max() + 1e-8)
        
        return importance


def create_heatmap_visualization(
    image: Image.Image, 
    cam: np.ndarray, 
    emotion: str,
    confidence: float
) -> str:
    """
    Create visualization with heatmap overlay on original image
    
    Args:
        image: Original PIL image
        cam: CAM heatmap (224, 224)
        emotion: Predicted emotion
        confidence: Confidence score
        
    Returns:
        Base64 encoded image string
    """
    # Resize CAM to image size
    img_array = np.array(image)
    original_height, original_width = img_array.shape[:2]
    
    # Resize CAM
    cam_resized = cv2.resize(cam, (original_width, original_height))
    
    # Convert CAM to heatmap
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    
    # Convert to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend with original image (70% original, 30% heatmap)
    overlay = cv2.addWeighted(img_array, 0.7, heatmap, 0.3, 0)
    
    # Add text annotation
    text = f"{emotion.upper()} ({confidence*100:.1f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color = (0, 255, 0)  # Green
    
    # Add background for text
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x, y = 10, 30
    overlay = cv2.rectangle(
        overlay,
        (x - 5, y - text_size[1] - 5),
        (x + text_size[0] + 5, y + 5),
        (0, 0, 0),
        -1
    )
    overlay = cv2.putText(
        overlay,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness
    )
    
    # Convert to PIL Image
    result_image = Image.fromarray(overlay)
    
    # Convert to base64
    buffered = BytesIO()
    result_image.save(buffered, format="PNG")
    buffered.seek(0)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{img_base64}"


def generate_grad_cam_for_prediction(
    image: Image.Image,
    model: torch.nn.Module,
    processor,
    emotion: str,
    emotion_list: list,
    confidence: float,
    device: str = 'cpu'
) -> dict:
    """
    Generate Grad-CAM visualization for a facial emotion prediction
    
    Args:
        image: PIL Image
        model: ViT model
        processor: Image processor
        emotion: Predicted emotion
        emotion_list: List of all emotions
        confidence: Prediction confidence
        device: 'cpu' or 'cuda'
        
    Returns:
        Dict with grad_cam_image (base64) and metadata
    """
    try:
        # Get emotion class index
        target_class = emotion_list.index(emotion)
        
        # Prepare input
        inputs = processor(image, return_tensors='pt')
        input_tensor = inputs['pixel_values'].to(device)
        
        # Generate Grad-CAM
        grad_cam = GradCAM(model, device=device)
        cam = grad_cam.generate_cam(input_tensor, target_class)
        
        # Create visualization
        grad_cam_image = create_heatmap_visualization(
            image, 
            cam, 
            emotion, 
            confidence
        )
        
        return {
            "success": True,
            "grad_cam_image": grad_cam_image,
            "heatmap_description": f"Red regions indicate high importance for {emotion} emotion prediction"
        }
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return {
            "success": False,
            "error": str(e),
            "grad_cam_image": None
        }
