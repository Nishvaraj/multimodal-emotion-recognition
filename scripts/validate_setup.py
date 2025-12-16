import sys

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")

try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV: {e}")

try:
    import librosa
    print(f"✓ Librosa: {librosa.__version__}")
except ImportError as e:
    print(f"✗ Librosa: {e}")

try:
    from transformers import AutoModel
    print("✓ Transformers available")
except ImportError as e:
    print(f"✗ Transformers: {e}")

try:
    import numpy
    print(f"✓ NumPy: {numpy.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")

try:
    import pandas
    print(f"✓ Pandas: {pandas.__version__}")
except ImportError as e:
    print(f"✗ Pandas: {e}")

try:
    import scipy
    print(f"✓ SciPy: {scipy.__version__}")
except ImportError as e:
    print(f"✗ SciPy: {e}")

print("\n✅ Core dependencies check complete!")
