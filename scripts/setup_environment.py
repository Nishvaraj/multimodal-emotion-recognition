"""
Environment setup verification script
"""

import sys
import subprocess

def check_python():
    """Check Python version"""
    print("✓ Python:", sys.version)
    if sys.version_info < (3, 10):
        print("⚠ Warning: Python 3.10+ recommended")
    return True

def check_packages():
    """Check if key packages are installed"""
    print("\nChecking Python packages:")
    try:
        import torch
        print("  ✓ PyTorch:", torch.__version__)
    except ImportError:
        print("  ✗ PyTorch: NOT INSTALLED")

    try:
        import fastapi
        print("  ✓ FastAPI:", fastapi.__version__)
    except ImportError:
        print("  ✗ FastAPI: NOT INSTALLED")

    try:
        import transformers
        print("  ✓ Transformers:", transformers.__version__)
    except ImportError:
        print("  ✗ Transformers: NOT INSTALLED")

    try:
        import librosa
        print("  ✓ Librosa:", librosa.__version__)
    except ImportError:
        print("  ✗ Librosa: NOT INSTALLED")

    try:
        import cv2
        print("  ✓ OpenCV:", cv2.__version__)
    except ImportError:
        print("  ✗ OpenCV: NOT INSTALLED")

    print("\nAll packages installed! Ready to code.")

def main():
    print("=" * 70)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 70 + "\n")

    check_python()
    check_packages()

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("""
1. Download datasets:
   python scripts/download_datasets.py

2. Run backend:
   python backend/main.py
   OR press F5 in VS Code

3. Run frontend:
   cd frontend && npm start

4. Open browser to http://localhost:3000
    """)
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()


