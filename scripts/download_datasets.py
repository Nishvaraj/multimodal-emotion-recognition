import os
import urllib.request
import zipfile

datasets_dir = "datasets/raw"
os.makedirs(datasets_dir, exist_ok=True)

print("=" * 70)
print("DATASET DOWNLOAD INSTRUCTIONS")
print("=" * 70)
print("\nNote: FER2013 and RAVDESS require manual download from Kaggle")
print("\n📥 FER2013 (Facial Expression Recognition):")
print("   URL: https://www.kaggle.com/datasets/msambare/fer2013")
print(f"   Extract to: {datasets_dir}/fer2013/")
print("\n📥 RAVDESS (Audio Speech Emotion):")
print("   URL: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio")
print(f"   Extract to: {datasets_dir}/ravdess/")
print("\n" + "=" * 70)
print("\nAfter downloading:")
print(f"  1. Extract FER2013 to: {os.path.abspath(datasets_dir)}/fer2013/")
print(f"  2. Extract RAVDESS to: {os.path.abspath(datasets_dir)}/ravdess/")
print("\nExpected structure:")
print(f"  {datasets_dir}/")
print("    ├── fer2013/")
print("    │   ├── train/")
print("    │   ├── val/")
print("    │   └── test/")
print("    └── ravdess/")
print("        └── *.wav files")
print("=" * 70)
