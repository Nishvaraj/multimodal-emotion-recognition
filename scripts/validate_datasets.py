"""
Dataset Structure Validation Script
Verifies FER2013 and RAVDESS datasets are correctly organized
"""
import os
from pathlib import Path
from collections import Counter

def validate_fer2013_structure():
    """Validate FER2013 dataset structure"""
    print("=" * 60)
    print("VALIDATING FER2013 DATASET STRUCTURE")
    print("=" * 60)
    
    base_path = Path("data/raw/fer2013")
    expected_splits = ["train", "test"]
    expected_emotions = ["happy", "sad", "angry", "fear", "surprise", "neutral", "disgust"]
    
    # Check base directory exists
    if not base_path.exists():
        print(f"❌ ERROR: {base_path} does not exist!")
        return False
    
    all_valid = True
    
    # Check each split
    for split in expected_splits:
        split_path = base_path / split
        print(f"\n📁 Checking {split}/ directory...")
        
        if not split_path.exists():
            print(f"   ❌ Missing: {split}/ directory")
            all_valid = False
            continue
        
        # Check emotion subdirectories
        emotion_counts = {}
        for emotion in expected_emotions:
            emotion_path = split_path / emotion
            if not emotion_path.exists():
                print(f"   ⚠️  Missing emotion folder: {emotion}/")
                emotion_counts[emotion] = 0
                all_valid = False
            else:
                # Count images
                image_files = list(emotion_path.glob("*.jpg")) + list(emotion_path.glob("*.png"))
                emotion_counts[emotion] = len(image_files)
                print(f"   ✓ {emotion}: {len(image_files)} images")
        
        total_images = sum(emotion_counts.values())
        print(f"   📊 Total images in {split}: {total_images}")
    
    if all_valid:
        print("\n✅ FER2013 structure is VALID")
    else:
        print("\n❌ FER2013 structure has ISSUES")
    
    return all_valid

def validate_ravdess_structure():
    """Validate RAVDESS dataset structure"""
    print("\n" + "=" * 60)
    print("VALIDATING RAVDESS DATASET STRUCTURE")
    print("=" * 60)
    
    base_path = Path("data/raw/ravdess")
    
    # Check base directory exists
    if not base_path.exists():
        print(f"❌ ERROR: {base_path} does not exist!")
        return False
    
    # Find all .wav files
    wav_files = list(base_path.glob("**/*.wav"))
    
    if len(wav_files) == 0:
        print(f"❌ No .wav files found in {base_path}")
        return False
    
    print(f"\n📁 Found {len(wav_files)} audio files")
    
    # Parse RAVDESS emotion codes from filenames
    # Format: 03-01-XX-01-01-01-01.wav (XX is emotion code)
    emotion_map = {
        1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
        5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
    }
    
    emotion_counts = Counter()
    valid_format = 0
    
    for wav_file in wav_files:
        parts = wav_file.stem.split('-')
        if len(parts) >= 3:
            try:
                emotion_code = int(parts[2])
                emotion = emotion_map.get(emotion_code, 'unknown')
                emotion_counts[emotion] += 1
                valid_format += 1
            except ValueError:
                pass
    
    print(f"\n✓ Valid RAVDESS format files: {valid_format}/{len(wav_files)}")
    print("\n📊 Emotion distribution:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"   {emotion}: {count} files")
    
    if valid_format > 0:
        print("\n✅ RAVDESS structure is VALID")
        return True
    else:
        print("\n❌ RAVDESS structure has ISSUES")
        return False

def check_dataset_sizes():
    """Check if datasets meet minimum size requirements"""
    print("\n" + "=" * 60)
    print("DATASET SIZE REQUIREMENTS CHECK")
    print("=" * 60)
    
    requirements = {
        "FER2013 train": ("data/raw/fer2013/train", 20000),
        "FER2013 test": ("data/raw/fer2013/test", 3000),
        "RAVDESS": ("data/raw/ravdess", 1000)
    }
    
    all_meet_requirements = True
    
    for name, (path, min_size) in requirements.items():
        path_obj = Path(path)
        if path_obj.exists():
            if "ravdess" in name.lower():
                count = len(list(path_obj.glob("**/*.wav")))
            else:
                count = len(list(path_obj.glob("**/*.jpg"))) + len(list(path_obj.glob("**/*.png")))
            
            status = "✅" if count >= min_size else "⚠️"
            print(f"{status} {name}: {count} (required: {min_size})")
            
            if count < min_size:
                all_meet_requirements = False
        else:
            print(f"❌ {name}: Path not found")
            all_meet_requirements = False
    
    return all_meet_requirements

def main():
    """Run all validation checks"""
    print("\n🔍 DATASET VALIDATION STARTING...\n")
    
    fer_valid = validate_fer2013_structure()
    ravdess_valid = validate_ravdess_structure()
    sizes_valid = check_dataset_sizes()
    
    print("\n" + "=" * 60)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 60)
    print(f"FER2013 Structure: {'✅ PASS' if fer_valid else '❌ FAIL'}")
    print(f"RAVDESS Structure: {'✅ PASS' if ravdess_valid else '❌ FAIL'}")
    print(f"Dataset Sizes: {'✅ PASS' if sizes_valid else '⚠️  WARNING'}")
    
    if fer_valid and ravdess_valid:
        print("\n🎉 ALL VALIDATION CHECKS PASSED!")
        print("✅ Ready to proceed to Phase 1: Data Preparation")
    else:
        print("\n⚠️  VALIDATION ISSUES DETECTED")
        print("Please fix the issues above before proceeding to Phase 1")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
