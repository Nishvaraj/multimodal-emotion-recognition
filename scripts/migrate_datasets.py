#!/usr/bin/env python3
"""
Script to move and organize datasets from Desktop to project data directory.
Handles FER2013 (archive) and RAVDESS (archive-2) datasets.
"""

import os
import shutil
import zipfile
from pathlib import Path

# Paths
DESKTOP = Path.home() / "Desktop"
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

# Dataset archives on Desktop
FER2013_ARCHIVE = DESKTOP / "archive"
RAVDESS_ARCHIVE = DESKTOP / "archive-2"

# Target directories
FER2013_TARGET = DATA_RAW / "fer2013"
RAVDESS_TARGET = DATA_RAW / "ravdess"

def setup_directories():
    """Create necessary directories."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    FER2013_TARGET.mkdir(parents=True, exist_ok=True)
    RAVDESS_TARGET.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directories in {DATA_RAW}")

def move_fer2013():
    """Move FER2013 dataset from Desktop/archive to data/raw/fer2013/"""
    print("\n📂 Processing FER2013 dataset (archive)...")

    if not FER2013_ARCHIVE.exists():
        print(f"❌ FER2013 archive not found at: {FER2013_ARCHIVE}")
        return False

    # Check if it's a zip file
    if FER2013_ARCHIVE.suffix == '.zip':
        print(f"  Extracting {FER2013_ARCHIVE}...")
        with zipfile.ZipFile(FER2013_ARCHIVE, 'r') as zip_ref:
            zip_ref.extractall(FER2013_TARGET)
    elif FER2013_ARCHIVE.is_dir():
        # It's already extracted, copy contents
        print(f"  Copying from {FER2013_ARCHIVE}...")
        for item in FER2013_ARCHIVE.iterdir():
            dest = FER2013_TARGET / item.name
            if item.is_dir():
                if not dest.exists():
                    shutil.copytree(item, dest)
                    print(f"  ✓ Copied directory: {item.name}")
            else:
                shutil.copy2(item, dest)
                print(f"  ✓ Copied file: {item.name}")

    # Verify structure
    expected_dirs = ["train", "test", "val"]
    found_dirs = [d.name for d in FER2013_TARGET.iterdir() if d.is_dir()]

    if any(exp in found_dirs for exp in expected_dirs):
        print(f"✅ FER2013 dataset ready at: {FER2013_TARGET}")
        print(f"   Found subdirectories: {', '.join(found_dirs)}")
        return True
    else:
        print(f"⚠️  Warning: Expected train/test/val folders not found")
        print(f"   Found: {', '.join(found_dirs)}")
        return True

def move_ravdess():
    """Move RAVDESS dataset from Desktop/archive-2 to data/raw/ravdess/"""
    print("\n📂 Processing RAVDESS dataset (archive-2)...")

    if not RAVDESS_ARCHIVE.exists():
        print(f"❌ RAVDESS archive not found at: {RAVDESS_ARCHIVE}")
        return False

    # Check if it's a zip file
    if RAVDESS_ARCHIVE.suffix == '.zip':
        print(f"  Extracting {RAVDESS_ARCHIVE}...")
        with zipfile.ZipFile(RAVDESS_ARCHIVE, 'r') as zip_ref:
            zip_ref.extractall(RAVDESS_TARGET)
    elif RAVDESS_ARCHIVE.is_dir():
        # It's already extracted, copy contents
        print(f"  Copying from {RAVDESS_ARCHIVE}...")
        for item in RAVDESS_ARCHIVE.rglob('*.wav'):
            dest = RAVDESS_TARGET / item.name
            shutil.copy2(item, dest)
            print(f"  ✓ Copied: {item.name}")

    # Count audio files
    wav_files = list(RAVDESS_TARGET.glob('*.wav'))
    if wav_files:
        print(f"✅ RAVDESS dataset ready at: {RAVDESS_TARGET}")
        print(f"   Found {len(wav_files)} audio files")
        return True
    else:
        print(f"⚠️  Warning: No .wav files found in {RAVDESS_TARGET}")
        return False

def main():
    print("=" * 70)
    print("DATASET MIGRATION TOOL")
    print("=" * 70)
    print(f"\nSource locations:")
    print(f"  FER2013:  {FER2013_ARCHIVE}")
    print(f"  RAVDESS:  {RAVDESS_ARCHIVE}")
    print(f"\nTarget location:")
    print(f"  {DATA_RAW}")
    print("\n" + "=" * 70)

    # Create directories
    setup_directories()

    # Move datasets
    fer_success = move_fer2013()
    ravdess_success = move_ravdess()

    # Summary
    print("\n" + "=" * 70)
    print("MIGRATION SUMMARY")
    print("=" * 70)
    print(f"FER2013:  {'✅ Success' if fer_success else '❌ Failed'}")
    print(f"RAVDESS:  {'✅ Success' if ravdess_success else '❌ Failed'}")
    print("=" * 70)

    if fer_success and ravdess_success:
        print("\n🎉 All datasets migrated successfully!")
        print("\nNext steps:")
        print("  1. Verify data structure:")
        print(f"     ls -R {DATA_RAW}")
        print("  2. Proceed to Phase 1: Data Preparation")
    else:
        print("\n⚠️  Some datasets could not be migrated.")
        print("Please check the paths and try again.")

if __name__ == "__main__":
    main()
