"""
Phase 1 Quick Test Script
Tests data loaders before running full EDA notebook
"""
import sys
from pathlib import Path


def test_phase1():
    """Run all Phase 1 tests"""
    print("=" * 70)
    print("PHASE 1: DATA LOADER VALIDATION")
    print("=" * 70)

    # Test 1: Check directories
    print("\n1️⃣ Checking dataset directories...")

    fer_train = Path("data/raw/fer2013/train")
    fer_test = Path("data/raw/fer2013/test")
    ravdess = Path("data/raw/ravdess")

    checks = {
        "FER2013 train": fer_train.exists(),
        "FER2013 test": fer_test.exists(),
        "RAVDESS": ravdess.exists()
    }

    all_exist = True
    for name, exists in checks.items():
        status = "✅" if exists else "❌"
        print(f"   {status} {name}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\n❌ Dataset directories missing!")
        return False

    # Test 2: Import module
    print("\n2️⃣ Testing imports...")
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "services"))
        from data_loader import FER2013Dataset, RAVDESSDataset, create_dataloaders
        print("   ✅ Data loader module imported")
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False

    # Test 3: Load FER2013
    print("\n3️⃣ Testing FER2013 loading...")
    try:
        fer_train_ds = FER2013Dataset('data/raw/fer2013/train', augment=False)
        print(f"   ✅ Loaded {len(fer_train_ds)} training images")

        sample = fer_train_ds[0]
        print(f"   ✅ Sample shape: {sample['image'].shape}")
        print(f"   ✅ Sample label: {sample['label']} ({fer_train_ds.get_emotion_name(sample['label'])})")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    # Test 4: Load RAVDESS
    print("\n4️⃣ Testing RAVDESS loading...")
    try:
        ravdess_ds = RAVDESSDataset('data/raw/ravdess')
        print(f"   ✅ Loaded {len(ravdess_ds)} audio files")

        sample = ravdess_ds[0]
        print(f"   ✅ Waveform shape: {sample['waveform'].shape}")
        print(f"   ✅ MFCC shape: {sample['mfcc'].shape}")
        print(f"   ✅ Sample emotion: {sample['emotion']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    # Test 5: DataLoaders
    print("\n5️⃣ Testing DataLoaders...")
    try:
        loaders = create_dataloaders(batch_size=4, num_workers=0)

        if loaders['fer_train']:
            batch = next(iter(loaders['fer_train']))
            print(f"   ✅ FER2013 batch shape: {batch['image'].shape}")

        if loaders['ravdess']:
            batch = next(iter(loaders['ravdess']))
            print(f"   ✅ RAVDESS batch shape: {batch['waveform'].shape}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    # Summary
    print("\n" + "=" * 70)
    print("✅ ALL PHASE 1 TESTS PASSED!")
    print("=" * 70)
    print("\n🎯 Next Step: Run Jupyter notebook for EDA")
    print("   Command: jupyter notebook")
    print("   Then open: notebooks/01_data_exploration.ipynb")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = test_phase1()
    sys.exit(0 if success else 1)
