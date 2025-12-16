"""
Data Loader Module for Phase 1
Handles loading and preprocessing of FER2013 and RAVDESS datasets
"""
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import librosa


class FER2013Dataset(Dataset):
    """FER2013 Facial Expression Dataset Loader"""

    def __init__(self, image_dir, label_mapping=None, augment=False, transform=None):
        """
        Args:
            image_dir: Path to FER2013 images (e.g., 'data/raw/fer2013/train')
            label_mapping: Dict mapping emotion names to indices
            augment: Whether to apply data augmentation
            transform: Custom transform (if None, uses default)
        """
        self.image_dir = Path(image_dir)
        self.augment = augment

        # Default emotion mapping (7 emotions)
        self.label_mapping = label_mapping or {
            'angry': 0,
            'disgust': 1,
            'fear': 2,
            'happy': 3,
            'neutral': 4,
            'sad': 5,
            'surprise': 6
        }

        self.images = []
        self.labels = []
        self._load_images()

        # Transforms for preprocessing
        if transform is None:
            if augment:
                # Training augmentations
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
            else:
                # Validation/test transforms (no augmentation)
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

    def _load_images(self):
        """Load image paths and labels from directory structure"""
        if not self.image_dir.exists():
            raise ValueError(f"Directory not found: {self.image_dir}")

        for emotion_dir in self.image_dir.iterdir():
            if emotion_dir.is_dir():
                emotion = emotion_dir.name.lower()
                label = self.label_mapping.get(emotion)

                if label is not None:
                    # Find all image files
                    for ext in ['*.jpg', '*.jpeg', '*.png']:
                        for img_file in emotion_dir.glob(ext):
                            self.images.append(str(img_file))
                            self.labels.append(label)

        if len(self.images) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

        print(f"Loaded {len(self.images)} images from {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Get single sample"""
        img_path = self.images[idx]
        label = self.labels[idx]

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        image = self.transform(image)

        return {
            'image': image,
            'label': label,
            'path': img_path
        }

    def get_emotion_name(self, label_idx):
        """Convert label index to emotion name"""
        inv_mapping = {v: k for k, v in self.label_mapping.items()}
        return inv_mapping.get(label_idx, 'unknown')


class RAVDESSDataset(Dataset):
    """RAVDESS Speech Emotion Dataset Loader"""

    def __init__(self, audio_dir, sr=16000, max_length=5.0):
        """
        Args:
            audio_dir: Path to RAVDESS audio files
            sr: Sample rate for audio loading
            max_length: Maximum audio length in seconds
        """
        self.audio_dir = Path(audio_dir)
        self.sr = sr
        self.max_length = max_length
        self.max_samples = int(sr * max_length)

        self.audio_files = []
        self.emotions = []
        self.emotion_labels = []

        # RAVDESS emotion mapping (from filename code)
        self.emotion_map = {
            1: 'neutral',
            2: 'calm',
            3: 'happy',
            4: 'sad',
            5: 'angry',
            6: 'fearful',
            7: 'disgust',
            8: 'surprised'
        }

        # Map to 7 emotions (merge calm with neutral)
        self.emotion_to_label = {
            'neutral': 0,
            'calm': 0,  # Merge with neutral
            'happy': 1,
            'sad': 2,
            'angry': 3,
            'fearful': 4,
            'disgust': 5,
            'surprised': 6
        }

        self._load_audio_files()

    def _load_audio_files(self):
        """Parse RAVDESS filenames and load metadata"""
        if not self.audio_dir.exists():
            raise ValueError(f"Directory not found: {self.audio_dir}")

        # Find all .wav files recursively
        wav_files = list(self.audio_dir.glob('**/*.wav'))

        for audio_file in wav_files:
            # RAVDESS filename format: 03-01-05-01-01-01-01.wav
            # Position 3 (0-indexed position 2) = emotion code
            parts = audio_file.stem.split('-')

            if len(parts) >= 3:
                try:
                    emotion_id = int(parts[2])
                    emotion = self.emotion_map.get(emotion_id)

                    if emotion:
                        label = self.emotion_to_label[emotion]
                        self.audio_files.append(str(audio_file))
                        self.emotions.append(emotion)
                        self.emotion_labels.append(label)
                except ValueError:
                    print(f"Warning: Could not parse {audio_file.name}")

        if len(self.audio_files) == 0:
            raise ValueError(f"No valid audio files found in {self.audio_dir}")

        print(f"Loaded {len(self.audio_files)} audio files from {self.audio_dir}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        """Get single audio sample"""
        audio_path = self.audio_files[idx]
        emotion = self.emotions[idx]
        label = self.emotion_labels[idx]

        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr)

        # Pad or truncate to fixed length
        if len(y) > self.max_samples:
            y = y[:self.max_samples]
        else:
            y = np.pad(y, (0, self.max_samples - len(y)), mode='constant')

        # Extract features
        # MFCCs (13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return {
            'audio_path': audio_path,
            'waveform': torch.FloatTensor(y),
            'mfcc': torch.FloatTensor(mfcc),
            'mel_spectrogram': torch.FloatTensor(mel_spec_db),
            'emotion': emotion,
            'label': label,
            'sr': sr
        }

    def get_emotion_name(self, label_idx):
        """Convert label index to emotion name"""
        label_to_emotion = {
            0: 'neutral/calm',
            1: 'happy',
            2: 'sad',
            3: 'angry',
            4: 'fearful',
            5: 'disgust',
            6: 'surprised'
        }
        return label_to_emotion.get(label_idx, 'unknown')


def create_dataloaders(batch_size=32, num_workers=4):
    """
    Create train/val/test dataloaders for both datasets

    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading

    Returns:
        Dictionary containing all dataloaders
    """
    print("Creating dataloaders...")

    # FER2013 dataloaders
    try:
        fer_train = FER2013Dataset('data/raw/fer2013/train', augment=True)
        fer_test = FER2013Dataset('data/raw/fer2013/test', augment=False)

        fer_train_loader = DataLoader(
            fer_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        fer_test_loader = DataLoader(
            fer_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        print(f"✓ FER2013 loaders created:")
        print(f"  Train: {len(fer_train)} samples")
        print(f"  Test: {len(fer_test)} samples")

    except Exception as e:
        print(f"⚠ Warning: Could not create FER2013 loaders: {e}")
        fer_train_loader = fer_test_loader = None

    # RAVDESS dataloader
    try:
        ravdess = RAVDESSDataset('data/raw/ravdess')
        ravdess_loader = DataLoader(
            ravdess,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        print(f"✓ RAVDESS loader created: {len(ravdess)} samples")

    except Exception as e:
        print(f"⚠ Warning: Could not create RAVDESS loader: {e}")
        ravdess_loader = None

    return {
        'fer_train': fer_train_loader,
        'fer_test': fer_test_loader,
        'ravdess': ravdess_loader
    }


if __name__ == "__main__":
    # Test data loaders
    print("Testing data loaders...\n")

    loaders = create_dataloaders(batch_size=16, num_workers=0)

    # Test FER2013
    if loaders['fer_train'] is not None:
        print("\nTesting FER2013 train loader:")
        batch = next(iter(loaders['fer_train']))
        print(f"  Batch image shape: {batch['image'].shape}")
        print(f"  Batch labels shape: {batch['label'].shape}")
        print(f"  Sample labels: {batch['label'][:5]}")

    # Test RAVDESS
    if loaders['ravdess'] is not None:
        print("\nTesting RAVDESS loader:")
        batch = next(iter(loaders['ravdess']))
        print(f"  Waveform shape: {batch['waveform'].shape}")
        print(f"  MFCC shape: {batch['mfcc'].shape}")
        print(f"  Mel spectrogram shape: {batch['mel_spectrogram'].shape}")
        print(f"  Sample labels: {batch['label'][:5]}")

    print("\n✓ Data loaders working correctly!")
