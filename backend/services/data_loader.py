"""Dataset loaders used by the training and experimentation workflows."""

# --- Imports ---
import os
import numpy as np
import cv2
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# --- Facial Dataset ---
class FER2013Dataset(Dataset):
    """FER2013 facial emotion dataset loader."""

    def __init__(self, root_dir: str, split: str = "train", transform=None):
        """
        Initialize FER2013 dataset.

        Args:
            root_dir: Root directory containing 'train' and 'test' folders
            split: 'train' or 'test'
            transform: Torchvision transforms to apply
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.emotion2idx = {e: i for i, e in enumerate(self.emotions)}

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        """Load all image paths and labels."""
        split_dir = os.path.join(self.root_dir, self.split)

        for emotion in self.emotions:
            emotion_dir = os.path.join(split_dir, emotion)
            if not os.path.exists(emotion_dir):
                continue

            for img_file in os.listdir(emotion_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(emotion_dir, img_file)
                    self.samples.append((img_path, self.emotion2idx[emotion]))

            # Stable ordering keeps experiments reproducible across runs.
            self.samples.sort(key=lambda item: item[0])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return torch.zeros(3, 224, 224), torch.tensor(label, dtype=torch.long)

        # Convert to RGB (3 channels)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform:
            image = self.transform(image)
        else:
            # Default transform
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image, torch.tensor(label, dtype=torch.long)


# --- Audio Dataset ---
class RAVDESSDataset(Dataset):
    """RAVDESS audio emotion dataset loader."""

    def __init__(self, root_dir: str, n_mfcc: int = 13, target_sr: int = 22050):
        """
        Initialize RAVDESS dataset.

        Args:
            root_dir: Root directory containing audio files
            n_mfcc: Number of MFCCs to extract
            target_sr: Target sampling rate
        """
        self.root_dir = root_dir
        self.n_mfcc = n_mfcc
        self.target_sr = target_sr
        self.emotion_map = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fear',
            '07': 'disgust',
            '08': 'surprise'
        }
        self.emotion2idx = {v: i for i, v in enumerate(set(self.emotion_map.values()))}

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        """Load all audio file paths and labels."""
        for file in os.listdir(self.root_dir):
            if file.endswith('.wav'):
                emotion_code = file.split('-')[2]
                if emotion_code in self.emotion_map:
                    emotion = self.emotion_map[emotion_code]
                    audio_path = os.path.join(self.root_dir, file)
                    self.samples.append((audio_path, self.emotion2idx[emotion]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]

        try:
            y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)

            # Normalize MFCC
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

            # Pad or truncate to fixed size (100 time steps)
            if mfcc.shape[1] < 100:
                mfcc = np.pad(mfcc, ((0, 0), (0, 100 - mfcc.shape[1])), mode='constant')
            else:
                mfcc = mfcc[:, :100]

            return torch.from_numpy(mfcc).float(), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(self.n_mfcc, 100), torch.tensor(label, dtype=torch.long)


# --- Dataloader Factory ---
def create_dataloaders(
    fer2013_dir: str = None,
    ravdess_dir: str = None,
    batch_size: int = 32,
    num_workers: int = 0,
    img_size: int = 224
) -> dict:
    """
    Create dataloaders for FER2013 and RAVDESS datasets.

    Args:
        fer2013_dir: Path to FER2013 dataset root
        ravdess_dir: Path to RAVDESS dataset root
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        img_size: Image size for FER2013

    Returns:
        Dictionary with dataloaders for each dataset
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    dataloaders = {}

    if fer2013_dir and os.path.exists(fer2013_dir):
        # FER2013 is split into train/test folders and exposed as two separate loaders.
        train_dataset = FER2013Dataset(fer2013_dir, split='train', transform=transform)
        test_dataset = FER2013Dataset(fer2013_dir, split='test', transform=transform)

        dataloaders['fer2013_train'] = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        dataloaders['fer2013_test'] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    if ravdess_dir and os.path.exists(ravdess_dir):
        # RAVDESS loader provides one shuffled stream for speech model training.
        audio_dataset = RAVDESSDataset(ravdess_dir)
        dataloaders['ravdess'] = DataLoader(
            audio_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

    return dataloaders
