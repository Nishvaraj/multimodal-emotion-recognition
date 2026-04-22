"""
Dataset loader tests for FER2013 and RAVDESS preprocessing behavior.

Given synthetic files and monkeypatched librosa calls,
when dataset classes and dataloader factories are exercised,
then padding, truncation, fallback tensors, and split construction remain deterministic.

Mocking strategy note:
- External model hubs (for example Hugging Face checkpoints) are not touched in this suite.
- Supabase is not used by dataset loaders and is therefore intentionally excluded.
"""

# --- Imports ---
import cv2
import numpy as np
import torch

from backend.services import data_loader
from backend.services.data_loader import FER2013Dataset, create_dataloaders


# --- Test Helpers ---
def _write_gray_image(path, value=128):
    image = np.full((48, 48), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


# --- Given-When-Then: FER2013 Loading Paths ---
def test_fer2013_dataset_loads_images_and_returns_tensors(tmp_path):
    train_happy = tmp_path / "train" / "happy"
    train_angry = tmp_path / "train" / "angry"
    train_happy.mkdir(parents=True)
    train_angry.mkdir(parents=True)

    _write_gray_image(train_happy / "img2.png", value=180)
    _write_gray_image(train_angry / "img1.jpg", value=80)

    dataset = FER2013Dataset(str(tmp_path), split="train")

    assert len(dataset) == 2

    image_tensor, label_tensor = dataset[0]
    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.shape == (3, 48, 48)
    assert isinstance(label_tensor, torch.Tensor)
    assert label_tensor.dtype == torch.long


def test_fer2013_dataset_handles_unreadable_images(tmp_path):
    train_happy = tmp_path / "train" / "happy"
    train_happy.mkdir(parents=True)

    unreadable = train_happy / "broken.jpg"
    unreadable.write_bytes(b"not-an-image")

    dataset = FER2013Dataset(str(tmp_path), split="train")
    image_tensor, label_tensor = dataset[0]

    assert image_tensor.shape == (3, 224, 224)
    assert label_tensor.dtype == torch.long


def test_create_dataloaders_builds_fer2013_train_and_test(tmp_path):
    train_happy = tmp_path / "train" / "happy"
    test_happy = tmp_path / "test" / "happy"
    train_happy.mkdir(parents=True)
    test_happy.mkdir(parents=True)

    _write_gray_image(train_happy / "train_img.png", value=120)
    _write_gray_image(test_happy / "test_img.png", value=140)

    dataloaders = create_dataloaders(
        fer2013_dir=str(tmp_path),
        ravdess_dir=None,
        batch_size=1,
        num_workers=0,
        img_size=64,
    )

    assert "fer2013_train" in dataloaders
    assert "fer2013_test" in dataloaders

    train_batch = next(iter(dataloaders["fer2013_train"]))
    images, labels = train_batch
    assert images.shape[-2:] == (64, 64)
    assert labels.dtype == torch.long


# --- Given-When-Then: RAVDESS Loading Paths ---
def test_ravdess_dataset_loads_supported_files_only(tmp_path):
    # File naming matches the RAVDESS pattern where the third token is emotion code.
    valid_1 = tmp_path / "03-01-05-01-01-01-01.wav"
    valid_2 = tmp_path / "03-01-03-01-01-01-02.wav"
    invalid_code = tmp_path / "03-01-99-01-01-01-03.wav"
    ignored_ext = tmp_path / "03-01-05-01-01-01-04.mp3"

    valid_1.write_bytes(b"x")
    valid_2.write_bytes(b"x")
    invalid_code.write_bytes(b"x")
    ignored_ext.write_bytes(b"x")

    dataset = data_loader.RAVDESSDataset(str(tmp_path))

    assert len(dataset.samples) == 2


def test_ravdess_getitem_returns_tensor_with_padding(monkeypatch, tmp_path):
    wav = tmp_path / "03-01-05-01-01-01-01.wav"
    wav.write_bytes(b"x")
    dataset = data_loader.RAVDESSDataset(str(tmp_path), n_mfcc=13, target_sr=16000)

    def fake_load(path, sr, mono):
        assert path.endswith(".wav")
        assert sr == 16000
        assert mono is True
        return np.ones(1600, dtype=np.float32), sr

    def fake_mfcc(y, sr, n_mfcc):
        assert n_mfcc == 13
        # 20 time steps triggers the pad branch to 100.
        return np.ones((13, 20), dtype=np.float32)

    monkeypatch.setattr(data_loader.librosa, "load", fake_load)
    monkeypatch.setattr(data_loader.librosa.feature, "mfcc", fake_mfcc)

    features, label = dataset[0]
    assert isinstance(features, torch.Tensor)
    assert features.shape == (13, 100)
    assert label.dtype == torch.long


def test_ravdess_getitem_returns_tensor_with_truncation(monkeypatch, tmp_path):
    wav = tmp_path / "03-01-05-01-01-01-01.wav"
    wav.write_bytes(b"x")
    dataset = data_loader.RAVDESSDataset(str(tmp_path), n_mfcc=13, target_sr=22050)

    monkeypatch.setattr(
        data_loader.librosa,
        "load",
        lambda path, sr, mono: (np.ones(2400, dtype=np.float32), sr),
    )
    monkeypatch.setattr(
        data_loader.librosa.feature,
        "mfcc",
        lambda y, sr, n_mfcc: np.ones((13, 140), dtype=np.float32),
    )

    features, label = dataset[0]
    assert features.shape == (13, 100)
    assert label.dtype == torch.long


def test_ravdess_getitem_handles_exceptions(monkeypatch, tmp_path):
    wav = tmp_path / "03-01-05-01-01-01-01.wav"
    wav.write_bytes(b"x")
    dataset = data_loader.RAVDESSDataset(str(tmp_path), n_mfcc=20, target_sr=22050)

    def boom(*args, **kwargs):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(data_loader.librosa, "load", boom)

    features, label = dataset[0]
    assert torch.equal(features, torch.zeros(20, 100))
    assert label.dtype == torch.long


# --- Given-When-Then: Dataloader Factory Paths ---
def test_create_dataloaders_returns_empty_when_paths_missing(tmp_path):
    dataloaders = create_dataloaders(
        fer2013_dir=str(tmp_path / "missing-fer"),
        ravdess_dir=str(tmp_path / "missing-ravdess"),
    )
    assert dataloaders == {}


def test_create_dataloaders_includes_ravdess(monkeypatch, tmp_path):
    ravdess = tmp_path / "ravdess"
    ravdess.mkdir(parents=True)
    (ravdess / "03-01-05-01-01-01-01.wav").write_bytes(b"x")

    monkeypatch.setattr(
        data_loader.librosa,
        "load",
        lambda path, sr, mono: (np.ones(2400, dtype=np.float32), sr),
    )
    monkeypatch.setattr(
        data_loader.librosa.feature,
        "mfcc",
        lambda y, sr, n_mfcc: np.ones((n_mfcc, 101), dtype=np.float32),
    )

    dataloaders = create_dataloaders(
        fer2013_dir=None,
        ravdess_dir=str(ravdess),
        batch_size=1,
        num_workers=0,
    )

    assert "ravdess" in dataloaders
    batch_features, batch_labels = next(iter(dataloaders["ravdess"]))
    assert batch_features.shape[-2:] == (13, 100)
    assert batch_labels.dtype == torch.long
