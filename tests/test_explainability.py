"""
Unit tests for facial/audio explainability utilities in the multimodal inference stack.

Given lightweight fake tensors, processors, and model wrappers,
when Grad-CAM/audio saliency helpers execute under success and failure paths,
then outputs remain deterministic and local-test safe without remote dependencies.

Mocking strategy note:
- Hugging Face-scale model objects are replaced with minimal fake classes and monkeypatches.
- Supabase is not used in explainability generation and is intentionally out of scope here.
"""

# --- Imports ---
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from backend.services import explainability


# --- Test Doubles ---
class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeModel:
    def __init__(self):
        layer = SimpleNamespace(layernorm_after=object())
        self.vit = SimpleNamespace(encoder=SimpleNamespace(layer=[layer, layer, layer]))

    def __call__(self, pixel_values):
        logits = torch.zeros((pixel_values.shape[0], 8), dtype=torch.float32)
        return SimpleNamespace(logits=logits)


class _AudioModel:
    def zero_grad(self):
        return None

    def __call__(self, input_values):
        # Make logits depend on input so autograd can propagate saliency.
        logits = input_values.mean(dim=1, keepdim=True).repeat(1, 8)
        return SimpleNamespace(logits=logits)


# --- Given-When-Then: Grad-CAM Paths ---
def test_vit_logits_wrapper_forward_returns_logits():
    wrapped = explainability.ViTLogitsWrapper(_FakeModel())
    out = wrapped(torch.zeros((1, 3, 32, 32), dtype=torch.float32))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 8)


def test_generate_grad_cam_success(monkeypatch):
    image = Image.new("RGB", (64, 64), color=(120, 120, 120))

    processor = lambda img, return_tensors: _FakeBatch(
        {"pixel_values": torch.zeros((1, 3, 32, 32), dtype=torch.float32)}
    )

    class _GradCamOK:
        def __init__(self, **kwargs):
            pass

        def __call__(self, input_tensor, targets):
            return np.array([np.ones((8, 8), dtype=np.float32)])

    monkeypatch.setattr(explainability, "GradCAM", _GradCamOK)

    orig_b64, blend_b64 = explainability.generate_grad_cam(
        image=image,
        model=_FakeModel(),
        processor=processor,
        emotion_idx=1,
        emotions_list=["a", "b", "c"],
        device=torch.device("cpu"),
    )

    assert isinstance(orig_b64, str) and len(orig_b64) > 0
    assert isinstance(blend_b64, str) and len(blend_b64) > 0


def test_generate_grad_cam_falls_back_to_eigen(monkeypatch):
    image = Image.new("RGB", (48, 48), color=(80, 80, 80))

    processor = lambda img, return_tensors: _FakeBatch(
        {"pixel_values": torch.zeros((1, 3, 32, 32), dtype=torch.float32)}
    )

    class _GradCamZero:
        def __init__(self, **kwargs):
            pass

        def __call__(self, input_tensor, targets):
            return np.array([np.zeros((8, 8), dtype=np.float32)])

    class _EigenCamOK:
        def __init__(self, **kwargs):
            pass

        def __call__(self, input_tensor):
            return np.array([np.ones((8, 8), dtype=np.float32)])

    monkeypatch.setattr(explainability, "GradCAM", _GradCamZero)
    monkeypatch.setattr(explainability, "EigenCAM", _EigenCamOK)

    orig_b64, blend_b64 = explainability.generate_grad_cam(
        image=image,
        model=_FakeModel(),
        processor=processor,
        emotion_idx=0,
        emotions_list=["x", "y"],
        device=torch.device("cpu"),
    )

    assert isinstance(orig_b64, str) and len(orig_b64) > 0
    assert isinstance(blend_b64, str) and len(blend_b64) > 0


def test_generate_grad_cam_returns_none_on_total_failure(monkeypatch):
    image = Image.new("RGB", (32, 32), color=(30, 30, 30))

    processor = lambda img, return_tensors: _FakeBatch(
        {"pixel_values": torch.zeros((1, 3, 32, 32), dtype=torch.float32)}
    )

    class _GradCamFail:
        def __init__(self, **kwargs):
            pass

        def __call__(self, input_tensor, targets):
            raise RuntimeError("gradcam failed")

    class _EigenCamFail:
        def __init__(self, **kwargs):
            pass

        def __call__(self, input_tensor):
            raise RuntimeError("eigencam failed")

    monkeypatch.setattr(explainability, "GradCAM", _GradCamFail)
    monkeypatch.setattr(explainability, "EigenCAM", _EigenCamFail)

    orig_b64, blend_b64 = explainability.generate_grad_cam(
        image=image,
        model=_FakeModel(),
        processor=processor,
        emotion_idx=2,
        emotions_list=["a", "b", "c"],
        device=torch.device("cpu"),
    )

    assert orig_b64 is None
    assert blend_b64 is None


# --- Given-When-Then: Audio Saliency Paths ---
def test_generate_audio_saliency_success(monkeypatch):
    audio = np.random.rand(1600).astype(np.float32)
    processor = lambda audio, sampling_rate, return_tensors, padding: {
        "input_values": torch.ones((1, 160), dtype=torch.float32)
    }

    def _fake_specshow(S_db, sr, x_axis, y_axis, ax):
        return ax.imshow(S_db, aspect="auto", origin="lower")

    monkeypatch.setattr(
        explainability.librosa.feature,
        "melspectrogram",
        lambda y, sr, n_mels: np.ones((32, 20), dtype=np.float32),
    )
    monkeypatch.setattr(
        explainability.librosa,
        "power_to_db",
        lambda S, ref: S,
    )
    monkeypatch.setattr(explainability.librosa.display, "specshow", _fake_specshow)

    spec_b64, sal_b64 = explainability.generate_audio_saliency(
        audio=audio,
        model=_AudioModel(),
        processor=processor,
        emotion_idx=1,
        emotions_list=["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"],
        device=torch.device("cpu"),
        sr=16000,
    )

    assert isinstance(spec_b64, str) and len(spec_b64) > 0
    assert isinstance(sal_b64, str) and len(sal_b64) > 0


def test_generate_audio_saliency_returns_none_for_empty_audio():
    processor = lambda audio, sampling_rate, return_tensors, padding: {
        "input_values": torch.ones((1, 160), dtype=torch.float32)
    }

    spec_b64, sal_b64 = explainability.generate_audio_saliency(
        audio=np.array([], dtype=np.float32),
        model=_AudioModel(),
        processor=processor,
        emotion_idx=0,
        emotions_list=["x"],
        device=torch.device("cpu"),
        sr=16000,
    )

    assert spec_b64 is None
    assert sal_b64 is None


# --- Given-When-Then: Combined Visualization Paths ---
def test_create_combined_visualization_success():
    combined = explainability.create_combined_visualization(
        grad_cam_base64="AAA",
        saliency_base64="BBB",
        facial_emotion="happy",
        speech_emotion="happy",
        concordance="MATCH",
    )
    assert isinstance(combined, str)
    assert len(combined) > 0


def test_create_combined_visualization_handles_exceptions(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("encode failed")

    monkeypatch.setattr(explainability.base64, "b64encode", _boom)

    combined = explainability.create_combined_visualization(
        grad_cam_base64="AAA",
        saliency_base64="BBB",
        facial_emotion="happy",
        speech_emotion="sad",
        concordance="MISMATCH",
    )
    assert combined is None
