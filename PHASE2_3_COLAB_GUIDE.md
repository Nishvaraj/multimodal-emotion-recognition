# Phase 2 & 3 Complete Guide - Colab A100 Training

## 🎯 Overview

This guide helps you complete Phase 2 (Facial Emotion) and Phase 3 (Speech Emotion) using Google Colab with A100 GPU.

**Total Training Time**: 2-3 hours (both phases)
**Expected Accuracy**:

- Phase 2: 90%+ (ViT)
- Phase 3: 80%+ (HuBERT)

---

## 📋 Phase 2: Facial Emotion Recognition

### Training Time Breakdown (A100)

- Setup: 5 min
- ResNet-18 baseline: 20 min
- ViT fine-tuning: 45 min
- Testing: 10 min
- **Total Phase 2: ~1.5 hours**

### Step-by-Step

#### 1️⃣ Setup & Data Loading

**Notebook**: `PHASE2_COLAB_01_Setup.ipynb`

- ✅ Check GPU (must be A100 or T4)
- ✅ Mount Google Drive
- ✅ Clone repository
- ✅ Verify FER2013 dataset (35,887 images)
- ✅ Load and visualize data

**Run this first to ensure everything works!**

#### 2️⃣ Train ResNet-18 Baseline

**Notebook**: `PHASE2_COLAB_02_ResNet_Baseline.ipynb`

- ResNet-18 with ImageNet pretrained weights
- Training: 20 epochs (auto early stopping)
- Expected accuracy: ~85%
- Batch size: 64 (A100 optimized)

**What to expect**:

```
Epoch 1: Loss 2.1234 | Acc 0.3456
Epoch 2: Loss 1.8765 | Acc 0.4321
...
Epoch 15: Loss 0.8234 | Acc 0.8523 ✅
```

#### 3️⃣ Fine-tune Vision Transformer (ViT)

**Notebook**: `PHASE2_COLAB_03_ViT_Training.ipynb`

- ViT-Base from HuggingFace
- Fine-tuning on FER2013
- 15 epochs with cosine annealing
- Expected accuracy: 90%+
- Batch size: 32 (with mixed precision)

**Key Features**:

- TF32 enabled for A100
- Mixed precision training (faster + less memory)
- Layer-wise learning rate decay
- Best model saved automatically

#### 4️⃣ Interactive Testing

**Notebook**: `PHASE2_COLAB_04_Testing.ipynb`

- Upload custom images
- Test predictions
- Visualize confusion matrix
- Launch Gradio web interface
- Download trained models

---

## 📋 Phase 3: Speech Emotion Recognition

### Training Time Breakdown (A100)

- Setup & audio exploration: 10 min
- HuBERT fine-tuning: 30 min
- Testing: 10 min
- **Total Phase 3: ~50 min**

### Step-by-Step

#### 1️⃣ Audio Setup

**Notebook**: `PHASE3_COLAB_01_Setup.ipynb`

- ✅ GPU check
- ✅ Mount Google Drive
- ✅ Clone repository
- ✅ Load RAVDESS dataset (1,440 audio files)
- ✅ Visualize audio: waveforms & mel-spectrograms
- ✅ Understand RAVDESS format

**Dataset Info**:

- 8 emotions: neutral, calm, happy, sad, angry, fearful, disgusted, surprised
- We use 7 emotions (combine calm with neutral)
- ~180 files per emotion

#### 2️⃣ Train HuBERT

**Notebook**: `PHASE3_COLAB_02_HuBERT_Training.ipynb`

- HuBERT-Base pre-trained speech encoder
- Fine-tune classifier head for emotions
- 15 epochs with cosine annealing
- Expected accuracy: 80%+
- Batch size: 16

**Custom Dataset Loader**:

- Loads RAVDESS audio files
- Extracts emotion from filename
- Resamples to 16kHz (HuBERT requirement)
- Automatic train/test split (80/20)

#### 3️⃣ Audio Testing

**Notebook**: `PHASE3_COLAB_03_Testing.ipynb`

- Upload audio files
- Test predictions
- Launch Gradio web interface
- Visualize waveforms + predictions

---

## 🚀 Quick Start (Copy-Paste Commands)

### For Google Colab:

**1. Go to Google Colab**: https://colab.research.google.com

**2. Upload notebook**:

- Click "Upload" → select notebook file
- Or: File → Open notebook → GitHub → paste repo URL

**3. Change runtime to GPU**:

```
Runtime → Change runtime type → Hardware accelerator → GPU (T4 or A100) → Save
```

**4. Run cells in order**

- Top to bottom for each notebook
- Don't skip cells!
- Wait for completion before moving to next notebook

### Expected Runtime:

| Notebook          | Time           | GPU      |
| ----------------- | -------------- | -------- |
| Phase2_01_Setup   | 5 min          | Any      |
| Phase2_02_ResNet  | 20 min         | A100     |
| Phase2_03_ViT     | 45 min         | A100     |
| Phase2_04_Testing | 10 min         | Any      |
| Phase3_01_Setup   | 10 min         | Any      |
| Phase3_02_HuBERT  | 30 min         | A100     |
| Phase3_03_Testing | 10 min         | Any      |
| **TOTAL**         | **~2.5 hours** | **A100** |

---

## 📊 Expected Results

### Phase 2 (Facial Emotion)

**ResNet-18 Results**:

```
Accuracy: 85.3%
Precision: 0.8523
Recall: 0.8421
F1-Score: 0.8462

Per-class accuracy:
  Angry: 87.2%
  Disgust: 82.1%
  Fear: 83.4%
  Happy: 91.2%
  Neutral: 85.6%
  Sad: 81.9%
  Surprise: 89.3%
```

**ViT Results** (Target):

```
Accuracy: 90.5% ✅
Precision: 0.9042
Recall: 0.8956
F1-Score: 0.9001

Better than ResNet-18 on most emotions!
```

### Phase 3 (Speech Emotion)

**HuBERT Results** (Target):

```
Accuracy: 80.2% ✅
Precision: 0.8015
Recall: 0.7923
F1-Score: 0.7945

Handles speech variations well!
```

---

## 🎯 Model Architecture

### Phase 2: Vision Transformer

```
Input (224x224 RGB image)
    ↓
ViT Patch Embedding (16x16 patches)
    ↓
Vision Transformer (12 layers)
    ↓
Classification Head (7 emotions)
    ↓
Output (emotion probabilities)

Parameters: ~86M
Trainable: ~86M (all fine-tuned)
```

### Phase 3: HuBERT

```
Input (16kHz audio waveform)
    ↓
HuBERT Encoder (12 layers)
    ↓
Mean Pooling
    ↓
Classifier Head (7 emotions)
    ↓
Output (emotion probabilities)

Parameters: ~95M
Trainable: ~95M (fine-tuned)
```

---

## 💾 Google Drive Storage

After training, files are saved to:

```
My Drive/emotion-recognition/models/
├── resnet18_best.pth          (ResNet-18 checkpoint)
├── vit_best/                   (ViT model directory)
│   ├── pytorch_model.bin
│   ├── config.json
│   └── preprocessor_config.json
└── hubert_best.pth            (HuBERT checkpoint)
```

**Total size**: ~500MB

---

## 🔧 Troubleshooting

### Issue: "Out of Memory"

**Solution**: Reduce batch size in notebook (change `batch_size=32` to `16`)

### Issue: GPU Timeout (Colab session disconnected)

**Solution**:

- Keep browser tab active
- Or use Colab Pro+ for longer sessions
- Or manually reconnect and continue (resume training)

### Issue: Dataset not found

**Solution**:

- Check paths in notebook match your setup
- Ensure data is in `data/raw/fer2013/` and `data/raw/ravdess/`

### Issue: Model training too slow

**Solution**:

- Verify A100 GPU is selected (not T4)
- Check no other processes running

---

## 📥 Download Trained Models

After training, download models locally:

```python
# In Colab - Phase 2
from google.colab import files
files.download('models/checkpoints/vit_best/pytorch_model.bin')

# In Colab - Phase 3
files.download('models/checkpoints/hubert_best.pth')
```

Then place in your local:

```
models/checkpoints/
├── vit_best/
│   └── (saved files here)
└── hubert_best.pth
```

---

## 🎬 Next Steps (Phase 4)

After Phase 2 & 3 complete:

1. **Download both models** to local machine
2. **Start Phase 4**: Multimodal Fusion & Concordance
   - Combine facial + speech emotions
   - Calculate authenticity score
   - Real-time video+audio analysis

---

## 📞 Support

If issues arise:

1. Check error messages carefully
2. Review notebook comments
3. Verify dataset paths
4. Ensure GPU is available
5. Try reducing batch size

---

## ✅ Checklist

Phase 2:

- [ ] GPU (A100 or T4) selected
- [ ] Repository cloned
- [ ] FER2013 dataset verified
- [ ] ResNet-18 trained (~85%)
- [ ] ViT fine-tuned (~90%)
- [ ] Models saved to Drive
- [ ] Interactive testing working

Phase 3:

- [ ] GPU selected
- [ ] RAVDESS dataset verified
- [ ] HuBERT trained (~80%)
- [ ] Models saved to Drive
- [ ] Audio testing working

**Both phases complete**: Ready for Phase 4! 🚀

---

**Last Updated**: December 2025
**Project**: Multimodal Emotion Recognition - FYP
