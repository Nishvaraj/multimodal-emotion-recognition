# Phase 2 & 3 Quick Reference Card

## 🚀 30-Second Start

1. Go to: **colab.research.google.com**
2. Upload first notebook: `PHASE2_COLAB_01_Setup.ipynb`
3. Runtime → GPU → Run all cells
4. Follow sequence below ↓

---

## 📋 Execution Sequence (2-3 hours total)

```
🟢 PHASE 2 - FACIAL EMOTION (1.5 hours)

1. PHASE2_COLAB_01_Setup.ipynb (5 min)
   ✓ GPU check
   ✓ Mount Drive
   ✓ Load FER2013
   ✓ Visualize data

2. PHASE2_COLAB_02_ResNet_Baseline.ipynb (20 min)
   ✓ Train ResNet-18
   ✓ Target: 85% accuracy
   ✓ Save checkpoint

3. PHASE2_COLAB_03_ViT_Training.ipynb (45 min)
   ✓ Fine-tune ViT
   ✓ Target: 90%+ accuracy ✨
   ✓ Save to Drive

4. PHASE2_COLAB_04_Testing.ipynb (10 min)
   ✓ Upload test image
   ✓ Launch Gradio demo
   ✓ Download models


🟢 PHASE 3 - SPEECH EMOTION (1 hour)

5. PHASE3_COLAB_01_Setup.ipynb (10 min)
   ✓ Load RAVDESS
   ✓ Explore audio
   ✓ Visualize spectrograms

6. PHASE3_COLAB_02_HuBERT_Training.ipynb (30 min)
   ✓ Train HuBERT
   ✓ Target: 80%+ accuracy ✨
   ✓ Save to Drive

7. PHASE3_COLAB_03_Testing.ipynb (10 min)
   ✓ Upload audio file
   ✓ Launch Gradio demo
   ✓ Download models

🎉 BOTH PHASES COMPLETE!
```

---

## 🎯 Key Targets

| Phase | Model         | Target Accuracy | Time   |
| ----- | ------------- | --------------- | ------ |
| 2     | ResNet-18     | 85%             | 20 min |
| 2     | **ViT** ⭐    | **90%+**        | 45 min |
| 3     | **HuBERT** ⭐ | **80%+**        | 30 min |

---

## ⚡ Critical Setup Steps

```python
# 1. GPU SELECTION (MUST DO FIRST!)
Runtime → Change runtime type → GPU → Save

# 2. Each notebook will:
   - Mount Google Drive
   - Clone repository
   - Load datasets
   - Train model
   - Save to Drive

# 3. Just run each cell in order!
   - No manual changes needed
   - Models auto-save
   - Colab handles everything
```

---

## 📊 Expected Console Output

### Phase 2 - ResNet Training

```
Epoch 1/20: Loss 2.1234 | Train Acc 0.3456 | Val Acc 0.4231
Epoch 2/20: Loss 1.8765 | Train Acc 0.5123 | Val Acc 0.5432
...
Epoch 15/20: Loss 0.6234 | Train Acc 0.8523 | Val Acc 0.8521
✅ Training completed!
   Best validation accuracy: 0.8521 (85.21%)
```

### Phase 2 - ViT Training

```
Epoch 1/15: Loss 1.8234 | Train Acc 0.5234 | Val Acc 0.6123
Epoch 2/15: Loss 1.2345 | Train Acc 0.7123 | Val Acc 0.7321
...
Epoch 12/15: Loss 0.3234 | Train Acc 0.9134 | Val Acc 0.9012
✅ ViT Training completed!
   Best validation accuracy: 0.9012 (90.12%)
```

### Phase 3 - HuBERT Training

```
Epoch 1/15: Loss 1.5234 | Train Acc 0.6123 | Val Acc 0.6421
Epoch 2/15: Loss 1.1234 | Train Acc 0.7345 | Val Acc 0.7531
...
Epoch 10/15: Loss 0.5123 | Train Acc 0.8234 | Val Acc 0.8021
✅ HuBERT Training completed!
   Best validation accuracy: 0.8021 (80.21%)
```

---

## 🎭 What Each Notebook Does

### Setup Notebooks (01)

- Check GPU available
- Mount Google Drive
- Clone git repository
- Load and verify datasets
- Visualize samples

### Training Notebooks (02 & 03)

- Load pre-trained model
- Create DataLoader
- Setup optimizer
- Training loop (epochs)
- Early stopping
- Save best checkpoint

### Testing Notebooks (04)

- Load trained model
- Upload custom image/audio
- Get prediction + confidence
- Visualize results
- Launch Gradio demo

---

## 💾 Google Drive Structure (After Training)

```
My Drive/
└── emotion-recognition/
    └── models/
        ├── resnet18_best.pth       (90 MB)
        ├── vit_best/               (350 MB)
        │   ├── pytorch_model.bin
        │   ├── config.json
        │   └── preprocessor_config.json
        └── hubert_best.pth         (160 MB)

Total: ~600 MB
```

---

## 🔍 Troubleshooting in 30 Seconds

| Problem             | Solution                    |
| ------------------- | --------------------------- |
| "No GPU"            | Runtime → GPU → Save        |
| "Out of Memory"     | Reduce batch_size to 16     |
| "Dataset not found" | Check paths in notebook     |
| "Colab timeout"     | Use Colab Pro+ or reconnect |
| "Slow training"     | Verify A100 GPU selected    |

---

## 📥 After Training Complete

```python
# Download models from Colab
from google.colab import files

# Phase 2
files.download('models/checkpoints/vit_best/pytorch_model.bin')

# Phase 3
files.download('models/checkpoints/hubert_best.pth')

# Then use locally for Phase 4!
```

---

## ✅ Minimal Checklist

- [ ] Use A100 GPU
- [ ] Run notebooks 1-7 in order
- [ ] ResNet: ~85% ✅
- [ ] ViT: ~90%+ ✅
- [ ] HuBERT: ~80%+ ✅
- [ ] Download models
- [ ] Ready for Phase 4!

---

## 🎊 That's It!

**Phase 2 & 3 complete in 2-3 hours!**

All 7 notebooks are ready to use.
Just upload to Colab and run!

Good luck! 🚀

---

**Files Ready**:

- ✅ 7 Colab notebooks
- ✅ Comprehensive guide
- ✅ This quick reference
- ✅ Implementation summary

**Status**: 🟢 Ready to train!
