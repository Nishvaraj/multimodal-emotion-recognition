# 🎯 TODO - Next Steps & Action Items

**Date**: January 28, 2026  
**Current Progress**: 65% Complete
**Deadline**: April 20, 2026 (13 weeks)  
**PPRS Target**: 85-90% compliance

---

## 📋 PRIORITY ROADMAP

### 🟢 COMPLETED (Jan 28, 2026)

#### ✅ Add `/api/predict/combined` Endpoint
- **Status**: COMPLETED ✅
- **Implementation**: Backend endpoint handles both image + audio
- **Impact**: Combined analysis fully functional

#### ✅ FFmpeg Installation & Video Processing
- **Status**: COMPLETED ✅ (Jan 28, 2026)
- **What was done**:
  - Installed FFmpeg via Homebrew
  - Implemented `extract_frame_and_audio_from_video()` function
  - Proper audio extraction: PCM 16-bit, mono, 16kHz resampling
  - Frame extraction from middle of video
  
- **Tested**: Video mode in Tab 3 working
- **Impact**: Video analysis feature fully functional

#### ✅ Gradio UI Enhancements
- **Status**: COMPLETED ✅ (Jan 28, 2026)
- **What was done**:
  - Added radio button for mode selection in Combined Analysis
  - Implemented `toggle_mode()` function for dynamic visibility
  - Dual-mode interface: Video OR Separate inputs
  - Audio waveform visualization in Speech tab
  
- **Tested**: All 4 tabs operational
- **Impact**: Production-ready Gradio demo

---

### 🔴 CRITICAL PRIORITY (This Week - Est. 2-3 days)

#### 1. Improve Facial Model Accuracy (BIGGEST BLOCKER)
- **Current**: 71.29% | **Target**: 90% | **Gap**: -18.71% ⚠️
- **Priority**: 🔴 CRITICAL - Biggest blocker for PPRS compliance
- **What to do**:
  - Analyze FER2013 dataset for quality issues
  - Implement stronger data augmentation:
    - Random rotation (±15°)
    - Color jittering
    - Gaussian blur
    - Cutout augmentation
  - Consider different architectures:
    - EfficientNet-B4 (lighter than ViT, better accuracy)
    - ResNet-152 (proven for facial emotion)
  - Retrain model with better hyperparameters
  
- **Why**: BIGGEST PPRS blocker (18.71% gap is critical)
- **Difficulty**: Hard (2-3 days work)
- **Impact**: HIGH - Without this, PPRS compliance won't meet target
- **Deadline**: Should start ASAP - this is critical path

#### 2. End-to-End Testing of All 4 Tabs
- **Status**: READY TO TEST ✅
- **What to test**:
  - [ ] Tab 1: Facial emotion (webcam + upload)
  - [ ] Tab 2: Speech emotion (audio record + upload) + waveform
  - [ ] Tab 3: Combined analysis video mode (NEW!)
  - [ ] Tab 3: Combined analysis separate mode
  - [ ] Tab 4: Model information display
  
- **Why**: Ensure everything works after recent changes
- **Difficulty**: Easy (1 hour manual testing)
- **Acceptance criteria**: All 4 tabs functional, no errors
- **Test location**: http://127.0.0.1:7860

---

### 🟠 HIGH PRIORITY (Week 1-2 - Est. 3-4 days)

#### 3. Implement Grad-CAM Visualization (Facial Explainability)
- **New file**: `backend/services/explainability.py`
- **What to do**:
  - Implement Grad-CAM to generate attention heatmaps
  - Show which facial regions contribute to emotion prediction
  - Return heatmap + original image overlaid
  - Integrate response into `/api/predict/facial` endpoint
  
- **Implementation steps**:
  1. Create `visualize_grad_cam()` function
  2. Hook into ViT model's last attention layer
  3. Generate heatmap (128x128 image)
  4. Overlay on original face region
  5. Return as base64-encoded PNG in response
  
- **Why**: PPRS requirement - must show explainability
- **Difficulty**: Medium (2 days)
- **Testing**: Verify heatmaps highlight relevant facial features
- **Expected output**: JSON with `"grad_cam": "data:image/png;base64,..."`

#### 4. Implement Audio Saliency Maps (Speech Explainability)
- **New file**: `backend/services/audio_explainability.py`
- **What to do**:
  - Generate frequency importance visualization
  - Show which frequencies are most important for emotion
  - Create spectrogram with highlighted important regions
  - Integrate response into `/api/predict/speech` endpoint
  
- **Implementation steps**:
  1. Create `visualize_audio_saliency()` function
  2. Generate spectrogram from audio
  3. Compute feature importance for HuBERT
  4. Highlight important frequency ranges
  5. Return as base64-encoded PNG in response
  
- **Why**: PPRS requirement - must show explainability for audio
- **Difficulty**: Medium (1-2 days)
- **Testing**: Verify highlighted regions correspond to emotion
- **Expected output**: JSON with `"saliency_map": "data:image/png;base64,..."`

#### 5. Setup SQLite Database for Session Storage
- **New file**: `backend/services/database.py`
- **What to do**:
  - Create SQLite database schema:
    - Sessions table (id, timestamp, user_info)
    - Predictions table (id, session_id, emotion, confidence, modality)
  - Create database utility functions:
    - `create_session()`
    - `save_prediction()`
    - `get_session_history()`
    - `export_session()`
  
- **Database schema**:
  ```sql
  CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    timestamp DATETIME,
    user_id TEXT,
    total_predictions INT
  );
  
  CREATE TABLE predictions (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    modality TEXT (facial/speech/video),
    emotion TEXT,
    confidence FLOAT,
    timestamp DATETIME,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
  );
  ```
  
- **Why**: PPRS requirement - must store session data
- **Difficulty**: Medium (1-2 days)
- **Database location**: `data/sessions.db`

---

### 🟡 MEDIUM PRIORITY (Week 2-3 - Est. 2-3 days)

#### 7. Add Session Management Endpoints
- **New endpoints in `backend/main.py`**:
  - `POST /api/sessions/create` - Start new session
  - `GET /api/sessions/{session_id}` - Get session data
  - `GET /api/sessions` - List all sessions
  - `POST /api/sessions/{session_id}/export` - Export as CSV/JSON
  - `DELETE /api/sessions/{session_id}` - Delete session
  
- **What to do**:
  - Integrate database layer with FastAPI
  - Create session ID (UUID) for each new session
  - Modify predict endpoints to save results
  - Create export functionality (CSV + JSON)
  
- **Why**: Complete session storage feature
- **Difficulty**: Medium (1-2 days)
- **Testing**: Verify data persists across requests

#### 8. Add Session History UI to Frontend
- **File to modify**: `frontend/src/App.js`
- **What to add**:
  - Session history sidebar
  - Export button (CSV/JSON)
  - Session details view
  - Delete session option
  
- **Why**: Users can view past predictions
- **Difficulty**: Medium (1-2 days)

#### 9. Add Confidence Threshold Settings
- **New feature**: Allow users to adjust emotion confidence threshold
- **Implementation**:
  - Add slider in Tab 4 (Model Info)
  - Update predictions when threshold changes
  - Show only emotions above threshold
  
- **Why**: More control over predictions
- **Difficulty**: Easy (1 hour)

---

### 🟢 LOW PRIORITY (Week 3+ - Polish & Polish)

#### 10. Performance Optimization
- **What to do**:
  - Cache model weights in memory
  - Batch multiple predictions
  - Optimize image processing (resize, normalize)
  - Add GPU support (if available)
  
- **Why**: Faster inference
- **Difficulty**: Medium

#### 11. Mobile App Responsiveness
- **What to test**:
  - All tabs work on mobile (iPhone, Android)
  - Touch-friendly buttons
  - Responsive webcam preview
  
- **Why**: Better user experience
- **Difficulty**: Easy (1 hour)

#### 12. Add Error Logging
- **What to do**:
  - Log all API errors to file
  - Create `/api/logs` endpoint for debugging
  - Add request/response logging
  
- **Why**: Easier debugging in production
- **Difficulty**: Easy (1 hour)

---

## 📊 WORK BREAKDOWN

## 📊 WORK BREAKDOWN

| Priority | Task | Est. Time | Status | Dependencies |
|----------|------|-----------|--------|--------------|
| 🔴 Critical | Combined endpoint | 2-3 hrs | ✅ DONE | None |
| 🔴 Critical | Improve facial acc | 2-3 days | 🚀 NEXT | None |
| 🔴 Critical | Test all tabs | 1 hr | Ready | Combined endpoint |
| 🟠 High | Grad-CAM vis | 2 days | Pending | Facial model |
| 🟠 High | Audio saliency | 1-2 days | Pending | Speech model |
| 🟠 High | SQLite setup | 1-2 days | Pending | None |
| 🟡 Medium | Session endpoints | 1-2 days | Pending | SQLite |
| 🟡 Medium | Session UI | 1-2 days | Pending | Session endpoints |
| 🟡 Medium | Confidence threshold | 1 hr | Pending | None |
| 🟢 Low | Performance opt | 1-2 days | Pending | All critical done |
| 🟢 Low | Mobile responsive | 1 hr | Pending | None |
| 🟢 Low | Error logging | 1 hr | Pending | None |

---

## 📅 SUGGESTED SPRINT SCHEDULE

### Week 1 (Jan 28 - Feb 3) - NOW
- [x] **Monday (Jan 28)**: Add combined endpoint + test Tab 2 ✅ DONE
- [ ] **Tuesday-Thursday**: Improve facial accuracy (try augmentation + EfficientNet)
- [ ] **Friday**: Test all tabs, create PR

**Current Status**: Combined endpoint complete! Tab 2 now works! 🎉

**Next**: Start facial accuracy improvement (CRITICAL PATH)

### Week 2 (Feb 4 - Feb 10)
- [ ] **Mon-Tue**: Implement Grad-CAM
- [ ] **Wed-Thu**: Implement audio saliency
- [ ] **Friday**: Integration testing

**Deliverable**: Both explainability features working

### Week 3 (Feb 11 - Feb 17)
- [ ] **Mon-Tue**: SQLite + session endpoints
- [ ] **Wed-Thu**: Frontend session UI
- [ ] **Friday**: End-to-end session testing

**Deliverable**: Full session storage + history feature

### Week 4+ (Feb 18 onwards)
- [ ] Performance optimization
- [ ] Mobile testing
- [ ] Error logging
- [ ] Final PPRS compliance audit

**Deliverable**: Production-ready system at 85-90% PPRS compliance

---

## ✅ ACCEPTANCE CRITERIA FOR DONE

### Must Have (Critical Path to PPRS)
- [x] `/api/predict/combined` endpoint working ✅ DONE
- [ ] All 4 frontend tabs functional (Tab 2 now works! ✅)
- [ ] Facial accuracy ≥ 80% (target 90%) - **NEXT PRIORITY**
- [ ] Grad-CAM visualization implemented
- [ ] Audio saliency maps implemented
- [ ] SQLite session storage working
- [ ] Session export (CSV + JSON) working

### Should Have (Better PPRS Score)
- [ ] Facial accuracy ≥ 85%
- [ ] Session history UI implemented
- [ ] Performance optimized
- [ ] Mobile responsive

### Nice to Have
- [ ] Confidence threshold settings
- [ ] Error logging
- [ ] API documentation

---

## 🚀 HOW TO GET STARTED

1. **Pick first task**: Add combined endpoint (easiest, unblocks Tab 2)
2. **Create branch**: `git checkout -b feature/combined-endpoint`
3. **Implement**: Add route to `backend/main.py`
4. **Test**: Run frontend, verify Tab 2 works
5. **Commit**: `git add . && git commit -m "Add combined prediction endpoint"`

Then move to accuracy improvement (biggest blocker).

---

## 💡 TIPS

- **Facial accuracy**: Start with stronger augmentation before trying new models
- **Grad-CAM**: Use existing libraries (torch-cam, pytorch-grad-cam)
- **Audio saliency**: Use SHAP or integrated gradients for HuBERT
- **Session storage**: Start simple (SQLite), can migrate to PostgreSQL later
- **Testing**: Run frontend against each new feature before moving on

---

## 📞 DEPENDENCIES & RESOURCES

- **Grad-CAM library**: `pip install pytorch-grad-cam`
- **Audio visualization**: `matplotlib`, `librosa` (already installed)
- **Database**: SQLite (built-in with Python)
- **UUID**: `uuid` module (built-in)
- **Image overlay**: `PIL/Pillow` (already installed)

All dependencies already in `requirements.txt`
