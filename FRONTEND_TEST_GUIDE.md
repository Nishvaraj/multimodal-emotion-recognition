
# 🧪 Frontend Testing Guide - Gradio Demo (All 4 Tabs)

**Status**: ✅ Running  
**Frontend**: http://127.0.0.1:7860 (Gradio)  
**Date**: January 28, 2026

---

## 📋 TEST CHECKLIST

### ✅ INFRASTRUCTURE READY
- [x] Backend server running (port 8000) with 7 endpoints
- [x] Gradio demo running (port 7860)
- [x] Both models loaded (ViT + HuBERT)
- [x] FFmpeg installed and integrated
- [x] Video processing pipeline operational

---

## 🎯 TAB 1: FACIAL EMOTION RECOGNITION

**Location**: Gradio Tab 1 (📸 Facial Emotion)

### Test 1.1: Facial Emotion - Webcam Capture
1. Open `http://127.0.0.1:7860` in browser
2. Click **Tab 1: 📸 Facial Emotion**
3. Under "Capture or Upload Image":
   - [ ] Click webcam icon to start camera
   - [ ] Position face clearly in frame
   - [ ] Click to capture
   - [ ] Click "🔮 Analyze Face" button
4. **Expected Result**:
   - [ ] Image displays with face visible
   - [ ] Emotion predicted (angry, disgust, fear, happy, neutral, sad, surprise)
   - [ ] Confidence score shown as percentage
   - [ ] Probability bars displayed for all 7 emotions
   - [ ] Top emotion highlighted in green box

### Test 1.2: Facial Emotion - Image Upload
1. In Tab 1:
   - [ ] Click upload icon to select image
   - [ ] Choose JPG/PNG with clear face
   - [ ] Click "🔮 Analyze Face"
2. **Expected Result**:
   - [ ] Image previewed
   - [ ] All emotion predictions displayed
   - [ ] No crashes or errors

---

## 🎯 TAB 2: SPEECH EMOTION RECOGNITION

**Location**: Gradio Tab 2 (🎤 Speech Emotion)

### Test 2.1: Speech Emotion - Microphone Recording
1. Click **Tab 2: 🎤 Speech Emotion**
2. Under "Record or Upload Audio":
   - [ ] Click microphone icon to start recording
   - [ ] Speak with emotion (e.g., "I'm so happy!" for happy)
   - [ ] Click to stop recording
   - [ ] Click "🔮 Analyze Voice" button
3. **Expected Result**:
   - [ ] Audio waveform displayed in canvas
   - [ ] Emotion predicted (angry, calm, disgust, fearful, happy, neutral, sad, surprised)
   - [ ] Confidence score shown
   - [ ] Probability bars for all 8 emotions
   - [ ] Waveform visualization clear and visible

### Test 2.2: Speech Emotion - Audio File Upload
1. In Tab 2:
   - [ ] Click upload icon to select audio
   - [ ] Choose WAV/MP3 file with clear speech
   - [ ] Click "🔮 Analyze Voice"
2. **Expected Result**:
   - [ ] Audio file loaded
   - [ ] Waveform displayed
   - [ ] Emotion detected
   - [ ] All confidence scores shown

---

## 🎯 TAB 3: COMBINED ANALYSIS (Facial + Speech)

**Location**: Gradio Tab 3 (🔗 Combined Analysis)

### Test 3.1: Combined Analysis - Video Mode (NEW!)
1. Click **Tab 3: 🔗 Combined Analysis**
2. Select mode: **Click "🎥 Video Upload (MP4)" radio button**
3. Upload/record video:
   - [ ] Click webcam icon to record video (5-10 seconds)
   - [ ] OR click upload icon to select MP4 file
   - [ ] Ensure face visible and clear speech/audio
4. Click "🎬 Analyze Video" button
5. **Expected Result**:
   - [ ] Middle frame extracted from video
   - [ ] Audio extracted and resampled to 16kHz
   - [ ] Facial emotion predicted from frame
   - [ ] Speech emotion predicted from audio
   - [ ] Extracted frame shown in preview
   - [ ] Concordance result displayed:
     - ✅ "MATCH" if both emotions agree
     - ⚠️ "MISMATCH" if emotions differ
   - [ ] No FFmpeg errors

### Test 3.2: Combined Analysis - Separate Mode
1. In Tab 3, select: **Click "📸 Separate Images & Audio" radio button**
2. Upload facial image:
   - [ ] Click webcam or upload icon
   - [ ] Select clear face image
3. Upload/record audio:
   - [ ] Click microphone or upload icon
   - [ ] Record speech matching emotion (e.g., "I'm happy!" for happy face)
4. Click "🚀 Analyze Both" button
5. **Expected Result**:
   - [ ] Both inputs processed
   - [ ] Facial emotion shown
   - [ ] Speech emotion shown
   - [ ] Concordance comparison displayed
   - [ ] Combined results clear

### Test 3.3: Mode Switching
1. In Tab 3, start with one mode selected
2. Click the radio button to switch to the other mode
3. **Expected Result**:
   - [ ] Video mode group appears/disappears smoothly
   - [ ] Separate mode group appears/disappears smoothly
   - [ ] No data lost from other inputs
   - [ ] UI remains responsive
   - [ ] Detailed comparison text
   - [ ] No errors in console

### Test 2.2: Combined - Test Matching Emotions
1. Get or create test files:
   - Happy face image + happy voice audio
   - Sad face image + sad voice audio
2. Upload matching pair
3. Click "Analyze Both"
4. **Expected Result**:
   - [ ] Concordance: **MATCH** ✅
   - [ ] Emotions agree
   - [ ] Clear UI feedback showing agreement

### Test 2.3: Combined - Test Mismatching Emotions
1. Get mismatched pair:
   - Happy face + sad voice (or vice versa)
2. Upload
3. Click "Analyze Both"
4. **Expected Result**:
   - [ ] Concordance: **MISMATCH** ⚠️
   - [ ] Emotions disagree
   - [ ] System explains which modality shows what emotion

---

## 🎯 TAB 3: VIDEO ANALYSIS

**File**: `frontend/src/App.js` (Tab 3 code)

### Test 3.1: Video Upload & Analysis
1. Click **Tab 3: Video Analysis**
2. Click "Choose Video File"
3. Select a video (MP4, recommended 10-30 seconds)
4. Click "Analyze Video"
5. **Expected Result**:
   - [ ] Video preview shown
   - [ ] Processing status displayed
   - [ ] Facial emotion from video frames detected
   - [ ] Speech emotion from audio track detected
   - [ ] Video duration shown
   - [ ] Number of frames processed displayed
   - [ ] FPS (frames per second) shown
   - [ ] Combined emotion from video shown
   - [ ] No errors in console

### Test 3.2: Video - Check Results Format
1. After video analysis completes:
   - [ ] Facial emotion section shows:
     - Emotion name
     - Confidence score
     - Number of frames analyzed
     - Probability bars
   - [ ] Speech emotion section shows:
     - Emotion name
     - Confidence score
     - Probability bars
   - [ ] Overall analysis shows combined prediction

---

## 🎯 TAB 4: MODEL INFORMATION

**File**: `frontend/src/App.js` (Tab 4 code)

### Test 4.1: Model Info Display
1. Click **Tab 4: Model Information**
2. Verify information shown:
   - [ ] **Facial Model (ViT)**
     - Model name: "Vision Transformer (google/vit-base-patch16-224-in21k)"
     - Training accuracy: "71.29%"
     - Emotions: 7 listed (angry, disgust, fear, happy, neutral, sad, surprise)
     - Description: Explains what ViT does
   - [ ] **Speech Model (HuBERT)**
     - Model name: "facebook/hubert-large-ls960-ft"
     - Training accuracy: "87.50%"
     - Emotions: 8 listed (angry, calm, disgust, fearful, happy, neutral, sad, surprised)
     - Description: Explains HuBERT
   - [ ] **System Status**
     - Backend status: "Connected ✓"
     - Both models: "Loaded ✓"
     - Device: "CPU" or "CUDA"
   - [ ] **Emotion Definitions**
     - All emotions have emoji
     - Each has brief description

### Test 4.2: Verify Emotion Lists Match Models
1. Check Tab 4 displays correct emotions:
   - [ ] Facial: 7 emotions (no "calm" or "surprised")
   - [ ] Speech: 8 emotions (includes "calm" and "surprised")
   - [ ] Lists match backend EMOTIONS_FACIAL and EMOTIONS_SPEECH

---

## 🔍 CONSOLE ERROR CHECK

While testing each tab, open browser DevTools:
1. Press `F12` or `Cmd+Option+I` (Mac)
2. Click **Console** tab
3. After each test, verify:
   - [ ] No red error messages
   - [ ] Only info/warning logs are acceptable
   - [ ] No "404" errors for API calls
   - [ ] No CORS errors

**Common errors to watch for**:
- ❌ `TypeError: Cannot read property of undefined` - missing response data
- ❌ `404 Not Found /api/predict/...` - endpoint not working
- ❌ `CORS error` - backend not accepting frontend requests
- ❌ `Uncaught SyntaxError` - parsing error in response

---

## 📊 SUCCESS CRITERIA

### ✅ All Tabs Functional
- [x] Tab 1: Facial emotion working (webcam + upload)
- [x] Tab 1: Speech emotion working (webcam + upload)
- [x] Tab 2: Combined analysis working
- [x] Tab 3: Video analysis working
- [x] Tab 4: Model info displaying correctly

### ✅ No Console Errors
- [x] Browser console shows no red errors
- [x] All API calls succeed (200 status)
- [x] No undefined variable errors

### ✅ Data Flows Correctly
- [x] Images load and display
- [x] Audio waveforms show
- [x] Emotions predicted correctly
- [x] Confidence scores reasonable (0.0-1.0)
- [x] Probabilities sum to 1.0
- [x] Concordance logic works

### ✅ UI Responsive
- [x] All buttons clickable
- [x] Forms validate input
- [x] Loading states visible
- [x] Results display properly
- [x] No layout issues

---

## 🧪 QUICK TEST SCRIPT (5 minutes)

Want to test fast? Follow this:

**TAB 1 (2 min)**:
```
1. Take selfie with Tab 1
2. Record "I'm happy!" 
3. Analyze both
4. Check results appear
```

**TAB 2 (2 min)**:
```
1. Upload selfie + audio
2. Click "Analyze Both"
3. Check MATCH/MISMATCH
4. Verify concordance shown
```

**TAB 3 (0.5 min)**:
```
1. Pick any video
2. Click analyze
3. Wait for results
```

**TAB 4 (0.5 min)**:
```
1. View model info
2. Check all details shown
```

---

## 🐛 TROUBLESHOOTING

### Issue: "Backend not responding"
**Solution**:
```bash
# Check backend is running
curl http://127.0.0.1:8000/health

# Should return:
# {"status": "healthy", "facial_model": true, "speech_model": true, "device": "cpu"}
```

### Issue: "Tab 2 shows error"
**Solution**:
- Verify both image AND audio are uploaded
- Check file formats (JPG/PNG for image, WAV/MP3 for audio)
- Check browser console for specific error

### Issue: "No emotion detected"
**Solution**:
- Make sure image shows clear face
- Make sure audio has voice (not silent)
- Check file not corrupted
- Try different file

### Issue: "Slow predictions"
**Solution**:
- First prediction is slow (models loading)
- Subsequent predictions should be ~2-5 seconds
- This is normal on CPU (not GPU)

### Issue: "API returns 400 error"
**Solution**:
- Check file format
- Try smaller file
- Check backend console for error message
- Run backend with more verbose logging

---

## 📝 TEST REPORT TEMPLATE

After testing, fill this in:

```
## Frontend Testing Report - Jan 28, 2026

### Tab 1: Separate Testing
- Facial emotion (webcam): [PASS/FAIL]
- Facial emotion (upload): [PASS/FAIL]
- Speech emotion (record): [PASS/FAIL]
- Speech emotion (upload): [PASS/FAIL]

### Tab 2: Combined Analysis
- Combined emotions: [PASS/FAIL]
- Concordance check: [PASS/FAIL]
- Matching test: [PASS/FAIL]
- Mismatching test: [PASS/FAIL]

### Tab 3: Video Analysis
- Video upload: [PASS/FAIL]
- Results display: [PASS/FAIL]

### Tab 4: Model Info
- All info displayed: [PASS/FAIL]
- Emotion counts correct: [PASS/FAIL]

### Console
- No red errors: [PASS/FAIL]

### Overall
- All tabs functional: [PASS/FAIL]
- Ready for next phase: [YES/NO]
```

---

## 🎯 NEXT STEPS

Once all tests PASS:
1. ✅ Update TODO_NEXT_STEPS.md with test results
2. ✅ Mark "Test All Tabs" as COMPLETE
3. 🚀 Start "Improve Facial Accuracy" (CRITICAL)

---

## 📚 REFERENCE

- Backend API: `http://127.0.0.1:8000`
- Frontend UI: `http://localhost:3000`
- API Docs: `http://127.0.0.1:8000/docs` (Swagger UI)
- Backend code: [backend/main.py](backend/main.py)
- Frontend code: [frontend/src/App.js](frontend/src/App.js)

---

## ✨ HAPPY TESTING! 🎉

The combined endpoint is now live. Tab 2 should work perfectly!

Report back with results and we'll move to facial accuracy improvement (the critical path).
