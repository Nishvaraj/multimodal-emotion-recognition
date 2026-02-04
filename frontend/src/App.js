import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

const EMOTION_EMOJIS = {
  'angry': '😠',
  'disgust': '🤢',
  'fear': '😨',
  'happy': '😊',
  'neutral': '😐',
  'sad': '😢',
  'surprise': '😲',
  'calm': '😌',
  'fearful': '😨',
  'surprised': '😲'
};

const EMOTIONS_FACIAL = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];
const EMOTIONS_SPEECH = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'];
const API_BASE = 'http://127.0.0.1:8000';

// ============== FACIAL EMOTION TAB ==============
function FacialTab() {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [emotion, setEmotion] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [probabilities, setProbabilities] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [showGradCAM, setShowGradCAM] = useState(false);
  const [gradCam, setGradCam] = useState(null);

  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = (evt) => {
        setImagePreview(evt.target.result);
        setEmotion(null);
        setGradCam(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraOn(true);
        setError(null);
      }
    } catch (err) {
      setError('Cannot access camera');
    }
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
      canvasRef.current.toBlob((blob) => {
        setImageFile(blob);
        const reader = new FileReader();
        reader.onload = (evt) => setImagePreview(evt.target.result);
        reader.readAsDataURL(blob);
        stopCamera();
      });
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      setIsCameraOn(false);
    }
  };

  const analyzeFacial = async () => {
    if (!imageFile) {
      setError('Please select or capture an image');
      return;
    }
    setLoading(true);
    setError(null);
    setGradCam(null);
    try {
      const formData = new FormData();
      formData.append('file', imageFile);
      
      const explainParam = showGradCAM ? '?explain=true' : '';
      const response = await axios.post(`${API_BASE}/api/predict/facial${explainParam}`, formData);
      if (response.data.success) {
        setEmotion(response.data.emotion);
        setConfidence(response.data.confidence);
        setProbabilities(response.data.probabilities);
        if (response.data.grad_cam) {
          setGradCam(response.data.grad_cam);
        }
      }
    } catch (err) {
      setError('API Error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Left Column - Input */}
      <div className="space-y-4">
        {/* Image Capture Card */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
          <div className="bg-purple-900 px-4 py-2 flex items-center gap-2">
            <span className="text-purple-300 text-sm font-medium">📷 Capture or Upload Image</span>
          </div>
          <div className="p-4">
            {!isCameraOn && !imagePreview && (
              <div className="border-2 border-dashed border-slate-700 rounded-lg p-8 text-center">
                <div className="text-6xl mb-4">📸</div>
                <p className="text-slate-400 mb-4">Click to Access Webcam</p>
                <button
                  onClick={startCamera}
                  className="bg-gradient-to-br from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white px-6 py-2 rounded-lg font-medium transition-all duration-200 hover:shadow-lg"
                >
                  Start Webcam
                </button>
                <div className="mt-4 pt-4 border-t border-slate-700">
                  <label className="cursor-pointer inline-block">
                    <div className="text-slate-400 hover:text-slate-300 flex items-center justify-center gap-2">
                      <span className="text-2xl">📁</span>
                      <span>Or Upload from Files</span>
                    </div>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImageSelect}
                      className="hidden"
                    />
                  </label>
                </div>
              </div>
            )}
            {isCameraOn && (
              <div>
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  className="w-full rounded-lg mb-3"
                />
                <canvas ref={canvasRef} width="320" height="240" className="hidden" />
                <div className="flex gap-2">
                  <button
                    onClick={capturePhoto}
                    className="flex-1 bg-green-600 hover:bg-green-500 text-white px-4 py-2 rounded-lg font-medium transition-colors"
                  >
                    📸 Capture
                  </button>
                  <button
                    onClick={stopCamera}
                    className="flex-1 bg-red-600 hover:bg-red-500 text-white px-4 py-2 rounded-lg font-medium transition-colors"
                  >
                    ✖️ Cancel
                  </button>
                </div>
              </div>
            )}
            {imagePreview && !isCameraOn && (
              <div>
                <img src={imagePreview} alt="Preview" className="w-full rounded-lg mb-3" />
                <label className="cursor-pointer inline-block w-full text-center">
                  <div className="text-slate-400 hover:text-slate-300 text-sm">
                    📁 Change Image
                  </div>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageSelect}
                    className="hidden"
                  />
                </label>
              </div>
            )}
          </div>
        </div>

        {/* Grad-CAM Checkbox */}
        {imagePreview && (
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
            <label className="flex items-start gap-3 cursor-pointer group">
              <input
                type="checkbox"
                checked={showGradCAM}
                onChange={(e) => setShowGradCAM(e.target.checked)}
                className="mt-1 w-5 h-5 rounded border-slate-600 bg-slate-700 text-purple-600 focus:ring-purple-500 focus:ring-offset-slate-800"
              />
              <div>
                <div className="text-slate-50 font-medium">
                  🔥 Show Grad-CAM Heatmap
                </div>
                <div className="text-slate-400 text-sm mt-1">
                  See which facial regions influenced the prediction
                </div>
              </div>
            </label>
          </div>
        )}

        {/* Analyze Button */}
        {imagePreview && (
          <button
            onClick={analyzeFacial}
            disabled={loading}
            className="w-full bg-gradient-to-br from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white px-6 py-3 rounded-lg font-medium text-lg transition-all duration-200 hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? '⏳ Analyzing...' : '🔮 Analyze Face'}
          </button>
        )}

        {error && (
          <div className="bg-red-900/50 border border-red-700 rounded-lg p-4 text-red-200">
            {error}
          </div>
        )}
      </div>

      {/* Right Column - Results */}
      <div className="space-y-4">
        {!emotion && (
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-8 text-center">
            <div className="text-6xl mb-4">⏳</div>
            <p className="text-slate-400">Waiting for input...</p>
          </div>
        )}

        {/* Confidence Scores */}
        {emotion && probabilities && (
          <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
            <div className="bg-purple-900 px-4 py-2 flex items-center gap-2">
              <span className="text-purple-300 text-sm font-medium">📊 Confidence Scores</span>
            </div>
            <div className="p-4 space-y-3">
              <div className="text-center mb-4">
                <div className="text-5xl mb-2">{EMOTION_EMOJIS[emotion]}</div>
                <div className="text-2xl font-bold text-slate-50">{emotion.toUpperCase()}</div>
                <div className="text-lg text-green-400">{(confidence * 100).toFixed(1)}%</div>
              </div>
              {EMOTIONS_FACIAL.map((emo) => (
                <div key={emo} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-300">{EMOTION_EMOJIS[emo]} {emo}</span>
                    <span className="text-slate-400">{((probabilities[emo] || 0) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-purple-500 to-indigo-500 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${(probabilities[emo] || 0) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Annotated Result */}
        {emotion && imagePreview && (
          <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
            <div className="bg-purple-900 px-4 py-2 flex items-center gap-2">
              <span className="text-purple-300 text-sm font-medium">✨ Annotated Result</span>
            </div>
            <div className="p-4">
              <img src={imagePreview} alt="Annotated" className="w-full rounded-lg" />
            </div>
          </div>
        )}

        {/* Grad-CAM Heatmap */}
        {gradCam && (
          <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
            <div className="bg-purple-900 px-4 py-2 flex items-center gap-2">
              <span className="text-purple-300 text-sm font-medium">🔥 Grad-CAM Heatmap</span>
            </div>
            <div className="p-4">
              <p className="text-slate-400 text-sm mb-3">
                Red regions = areas the model focused on | Blue regions = less important
              </p>
              <img
                src={`data:image/png;base64,${gradCam}`}
                alt="Grad-CAM Heatmap"
                className="w-full rounded-lg"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ============== SPEECH EMOTION TAB ==============
function SpeechTab() {
  const [audioFile, setAudioFile] = useState(null);
  const [emotion, setEmotion] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [probabilities, setProbabilities] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const streamRef = useRef(null);
  const [showSaliency, setShowSaliency] = useState(false);
  const [saliency, setSaliency] = useState(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];
      
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.ondataavailable = (e) => chunksRef.current.push(e.data);
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/wav' });
        setAudioFile(audioBlob);
      };
      mediaRecorder.start();
      setIsRecording(true);
      setError(null);
    } catch (err) {
      setError('Cannot access microphone');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (streamRef.current) streamRef.current.getTracks().forEach(track => track.stop());
    }
  };

  const handleAudioSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setAudioFile(file);
      setEmotion(null);
      setSaliency(null);
    }
  };

  const analyzeSpeech = async () => {
    if (!audioFile) {
      setError('Please record or upload audio');
      return;
    }
    setLoading(true);
    setError(null);
    setSaliency(null);
    try {
      const formData = new FormData();
      formData.append('file', audioFile);
      
      const explainParam = showSaliency ? '?explain=true' : '';
      const response = await axios.post(`${API_BASE}/api/predict/speech${explainParam}`, formData);
      if (response.data.success) {
        setEmotion(response.data.emotion);
        setConfidence(response.data.confidence);
        setProbabilities(response.data.probabilities);
        if (response.data.saliency) {
          setSaliency(response.data.saliency);
        }
      }
    } catch (err) {
      setError('API Error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Left Column - Input */}
      <div className="space-y-4">
        {/* Audio Record/Upload Card */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
          <div className="bg-purple-900 px-4 py-2 flex items-center gap-2">
            <span className="text-purple-300 text-sm font-medium">🎤 Record or Upload Audio</span>
          </div>
          <div className="p-4">
            {!isRecording && !audioFile && (
              <div className="border-2 border-dashed border-slate-700 rounded-lg p-8 text-center">
                <div className="text-6xl mb-4">🎙️</div>
                <p className="text-slate-400 mb-4">Click to Record Audio</p>
                <button
                  onClick={startRecording}
                  className="bg-gradient-to-br from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white px-6 py-2 rounded-lg font-medium transition-all duration-200 hover:shadow-lg"
                >
                  Start Recording
                </button>
                <div className="mt-4 pt-4 border-t border-slate-700">
                  <label className="cursor-pointer inline-block">
                    <div className="text-slate-400 hover:text-slate-300 flex items-center justify-center gap-2">
                      <span className="text-2xl">📁</span>
                      <span>Or Upload from Files</span>
                    </div>
                    <input
                      type="file"
                      accept="audio/*"
                      onChange={handleAudioSelect}
                      className="hidden"
                    />
                  </label>
                </div>
              </div>
            )}
            {isRecording && (
              <div className="text-center">
                <div className="text-red-500 text-xl font-semibold mb-4 animate-pulse">
                  🔴 Recording...
                </div>
                <div className="bg-slate-700 rounded-lg h-20 mb-4 flex items-center justify-center">
                  <div className="flex gap-1">
                    {[...Array(20)].map((_, i) => (
                      <div
                        key={i}
                        className="w-1 bg-purple-500 rounded-full animate-pulse"
                        style={{
                          height: `${Math.random() * 60 + 20}px`,
                          animationDelay: `${i * 0.05}s`
                        }}
                      />
                    ))}
                  </div>
                </div>
                <button
                  onClick={stopRecording}
                  className="bg-red-600 hover:bg-red-500 text-white px-6 py-2 rounded-lg font-medium transition-colors"
                >
                  ⏹️ Stop Recording
                </button>
              </div>
            )}
            {audioFile && !isRecording && (
              <div className="text-center">
                <div className="text-green-400 text-lg font-semibold mb-3">
                  ✅ Audio Ready
                </div>
                <div className="bg-slate-700 rounded-lg p-4 mb-3">
                  <div className="text-4xl mb-2">🎵</div>
                  <div className="text-slate-300 text-sm">Audio file loaded</div>
                </div>
                <label className="cursor-pointer inline-block">
                  <div className="text-slate-400 hover:text-slate-300 text-sm">
                    📁 Change Audio
                  </div>
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={handleAudioSelect}
                    className="hidden"
                  />
                </label>
              </div>
            )}
          </div>
        </div>

        {/* Saliency Checkbox */}
        {audioFile && (
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
            <label className="flex items-start gap-3 cursor-pointer group">
              <input
                type="checkbox"
                checked={showSaliency}
                onChange={(e) => setShowSaliency(e.target.checked)}
                className="mt-1 w-5 h-5 rounded border-slate-600 bg-slate-700 text-purple-600 focus:ring-purple-500 focus:ring-offset-slate-800"
              />
              <div>
                <div className="text-slate-50 font-medium">
                  📊 Show Audio Saliency Map
                </div>
                <div className="text-slate-400 text-sm mt-1">
                  See which frequencies influenced the prediction
                </div>
              </div>
            </label>
          </div>
        )}

        {/* Analyze Button */}
        {audioFile && (
          <button
            onClick={analyzeSpeech}
            disabled={loading}
            className="w-full bg-gradient-to-br from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white px-6 py-3 rounded-lg font-medium text-lg transition-all duration-200 hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? '⏳ Analyzing...' : '🎤 Analyze Voice'}
          </button>
        )}

        {error && (
          <div className="bg-red-900/50 border border-red-700 rounded-lg p-4 text-red-200">
            {error}
          </div>
        )}
      </div>

      {/* Right Column - Results */}
      <div className="space-y-4">
        {!emotion && (
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-8 text-center">
            <div className="text-6xl mb-4">⏳</div>
            <p className="text-slate-400">Waiting for input...</p>
          </div>
        )}

        {/* Confidence Scores */}
        {emotion && probabilities && (
          <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
            <div className="bg-purple-900 px-4 py-2 flex items-center gap-2">
              <span className="text-purple-300 text-sm font-medium">📊 Confidence Scores</span>
            </div>
            <div className="p-4 space-y-3">
              <div className="text-center mb-4">
                <div className="text-5xl mb-2">{EMOTION_EMOJIS[emotion]}</div>
                <div className="text-2xl font-bold text-slate-50">{emotion.toUpperCase()}</div>
                <div className="text-lg text-green-400">{(confidence * 100).toFixed(1)}%</div>
              </div>
              {EMOTIONS_SPEECH.map((emo) => (
                <div key={emo} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-300">{EMOTION_EMOJIS[emo]} {emo}</span>
                    <span className="text-slate-400">{((probabilities[emo] || 0) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-purple-500 to-indigo-500 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${(probabilities[emo] || 0) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Audio Saliency Map */}
        {saliency && (
          <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
            <div className="bg-purple-900 px-4 py-2 flex items-center gap-2">
              <span className="text-purple-300 text-sm font-medium">📊 Audio Saliency Map</span>
            </div>
            <div className="p-4">
              <p className="text-slate-400 text-sm mb-3">
                Red frequencies = important for prediction | Blue frequencies = less important
              </p>
              <img
                src={`data:image/png;base64,${saliency}`}
                alt="Audio Saliency"
                className="w-full rounded-lg"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ============== COMBINED ANALYSIS TAB ==============
function CombinedTab() {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [facialEmotion, setFacialEmotion] = useState(null);
  const [speechEmotion, setSpeechEmotion] = useState(null);
  const [concordance, setConcordance] = useState(null);
  const [facialProbs, setFacialProbs] = useState(null);
  const [speechProbs, setSpeechProbs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showGradCAM, setShowGradCAM] = useState(false);
  const [showSaliency, setShowSaliency] = useState(false);
  const [gradCam, setGradCam] = useState(null);
  const [saliency, setSaliency] = useState(null);

  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = (evt) => setImagePreview(evt.target.result);
      reader.readAsDataURL(file);
    }
  };

  const handleAudioSelect = (e) => {
    const file = e.target.files[0];
    if (file) setAudioFile(file);
  };

  const analyzeCombined = async () => {
    if (!imageFile || !audioFile) {
      setError('Please provide both image and audio');
      return;
    }
    setLoading(true);
    setError(null);
    setGradCam(null);
    setSaliency(null);
    try {
      const formData = new FormData();
      formData.append('image_file', imageFile);
      formData.append('audio_file', audioFile);
      
      const explainParam = (showGradCAM || showSaliency) ? '?explain=true' : '';
      const response = await axios.post(`${API_BASE}/api/predict/combined${explainParam}`, formData);
      if (response.data.success) {
        setFacialEmotion(response.data.facial_emotion.emotion);
        setSpeechEmotion(response.data.speech_emotion.emotion);
        setConcordance(response.data.concordance);
        setFacialProbs(response.data.facial_emotion.probabilities);
        setSpeechProbs(response.data.speech_emotion.probabilities);
        
        if (response.data.explainability) {
          if (response.data.explainability.grad_cam) {
            setGradCam(response.data.explainability.grad_cam);
          }
          if (response.data.explainability.saliency) {
            setSaliency(response.data.explainability.saliency);
          }
        }
      }
    } catch (err) {
      setError('API Error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Input Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Image Upload */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
          <div className="bg-purple-900 px-4 py-2 flex items-center gap-2">
            <span className="text-purple-300 text-sm font-medium">📷 Image Input</span>
          </div>
          <div className="p-4">
            {!imagePreview ? (
              <label className="cursor-pointer block">
                <div className="border-2 border-dashed border-slate-700 rounded-lg p-8 text-center hover:border-slate-600 transition-colors">
                  <div className="text-6xl mb-4">📸</div>
                  <p className="text-slate-400 mb-2">Upload Image</p>
                  <p className="text-slate-500 text-sm">Click or drag to upload</p>
                </div>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageSelect}
                  className="hidden"
                />
              </label>
            ) : (
              <div>
                <img src={imagePreview} alt="Preview" className="w-full rounded-lg mb-3" />
                <label className="cursor-pointer block text-center">
                  <div className="text-slate-400 hover:text-slate-300 text-sm">
                    📁 Change Image
                  </div>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageSelect}
                    className="hidden"
                  />
                </label>
              </div>
            )}
          </div>
        </div>

        {/* Audio Upload */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
          <div className="bg-purple-900 px-4 py-2 flex items-center gap-2">
            <span className="text-purple-300 text-sm font-medium">🎤 Audio Input</span>
          </div>
          <div className="p-4">
            {!audioFile ? (
              <label className="cursor-pointer block">
                <div className="border-2 border-dashed border-slate-700 rounded-lg p-8 text-center hover:border-slate-600 transition-colors">
                  <div className="text-6xl mb-4">🎵</div>
                  <p className="text-slate-400 mb-2">Upload Audio</p>
                  <p className="text-slate-500 text-sm">Click or drag to upload</p>
                </div>
                <input
                  type="file"
                  accept="audio/*"
                  onChange={handleAudioSelect}
                  className="hidden"
                />
              </label>
            ) : (
              <div className="text-center">
                <div className="bg-slate-700 rounded-lg p-8 mb-3">
                  <div className="text-4xl mb-2">✅</div>
                  <div className="text-green-400 font-semibold">Audio Loaded</div>
                </div>
                <label className="cursor-pointer block">
                  <div className="text-slate-400 hover:text-slate-300 text-sm">
                    📁 Change Audio
                  </div>
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={handleAudioSelect}
                    className="hidden"
                  />
                </label>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Explainability Options */}
      {imageFile && audioFile && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
            <label className="flex items-start gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={showGradCAM}
                onChange={(e) => setShowGradCAM(e.target.checked)}
                className="mt-1 w-5 h-5 rounded border-slate-600 bg-slate-700 text-purple-600 focus:ring-purple-500"
              />
              <div>
                <div className="text-slate-50 font-medium">🔥 Facial Grad-CAM</div>
                <div className="text-slate-400 text-sm mt-1">Visualize facial attention</div>
              </div>
            </label>
          </div>
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
            <label className="flex items-start gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={showSaliency}
                onChange={(e) => setShowSaliency(e.target.checked)}
                className="mt-1 w-5 h-5 rounded border-slate-600 bg-slate-700 text-purple-600 focus:ring-purple-500"
              />
              <div>
                <div className="text-slate-50 font-medium">📊 Audio Saliency</div>
                <div className="text-slate-400 text-sm mt-1">Visualize audio frequencies</div>
              </div>
            </label>
          </div>
        </div>
      )}

      {/* Analyze Button */}
      {imageFile && audioFile && (
        <button
          onClick={analyzeCombined}
          disabled={loading}
          className="w-full bg-gradient-to-br from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white px-6 py-3 rounded-lg font-medium text-lg transition-all duration-200 hover:shadow-lg disabled:opacity-50"
        >
          {loading ? '⏳ Analyzing...' : '🚀 Analyze Both'}
        </button>
      )}

      {error && (
        <div className="bg-red-900/50 border border-red-700 rounded-lg p-4 text-red-200">
          {error}
        </div>
      )}

      {/* Results */}
      {facialEmotion && speechEmotion && (
        <div className="space-y-4">
          {/* Concordance Banner */}
          <div className={`rounded-xl p-4 text-center font-semibold text-lg ${
            concordance === 'MATCH' 
              ? 'bg-green-900/50 border border-green-700 text-green-200' 
              : 'bg-yellow-900/50 border border-yellow-700 text-yellow-200'
          }`}>
            {concordance === 'MATCH' ? '✅ EMOTIONS MATCH!' : '⚠️ EMOTIONS DIFFER'}
          </div>

          {/* Results Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Facial Results */}
            <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
              <div className="bg-purple-900 px-4 py-2">
                <span className="text-purple-300 text-sm font-medium">📸 Facial Emotion</span>
              </div>
              <div className="p-4">
                <div className="text-center mb-4">
                  <div className="text-5xl mb-2">{EMOTION_EMOJIS[facialEmotion]}</div>
                  <div className="text-xl font-bold text-slate-50">{facialEmotion.toUpperCase()}</div>
                </div>
                {facialProbs && (
                  <div className="space-y-2">
                    {Object.entries(facialProbs).slice(0, 4).map(([emo, prob]) => (
                      <div key={emo} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-300">{EMOTION_EMOJIS[emo]} {emo}</span>
                          <span className="text-slate-400">{(prob * 100).toFixed(0)}%</span>
                        </div>
                        <div className="w-full bg-slate-700 rounded-full h-1.5">
                          <div
                            className="bg-gradient-to-r from-purple-500 to-indigo-500 h-1.5 rounded-full"
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Speech Results */}
            <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
              <div className="bg-purple-900 px-4 py-2">
                <span className="text-purple-300 text-sm font-medium">🎤 Speech Emotion</span>
              </div>
              <div className="p-4">
                <div className="text-center mb-4">
                  <div className="text-5xl mb-2">{EMOTION_EMOJIS[speechEmotion]}</div>
                  <div className="text-xl font-bold text-slate-50">{speechEmotion.toUpperCase()}</div>
                </div>
                {speechProbs && (
                  <div className="space-y-2">
                    {Object.entries(speechProbs).slice(0, 4).map(([emo, prob]) => (
                      <div key={emo} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-300">{EMOTION_EMOJIS[emo]} {emo}</span>
                          <span className="text-slate-400">{(prob * 100).toFixed(0)}%</span>
                        </div>
                        <div className="w-full bg-slate-700 rounded-full h-1.5">
                          <div
                            className="bg-gradient-to-r from-purple-500 to-indigo-500 h-1.5 rounded-full"
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Annotated Face */}
          {imagePreview && (
            <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
              <div className="bg-purple-900 px-4 py-2">
                <span className="text-purple-300 text-sm font-medium">✨ Annotated Face</span>
              </div>
              <div className="p-4">
                <img src={imagePreview} alt="Annotated" className="w-full rounded-lg" />
              </div>
            </div>
          )}

          {/* Explainability Visualizations */}
          {(gradCam || saliency) && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {gradCam && (
                <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
                  <div className="bg-purple-900 px-4 py-2">
                    <span className="text-purple-300 text-sm font-medium">🔥 Facial Grad-CAM</span>
                  </div>
                  <div className="p-4">
                    <img src={`data:image/png;base64,${gradCam}`} alt="Grad-CAM" className="w-full rounded-lg" />
                  </div>
                </div>
              )}
              {saliency && (
                <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
                  <div className="bg-purple-900 px-4 py-2">
                    <span className="text-purple-300 text-sm font-medium">📊 Audio Saliency</span>
                  </div>
                  <div className="p-4">
                    <img src={`data:image/png;base64,${saliency}`} alt="Saliency" className="w-full rounded-lg" />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ============== MODEL INFORMATION TAB ==============
function ModelInfoTab() {
  return (
    <div className="space-y-4">
      {/* Model Details */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
        <div className="bg-purple-900 px-4 py-2">
          <span className="text-purple-300 text-sm font-medium">📊 Model Details</span>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Facial Model */}
            <div className="space-y-3">
              <h3 className="text-xl font-bold text-slate-50 mb-4">📸 Facial Emotion Recognition</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-400">Architecture:</span>
                  <span className="text-slate-200 font-medium">Vision Transformer (ViT)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Training Data:</span>
                  <span className="text-slate-200 font-medium">FER2013 (35,887 images)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Emotions:</span>
                  <span className="text-slate-200 font-medium">7 classes</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Accuracy:</span>
                  <span className="text-green-400 font-bold">71.29%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Model Size:</span>
                  <span className="text-slate-200 font-medium">327 MB</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Input:</span>
                  <span className="text-slate-200 font-medium">224×224 RGB</span>
                </div>
              </div>
            </div>

            {/* Speech Model */}
            <div className="space-y-3">
              <h3 className="text-xl font-bold text-slate-50 mb-4">🎤 Speech Emotion Recognition</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-400">Architecture:</span>
                  <span className="text-slate-200 font-medium">HuBERT Large</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Training Data:</span>
                  <span className="text-slate-200 font-medium">RAVDESS (1,440 files)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Emotions:</span>
                  <span className="text-slate-200 font-medium">8 classes</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Accuracy:</span>
                  <span className="text-green-400 font-bold">87.50%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Model Size:</span>
                  <span className="text-slate-200 font-medium">~360 MB</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Input:</span>
                  <span className="text-slate-200 font-medium">16kHz Mono</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* System Info */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
        <div className="bg-purple-900 px-4 py-2">
          <span className="text-purple-300 text-sm font-medium">💻 System Info</span>
        </div>
        <div className="p-6">
          <div className="space-y-3 text-sm">
            <div className="flex justify-between items-center">
              <span className="text-slate-400">Device:</span>
              <span className="text-slate-200 font-medium">CPU/GPU (Auto-detected)</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-400">Facial Model Status:</span>
              <span className="text-green-400 font-medium">✅ Loaded</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-400">Speech Model Status:</span>
              <span className="text-green-400 font-medium">✅ Loaded</span>
            </div>
          </div>
        </div>
      </div>

      {/* How to Use */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
        <div className="bg-purple-900 px-4 py-2">
          <span className="text-purple-300 text-sm font-medium">🎯 How to Use</span>
        </div>
        <div className="p-6">
          <div className="space-y-4 text-sm text-slate-300">
            <div>
              <h4 className="font-semibold text-slate-50 mb-2">📸 Facial Emotion Tab</h4>
              <p className="text-slate-400">
                Upload an image or use your webcam to capture a photo. The model will analyze facial expressions and predict emotions.
                Enable Grad-CAM to see which facial regions influenced the prediction.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-slate-50 mb-2">🎤 Speech Emotion Tab</h4>
              <p className="text-slate-400">
                Record audio or upload an audio file. The model will analyze voice tone and predict emotions.
                Enable Audio Saliency to see which frequencies were most important.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-slate-50 mb-2">🔗 Combined Analysis Tab</h4>
              <p className="text-slate-400">
                Upload both an image and audio file to analyze facial and vocal emotions simultaneously.
                The system will compare results and show if emotions match or differ.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============== MAIN APP COMPONENT ==============
function App() {
  const [activeTab, setActiveTab] = useState(0);

  const tabs = [
    { id: 0, emoji: '🎭', label: 'Facial Emotion' },
    { id: 1, emoji: '🎤', label: 'Speech Emotion' },
    { id: 2, emoji: '🔗', label: 'Combined Analysis' },
    { id: 3, emoji: '📊', label: 'Model Information' }
  ];

  return (
    <div className="min-h-screen bg-slate-900 text-slate-50">
      {/* Header */}
      <header className="bg-slate-900 border-b border-slate-700 px-6 py-8">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
            <span className="text-5xl">🎭</span>
            <span>Unified Facial + Voice Emotion Recognition</span>
          </h1>
          <p className="text-slate-400 text-lg mb-4">
            Test both facial expressions and voice tone simultaneously!
          </p>
          <div className="flex flex-wrap gap-4 text-sm">
            <div className="bg-purple-900 text-purple-300 px-3 py-1 rounded-full flex items-center gap-2">
              <span>📸 ViT</span>
              <span className="text-purple-200 font-semibold">71.29% accuracy</span>
            </div>
            <div className="bg-purple-900 text-purple-300 px-3 py-1 rounded-full flex items-center gap-2">
              <span>🎤 HuBERT</span>
              <span className="text-purple-200 font-semibold">87.50% accuracy</span>
            </div>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <div className="bg-slate-900 border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-2 overflow-x-auto">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-3 font-medium transition-all whitespace-nowrap ${
                  activeTab === tab.id
                    ? 'text-purple-400 border-b-2 border-purple-400'
                    : 'text-slate-400 hover:text-slate-300'
                }`}
              >
                <span className="mr-2">{tab.emoji}</span>
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === 0 && <FacialTab />}
        {activeTab === 1 && <SpeechTab />}
        {activeTab === 2 && <CombinedTab />}
        {activeTab === 3 && <ModelInfoTab />}
      </main>

      {/* Footer */}
      <footer className="bg-slate-900 border-t border-slate-700 px-6 py-6 mt-12">
        <div className="max-w-7xl mx-auto text-center text-slate-400 text-sm">
          <p>Use via API 🚀 · Built with Gradio 🍊 · Settings ⚙️</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
