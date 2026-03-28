import React, { useState, useRef, useEffect, useMemo } from 'react';
import { BrowserRouter, Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { jsPDF } from 'jspdf';
import { supabase } from './supabaseClient';
import { saveAnalysisToSupabase, loadAnalysisHistoryFromSupabase, updateAnalysisNote, toggleAnalysisPin, deleteAnalysisRecord } from './supabaseHistoryService';
import logoImage from './assets/logo.png';

const EMOTION_EMOJIS = {
  'angry': 'AN',
  'disgust': 'DI',
  'fear': 'FE',
  'happy': 'HA',
  'neutral': 'NE',
  'sad': 'SA',
  'surprise': 'SU',
  'calm': 'CA',
  'fearful': 'FE',
  'surprised': 'SU'
};

const EMOTIONS_FACIAL = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];
const EMOTIONS_SPEECH = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'];
const API_BASE = process.env.REACT_APP_API_BASE || 'http://127.0.0.1:8000';

const CONCORDANCE_SCORE_MAP = {
  MATCH: 100,
  PARTIAL: 65,
  MISMATCH: 30
};

function parseRecordTimestamp(row) {
  const time = new Date(row?.createdAt || 0).getTime();
  return Number.isFinite(time) ? time : 0;
}

function getConcordanceScore(row) {
  if (row?.concordance && CONCORDANCE_SCORE_MAP[row.concordance]) {
    return CONCORDANCE_SCORE_MAP[row.concordance];
  }
  const confidence = Number(row?.confidence || 0);
  return Math.max(0, Math.min(100, confidence * 100));
}

function splitMultimodalEmotions(row) {
  if (!row?.emotion || !String(row.emotion).includes('|')) return null;
  const [facial, speech] = String(row.emotion).split('|').map((item) => item.trim().toLowerCase());
  if (!facial || !speech) return null;
  return { facial, speech };
}

function formatPercent(value) {
  if (!Number.isFinite(value)) return '0%';
  return `${Math.round(value)}%`;
}

function formatDayMonth(dateObj) {
  return dateObj.toLocaleDateString(undefined, { day: '2-digit', month: 'short' });
}

function getSessionDatesSet(history) {
  const set = new Set();
  history.forEach((row) => {
    const d = new Date(parseRecordTimestamp(row));
    const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
    set.add(key);
  });
  return set;
}

function buildHeatmapData(history) {
  const now = new Date();
  now.setHours(0, 0, 0, 0);
  const start = new Date(now);
  start.setDate(now.getDate() - 83);

  const countsByDate = new Map();
  history.forEach((row) => {
    const d = new Date(parseRecordTimestamp(row));
    d.setHours(0, 0, 0, 0);
    if (d < start || d > now) return;
    const key = d.toISOString().slice(0, 10);
    countsByDate.set(key, (countsByDate.get(key) || 0) + 1);
  });

  const cells = Array.from({ length: 7 }, () => Array(12).fill(0));
  let maxCount = 0;

  for (let dayOffset = 0; dayOffset < 84; dayOffset += 1) {
    const d = new Date(start);
    d.setDate(start.getDate() + dayOffset);
    const key = d.toISOString().slice(0, 10);
    const count = countsByDate.get(key) || 0;
    const weekIndex = Math.floor(dayOffset / 7);
    const dayIndex = (d.getDay() + 6) % 7;
    cells[dayIndex][weekIndex] = count;
    if (count > maxCount) maxCount = count;
  }

  const toLevel = (count) => {
    if (!count || maxCount === 0) return 0;
    const ratio = count / maxCount;
    if (ratio < 0.25) return 1;
    if (ratio < 0.5) return 2;
    if (ratio < 0.75) return 3;
    return 4;
  };

  return cells.map((row) => row.map((count) => ({ count, level: toLevel(count) })));
}

function buildTrendSeries(history) {
  const now = new Date();
  const points = Array.from({ length: 12 }, (_, i) => {
    const end = new Date(now);
    end.setDate(now.getDate() - ((11 - i) * 7));
    const start = new Date(end);
    start.setDate(end.getDate() - 6);
    const weekRows = history.filter((row) => {
      const t = parseRecordTimestamp(row);
      return t >= start.getTime() && t <= end.getTime();
    });
    if (!weekRows.length) return 0;
    const avg = weekRows.reduce((sum, row) => sum + getConcordanceScore(row), 0) / weekRows.length;
    return Math.round(avg);
  });
  return points;
}

// ============== FACIAL EMOTION TAB ==============
function FacialTab({ onResult }) {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [emotion, setEmotion] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [probabilities, setProbabilities] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const cameraStreamRef = useRef(null);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [showGradCAM, setShowGradCAM] = useState(false);
  const [gradCam, setGradCam] = useState(null);
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const [faceDetected, setFaceDetected] = useState(false);

  useEffect(() => {
    if (isCameraOn && videoRef.current && cameraStreamRef.current) {
      videoRef.current.srcObject = cameraStreamRef.current;
    }
  }, [isCameraOn]);

  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = (evt) => {
        setImagePreview(evt.target.result);
        setEmotion(null);
        setGradCam(null);
        setAnnotatedImage(null);
        setFaceDetected(false);
      };
      reader.readAsDataURL(file);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      cameraStreamRef.current = stream;
      setIsCameraOn(true);
      setError(null);
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
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (cameraStreamRef.current) {
      cameraStreamRef.current.getTracks().forEach(track => track.stop());
      cameraStreamRef.current = null;
    }
    setIsCameraOn(false);
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
        setAnnotatedImage(response.data.annotated_image || null);
        setFaceDetected(Boolean(response.data.face_detected));
        if (response.data.grad_cam) {
          setGradCam(response.data.grad_cam);
        }
        if (onResult) {
          onResult({
            id: `facial-${Date.now()}`,
            modality: 'facial',
            emotion: response.data.emotion,
            confidence: response.data.confidence,
            probabilities: response.data.probabilities,
            explainability: response.data.grad_cam ? 'grad-cam' : 'none',
            createdAt: new Date().toISOString(),
            note: '',
            pinned: false
          });
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
            <span className="text-purple-300 text-sm font-medium">Image Source</span>
          </div>
          <div className="p-4">
            {!isCameraOn && !imagePreview && (
              <div className="border-2 border-dashed border-slate-700 rounded-lg p-8 text-center">
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
                    Capture
                  </button>
                  <button
                    onClick={stopCamera}
                    className="flex-1 bg-red-600 hover:bg-red-500 text-white px-4 py-2 rounded-lg font-medium transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
            {imagePreview && !isCameraOn && (
              <div>
                <img src={imagePreview} alt="Preview" className="w-full rounded-lg mb-3" />
                <label className="cursor-pointer inline-block w-full text-center">
                  <div className="text-slate-400 hover:text-slate-300 text-sm">
                    Change Image
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
                  Enable Grad-CAM Heatmap
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
            {loading ? 'Analyzing...' : 'Analyze Face'}
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
            <p className="text-slate-400">Waiting for input...</p>
          </div>
        )}

        {/* Confidence Scores */}
        {emotion && probabilities && (
          <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
            <div className="bg-purple-900 px-4 py-2 flex items-center gap-2">
              <span className="text-purple-300 text-sm font-medium">Confidence Scores</span>
            </div>
            <div className="p-4 space-y-3">
              <div className="text-center mb-4">
                <div className="text-xs font-bold tracking-[0.2em] text-slate-400 mb-2">{EMOTION_EMOJIS[emotion]}</div>
                <div className="text-2xl font-bold text-slate-50">{emotion.toUpperCase()}</div>
                <div className="text-lg text-green-400">{(confidence * 100).toFixed(1)}%</div>
              </div>
              {EMOTIONS_FACIAL.map((emo) => (
                <div key={emo} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-300">{EMOTION_EMOJIS[emo]} - {emo}</span>
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
        {emotion && (annotatedImage || imagePreview) && (
          <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
            <div className="bg-purple-900 px-4 py-2 flex items-center gap-2">
              <span className="text-purple-300 text-sm font-medium">Face Detection</span>
            </div>
            <div className="p-4">
              <p className="text-slate-400 text-sm mb-3">
                {faceDetected
                  ? 'Yellow box marks detected face before emotion reasoning.'
                  : 'No face box was detected; model used the full image.'}
              </p>
              <img src={annotatedImage ? `data:image/png;base64,${annotatedImage}` : imagePreview} alt="Annotated" className="w-full rounded-lg" />
            </div>
          </div>
        )}

        {/* Grad-CAM Heatmap */}
        {gradCam && (
          <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
            <div className="bg-purple-900 px-4 py-2 flex items-center gap-2">
              <span className="text-purple-300 text-sm font-medium">Grad-CAM Heatmap</span>
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
function SpeechTab({ onResult }) {
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
  const [waveform, setWaveform] = useState(null);

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
      setWaveform(null);
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
    setWaveform(null);
    try {
      const formData = new FormData();
      formData.append('file', audioFile);
      
      const explainParam = showSaliency ? '?explain=true' : '';
      const response = await axios.post(`${API_BASE}/api/predict/speech${explainParam}`, formData);
      if (response.data.success) {
        setEmotion(response.data.emotion);
        setConfidence(response.data.confidence);
        setProbabilities(response.data.probabilities);
        if (response.data.waveform) {
          setWaveform(response.data.waveform);
        }
        if (response.data.saliency) {
          setSaliency(response.data.saliency);
        }
        if (onResult) {
          onResult({
            id: `speech-${Date.now()}`,
            modality: 'speech',
            emotion: response.data.emotion,
            confidence: response.data.confidence,
            probabilities: response.data.probabilities,
            explainability: response.data.saliency ? 'saliency' : 'none',
            createdAt: new Date().toISOString(),
            note: '',
            pinned: false
          });
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
            <span className="text-purple-300 text-sm font-medium">Audio Source</span>
          </div>
          <div className="p-4">
            {!isRecording && !audioFile && (
              <div className="border-2 border-dashed border-slate-700 rounded-lg p-8 text-center">
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
                  Recording in progress...
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
                  Stop Recording
                </button>
              </div>
            )}
            {audioFile && !isRecording && (
              <div className="text-center">
                <div className="text-green-400 text-lg font-semibold mb-3">
                  Audio Ready
                </div>
                <div className="bg-slate-700 rounded-lg p-4 mb-3">
                  <div className="text-slate-300 text-sm">Audio file loaded</div>
                </div>
                <label className="cursor-pointer inline-block">
                  <div className="text-slate-400 hover:text-slate-300 text-sm">
                    Change Audio
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
                  Enable Audio Saliency Map
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
            {loading ? 'Analyzing...' : 'Analyze Voice'}
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
            <p className="text-slate-400">Waiting for input...</p>
          </div>
        )}

        {/* Confidence Scores */}
        {emotion && probabilities && (
          <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
            <div className="bg-purple-900 px-4 py-2 flex items-center gap-2">
              <span className="text-purple-300 text-sm font-medium">Confidence Scores</span>
            </div>
            <div className="p-4 space-y-3">
              <div className="text-center mb-4">
                <div className="text-xs font-bold tracking-[0.2em] text-slate-400 mb-2">{EMOTION_EMOJIS[emotion]}</div>
                <div className="text-2xl font-bold text-slate-50">{emotion.toUpperCase()}</div>
                <div className="text-lg text-green-400">{(confidence * 100).toFixed(1)}%</div>
              </div>
              {EMOTIONS_SPEECH.map((emo) => (
                <div key={emo} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-300">{EMOTION_EMOJIS[emo]} - {emo}</span>
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

        {/* Audio Waveform */}
        {waveform && (
          <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
            <div className="bg-purple-900 px-4 py-2 flex items-center gap-2">
              <span className="text-purple-300 text-sm font-medium">Audio Waveform</span>
            </div>
            <div className="p-4">
              <p className="text-slate-400 text-sm mb-3">
                Line plot of the uploaded audio signal over time.
              </p>
              <img
                src={`data:image/png;base64,${waveform}`}
                alt="Audio Waveform"
                className="w-full rounded-lg"
              />
            </div>
          </div>
        )}

        {/* Audio Saliency Map */}
        {saliency && (
          <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
            <div className="bg-purple-900 px-4 py-2 flex items-center gap-2">
              <span className="text-purple-300 text-sm font-medium">Audio Saliency Map</span>
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
function CombinedTab({ onResult }) {
  const [inputMode, setInputMode] = useState('separate');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [videoFile, setVideoFile] = useState(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState(null);

  const [isCameraOn, setIsCameraOn] = useState(false);
  const imageVideoRef = useRef(null);
  const imageCanvasRef = useRef(null);
  const imageCameraStreamRef = useRef(null);

  const [isAudioRecording, setIsAudioRecording] = useState(false);
  const audioRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioStreamRef = useRef(null);

  const [isVideoCameraOn, setIsVideoCameraOn] = useState(false);
  const [isVideoRecording, setIsVideoRecording] = useState(false);
  const videoLiveRef = useRef(null);
  const videoRecorderRef = useRef(null);
  const videoChunksRef = useRef([]);
  const videoStreamRef = useRef(null);

  const [facialEmotion, setFacialEmotion] = useState(null);
  const [speechEmotion, setSpeechEmotion] = useState(null);
  const [concordance, setConcordance] = useState(null);
  const [facialProbs, setFacialProbs] = useState(null);
  const [speechProbs, setSpeechProbs] = useState(null);
  const [videoResult, setVideoResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showExplainability, setShowExplainability] = useState(false);
  const [gradCam, setGradCam] = useState(null);
  const [saliency, setSaliency] = useState(null);
  const [waveform, setWaveform] = useState(null);
  const [annotatedFace, setAnnotatedFace] = useState(null);
  const [faceDetected, setFaceDetected] = useState(false);
  const [explainabilityStatus, setExplainabilityStatus] = useState(null);

  useEffect(() => {
    return () => {
      if (videoPreviewUrl) {
        URL.revokeObjectURL(videoPreviewUrl);
      }
      if (imageCameraStreamRef.current) {
        imageCameraStreamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (audioStreamRef.current) {
        audioStreamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (videoStreamRef.current) {
        videoStreamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, [videoPreviewUrl]);

  useEffect(() => {
    if (isCameraOn && imageVideoRef.current && imageCameraStreamRef.current) {
      imageVideoRef.current.srcObject = imageCameraStreamRef.current;
    }
  }, [isCameraOn]);

  useEffect(() => {
    if (isVideoCameraOn && videoLiveRef.current && videoStreamRef.current) {
      videoLiveRef.current.srcObject = videoStreamRef.current;
    }
  }, [isVideoCameraOn]);

  const resetResults = () => {
    setFacialEmotion(null);
    setSpeechEmotion(null);
    setConcordance(null);
    setFacialProbs(null);
    setSpeechProbs(null);
    setVideoResult(null);
    setGradCam(null);
    setSaliency(null);
    setWaveform(null);
    setAnnotatedFace(null);
    setFaceDetected(false);
    setExplainabilityStatus(null);
  };

  const switchMode = (mode) => {
    setInputMode(mode);
    setError(null);
    resetResults();
  };

  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = (evt) => {
        setImagePreview(evt.target.result);
        resetResults();
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAudioSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setAudioFile(file);
      resetResults();
    }
  };

  const startImageCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      imageCameraStreamRef.current = stream;
      setIsCameraOn(true);
      setError(null);
    } catch (err) {
      setError('Cannot access webcam for image capture');
    }
  };

  const stopImageCamera = () => {
    if (imageVideoRef.current) {
      imageVideoRef.current.srcObject = null;
    }
    if (imageCameraStreamRef.current) {
      imageCameraStreamRef.current.getTracks().forEach((track) => track.stop());
      imageCameraStreamRef.current = null;
    }
    setIsCameraOn(false);
  };

  const captureImage = () => {
    if (!imageVideoRef.current || !imageCanvasRef.current) return;
    const ctx = imageCanvasRef.current.getContext('2d');
    ctx.drawImage(imageVideoRef.current, 0, 0, imageCanvasRef.current.width, imageCanvasRef.current.height);
    imageCanvasRef.current.toBlob((blob) => {
      if (!blob) return;
      const captured = new File([blob], `capture-${Date.now()}.png`, { type: 'image/png' });
      setImageFile(captured);
      const reader = new FileReader();
      reader.onload = (evt) => {
        setImagePreview(evt.target.result);
        resetResults();
      };
      reader.readAsDataURL(captured);
      stopImageCamera();
    }, 'image/png');
  };

  const startAudioRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioStreamRef.current = stream;
      audioChunksRef.current = [];
      const recorder = new MediaRecorder(stream);
      audioRecorderRef.current = recorder;
      recorder.ondataavailable = (e) => audioChunksRef.current.push(e.data);
      recorder.onstop = () => {
        const mimeType = audioChunksRef.current[0]?.type || 'audio/webm';
        const blob = new Blob(audioChunksRef.current, { type: mimeType });
        const ext = mimeType.includes('ogg') ? 'ogg' : 'webm';
        const recorded = new File([blob], `recorded-audio-${Date.now()}.${ext}`, { type: mimeType });
        setAudioFile(recorded);
        resetResults();
      };
      recorder.start();
      setIsAudioRecording(true);
      setError(null);
    } catch (err) {
      setError('Cannot access microphone');
    }
  };

  const stopAudioRecording = () => {
    if (audioRecorderRef.current && isAudioRecording) {
      audioRecorderRef.current.stop();
      setIsAudioRecording(false);
      if (audioStreamRef.current) {
        audioStreamRef.current.getTracks().forEach((track) => track.stop());
        audioStreamRef.current = null;
      }
    }
  };

  const handleVideoSelect = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
    setVideoFile(file);
    setVideoPreviewUrl(URL.createObjectURL(file));
    resetResults();
  };

  const startVideoCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      videoStreamRef.current = stream;
      setIsVideoCameraOn(true);
      setError(null);
    } catch (err) {
      setError('Cannot access camera/microphone for video recording');
    }
  };

  const stopVideoCamera = () => {
    if (videoLiveRef.current) {
      videoLiveRef.current.srcObject = null;
    }
    if (videoStreamRef.current) {
      videoStreamRef.current.getTracks().forEach((track) => track.stop());
      videoStreamRef.current = null;
    }
    setIsVideoCameraOn(false);
    setIsVideoRecording(false);
  };

  const startVideoRecording = () => {
    if (!videoStreamRef.current) return;
    videoChunksRef.current = [];
    const recorder = new MediaRecorder(videoStreamRef.current);
    videoRecorderRef.current = recorder;
    recorder.ondataavailable = (e) => videoChunksRef.current.push(e.data);
    recorder.onstop = () => {
      const mimeType = videoChunksRef.current[0]?.type || 'video/webm';
      const blob = new Blob(videoChunksRef.current, { type: mimeType });
      const ext = mimeType.includes('mp4') ? 'mp4' : 'webm';
      const recorded = new File([blob], `recorded-video-${Date.now()}.${ext}`, { type: mimeType });
      if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
      setVideoFile(recorded);
      setVideoPreviewUrl(URL.createObjectURL(recorded));
      resetResults();
    };
    recorder.start();
    setIsVideoRecording(true);
  };

  const stopVideoRecording = () => {
    if (videoRecorderRef.current && isVideoRecording) {
      videoRecorderRef.current.stop();
      setIsVideoRecording(false);
    }
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
    setWaveform(null);
    setAnnotatedFace(null);
    setFaceDetected(false);
    setExplainabilityStatus(null);
    try {
      const formData = new FormData();
      formData.append('image_file', imageFile);
      formData.append('audio_file', audioFile);
      
      const explainParam = showExplainability ? '?explain=true' : '';
      const response = await axios.post(`${API_BASE}/api/predict/combined${explainParam}`, formData);
      if (response.data.success) {
        setFacialEmotion(response.data.facial_emotion.emotion);
        setSpeechEmotion(response.data.speech_emotion.emotion);
        setConcordance(response.data.concordance);
        setFacialProbs(response.data.facial_emotion.probabilities);
        setSpeechProbs(response.data.speech_emotion.probabilities);
        setAnnotatedFace(response.data.facial_emotion.annotated_image || null);
        setFaceDetected(Boolean(response.data.facial_emotion.face_detected));
        
        if (response.data.explainability) {
          if (response.data.explainability.grad_cam) {
            setGradCam(response.data.explainability.grad_cam);
          }
          if (response.data.explainability.saliency) {
            setSaliency(response.data.explainability.saliency);
          }
          if (response.data.explainability.waveform) {
            setWaveform(response.data.explainability.waveform);
          }
        }
        if (response.data.explainability_status) {
          setExplainabilityStatus(response.data.explainability_status);
        }
        if (onResult) {
          onResult({
            id: `combined-${Date.now()}`,
            modality: 'multimodal',
            emotion: `${response.data.facial_emotion.emotion} | ${response.data.speech_emotion.emotion}`,
            confidence: Math.max(
              response.data.facial_emotion.confidence || 0,
              response.data.speech_emotion.confidence || 0
            ),
            probabilities: {
              facial: response.data.facial_emotion.probabilities,
              speech: response.data.speech_emotion.probabilities
            },
            explainability: response.data.explainability ? 'enabled' : 'none',
            concordance: response.data.concordance,
            createdAt: new Date().toISOString(),
            note: '',
            pinned: false
          });
        }
      }
    } catch (err) {
      setError('API Error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const analyzeVideo = async () => {
    if (!videoFile) {
      setError('Please upload or record a video first');
      return;
    }
    setLoading(true);
    setError(null);
    setGradCam(null);
    setSaliency(null);
    setWaveform(null);
    setAnnotatedFace(null);
    setFaceDetected(false);
    setVideoResult(null);
    setExplainabilityStatus(null);
    try {
      const formData = new FormData();
      formData.append('file', videoFile, videoFile.name || `video-${Date.now()}.webm`);
      const explainParam = showExplainability ? '?explain=true' : '';
      const response = await axios.post(`${API_BASE}/api/predict/video${explainParam}`, formData);
      if (response.data.success) {
        const face = response.data.facial_emotion?.emotion || 'unknown';
        const speech = response.data.speech_emotion?.emotion || 'unknown';
        const videoConcordance = face === speech ? 'MATCH' : 'MISMATCH';

        setVideoResult(response.data);
        setFacialEmotion(face);
        setSpeechEmotion(speech);
        setConcordance(videoConcordance);
        setFacialProbs(response.data.facial_emotion?.probabilities || null);
        setSpeechProbs(response.data.speech_emotion?.probabilities || null);

        if (response.data.explainability) {
          if (response.data.explainability.grad_cam) {
            setGradCam(response.data.explainability.grad_cam);
          }
          if (response.data.explainability.saliency) {
            setSaliency(response.data.explainability.saliency);
          }
          if (response.data.explainability.waveform) {
            setWaveform(response.data.explainability.waveform);
          }
        }
        if (response.data.explainability_status) {
          setExplainabilityStatus(response.data.explainability_status);
        }

        if (onResult) {
          onResult({
            id: `video-${Date.now()}`,
            modality: 'multimodal',
            emotion: `${response.data.combined_emotion || 'unknown'} (video)`,
            confidence: Math.max(
              response.data.facial_emotion?.confidence || 0,
              response.data.speech_emotion?.confidence || 0
            ),
            probabilities: {
              facial: response.data.facial_emotion?.probabilities || {},
              speech: response.data.speech_emotion?.probabilities || {}
            },
            explainability: showExplainability ? 'requested' : 'none',
            concordance: videoConcordance,
            createdAt: new Date().toISOString(),
            note: '',
            pinned: false
          });
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
      <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
        <h3 className="text-slate-50 font-semibold mb-1">Analyze Face and Voice Together</h3>
        <p className="text-slate-400 text-sm">
          Choose one mode: use separate image + audio inputs (upload or live capture), or analyze a single uploaded/recorded video.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => switchMode('separate')}
          className={`ga-header-btn ${inputMode === 'separate' ? 'active' : ''}`}
        >
          📷 + 🎤 Separate Inputs
        </button>
        <button
          type="button"
          onClick={() => switchMode('video')}
          className={`ga-header-btn ${inputMode === 'video' ? 'active' : ''}`}
        >
          🎥 Video Input
        </button>
      </div>

      {/* Input Section */}
      {inputMode === 'separate' && (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Image Upload */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
          <div className="bg-purple-900 px-4 py-2 flex items-center gap-2">
            <span className="text-purple-300 text-sm font-medium">Image Input</span>
          </div>
          <div className="p-4">
            {!imagePreview && !isCameraOn && (
              <div className="space-y-3">
                <label className="cursor-pointer block">
                  <div className="border-2 border-dashed border-slate-700 rounded-lg p-8 text-center hover:border-slate-600 transition-colors">
                    <p className="text-slate-400 mb-2">Upload Image</p>
                    <p className="text-slate-500 text-sm">Drop Image Here or Click to Upload</p>
                  </div>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageSelect}
                    className="hidden"
                  />
                </label>
                <button
                  type="button"
                  onClick={startImageCamera}
                  className="w-full bg-slate-700 hover:bg-slate-600 text-slate-100 px-4 py-2 rounded-lg"
                >
                  Capture with Webcam
                </button>
              </div>
            )}

            {isCameraOn && (
              <div>
                <video ref={imageVideoRef} autoPlay playsInline className="w-full rounded-lg mb-3" />
                <canvas ref={imageCanvasRef} width="640" height="480" className="hidden" />
                <div className="flex gap-2">
                  <button type="button" onClick={captureImage} className="flex-1 bg-green-600 hover:bg-green-500 text-white px-4 py-2 rounded-lg">Capture</button>
                  <button type="button" onClick={stopImageCamera} className="flex-1 bg-red-600 hover:bg-red-500 text-white px-4 py-2 rounded-lg">Cancel</button>
                </div>
              </div>
            )}

            {imagePreview && !isCameraOn && (
              <div>
                <img src={imagePreview} alt="Preview" className="w-full rounded-lg mb-3" />
                <label className="cursor-pointer block text-center">
                  <div className="text-slate-400 hover:text-slate-300 text-sm">
                    Change Image
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
            <span className="text-purple-300 text-sm font-medium">Audio Input</span>
          </div>
          <div className="p-4">
            {!audioFile && !isAudioRecording && (
              <div className="space-y-3">
                <label className="cursor-pointer block">
                  <div className="border-2 border-dashed border-slate-700 rounded-lg p-8 text-center hover:border-slate-600 transition-colors">
                    <p className="text-slate-400 mb-2">Upload Audio</p>
                    <p className="text-slate-500 text-sm">Drop Audio Here or Click to Upload</p>
                  </div>
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={handleAudioSelect}
                    className="hidden"
                  />
                </label>
                <button
                  type="button"
                  onClick={startAudioRecording}
                  className="w-full bg-slate-700 hover:bg-slate-600 text-slate-100 px-4 py-2 rounded-lg"
                >
                  Record Live with Mic
                </button>
              </div>
            )}

            {isAudioRecording && (
              <div className="text-center">
                <div className="text-red-400 font-semibold mb-3">Recording...</div>
                <button type="button" onClick={stopAudioRecording} className="bg-red-600 hover:bg-red-500 text-white px-4 py-2 rounded-lg">
                  Stop Recording
                </button>
              </div>
            )}

            {audioFile && !isAudioRecording && (
              <div className="text-center">
                <div className="bg-slate-700 rounded-lg p-8 mb-3">
                  <div className="text-green-400 font-semibold">Audio Loaded</div>
                </div>
                <label className="cursor-pointer block">
                  <div className="text-slate-400 hover:text-slate-300 text-sm">
                    Change Audio
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
      )}

      {inputMode === 'video' && (
        <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
          <div className="bg-purple-900 px-4 py-2 flex items-center gap-2">
            <span className="text-purple-300 text-sm font-medium">Video Input</span>
          </div>
          <div className="p-4 space-y-4">
            <p className="text-slate-400 text-sm">Upload a video with face + voice, or record one live using webcam and microphone.</p>
            {!isVideoCameraOn && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                <label className="cursor-pointer block">
                  <div className="border-2 border-dashed border-slate-700 rounded-lg p-8 text-center hover:border-slate-600 transition-colors">
                    <p className="text-slate-400 mb-2">Upload Video</p>
                    <p className="text-slate-500 text-sm">Drop Video Here or Click to Upload</p>
                  </div>
                  <input type="file" accept="video/*" onChange={handleVideoSelect} className="hidden" />
                </label>
                <button
                  type="button"
                  onClick={startVideoCamera}
                  className="bg-slate-700 hover:bg-slate-600 text-slate-100 px-4 py-2 rounded-lg"
                >
                  Record Live Video (Cam + Mic)
                </button>
              </div>
            )}

            {isVideoCameraOn && (
              <div>
                <video ref={videoLiveRef} autoPlay playsInline muted className="w-full rounded-lg mb-3" />
                <div className="flex flex-wrap gap-2">
                  {!isVideoRecording ? (
                    <button type="button" onClick={startVideoRecording} className="bg-green-600 hover:bg-green-500 text-white px-4 py-2 rounded-lg">
                      Start Recording
                    </button>
                  ) : (
                    <button type="button" onClick={stopVideoRecording} className="bg-red-600 hover:bg-red-500 text-white px-4 py-2 rounded-lg">
                      Stop Recording
                    </button>
                  )}
                  <button type="button" onClick={stopVideoCamera} className="bg-slate-600 hover:bg-slate-500 text-white px-4 py-2 rounded-lg">
                    Close Camera
                  </button>
                </div>
              </div>
            )}

            {videoPreviewUrl && (
              <div>
                <video src={videoPreviewUrl} controls className="w-full rounded-lg" />
                <div className="text-slate-400 text-xs mt-2">Video ready for multimodal analysis.</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Explainability Options */}
      {((inputMode === 'separate' && imageFile && audioFile) || (inputMode === 'video' && videoFile)) && (
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
          <label className="flex items-start gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={showExplainability}
              onChange={(e) => setShowExplainability(e.target.checked)}
              className="mt-1 w-5 h-5 rounded border-slate-600 bg-slate-700 text-purple-600 focus:ring-purple-500"
            />
            <div>
              <div className="text-slate-50 font-medium">Enable Explainability Output</div>
              <div className="text-slate-400 text-sm mt-1">
                Returns Grad-CAM and/or saliency visualizations when supported by the selected analysis mode.
              </div>
            </div>
          </label>
        </div>
      )}

      {/* Analyze Button */}
      {inputMode === 'separate' && imageFile && audioFile && (
        <button
          onClick={analyzeCombined}
          disabled={loading}
          className="w-full bg-gradient-to-br from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white px-6 py-3 rounded-lg font-medium text-lg transition-all duration-200 hover:shadow-lg disabled:opacity-50"
        >
          {loading ? 'Analyzing...' : '🚀 Analyze Both'}
        </button>
      )}

      {inputMode === 'video' && videoFile && (
        <button
          onClick={analyzeVideo}
          disabled={loading}
          className="w-full bg-gradient-to-br from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white px-6 py-3 rounded-lg font-medium text-lg transition-all duration-200 hover:shadow-lg disabled:opacity-50"
        >
          {loading ? 'Analyzing...' : '🚀 Analyze Video'}
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
          <div className={`rounded-xl p-3 text-sm font-medium ${
            showExplainability
              ? (gradCam || saliency)
                ? 'bg-green-900/40 border border-green-700 text-green-200'
                : 'bg-amber-900/40 border border-amber-700 text-amber-200'
              : 'bg-slate-800 border border-slate-700 text-slate-300'
          }`}>
            {showExplainability
              ? (gradCam || saliency)
                ? 'Explainability generated successfully.'
                : explainabilityStatus?.errors?.length
                  ? `Explainability requested, but map generation failed: ${explainabilityStatus.errors.join(' | ')}`
                  : 'Explainability was requested but no maps were returned for this input.'
              : 'Explainability is off. Enable the toggle to request Grad-CAM and saliency outputs.'}
          </div>

          {/* Concordance Banner */}
          <div className={`rounded-xl p-4 text-center font-semibold text-lg ${
            concordance === 'MATCH' 
              ? 'bg-green-900/50 border border-green-700 text-green-200' 
              : 'bg-yellow-900/50 border border-yellow-700 text-yellow-200'
          }`}>
            {concordance === 'MATCH' ? 'Emotions Match' : 'Emotions Differ'}
          </div>

          {/* Results Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Facial Results */}
            <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
              <div className="bg-purple-900 px-4 py-2">
                <span className="text-purple-300 text-sm font-medium">Facial Emotion</span>
              </div>
              <div className="p-4">
                <div className="text-center mb-4">
                  <div className="text-xs font-bold tracking-[0.2em] text-slate-400 mb-2">{EMOTION_EMOJIS[facialEmotion]}</div>
                  <div className="text-xl font-bold text-slate-50">{facialEmotion.toUpperCase()}</div>
                </div>
                {facialProbs && (
                  <div className="space-y-2">
                    {Object.entries(facialProbs).slice(0, 4).map(([emo, prob]) => (
                      <div key={emo} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-300">{EMOTION_EMOJIS[emo]} - {emo}</span>
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
                <span className="text-purple-300 text-sm font-medium">Speech Emotion</span>
              </div>
              <div className="p-4">
                <div className="text-center mb-4">
                  <div className="text-xs font-bold tracking-[0.2em] text-slate-400 mb-2">{EMOTION_EMOJIS[speechEmotion]}</div>
                  <div className="text-xl font-bold text-slate-50">{speechEmotion.toUpperCase()}</div>
                </div>
                {speechProbs && (
                  <div className="space-y-2">
                    {Object.entries(speechProbs).slice(0, 4).map(([emo, prob]) => (
                      <div key={emo} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-300">{EMOTION_EMOJIS[emo]} - {emo}</span>
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
          {inputMode === 'separate' && (annotatedFace || imagePreview) && (
            <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
              <div className="bg-purple-900 px-4 py-2">
                <span className="text-purple-300 text-sm font-medium">Face Detection</span>
              </div>
              <div className="p-4">
                <p className="text-slate-400 text-sm mb-3">
                  {faceDetected
                    ? 'Detected face is boxed before explainability is computed.'
                    : 'Face box not found; full image was analyzed.'}
                </p>
                <img src={annotatedFace ? `data:image/png;base64,${annotatedFace}` : imagePreview} alt="Annotated" className="w-full rounded-lg" />
              </div>
            </div>
          )}

          {/* Explainability Visualizations */}
          {(gradCam || saliency || waveform) && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {gradCam && (
                <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
                  <div className="bg-purple-900 px-4 py-2">
                    <span className="text-purple-300 text-sm font-medium">Facial Grad-CAM</span>
                  </div>
                  <div className="p-4">
                    <img src={`data:image/png;base64,${gradCam}`} alt="Grad-CAM" className="w-full rounded-lg" />
                  </div>
                </div>
              )}
              {waveform && (
                <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
                  <div className="bg-purple-900 px-4 py-2">
                    <span className="text-purple-300 text-sm font-medium">Audio Waveform</span>
                  </div>
                  <div className="p-4">
                    <img src={`data:image/png;base64,${waveform}`} alt="Waveform" className="w-full rounded-lg" />
                  </div>
                </div>
              )}
              {saliency && (
                <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
                  <div className="bg-purple-900 px-4 py-2">
                    <span className="text-purple-300 text-sm font-medium">Audio Saliency</span>
                  </div>
                  <div className="p-4">
                    <img src={`data:image/png;base64,${saliency}`} alt="Saliency" className="w-full rounded-lg" />
                  </div>
                </div>
              )}
            </div>
          )}

          {inputMode === 'video' && videoResult && (
            <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
              <div className="bg-purple-900 px-4 py-2">
                <span className="text-purple-300 text-sm font-medium">Video Analysis Details</span>
              </div>
              <div className="p-4 text-sm text-slate-300 space-y-2">
                <div>Combined Emotion: <span className="text-slate-100 font-semibold">{videoResult.combined_emotion}</span></div>
                <div>Frames Processed: <span className="text-slate-100 font-semibold">{videoResult.frames_processed}</span></div>
                <div>Duration: <span className="text-slate-100 font-semibold">{Number(videoResult.video_duration || 0).toFixed(2)}s</span></div>
                <div>FPS: <span className="text-slate-100 font-semibold">{Number(videoResult.fps || 0).toFixed(2)}</span></div>
              </div>
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
          <span className="text-purple-300 text-sm font-medium">Model Details</span>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Facial Model */}
            <div className="space-y-3">
              <h3 className="text-xl font-bold text-slate-50 mb-4">Facial Emotion Recognition</h3>
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
                  <span className="text-slate-200 font-medium">224x224 RGB</span>
                </div>
              </div>
            </div>

            {/* Speech Model */}
            <div className="space-y-3">
              <h3 className="text-xl font-bold text-slate-50 mb-4">Speech Emotion Recognition</h3>
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
          <span className="text-purple-300 text-sm font-medium">System Info</span>
        </div>
        <div className="p-6">
          <div className="space-y-3 text-sm">
            <div className="flex justify-between items-center">
              <span className="text-slate-400">Device:</span>
              <span className="text-slate-200 font-medium">CPU/GPU (Auto-detected)</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-400">Facial Model Status:</span>
              <span className="text-green-400 font-medium">Loaded</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-400">Speech Model Status:</span>
              <span className="text-green-400 font-medium">Loaded</span>
            </div>
          </div>
        </div>
      </div>

      {/* How to Use */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
        <div className="bg-purple-900 px-4 py-2">
          <span className="text-purple-300 text-sm font-medium">How to Use</span>
        </div>
        <div className="p-6">
          <div className="space-y-4 text-sm text-slate-300">
            <div>
              <h4 className="font-semibold text-slate-50 mb-2">Facial Emotion Tab</h4>
              <p className="text-slate-400">
                Upload an image or use your webcam to capture a photo. The model will analyze facial expressions and predict emotions.
                Enable Grad-CAM to see which facial regions influenced the prediction.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-slate-50 mb-2">Speech Emotion Tab</h4>
              <p className="text-slate-400">
                Record audio or upload an audio file. The model will analyze voice tone and predict emotions.
                Enable Audio Saliency to see which frequencies were most important.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-slate-50 mb-2">Combined Analysis Tab</h4>
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
const AUTH_STORAGE_KEY = 'mmer_auth_user';

function AuthPage({ mode = 'login', onAuthSuccess }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const isSignup = mode === 'signup';

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email || !password || (isSignup && !name)) {
      setError('Please fill all required fields.');
      return;
    }

    setLoading(true);
    setError('');

    try {
      if (isSignup) {
        const { data, error: signUpError } = await supabase.auth.signUp({
          email,
          password,
          options: {
            data: { name: name || email.split('@')[0] }
          }
        });
        if (signUpError) {
          setError(signUpError.message);
          setLoading(false);
          return;
        }
        if (data?.user) {
          onAuthSuccess({ email: data.user.email, name: name || email.split('@')[0] });
          navigate('/app/dashboard', { replace: true });
        }
      } else {
        const { data, error: signInError } = await supabase.auth.signInWithPassword({ email, password });
        if (signInError) {
          setError(signInError.message);
          setLoading(false);
          return;
        }
        if (data?.user) {
          onAuthSuccess({ email: data.user.email, name: data.user.user_metadata?.name || email.split('@')[0] });
          navigate('/app/dashboard', { replace: true });
        }
      }
    } catch (err) {
      setError('An unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="ga-login">
      <form onSubmit={handleSubmit} className="ga-login-card">
        <div className="ga-login-title">{isSignup ? 'Create account' : 'Welcome back'}</div>
        <div className="ga-login-subtitle">
          {isSignup ? 'Start your multimodal analytics workspace' : 'Sign in to continue to your workspace'}
        </div>

        {isSignup && (
          <>
            <label className="ga-field-label">Full name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Your name"
              className="ga-input"
            />
          </>
        )}

        <label className="ga-field-label">Email</label>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="analyst@company.com"
          className="ga-input"
        />

        <label className="ga-field-label">Password</label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Enter password"
          className="ga-input"
        />

        {error && <div className="ga-error">{error}</div>}

        <button type="submit" className="ga-primary-btn" disabled={loading}>
          {loading ? 'Processing...' : (isSignup ? 'Create Account' : 'Sign In')}
        </button>

        <div className="ga-auth-footer-text">
          {isSignup ? 'Already have an account?' : 'Need an account?'}{' '}
          <button
            type="button"
            className="ga-text-btn"
            onClick={() => navigate(isSignup ? '/login' : '/signup')}
          >
            {isSignup ? 'Login' : 'Get Started'}
          </button>
        </div>
      </form>
    </div>
  );
}

function MarketingPage() {
  const navigate = useNavigate();
  const openExternal = (url) => window.open(url, '_blank', 'noopener,noreferrer');

  return (
    <div className="mmer-marketing">
      <header className="mmer-top-nav">
        <div className="mmer-logo-wrap">
          <img src={logoImage} alt="MMER logo" className="mmer-logo-mark" />
          <div className="mmer-logo">MMER Platform</div>
        </div>
        <div className="mmer-nav-actions">
          <button className="mmer-link-btn" onClick={() => navigate('/login')}>Login</button>
          <button className="mmer-primary-btn" onClick={() => navigate('/signup')}>Get Started</button>
        </div>
      </header>

      <section className="mmer-section mmer-hero">
        <p className="mmer-eyebrow">Multimodal Emotion Recognition</p>
        <h1>If Someone Says "I'm Fine." Are They Really?</h1>
        <p>
          MMER analyzes facial expressions and vocal tone at the same time, then computes a concordance score to show whether both signals align.
          It is built for explainable, privacy-first emotion intelligence.
        </p>
        <div className="mmer-trust-strip">200+ early-access signups from research and industry teams.</div>
        <div className="mmer-hero-actions">
          <button className="mmer-primary-btn" onClick={() => openExternal('https://huggingface.co/spaces/Nishvaraj/MMER')}>Try Demo Free</button>
          <button className="mmer-link-btn" onClick={() => openExternal('https://sites.google.com/view/mmer-webapp/how-it-works')}>See How It Works</button>
          <button className="mmer-link-btn" onClick={() => navigate('/signup')}>Create Workspace</button>
        </div>
      </section>

      <section className="mmer-section">
        <h2>What Makes MMER Different?</h2>
        <div className="mmer-feature-grid">
          <article className="mmer-feature-card">
            <h3>Dual Modality Analysis</h3>
            <p>Vision Transformer + HuBERT run together so face and voice are analyzed in one pass, not in isolation.</p>
          </article>
          <article className="mmer-feature-card">
            <h3>Novel Concordance Metric</h3>
            <p>Measure emotional authenticity by comparing face and voice agreement. Higher concordance means stronger alignment.</p>
          </article>
          <article className="mmer-feature-card">
            <h3>Privacy by Architecture</h3>
            <p>Designed for local-first processing with explainable outputs and no mandatory cloud storage workflow.</p>
          </article>
        </div>
      </section>

      <section className="mmer-section">
        <h2>Benchmarks and Coverage</h2>
        <div className="mmer-stat-grid">
          <article className="mmer-stat-card">
            <div className="mmer-stat-value">87.50%</div>
            <div className="mmer-stat-label">Speech Accuracy</div>
            <div className="mmer-stat-meta">HuBERT · RAVDESS</div>
          </article>
          <article className="mmer-stat-card">
            <div className="mmer-stat-value">71.29%</div>
            <div className="mmer-stat-label">Facial Accuracy</div>
            <div className="mmer-stat-meta">ViT · FER-2013</div>
          </article>
          <article className="mmer-stat-card">
            <div className="mmer-stat-value">7</div>
            <div className="mmer-stat-label">Emotion Classes</div>
            <div className="mmer-stat-meta">Happy · Sad · Angry · Fear · Disgust · Surprise · Neutral</div>
          </article>
          <article className="mmer-stat-card">
            <div className="mmer-stat-value">Novel</div>
            <div className="mmer-stat-label">Concordance Metric</div>
            <div className="mmer-stat-meta">First open-access implementation in this workflow</div>
          </article>
        </div>
      </section>

      <section className="mmer-section">
        <h2>Who Is MMER For?</h2>
        <div className="mmer-feature-grid">
          <article className="mmer-feature-card">
            <h3>Mental Health Professionals</h3>
            <p>Support therapy observations with an objective concordance signal and explainable model outputs.</p>
          </article>
          <article className="mmer-feature-card">
            <h3>HR and Wellness Teams</h3>
            <p>Enable consent-based emotional self-reflection workflows for coaching and leadership development.</p>
          </article>
          <article className="mmer-feature-card">
            <h3>Academic Researchers</h3>
            <p>Use open, reproducible multimodal analysis with visualization outputs suitable for technical reporting.</p>
          </article>
        </div>
      </section>

      <section className="mmer-section mmer-cta">
        <h2>Join the Early Access Queue</h2>
        <p>Receive development updates and priority access to new multimodal features.</p>
        <div className="mmer-hero-actions">
          <button className="mmer-primary-btn" onClick={() => openExternal('https://forms.gle/b5g3245J4Y4Ta3M37')}>Early Access</button>
          <button className="mmer-link-btn" onClick={() => navigate('/login')}>Sign In</button>
        </div>
      </section>
    </div>
  );
}

function DashboardIcon({ name, className = '' }) {
  const iconClass = `ga-glyph ${className}`.trim();
  switch (name) {
    case 'overview':
      return (
        <svg viewBox="0 0 24 24" className={iconClass} aria-hidden="true">
          <path d="M3 12l9-8 9 8" />
          <path d="M6 10v10h12V10" />
        </svg>
      );
    case 'facial':
      return (
        <svg viewBox="0 0 24 24" className={iconClass} aria-hidden="true">
          <rect x="3" y="5" width="18" height="14" rx="3" />
          <circle cx="12" cy="12" r="3.5" />
          <path d="M7.5 9.5h.01M16.5 9.5h.01" />
        </svg>
      );
    case 'speech':
      return (
        <svg viewBox="0 0 24 24" className={iconClass} aria-hidden="true">
          <rect x="9" y="3" width="6" height="11" rx="3" />
          <path d="M5 11a7 7 0 0014 0" />
          <path d="M12 18v3" />
          <path d="M8.5 21h7" />
        </svg>
      );
    case 'multimodal':
      return (
        <svg viewBox="0 0 24 24" className={iconClass} aria-hidden="true">
          <rect x="4" y="4" width="7" height="7" rx="2" />
          <rect x="13" y="4" width="7" height="7" rx="2" />
          <rect x="4" y="13" width="7" height="7" rx="2" />
          <path d="M14 14h6" />
          <path d="M17 11v6" />
        </svg>
      );
    case 'model':
      return (
        <svg viewBox="0 0 24 24" className={iconClass} aria-hidden="true">
          <rect x="7" y="7" width="10" height="10" rx="2" />
          <path d="M12 1v4M12 19v4M1 12h4M19 12h4M4.5 4.5l2.8 2.8M16.7 16.7l2.8 2.8M19.5 4.5l-2.8 2.8M7.3 16.7l-2.8 2.8" />
        </svg>
      );
    case 'history':
      return (
        <svg viewBox="0 0 24 24" className={iconClass} aria-hidden="true">
          <path d="M3 12a9 9 0 109-9" />
          <path d="M3 4v8h8" />
          <path d="M12 7v5l3 2" />
        </svg>
      );
    case 'users':
      return (
        <svg viewBox="0 0 24 24" className={iconClass} aria-hidden="true">
          <circle cx="9" cy="8" r="3" />
          <path d="M3.5 18a5.5 5.5 0 0111 0" />
          <circle cx="17" cy="9" r="2" />
          <path d="M15 18a4 4 0 014 0" />
        </svg>
      );
    case 'sessions':
      return (
        <svg viewBox="0 0 24 24" className={iconClass} aria-hidden="true">
          <rect x="3" y="5" width="18" height="14" rx="2" />
          <path d="M3 10h18" />
          <path d="M8 15h3M13 15h3" />
        </svg>
      );
    case 'conversion':
      return (
        <svg viewBox="0 0 24 24" className={iconClass} aria-hidden="true">
          <path d="M4 6h16" />
          <path d="M7 6v12h10V6" />
          <path d="M9 10l2.5 2.5L15 9" />
        </svg>
      );
    case 'bounce':
      return (
        <svg viewBox="0 0 24 24" className={iconClass} aria-hidden="true">
          <path d="M5 6h14" />
          <path d="M7 6v12h10" />
          <path d="M11 13l-3 3 3 3" />
          <path d="M8 16h8" />
        </svg>
      );
    default:
      return (
        <svg viewBox="0 0 24 24" className={iconClass} aria-hidden="true">
          <circle cx="12" cy="12" r="9" />
        </svg>
      );
  }
}

function KpiCard({ title, value, detail, detailClass = '' }) {
  return (
    <article className="ga-kpi-card ga-kpi-summary-card">
      <div className="ga-kpi-title">{title}</div>
      <div className="ga-kpi-value">{value}</div>
      <div className={`ga-kpi-detail ${detailClass}`}>{detail}</div>
    </article>
  );
}

function ActivityHeatmapCard({ heatmapData }) {
  return (
    <article className="ga-card ga-heatmap-card">
      <h3 className="ga-section-title ga-heatmap-title">Activity Heatmap · Last 12 Weeks</h3>
      <div className="ga-heatmap-grid" role="img" aria-label="Activity heatmap for the last 12 weeks">
        {heatmapData.flatMap((row, rowIdx) =>
          row.map((cell, colIdx) => (
            <span
              key={`${rowIdx}-${colIdx}`}
              className={`ga-heat-cell level-${cell.level}`}
              title={`Week ${colIdx + 1}, day ${rowIdx + 1}: ${cell.count} session${cell.count === 1 ? '' : 's'}`}
            />
          ))
        )}
      </div>
      <div className="ga-heatmap-scale">
        <span>12 weeks ago</span>
        <span>This week</span>
      </div>
    </article>
  );
}

function InsightCard({ icon, title, description }) {
  return (
    <article className="ga-card ga-insight-card">
      <div className="ga-insight-icon">{icon}</div>
      <h4 className="ga-insight-title">{title}</h4>
      <p className="ga-insight-text">{description}</p>
    </article>
  );
}

function OverviewTab({ history, analytics }) {
  const {
    totalSessions,
    monthDelta,
    avgConcordance,
    weeklyConcordanceDelta,
    bestSessionScore,
    bestSessionDate,
    streakDays,
    heatmapData,
    bestEmotionInsight,
    worstEmotionInsight
  } = analytics;

  const weekDeltaText = `${weeklyConcordanceDelta >= 0 ? '↑' : '↓'} ${Math.abs(weeklyConcordanceDelta)}% ${weeklyConcordanceDelta >= 0 ? 'improving' : 'declining'}`;

  return (
    <>
      <section className="ga-kpi-grid ga-kpi-summary-grid">
        <KpiCard
          title="Total Sessions"
          value={String(totalSessions)}
          detail={`${monthDelta >= 0 ? '↑' : '↓'} ${Math.abs(monthDelta)}% this month`}
          detailClass={monthDelta >= 0 ? 'up' : ''}
        />
        <KpiCard title="Avg Concordance" value={formatPercent(avgConcordance)} detail={weekDeltaText} detailClass={weeklyConcordanceDelta >= 0 ? 'up' : ''} />
        <KpiCard title="Best Session" value={formatPercent(bestSessionScore)} detail={`Match · ${bestSessionDate || '-'}`} />
        <KpiCard title="Streak" value={String(streakDays)} detail="days in a row" />
      </section>

      <ActivityHeatmapCard heatmapData={heatmapData} />

      <section className="ga-insights-grid">
        <InsightCard
          icon="↗"
          title="Concordance improving"
          description={`Your average score ${weeklyConcordanceDelta >= 0 ? 'rose' : 'dropped'} ${Math.abs(weeklyConcordanceDelta)}% this week. ${history.length ? 'Keep practising daily sessions.' : 'Run a few sessions to unlock trend insights.'}`}
        />
        <InsightCard
          icon="◎"
          title={`${bestEmotionInsight.emotion} most consistent`}
          description={`${bestEmotionInsight.emotion} has the highest facial-speech alignment at ${bestEmotionInsight.rate}% average match rate.`}
        />
        <InsightCard
          icon="⚑"
          title={`${worstEmotionInsight.emotion} needs attention`}
          description={`${worstEmotionInsight.emotion} shows lowest concordance (${worstEmotionInsight.rate}%). Your face and voice diverge on this emotion.`}
        />
      </section>
    </>
  );
}

function EmotionTrendsTab({ analytics }) {
  const trendPoints = analytics.weeklyTrend;
  const xStep = 500 / Math.max(1, trendPoints.length - 1);
  const pointString = trendPoints
    .map((score, i) => {
      const x = i * xStep;
      const y = 100 - score;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
  const areaString = `${pointString} 500,100 0,100`;

  return (
    <div className="ga-design-stack">
      <div className="ga-card">
        <div className="ga-section-title">Concordance score over time</div>
        <div className="ga-mini-chart-wrap">
          <svg viewBox="0 0 500 100" className="ga-mini-chart" preserveAspectRatio="none" role="img" aria-label="Concordance trend over last 30 days">
            <polyline points={pointString} className="ga-mini-line" />
            <polyline points={areaString} className="ga-mini-area" />
            <line x1="0" y1="40" x2="500" y2="40" className="ga-mini-target" />
          </svg>
        </div>
      </div>
      <div className="ga-design-two-col">
        <div className="ga-card">
          <div className="ga-section-title">Emotion Frequency</div>
          <div className="ga-bars">
            {analytics.emotionFrequency.map(([name, value]) => (
              <div key={name} className="ga-bar-row">
                <div className="ga-bar-label">{name}</div>
                <div className="ga-bar-track"><div className="ga-bar-fill" style={{ width: `${value}%` }} /></div>
                <div className="ga-bar-value">{value}%</div>
              </div>
            ))}
          </div>
        </div>
        <div className="ga-card">
          <div className="ga-section-title">Per-emotion concordance avg</div>
          <div className="ga-bars">
            {analytics.emotionConcordance.map(([name, value]) => (
              <div key={name} className="ga-bar-row">
                <div className="ga-bar-label">{name}</div>
                <div className="ga-bar-track"><div className="ga-bar-fill" style={{ width: `${value}%` }} /></div>
                <div className="ga-bar-value">{value}%</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function AIFeedbackTab({ analytics }) {
  const cards = analytics.feedbackCards;
  return (
    <div className="ga-design-stack">
      {cards.map((card) => (
        <div key={card.title} className={`ga-feedback-card ${card.cls}`}>
          <div className="ga-feedback-title">{card.title}</div>
          <div className="ga-feedback-text">{card.text}</div>
        </div>
      ))}
    </div>
  );
}

function CompareSessionsTab({ analytics }) {
  const [sessionA, sessionB] = analytics.latestComparableSessions;

  if (!sessionA || !sessionB) {
    return (
      <div className="ga-card">
        <p className="ga-empty">Need at least two multimodal sessions to compare. Run more combined analyses.</p>
      </div>
    );
  }

  const scoreDiff = Math.round(getConcordanceScore(sessionA) - getConcordanceScore(sessionB));

  return (
    <div className="ga-design-stack">
      <div className="ga-design-two-col">
        <div className="ga-card">
          <div className="ga-section-title">Session A · {formatDayMonth(new Date(parseRecordTimestamp(sessionA)))}</div>
          <div className="ga-compare-score good">{formatPercent(getConcordanceScore(sessionA))}</div>
          <div className="ga-compare-note">{sessionA.concordance || 'Session'}</div>
        </div>
        <div className="ga-card">
          <div className="ga-section-title">Session B · {formatDayMonth(new Date(parseRecordTimestamp(sessionB)))}</div>
          <div className="ga-compare-score bad">{formatPercent(getConcordanceScore(sessionB))}</div>
          <div className="ga-compare-note">{sessionB.concordance || 'Session'}</div>
        </div>
      </div>
      <div className="ga-card">
        <div className="ga-section-title">What Changed</div>
        <div className="ga-bars">
          <div className="ga-bar-row"><div className="ga-bar-label">Session A confidence</div><div className="ga-bar-track"><div className="ga-bar-fill" style={{ width: `${Math.round((sessionA.confidence || 0) * 100)}%` }} /></div><div className="ga-bar-value">{formatPercent((sessionA.confidence || 0) * 100)}</div></div>
          <div className="ga-bar-row"><div className="ga-bar-label">Session B confidence</div><div className="ga-bar-track"><div className="ga-bar-fill" style={{ width: `${Math.round((sessionB.confidence || 0) * 100)}%` }} /></div><div className="ga-bar-value">{formatPercent((sessionB.confidence || 0) * 100)}</div></div>
          <div className="ga-bar-row"><div className="ga-bar-label">Concordance delta</div><div className="ga-bar-track"><div className="ga-bar-fill" style={{ width: `${Math.min(100, Math.abs(scoreDiff))}%` }} /></div><div className="ga-bar-value">{scoreDiff >= 0 ? '+' : ''}{scoreDiff}%</div></div>
        </div>
      </div>
    </div>
  );
}

function ExportReportTab({ onExportCsv, onExportSummary, analytics }) {
  return (
    <div className="ga-design-stack">
      <div className="ga-card">
        <div className="ga-section-title">Summary · {new Date().toLocaleDateString(undefined, { month: 'long', year: 'numeric' })}</div>
        <div className="ga-report-grid">
          <div>Total sessions <strong>{analytics.totalSessions}</strong></div>
          <div>Average concordance <strong>{formatPercent(analytics.avgConcordance)}</strong></div>
          <div>Match sessions <strong>{analytics.matchCount}</strong></div>
          <div>Mismatch sessions <strong>{analytics.mismatchCount}</strong></div>
        </div>
      </div>
      <div className="ga-report-actions">
        <button className="ga-header-btn" onClick={onExportSummary}>Download PDF Report</button>
        <button className="ga-header-btn" onClick={onExportCsv}>Export CSV</button>
      </div>
    </div>
  );
}

function HistoryTab({ history, onTogglePin, onDelete, onUpdateNote }) {
  return (
    <div className="ga-card">
      <div className="ga-section-title">Analysis History</div>
      {history.length === 0 ? (
        <p className="ga-empty">No analysis records yet. Run facial, speech, or multimodal analysis to populate history.</p>
      ) : (
        <table className="ga-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Modality</th>
              <th>Result</th>
              <th>Confidence</th>
              <th>Explainability</th>
              <th>Notes</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {history.map((row) => (
              <tr key={row.id}>
                <td>{new Date(row.createdAt).toLocaleString()}</td>
                <td>
                  <span className={`ga-pill ${row.pinned ? 'pinned' : ''}`}>{row.modality}</span>
                </td>
                <td>{row.emotion}</td>
                <td>{`${((row.confidence || 0) * 100).toFixed(1)}%`}</td>
                <td>{row.explainability || 'none'}</td>
                <td>
                  <input
                    className="ga-note-input"
                    value={row.note || ''}
                    placeholder="Add note"
                    onChange={(e) => onUpdateNote(row.id, e.target.value)}
                  />
                </td>
                <td>
                  <div className="ga-row-actions">
                    <button className="ga-text-btn" onClick={() => onTogglePin(row.id)}>
                      {row.pinned ? 'Unpin' : 'Pin'}
                    </button>
                    <button className="ga-text-btn danger" onClick={() => onDelete(row.id)}>
                      Delete
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

const THEME_STORAGE_KEY = 'mmer_theme';

function DashboardConsole({ authUser, onLogout }) {
  const [activeTab, setActiveTab] = useState(0);
  const [search, setSearch] = useState('');
  const [dateValue, setDateValue] = useState('');
  const [history, setHistory] = useState([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const logoDataUrlRef = useRef(null);
  const [theme, setTheme] = useState(() => {
    const saved = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (saved === 'dark' || saved === 'light') {
      return saved;
    }
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
      return 'light';
    }
    return 'dark';
  });

  const tabs = [
    { id: 0, icon: 'overview', label: 'Dashboard' },
    { id: 1, icon: 'facial', label: 'Facial Upload' },
    { id: 2, icon: 'speech', label: 'Speech Upload' },
    { id: 3, icon: 'multimodal', label: 'Combined' },
    { id: 6, icon: 'overview', label: 'Emotion Trends' },
    { id: 7, icon: 'model', label: 'AI Feedback' },
    { id: 8, icon: 'multimodal', label: 'Compare Sessions' },
    { id: 5, icon: 'history', label: 'Session History' },
    { id: 9, icon: 'history', label: 'Export Report' },
    { id: 4, icon: 'model', label: 'Model Info' }
  ];

  const navSections = [
    { label: 'Overview', tabIds: [0] },
    { label: 'Analysis', tabIds: [1, 2, 3] },
    { label: 'Insights', tabIds: [6, 7, 8] },
    { label: 'Records', tabIds: [5, 9, 4] }
  ];

  const tabDescriptions = {
    0: 'Overview and key session metrics.',
    1: 'Upload or capture a face image and get explainable emotion predictions.',
    2: 'Upload or record voice to detect emotion with confidence and saliency cues.',
    3: 'Run combined face + voice analysis using separate inputs or a single video.',
    6: 'View emotion trend summaries across recent sessions.',
    7: 'Get AI-generated feedback and actionable tips.',
    8: 'Compare session-level changes and differences.',
    5: 'Track, annotate, pin, and export your analysis history.',
    9: 'Generate and export report summaries.',
    4: 'Review model architecture, accuracy, and system capabilities.'
  };

  // Load history from Supabase on mount
  useEffect(() => {
    const loadHistory = async () => {
      setIsLoadingHistory(true);
      try {
        const data = await loadAnalysisHistoryFromSupabase();
        setHistory(data);
      } catch (err) {
        console.error('Error loading history:', err);
        setHistory([]);
      } finally {
        setIsLoadingHistory(false);
      }
    };

    loadHistory();
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  const addHistoryRecord = async (record) => {
    // Save to Supabase
    const saved = await saveAnalysisToSupabase(record);
    if (saved) {
      // Reload history from Supabase
      const updated = await loadAnalysisHistoryFromSupabase();
      setHistory(updated);
    }
  };

  const togglePinHistory = async (id) => {
    const record = history.find((item) => item.id === id);
    if (record) {
      const toggled = await toggleAnalysisPin(id, record.pinned);
      if (toggled) {
        const updated = await loadAnalysisHistoryFromSupabase();
        setHistory(updated);
      }
    }
  };

  const deleteHistory = async (id) => {
    const deleted = await deleteAnalysisRecord(id);
    if (deleted) {
      const updated = await loadAnalysisHistoryFromSupabase();
      setHistory(updated);
    }
  };

  const updateHistoryNote = async (id, note) => {
    const updated = await updateAnalysisNote(id, note);
    if (updated) {
      setHistory((prev) => prev.map((item) => (item.id === id ? { ...item, note } : item)));
    }
  };

  const filteredHistory = history
    .filter((row) => (dateValue ? row.createdAt.startsWith(dateValue) : true))
    .filter((row) => {
      const q = search.trim().toLowerCase();
      if (!q) return true;
      return [row.modality, row.emotion, row.note, row.explainability]
        .filter(Boolean)
        .some((v) => String(v).toLowerCase().includes(q));
    })
    .sort((a, b) => {
      if (a.pinned && !b.pinned) return -1;
      if (!a.pinned && b.pinned) return 1;
      return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
    });

  const activeTabLabel = tabs.find((tab) => tab.id === activeTab)?.label || 'Dashboard';
  const activeDescription = tabDescriptions[activeTab] || '';
  const todayKey = new Date().toISOString().slice(0, 10);
  const todayRecords = history.filter((row) => row.createdAt.startsWith(todayKey)).length;
  const profileInitials = (() => {
    const base = authUser?.name || authUser?.email || 'User';
    const pieces = String(base).trim().split(/\s+/).filter(Boolean);
    if (pieces.length >= 2) return `${pieces[0][0]}${pieces[1][0]}`.toUpperCase();
    return String(base).slice(0, 2).toUpperCase();
  })();

  const exportHistoryCsv = () => {
    if (!filteredHistory.length) return;
    const header = ['Time', 'Modality', 'Result', 'Confidence', 'Explainability', 'Concordance', 'Note'];
    const rows = filteredHistory.map((row) => [
      new Date(row.createdAt).toISOString(),
      row.modality,
      row.emotion,
      `${((row.confidence || 0) * 100).toFixed(2)}%`,
      row.explainability || 'none',
      row.concordance || '-',
      (row.note || '').replace(/\n/g, ' ')
    ]);
    const csv = [header, ...rows]
      .map((line) => line.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(','))
      .join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mmer-history-${dateValue || 'all'}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const ensureLogoDataUrl = async () => {
    if (logoDataUrlRef.current) return logoDataUrlRef.current;
    const response = await fetch(logoImage);
    const blob = await response.blob();
    const dataUrl = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
    logoDataUrlRef.current = dataUrl;
    return dataUrl;
  };

  const exportSummaryReport = async () => {
    if (!filteredHistory.length) return;
    const byModality = filteredHistory.reduce((acc, cur) => {
      acc[cur.modality] = (acc[cur.modality] || 0) + 1;
      return acc;
    }, {});
    const doc = new jsPDF({ unit: 'pt', format: 'a4' });

    try {
      const logoDataUrl = await ensureLogoDataUrl();
      doc.addImage(logoDataUrl, 'PNG', 40, 34, 48, 48);
    } catch (err) {
      // Continue PDF generation even if logo embedding fails.
    }

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(18);
    doc.text('MMER User Summary Report', 100, 58);

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(11);
    doc.text(`User: ${authUser?.email || 'N/A'}`, 40, 104);
    doc.text(`Range: ${dateValue || 'All records'}`, 40, 122);
    doc.text(`Total records: ${filteredHistory.length}`, 40, 140);

    let y = 170;
    doc.setFont('helvetica', 'bold');
    doc.text('By Modality', 40, y);
    y += 18;

    doc.setFont('helvetica', 'normal');
    Object.entries(byModality).forEach(([key, value]) => {
      doc.text(`- ${key}: ${value}`, 52, y);
      y += 16;
    });

    y += 10;
    doc.setFont('helvetica', 'bold');
    doc.text('Latest Entries', 40, y);
    y += 18;
    doc.setFont('helvetica', 'normal');

    filteredHistory.slice(0, 20).forEach((row, idx) => {
      const line = `${idx + 1}. ${new Date(row.createdAt).toLocaleString()} | ${row.modality} | ${row.emotion} | ${((row.confidence || 0) * 100).toFixed(1)}%`;
      const wrapped = doc.splitTextToSize(line, 510);
      if (y > 780) {
        doc.addPage();
        y = 50;
      }
      doc.text(wrapped, 40, y);
      y += 16 * wrapped.length;
    });

    doc.save(`mmer-summary-${dateValue || 'all'}.pdf`);
  };

  const analytics = useMemo(() => {
    const safeHistory = [...history].sort((a, b) => parseRecordTimestamp(a) - parseRecordTimestamp(b));
    const totalSessions = safeHistory.length;

    const now = new Date();
    const currentMonth = now.getMonth();
    const currentYear = now.getFullYear();
    const prevMonthDate = new Date(currentYear, currentMonth - 1, 1);
    const prevMonth = prevMonthDate.getMonth();
    const prevYear = prevMonthDate.getFullYear();

    const currentMonthRows = safeHistory.filter((row) => {
      const d = new Date(parseRecordTimestamp(row));
      return d.getMonth() === currentMonth && d.getFullYear() === currentYear;
    });
    const prevMonthRows = safeHistory.filter((row) => {
      const d = new Date(parseRecordTimestamp(row));
      return d.getMonth() === prevMonth && d.getFullYear() === prevYear;
    });
    const monthDelta = prevMonthRows.length > 0
      ? Math.round(((currentMonthRows.length - prevMonthRows.length) / prevMonthRows.length) * 100)
      : (currentMonthRows.length > 0 ? 100 : 0);

    const avgConcordance = safeHistory.length
      ? safeHistory.reduce((sum, row) => sum + getConcordanceScore(row), 0) / safeHistory.length
      : 0;

    const nowTs = Date.now();
    const oneWeekMs = 7 * 24 * 60 * 60 * 1000;
    const thisWeek = safeHistory.filter((row) => parseRecordTimestamp(row) >= nowTs - oneWeekMs);
    const prevWeek = safeHistory.filter((row) => {
      const t = parseRecordTimestamp(row);
      return t >= nowTs - (2 * oneWeekMs) && t < nowTs - oneWeekMs;
    });
    const thisWeekAvg = thisWeek.length ? thisWeek.reduce((s, r) => s + getConcordanceScore(r), 0) / thisWeek.length : 0;
    const prevWeekAvg = prevWeek.length ? prevWeek.reduce((s, r) => s + getConcordanceScore(r), 0) / prevWeek.length : 0;
    const weeklyConcordanceDelta = Math.round(thisWeekAvg - prevWeekAvg);

    const bestRow = safeHistory.reduce((best, row) => {
      if (!best) return row;
      return getConcordanceScore(row) > getConcordanceScore(best) ? row : best;
    }, null);

    const bestSessionScore = bestRow ? getConcordanceScore(bestRow) : 0;
    const bestSessionDate = bestRow ? formatDayMonth(new Date(parseRecordTimestamp(bestRow))) : null;

    const sessionDates = Array.from(getSessionDatesSet(safeHistory)).sort((a, b) => (a < b ? 1 : -1));
    let streakDays = 0;
    if (sessionDates.length) {
      let cursor = new Date();
      cursor.setHours(0, 0, 0, 0);
      for (let i = 0; i < sessionDates.length + 1; i += 1) {
        const key = `${cursor.getFullYear()}-${String(cursor.getMonth() + 1).padStart(2, '0')}-${String(cursor.getDate()).padStart(2, '0')}`;
        if (sessionDates.includes(key)) {
          streakDays += 1;
          cursor.setDate(cursor.getDate() - 1);
        } else {
          break;
        }
      }
    }

    const multimodalRows = safeHistory.filter((row) => row.modality === 'multimodal' && splitMultimodalEmotions(row));
    const emotionCounts = {};
    const emotionConcordance = {};
    multimodalRows.forEach((row) => {
      const pair = splitMultimodalEmotions(row);
      if (!pair) return;
      [pair.facial, pair.speech].forEach((emotion) => {
        emotionCounts[emotion] = (emotionCounts[emotion] || 0) + 1;
      });
      const score = getConcordanceScore(row);
      emotionConcordance[pair.facial] = emotionConcordance[pair.facial] || { sum: 0, count: 0 };
      emotionConcordance[pair.facial].sum += score;
      emotionConcordance[pair.facial].count += 1;
    });

    const frequencyPairs = Object.entries(emotionCounts)
      .map(([emotion, count]) => [emotion, Math.round((count / Math.max(1, Object.values(emotionCounts).reduce((a, b) => a + b, 0))) * 100)])
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);

    const concordancePairs = Object.entries(emotionConcordance)
      .map(([emotion, stats]) => [emotion, Math.round(stats.sum / Math.max(1, stats.count))])
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);

    const bestEmotionInsight = concordancePairs[0] || ['Happy', 0];
    const worstEmotionInsight = concordancePairs[concordancePairs.length - 1] || ['Fear', 0];

    const matchCount = safeHistory.filter((row) => row.concordance === 'MATCH').length;
    const mismatchCount = safeHistory.filter((row) => row.concordance === 'MISMATCH').length;

    const feedbackCards = [
      {
        title: `${bestEmotionInsight[0]} is your strongest alignment emotion`,
        text: `Across multimodal sessions, ${bestEmotionInsight[0]} has your best average concordance at ${bestEmotionInsight[1]}%.`,
        cls: 'positive'
      },
      {
        title: `${worstEmotionInsight[0]} needs focused practice`,
        text: `${worstEmotionInsight[0]} currently shows your lowest concordance at ${worstEmotionInsight[1]}%. Try dedicated practice for this emotion.`,
        cls: 'warning'
      },
      {
        title: 'Suggestion · run more combined sessions',
        text: 'Combined facial + speech sessions improve trend accuracy and make your feedback more reliable.',
        cls: 'suggestion'
      }
    ];

    const latestComparableSessions = [...safeHistory]
      .filter((row) => row.modality === 'multimodal')
      .sort((a, b) => parseRecordTimestamp(b) - parseRecordTimestamp(a))
      .slice(0, 2);

    return {
      totalSessions,
      monthDelta,
      avgConcordance,
      weeklyConcordanceDelta,
      bestSessionScore,
      bestSessionDate,
      streakDays,
      heatmapData: buildHeatmapData(safeHistory),
      weeklyTrend: buildTrendSeries(safeHistory),
      emotionFrequency: frequencyPairs.length ? frequencyPairs : [['no-data', 0]],
      emotionConcordance: concordancePairs.length ? concordancePairs : [['no-data', 0]],
      bestEmotionInsight: { emotion: bestEmotionInsight[0], rate: bestEmotionInsight[1] },
      worstEmotionInsight: { emotion: worstEmotionInsight[0], rate: worstEmotionInsight[1] },
      feedbackCards,
      latestComparableSessions,
      matchCount,
      mismatchCount
    };
  }, [history]);

  return (
    <div className="ga-layout">
      <aside className="ga-sidebar">
        <div className="ga-brand-wrap">
          <div className="ga-brand-top">
            <div className="ga-brand-wordmark" aria-label="MMER">MMER</div>
          </div>
        </div>
        <nav className="ga-nav">
          {navSections.map((section) => (
            <div key={section.label} className="ga-nav-section">
              <div className="ga-nav-label">{section.label}</div>
              {section.tabIds.map((tabId) => {
                const tab = tabs.find((item) => item.id === tabId);
                if (!tab) return null;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`ga-nav-item ${activeTab === tab.id ? 'active' : ''}`}
                  >
                    <span className="ga-nav-icon"><DashboardIcon name={tab.icon} /></span>
                    <span className="ga-nav-item-text">{tab.label}</span>
                    {tab.badge && <span className="ga-nav-badge">{tab.badge}</span>}
                  </button>
                );
              })}
            </div>
          ))}
        </nav>
        <div className="ga-sidebar-footer">
          <div className="ga-user-chip">
            <div className="ga-user-avatar">{profileInitials}</div>
            <div className="ga-user-email">{authUser.email}</div>
          </div>
          <button className="ga-text-btn" onClick={onLogout}>Exit</button>
        </div>
      </aside>

      <div className="ga-content-wrap">
        <header className="ga-header">
          <div className="ga-top-title">{activeTabLabel}</div>
          <div className="ga-header-actions">
            <button className="ga-live-session-btn" onClick={() => setActiveTab(1)}>+ Start Facial Session</button>
            <button className="ga-header-btn" onClick={() => setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'))}>
              {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
            </button>
            <button className="ga-profile" title={authUser?.name || authUser?.email || 'Profile'}>{profileInitials}</button>
          </div>
        </header>

        <main className="ga-main">
          {(activeTab === 5 || activeTab === 9) && (
            <div className="ga-record-toolbar">
              <input
                className="ga-search"
                placeholder="Search emotions, notes, explainability..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
              <input type="date" className="ga-date" value={dateValue} onChange={(e) => setDateValue(e.target.value)} />
              <button className="ga-header-btn" onClick={exportHistoryCsv} disabled={!filteredHistory.length}>Export CSV</button>
              <button className="ga-header-btn" onClick={exportSummaryReport} disabled={!filteredHistory.length}>Export PDF</button>
            </div>
          )}

          {activeTab !== 5 && activeTab !== 9 && (
            <div className="ga-page-title-row">
              <div>
                <h1 className="ga-page-title">{activeTabLabel}</h1>
                <p className="ga-page-subtitle">{activeDescription}</p>
              </div>
              <div className="ga-context-chips">
                <span className="ga-chip">Today: {todayRecords} records</span>
                <span className="ga-chip">Total: {history.length} records</span>
              </div>
            </div>
          )}

          {activeTab === 0 && <OverviewTab history={history} analytics={analytics} />}
          {activeTab === 1 && <div className="ga-card ga-tab-shell"><FacialTab onResult={addHistoryRecord} /></div>}
          {activeTab === 2 && <div className="ga-card ga-tab-shell"><SpeechTab onResult={addHistoryRecord} /></div>}
          {activeTab === 3 && <div className="ga-card ga-tab-shell"><CombinedTab onResult={addHistoryRecord} /></div>}
          {activeTab === 6 && <EmotionTrendsTab analytics={analytics} />}
          {activeTab === 7 && <AIFeedbackTab analytics={analytics} />}
          {activeTab === 8 && <CompareSessionsTab analytics={analytics} />}
          {activeTab === 4 && <div className="ga-card ga-tab-shell"><ModelInfoTab /></div>}
          {activeTab === 9 && <ExportReportTab onExportCsv={exportHistoryCsv} onExportSummary={exportSummaryReport} analytics={analytics} />}
          {activeTab === 5 && isLoadingHistory && (
            <div className="ga-card">
              <p className="ga-empty">Loading your history...</p>
            </div>
          )}
          {activeTab === 5 && !isLoadingHistory && (
            <HistoryTab
              history={filteredHistory}
              onTogglePin={togglePinHistory}
              onDelete={deleteHistory}
              onUpdateNote={updateHistoryNote}
            />
          )}
        </main>
      </div>
    </div>
  );
}

function PublicOnlyRoute({ isAuthenticated, children }) {
  const location = useLocation();
  if (isAuthenticated) {
    return <Navigate to="/app/dashboard" replace state={{ from: location }} />;
  }
  return children;
}

function ProtectedRoute({ isAuthenticated, children }) {
  const location = useLocation();
  if (!isAuthenticated) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }
  return children;
}

function AppRouter() {
  const [authUser, setAuthUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const savedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (savedTheme === 'dark' || savedTheme === 'light') {
      document.documentElement.setAttribute('data-theme', savedTheme);
    } else {
      document.documentElement.setAttribute('data-theme', 'dark');
    }
  }, []);

  useEffect(() => {
    const initAuth = async () => {
      try {
        const { data: { session } } = await supabase.auth.getSession();
        if (session?.user) {
          setAuthUser({
            email: session.user.email,
            name: session.user.user_metadata?.name || session.user.email.split('@')[0]
          });
        }
      } catch (err) {
        console.error('Auth error:', err);
      } finally {
        setLoading(false);
      }
    };

    initAuth();

    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, session) => {
      if (session?.user) {
        setAuthUser({
          email: session.user.email,
          name: session.user.user_metadata?.name || session.user.email.split('@')[0]
        });
      } else {
        setAuthUser(null);
      }
    });

    return () => subscription?.unsubscribe();
  }, []);

  const handleAuthSuccess = (user) => {
    setAuthUser(user);
  };

  const handleLogout = async () => {
    await supabase.auth.signOut();
    setAuthUser(null);
  };

  const isAuthenticated = Boolean(authUser?.email);

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100vh',
        backgroundColor: '#f8f9fa'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '18px', color: '#202124', marginBottom: '12px' }}>Loading...</div>
          <div style={{ fontSize: '12px', color: '#5f6368' }}>Checking authentication</div>
        </div>
      </div>
    );
  }

  return (
    <Routes>
      <Route
        path="/"
        element={
          <PublicOnlyRoute isAuthenticated={isAuthenticated}>
            <MarketingPage />
          </PublicOnlyRoute>
        }
      />
      <Route
        path="/login"
        element={
          <PublicOnlyRoute isAuthenticated={isAuthenticated}>
            <AuthPage mode="login" onAuthSuccess={handleAuthSuccess} />
          </PublicOnlyRoute>
        }
      />
      <Route
        path="/signup"
        element={
          <PublicOnlyRoute isAuthenticated={isAuthenticated}>
            <AuthPage mode="signup" onAuthSuccess={handleAuthSuccess} />
          </PublicOnlyRoute>
        }
      />
      <Route
        path="/app/dashboard"
        element={
          <ProtectedRoute isAuthenticated={isAuthenticated}>
            <DashboardConsole authUser={authUser} onLogout={handleLogout} />
          </ProtectedRoute>
        }
      />
      <Route path="*" element={<Navigate to={isAuthenticated ? '/app/dashboard' : '/'} replace />} />
    </Routes>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppRouter />
    </BrowserRouter>
  );
}

export default App;
