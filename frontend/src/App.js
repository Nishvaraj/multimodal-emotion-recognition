import React, { useState, useRef, useEffect, useMemo } from 'react';
import { BrowserRouter, Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { jsPDF } from 'jspdf';
import { supabase } from './supabaseClient';
import { saveAnalysisToSupabase, loadAnalysisHistoryFromSupabase, updateAnalysisNote, toggleAnalysisPin, deleteAnalysisRecord } from './supabaseHistoryService';
import logoImage from './assets/logo.png';

// ============== MODEL + API CONSTANTS ==============
const EMOTIONS_FACIAL = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];
const EMOTIONS_SPEECH = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'];
const API_BASE = process.env.REACT_APP_API_BASE || 'http://127.0.0.1:8000';
const MIN_AUDIO_SECONDS = 5;
const RECOMMENDED_AUDIO_SECONDS = 10;

// ============== MEDIA + UI CONSTANTS ==============
const AUDIO_MIME_CANDIDATES = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/ogg'];
const VIDEO_MIME_CANDIDATES = ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm', 'video/mp4'];
const BTN_PRIMARY = 'bg-gradient-to-br from-blue-700 to-blue-900 hover:from-blue-600 hover:to-blue-800 text-white px-6 py-2 rounded-lg font-medium transition-all duration-200 hover:shadow-lg';
const BTN_SUCCESS = 'bg-gradient-to-br from-cyan-600 to-blue-700 hover:from-cyan-500 hover:to-blue-600 text-white px-4 py-2 rounded-lg font-medium transition-colors';
const BTN_DANGER = 'bg-red-600 hover:bg-red-500 text-white px-4 py-2 rounded-lg font-medium transition-colors';
const BTN_NEUTRAL = 'bg-slate-700 hover:bg-slate-600 text-slate-100 px-4 py-2 rounded-lg font-medium transition-colors';

// ============== GENERIC HELPERS ==============
function pickSupportedMimeType(candidates) {
  if (typeof MediaRecorder === 'undefined' || typeof MediaRecorder.isTypeSupported !== 'function') {
    return '';
  }
  return candidates.find((mime) => MediaRecorder.isTypeSupported(mime)) || '';
}

function getFileExtensionForMime(mimeType, fallback) {
  const mime = String(mimeType || '').toLowerCase();
  if (mime.includes('ogg')) return 'ogg';
  if (mime.includes('mp4')) return 'mp4';
  if (mime.includes('wav')) return 'wav';
  if (mime.includes('webm')) return 'webm';
  return fallback;
}

function getAudioDurationSeconds(file) {
  return new Promise((resolve, reject) => {
    if (!file) {
      reject(new Error('No audio file provided'));
      return;
    }

    const audio = document.createElement('audio');
    const objectUrl = URL.createObjectURL(file);
    audio.preload = 'metadata';
    audio.onloadedmetadata = () => {
      URL.revokeObjectURL(objectUrl);
      if (Number.isFinite(audio.duration)) {
        resolve(audio.duration);
      } else {
        reject(new Error('Could not determine audio duration'));
      }
    };
    audio.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error('Could not read audio duration'));
    };
    audio.src = objectUrl;
  });
}

function getApiErrorMessage(err, fallback = 'Request failed') {
  if (err?.response?.data?.error) return String(err.response.data.error);
  if (err?.response?.data?.detail) return String(err.response.data.detail);
  if (err?.message) return String(err.message);
  return fallback;
}

const CONCORDANCE_PERCENT_MAP = {
  MATCH: 88,
  PARTIAL: 62,
  MISMATCH: 28
};

const CONCORDANCE_CATEGORY_THRESHOLD = {
  mismatchMax: 3.9,
  partialMin: 4.0,
  partialMax: 7.4,
  matchMin: 7.5
};

function normalizeConcordancePercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  if (numeric <= 10) {
    return Math.max(1, Math.min(100, Math.round(numeric * 10)));
  }
  return Math.max(1, Math.min(100, Math.round(numeric)));
}

function getConcordancePercent(rowOrValue) {
  if (rowOrValue && typeof rowOrValue === 'object') {
    const row = rowOrValue;
    if (Number.isFinite(Number(row.concordance_score))) {
      return normalizeConcordancePercent(row.concordance_score);
    }
    if (row?.concordance && CONCORDANCE_PERCENT_MAP[row.concordance]) {
      return CONCORDANCE_PERCENT_MAP[row.concordance];
    }
    const confidence = Number(row?.confidence || 0);
    return Math.max(1, Math.min(100, Math.round(confidence * 100)));
  }

  return normalizeConcordancePercent(rowOrValue);
}

function getConcordanceScore(rowOrValue) {
  return getConcordancePercent(rowOrValue);
}

function getConcordanceScore10(rowOrValue) {
  const percent = getConcordancePercent(rowOrValue);
  return Math.max(0.1, Math.min(10, Number((percent / 10).toFixed(1))));
}

function getConcordanceCategory(rowOrValue) {
  const score = getConcordanceScore10(rowOrValue);
  if (score >= CONCORDANCE_CATEGORY_THRESHOLD.matchMin) {
    return { key: 'MATCH', label: 'Match' };
  }
  if (score >= CONCORDANCE_CATEGORY_THRESHOLD.partialMin) {
    return { key: 'PARTIAL', label: 'Partial Match' };
  }
  return { key: 'MISMATCH', label: 'Mismatch' };
}

function getConcordanceMetrics(rowOrValue) {
  const percent = getConcordancePercent(rowOrValue);
  const score = getConcordanceScore10(percent);
  const category = getConcordanceCategory(score);
  return {
    percent,
    score,
    categoryKey: category.key,
    categoryLabel: category.label
  };
}

function formatConcordanceValue(rowOrValue) {
  const metrics = getConcordanceMetrics(rowOrValue);
  return formatPercent(metrics.percent);
}

function getConcordanceToneClass(categoryKey) {
  if (categoryKey === 'MATCH') return 'good';
  if (categoryKey === 'PARTIAL') return 'partial';
  return 'bad';
}

function buildConcordanceExplainabilityText({ facialEmotion, speechEmotion, categoryLabel, score, percent }) {
  const faceText = formatEmotionLabel(facialEmotion || 'unknown');
  const speechText = formatEmotionLabel(speechEmotion || 'unknown');
  const categoryReason = {
    Match: 'Both modalities point to the same emotional direction, so the system treats the session as aligned.',
    'Partial Match': 'The predictions overlap but do not fully agree, so the score lands in the middle band.',
    Mismatch: 'The facial and speech predictions diverge, so the session is classified as low concordance.'
  }[categoryLabel] || 'The score is derived from how closely the facial and speech predictions align.';

  return {
    facialLine: `Facial emotion prediction: ${faceText}`,
    speechLine: `Speech emotion prediction: ${speechText}`,
    reasoningLine: `Reasoning: ${categoryReason}`,
    categoryLine: `Concordance category: ${categoryLabel}`,
    scoreLine: `Final concordance: ${formatPercent(percent)}`
  };
}

function splitMultimodalEmotions(row) {
  if (!row?.emotion || !String(row.emotion).includes('|')) return null;
  const [facial, speech] = String(row.emotion).split('|').map((item) => item.trim().toLowerCase());
  if (!facial || !speech) return null;
  return { facial, speech };
}

function extractEmotionLabels(row) {
  const raw = String(row?.emotion || '').trim().toLowerCase();
  if (!raw) return [];
  return raw
    .split('|')
    .map((item) => item.trim())
    .filter((label) => label && label !== 'unknown' && label !== 'no-data');
}

function formatEmotionLabel(label) {
  if (!label || label === 'no-data') return 'No Data';
  return String(label)
    .replace(/[_-]+/g, ' ')
    .split(' ')
    .filter(Boolean)
    .map((part) => part[0].toUpperCase() + part.slice(1))
    .join(' ');
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
  // Build a compact 12-week grid so the dashboard can render activity density by day.
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
    // Convert raw counts into one of five visual levels for the heatmap cells.
    if (!count || maxCount === 0) return 0;
    const ratio = count / maxCount;
    if (ratio < 0.25) return 1;
    if (ratio < 0.5) return 2;
    if (ratio < 0.75) return 3;
    return 4;
  };

  return cells.map((row) => row.map((count) => ({ count, level: toLevel(count) })));
}

function buildWeeklyVolumeCounts(history) {
  // Count analyses per week for the volume chart in the dashboard.
  const now = new Date();
  now.setHours(0, 0, 0, 0);
  const start = new Date(now);
  start.setDate(now.getDate() - 83);

  const counts = Array(12).fill(0);
  const msPerDay = 24 * 60 * 60 * 1000;

  history.forEach((row) => {
    const d = new Date(parseRecordTimestamp(row));
    d.setHours(0, 0, 0, 0);
    if (d < start || d > now) return;

    const dayOffset = Math.floor((d.getTime() - start.getTime()) / msPerDay);
    const weekIndex = Math.floor(dayOffset / 7);
    if (weekIndex >= 0 && weekIndex < 12) {
      counts[weekIndex] += 1;
    }
  });

  return counts;
}

function buildTrendSeries(history) {
  // Compute a rolling concordance trend so weekly changes are easy to compare.
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

function buildWeeklySeries(history, { metric = 'concordance', modality = 'all' } = {}) {
  // Reuse one builder for concordance, confidence, and volume by switching the metric mode.
  const scopedHistory = modality === 'all'
    ? history
    : history.filter((row) => row.modality === modality);

  if (metric === 'volume') {
    return buildWeeklyVolumeCounts(scopedHistory);
  }

  const now = new Date();
  const points = Array.from({ length: 12 }, (_, i) => {
    const end = new Date(now);
    end.setDate(now.getDate() - ((11 - i) * 7));
    const start = new Date(end);
    start.setDate(end.getDate() - 6);

    const weekRows = scopedHistory.filter((row) => {
      const t = parseRecordTimestamp(row);
      const inWindow = t >= start.getTime() && t <= end.getTime();
      return inWindow;
    });

    if (!weekRows.length) {
      return 0;
    }

    if (metric === 'confidence') {
      const avgConfidence = weekRows.reduce((sum, row) => sum + Number(row.confidence || 0), 0) / weekRows.length;
      return Math.round(avgConfidence * 100);
    }

    const avgConcordance = weekRows.reduce((sum, row) => sum + getConcordanceScore(row), 0) / weekRows.length;
    return Math.round(avgConcordance);
  });

  return points;
}

// ============== FACIAL EMOTION TAB ==============
function FacialTab({ onResult, clearSignal = 0 }) {
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

  const clearCurrentAnalysis = () => {
    // Reset every piece of facial state so the next analysis starts from a blank slate.
    setError(null);
    setShowGradCAM(false);
    setEmotion(null);
    setConfidence(null);
    setProbabilities(null);
    setGradCam(null);
    setAnnotatedImage(null);
    setFaceDetected(false);
    if (isCameraOn) {
      stopCamera();
    }
    if (imagePreview) {
      setImagePreview(null);
    }
    setImageFile(null);
  };

  useEffect(() => {
    if (clearSignal > 0) {
      clearCurrentAnalysis();
    }
  }, [clearSignal]);

  useEffect(() => {
    if (isCameraOn && videoRef.current && cameraStreamRef.current) {
      videoRef.current.srcObject = cameraStreamRef.current;
    }
  }, [isCameraOn]);

  const handleImageSelect = (e) => {
    // Convert the selected file into a previewable data URL while clearing stale results.
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
      // Request camera access only when the user explicitly asks for it.
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
      // Snapshot the live camera frame into a blob so the backend can analyze it like an upload.
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
    // Keep the upload/explainability request aligned with the checkbox state.
    setLoading(true);
    setError(null);
    setGradCam(null);
    try {
      const formData = new FormData();
      formData.append('file', imageFile);
      
      const explainParam = showGradCAM ? '?explain=true' : '';
      const response = await axios.post(`${API_BASE}/api/predict/facial${explainParam}`, formData, { timeout: 500000 });
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
      } else {
        setError(response.data.error || 'Prediction failed');
      }
    } catch (err) {
      setError('API Error: ' + getApiErrorMessage(err, 'Unable to analyze image'));
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
          <div className="bg-blue-900 px-4 py-2 flex items-center gap-2">
            <span className="text-blue-300 text-sm font-medium">Image Source</span>
          </div>
          <div className="p-4">
            {!isCameraOn && !imagePreview && (
              <div className="border-2 border-dashed border-slate-700 rounded-lg p-8 text-center">
                <p className="text-slate-400 mb-4">Click to Access Webcam</p>
                <button
                  onClick={startCamera}
                  className={BTN_PRIMARY}
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
                    className={`flex-1 ${BTN_SUCCESS}`}
                  >
                    Capture
                  </button>
                  <button
                    onClick={stopCamera}
                    className={`flex-1 ${BTN_DANGER}`}
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
                className="mt-1 w-5 h-5 rounded border-slate-600 bg-slate-700 text-blue-600 focus:ring-blue-500 focus:ring-offset-slate-800"
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
            className="w-full bg-gradient-to-br from-blue-700 to-blue-900 hover:from-blue-600 hover:to-blue-800 text-white px-6 py-3 rounded-lg font-medium text-lg transition-all duration-200 hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
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
            <div className="bg-blue-900 px-4 py-2 flex items-center gap-2">
              <span className="text-blue-300 text-sm font-medium">Confidence Scores</span>
            </div>
            <div className="p-4 space-y-3">
              <div className="text-center mb-4">
                <div className="text-2xl font-bold text-slate-50">{formatEmotionLabel(emotion)}</div>
                <div className="text-lg text-cyan-300">{(confidence * 100).toFixed(1)}%</div>
              </div>
              {EMOTIONS_FACIAL.map((emo) => (
                <div key={emo} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-300">{formatEmotionLabel(emo)}</span>
                    <span className="text-slate-400">{((probabilities[emo] || 0) * 100).toFixed(1)}%</span>
                  </div>
                    <div className="w-full bg-slate-700 rounded-full h-2">
                      <div
                        className="bg-gradient-to-r from-blue-600 to-blue-800 h-2 rounded-full transition-all duration-500"
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
            <div className="bg-blue-900 px-4 py-2 flex items-center gap-2">
              <span className="text-blue-300 text-sm font-medium">Face Detection</span>
            </div>
            <div className="p-4">
              <p className="text-slate-400 text-sm mb-3">
                {faceDetected
                  ? 'Detected face is shown with a tighter box before explainability is computed.'
                  : 'No face box was detected; model used the full image.'}
              </p>
              <img
                src={annotatedImage ? `data:image/png;base64,${annotatedImage}` : imagePreview}
                alt="Annotated"
                className="w-full rounded-lg"
                style={{ maxHeight: '300px', width: '100%', objectFit: 'contain', display: 'block', margin: '0 auto' }}
              />
            </div>
          </div>
        )}

        {/* Grad-CAM Heatmap */}
        {gradCam && (
          <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
            <div className="bg-blue-900 px-4 py-2 flex items-center gap-2">
              <span className="text-blue-300 text-sm font-medium">Facial Grad-CAM Heatmap</span>
            </div>
            <div className="p-4">
              <p className="text-slate-400 text-sm mb-3">
                Heat map shows where the facial model focused most. Red/orange areas had the strongest influence.
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
function SpeechTab({ onResult, clearSignal = 0 }) {
  const [audioFile, setAudioFile] = useState(null);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState(null);
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
  const recordingStartedAtRef = useRef(null);
  const suppressRecordingOnStopRef = useRef(false);

  useEffect(() => {
    return () => {
      if (audioPreviewUrl) {
        URL.revokeObjectURL(audioPreviewUrl);
      }
    };
  }, [audioPreviewUrl]);

  const startRecording = async () => {
    try {
      // Record from the microphone only after the user has granted permission.
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];
      recordingStartedAtRef.current = Date.now();

      const mimeType = pickSupportedMimeType(AUDIO_MIME_CANDIDATES);
      const mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.ondataavailable = (e) => chunksRef.current.push(e.data);
      mediaRecorder.onstop = () => {
        if (suppressRecordingOnStopRef.current) {
          suppressRecordingOnStopRef.current = false;
          return;
        }
        // Build a File object so the upload path and recorded path use the same backend contract.
        const recordingDuration = recordingStartedAtRef.current ? (Date.now() - recordingStartedAtRef.current) / 1000 : 0;
        const recorderType = mediaRecorder.mimeType || mimeType || 'audio/webm';
        const audioBlob = new Blob(chunksRef.current, { type: recorderType });
        const ext = getFileExtensionForMime(recorderType, 'webm');
        const recordedFile = new File([audioBlob], `recorded-audio-${Date.now()}.${ext}`, { type: recorderType });
        if (recordingDuration < MIN_AUDIO_SECONDS) {
          setError(`Audio must be at least ${MIN_AUDIO_SECONDS} seconds. Use 10+ seconds for better feedback.`);
          setAudioFile(null);
          setEmotion(null);
          setSaliency(null);
          setWaveform(null);
          return;
        }
        if (audioPreviewUrl) {
          URL.revokeObjectURL(audioPreviewUrl);
        }
        setAudioPreviewUrl(URL.createObjectURL(recordedFile));
        setAudioFile(recordedFile);
      };
      mediaRecorder.start();
      setIsRecording(true);
      setError(null);
    } catch (err) {
      setError('Cannot access microphone');
    }
  };

  const clearCurrentAnalysis = () => {
    // Reset every multimodal result so the next run starts clean.
    setError(null);
    setShowSaliency(false);
    setEmotion(null);
    setConfidence(null);
    setProbabilities(null);
    setSaliency(null);
    setWaveform(null);
    if (isRecording && mediaRecorderRef.current) {
      suppressRecordingOnStopRef.current = true;
      stopRecording();
    }
    if (audioPreviewUrl) {
      URL.revokeObjectURL(audioPreviewUrl);
      setAudioPreviewUrl(null);
    }
    setAudioFile(null);
  };

  useEffect(() => {
    if (clearSignal > 0) {
      clearCurrentAnalysis();
    }
  }, [clearSignal]);

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (streamRef.current) streamRef.current.getTracks().forEach(track => track.stop());
    }
  };

  const handleAudioSelect = async (e) => {
    const file = e.target.files[0];
    if (file) {
      try {
        const durationSeconds = await getAudioDurationSeconds(file);
        if (durationSeconds < MIN_AUDIO_SECONDS) {
          setError(`Audio must be at least ${MIN_AUDIO_SECONDS} seconds. Use 10+ seconds for better feedback.`);
          if (audioPreviewUrl) {
            URL.revokeObjectURL(audioPreviewUrl);
          }
          setAudioPreviewUrl(null);
          setAudioFile(null);
          setEmotion(null);
          setSaliency(null);
          setWaveform(null);
          e.target.value = '';
          return;
        }
      } catch (durationErr) {
        setError('Could not read audio duration. Please upload a different file.');
        e.target.value = '';
        return;
      }
      if (audioPreviewUrl) {
        URL.revokeObjectURL(audioPreviewUrl);
      }
      setAudioPreviewUrl(URL.createObjectURL(file));
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
    // The explainability checkbox controls whether the saliency map is requested from the backend.
    setLoading(true);
    setError(null);
    setSaliency(null);
    setWaveform(null);
    try {
      const formData = new FormData();
      formData.append('file', audioFile);
      
      const explainParam = showSaliency ? '?explain=true' : '';
      const response = await axios.post(`${API_BASE}/api/predict/speech${explainParam}`, formData, { timeout: 400000 });
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
      } else {
        setError(response.data.error || 'Prediction failed');
      }
    } catch (err) {
      setError('API Error: ' + getApiErrorMessage(err, 'Unable to analyze audio'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Left Column - Input */}
      <div className="space-y-4" style={{ minHeight: '400px' }}>
        {/* Audio Record/Upload Card */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
          <div className="bg-blue-900 px-4 py-2 flex items-center gap-2">
            <span className="text-blue-300 text-sm font-medium">Audio Source</span>
          </div>
          <div className="p-4">
            {!isRecording && !audioFile && (
              <div className="border-2 border-dashed border-slate-700 rounded-lg p-8 text-center">
                <p className="text-slate-400 mb-4">Click to Record Audio</p>
                <button
                  onClick={startRecording}
                  className={BTN_PRIMARY}
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
                        className="w-1 bg-blue-500 rounded-full animate-pulse"
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
                  className={BTN_DANGER}
                >
                  Stop Recording
                </button>
              </div>
            )}
            {audioFile && !isRecording && (
              <div className="text-center">
                <div className="text-cyan-300 text-lg font-semibold mb-3">
                  Audio Ready
                </div>
                <div className="bg-slate-700 rounded-lg p-4 mb-3">
                  <div className="text-slate-300 text-sm">Audio file loaded</div>
                </div>
                {audioPreviewUrl && (
                  <audio src={audioPreviewUrl} controls className="w-full mb-3" />
                )}
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
                className="mt-1 w-5 h-5 rounded border-slate-600 bg-slate-700 text-blue-600 focus:ring-blue-500 focus:ring-offset-slate-800"
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

        {audioFile && (
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
            <h4 className="text-slate-50 font-semibold mb-2">Analysis Tips</h4>
            <ul className="text-slate-300 text-sm space-y-1">
              <li>Use audio of at least 10 seconds for better results</li>
              <li>WAV or MP3 files work best</li>
              <li>Speak clearly for accurate speech emotion detection</li>
            </ul>
          </div>
        )}

        {/* Analyze Button */}
        {audioFile && (
          <button
            onClick={analyzeSpeech}
            disabled={loading}
            className="w-full bg-gradient-to-br from-blue-700 to-blue-900 hover:from-blue-600 hover:to-blue-800 text-white px-6 py-3 rounded-lg font-medium text-lg transition-all duration-200 hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
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
            <div className="bg-blue-900 px-4 py-2 flex items-center gap-2">
              <span className="text-blue-300 text-sm font-medium">Confidence Scores</span>
            </div>
            <div className="p-4 space-y-3">
              <div className="text-center mb-4">
                <div className="text-2xl font-bold text-slate-50">{formatEmotionLabel(emotion)}</div>
                <div className="text-lg text-cyan-300">{(confidence * 100).toFixed(1)}%</div>
              </div>
              {EMOTIONS_SPEECH.map((emo) => (
                <div key={emo} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-300">{formatEmotionLabel(emo)}</span>
                    <span className="text-slate-400">{((probabilities[emo] || 0) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-blue-600 to-blue-800 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${(probabilities[emo] || 0) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Audio Spectrogram */}
        {waveform && (
          <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
            <div className="bg-blue-900 px-4 py-2 flex items-center gap-2">
              <span className="text-blue-300 text-sm font-medium">Audio Spectrogram</span>
            </div>
            <div className="p-4">
              <p className="text-slate-400 text-sm mb-3">
                Mel-frequency spectrogram of the uploaded audio signal.
              </p>
              <img
                src={`data:image/png;base64,${waveform}`}
                alt="Audio Spectrogram"
                className="w-full rounded-lg"
                style={{ display: 'block', margin: '0 auto', maxWidth: '100%' }}
              />
            </div>
          </div>
        )}

        {/* Audio Saliency Map */}
        {saliency && (
          <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
            <div className="bg-blue-900 px-4 py-2 flex items-center gap-2">
              <span className="text-blue-300 text-sm font-medium">Audio Saliency Map</span>
            </div>
            <div className="p-2" style={{ paddingTop: 0, marginTop: 0 }}>
              <p className="text-slate-400 text-sm mb-3 mt-2">
                Red frequencies = important for prediction | Blue frequencies = less important
              </p>
              <img
                src={`data:image/png;base64,${saliency}`}
                alt="Audio Saliency"
                className="w-full rounded-lg"
                style={{ width: '100%', objectFit: 'contain', display: 'block', marginTop: 0, paddingTop: 0 }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ============== COMBINED ANALYSIS TAB ==============
function CombinedTab({ onResult, clearSignal = 0 }) {
  const [inputMode, setInputMode] = useState('separate');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState(null);
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
  const suppressAudioOnStopRef = useRef(false);

  const [isVideoCameraOn, setIsVideoCameraOn] = useState(false);
  const [isVideoRecording, setIsVideoRecording] = useState(false);
  const videoLiveRef = useRef(null);
  const videoRecorderRef = useRef(null);
  const videoChunksRef = useRef([]);
  const videoStreamRef = useRef(null);
  const suppressVideoOnStopRef = useRef(false);

  const [facialEmotion, setFacialEmotion] = useState(null);
  const [speechEmotion, setSpeechEmotion] = useState(null);
  const [concordance, setConcordance] = useState(null);
  const [concordanceScore, setConcordanceScore] = useState(null);
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

  const concordanceMetrics = getConcordanceMetrics({
    concordance,
    concordance_score: concordanceScore
  });
  const concordanceExplainabilityText = buildConcordanceExplainabilityText({
    facialEmotion,
    speechEmotion,
    categoryLabel: concordanceMetrics.categoryLabel,
    score: concordanceMetrics.score,
    percent: concordanceMetrics.percent
  });
  const audioRecordingStartedAtRef = useRef(null);

  useEffect(() => {
    return () => {
      if (videoPreviewUrl) {
        URL.revokeObjectURL(videoPreviewUrl);
      }
      if (audioPreviewUrl) {
        URL.revokeObjectURL(audioPreviewUrl);
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
  }, [videoPreviewUrl, audioPreviewUrl]);

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
    setConcordanceScore(null);
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

  const clearCurrentAnalysis = () => {
    setError(null);
    setShowExplainability(false);
    resetResults();

    if (inputMode === 'separate') {
      if (isCameraOn) {
        stopImageCamera();
      }
      if (isAudioRecording && audioRecorderRef.current) {
        suppressAudioOnStopRef.current = true;
        stopAudioRecording();
      }
      if (audioPreviewUrl) {
        URL.revokeObjectURL(audioPreviewUrl);
      }
      setImageFile(null);
      setImagePreview(null);
      setAudioFile(null);
      setAudioPreviewUrl(null);
      return;
    }

    if (isVideoRecording && videoRecorderRef.current) {
      suppressVideoOnStopRef.current = true;
      stopVideoRecording();
    }
    if (isVideoCameraOn) {
      stopVideoCamera();
    }
    if (videoPreviewUrl) {
      URL.revokeObjectURL(videoPreviewUrl);
    }
    setVideoFile(null);
    setVideoPreviewUrl(null);
  };

  useEffect(() => {
    if (clearSignal > 0) {
      clearCurrentAnalysis();
    }
  }, [clearSignal]);

  const handleImageSelect = (e) => {
    // Keep image selection simple: load a new preview and clear stale analysis.
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

  const handleAudioSelect = async (e) => {
    const file = e.target.files[0];
    if (file) {
      try {
        const durationSeconds = await getAudioDurationSeconds(file);
        if (durationSeconds < MIN_AUDIO_SECONDS) {
          setError(`Audio must be at least ${MIN_AUDIO_SECONDS} seconds. Use 10+ seconds for better feedback.`);
          setAudioFile(null);
          resetResults();
          e.target.value = '';
          return;
        }
      } catch (durationErr) {
        setError('Could not read audio duration. Please upload a different file.');
        e.target.value = '';
        return;
      }
      if (audioPreviewUrl) {
        URL.revokeObjectURL(audioPreviewUrl);
      }
      setAudioFile(file);
      setAudioPreviewUrl(URL.createObjectURL(file));
      resetResults();
    }
  };

  const startImageCamera = async () => {
    try {
      // Keep image capture separate from audio so the combined flow remains explicit.
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
    // Convert the webcam frame into a File so the combined endpoint receives the same payload type as uploads.
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
      // Record audio independently so the multimodal workflow can run without a video upload.
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioStreamRef.current = stream;
      audioChunksRef.current = [];
      audioRecordingStartedAtRef.current = Date.now();
      const mimeType = pickSupportedMimeType(AUDIO_MIME_CANDIDATES);
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      audioRecorderRef.current = recorder;
      recorder.ondataavailable = (e) => audioChunksRef.current.push(e.data);
      recorder.onstop = () => {
        if (suppressAudioOnStopRef.current) {
          suppressAudioOnStopRef.current = false;
          return;
        }
        // Package the recorded clip into a File object for the backend upload route.
        const recordingDuration = audioRecordingStartedAtRef.current ? (Date.now() - audioRecordingStartedAtRef.current) / 1000 : 0;
        if (recordingDuration < MIN_AUDIO_SECONDS) {
          setError(`Audio must be at least ${MIN_AUDIO_SECONDS} seconds. Use 10+ seconds for better feedback.`);
          setAudioFile(null);
          resetResults();
          return;
        }
        if (audioPreviewUrl) {
          URL.revokeObjectURL(audioPreviewUrl);
        }
        const recorderType = recorder.mimeType || mimeType || audioChunksRef.current[0]?.type || 'audio/webm';
        const blob = new Blob(audioChunksRef.current, { type: recorderType });
        const ext = getFileExtensionForMime(recorderType, 'webm');
        const recorded = new File([blob], `recorded-audio-${Date.now()}.${ext}`, { type: recorderType });
        setAudioFile(recorded);
        setAudioPreviewUrl(URL.createObjectURL(recorded));
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
  if (!navigator?.mediaDevices?.getUserMedia) {
    setError('Camera and microphone are not supported in this browser.');
    return;
  }
  try {
    // Request combined camera + microphone access only when the user selects video mode.
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: true
    });
    videoStreamRef.current = stream;
    setIsVideoCameraOn(true);
    setError(null);
  } catch (err) {
    setError('Cannot access camera/microphone. Check browser permissions.');
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
    const mimeType = pickSupportedMimeType(VIDEO_MIME_CANDIDATES);
    const recorder = mimeType ? new MediaRecorder(videoStreamRef.current, { mimeType }) : new MediaRecorder(videoStreamRef.current);
    videoRecorderRef.current = recorder;
    recorder.ondataavailable = (e) => videoChunksRef.current.push(e.data);
    recorder.onstop = () => {
      if (suppressVideoOnStopRef.current) {
        suppressVideoOnStopRef.current = false;
        return;
      }
      const recorderType = recorder.mimeType || mimeType || videoChunksRef.current[0]?.type || 'video/webm';
      const blob = new Blob(videoChunksRef.current, { type: recorderType });
      const ext = getFileExtensionForMime(recorderType, 'webm');
      const recorded = new File([blob], `recorded-video-${Date.now()}.${ext}`, { type: recorderType });
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
      // Stop the stream so the live preview shuts down immediately and releases devices.
      if (videoStreamRef.current) {
        videoStreamRef.current.getTracks().forEach((track) => track.stop());
        videoStreamRef.current = null;
      }
      if (videoLiveRef.current) {
        videoLiveRef.current.srcObject = null;
      }
      setIsVideoCameraOn(false);
    }
  };

  const analyzeCombined = async () => {
    if (!imageFile || !audioFile) {
      setError('Please provide both image and audio');
      return;
    }
    // Send both files together so the backend can calculate concordance from the two modalities.
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
      const response = await axios.post(`${API_BASE}/api/predict/combined${explainParam}`, formData, { timeout: 500000 });
      if (response.data.success) {
        setFacialEmotion(response.data.facial_emotion.emotion);
        setSpeechEmotion(response.data.speech_emotion.emotion);
        setConcordance(response.data.concordance);
        setConcordanceScore(getConcordancePercent(response.data));
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
          const combinedConcordancePercent = getConcordancePercent(response.data);
          const combinedConfidence = Number.isFinite(Number(response.data.combined_confidence))
            ? Number(response.data.combined_confidence)
            : Math.max(
              response.data.facial_emotion.confidence || 0,
              response.data.speech_emotion.confidence || 0
            );
          onResult({
            id: `combined-${Date.now()}`,
            modality: 'multimodal',
            emotion: `${response.data.facial_emotion.emotion} | ${response.data.speech_emotion.emotion}`,
            confidence: combinedConfidence,
            probabilities: {
              facial: response.data.facial_emotion.probabilities,
              speech: response.data.speech_emotion.probabilities
            },
            explainability: response.data.explainability ? 'enabled' : 'none',
            concordance: response.data.concordance,
            concordance_score: combinedConcordancePercent,
            createdAt: new Date().toISOString(),
            note: '',
            pinned: false
          });
        }
      } else {
        setError(response.data.error || 'Prediction failed');
      }
    } catch (err) {
      setError('API Error: ' + getApiErrorMessage(err, 'Unable to analyze combined input'));
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
      const response = await axios.post(`${API_BASE}/api/predict/video${explainParam}`, formData, { timeout: 500000 });
      if (response.data.success) {
        const face = response.data.facial_emotion?.emotion || 'unknown';
        const speech = response.data.speech_emotion?.emotion || 'unknown';
        const videoConcordance = response.data.concordance || ((face === 'unknown' || speech === 'unknown') ? 'UNKNOWN' : (face === speech ? 'MATCH' : 'MISMATCH'));

        setVideoResult(response.data);
        setFacialEmotion(face);
        setSpeechEmotion(speech);
        setConcordance(videoConcordance);
        setConcordanceScore(getConcordancePercent({
          concordance: videoConcordance,
          concordance_score: response.data.concordance_score,
          confidence: Math.max(
            response.data.facial_emotion?.confidence || 0,
            response.data.speech_emotion?.confidence || 0
          )
        }));
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
          const videoConcordancePercent = getConcordancePercent({
            concordance: videoConcordance,
            concordance_score: response.data.concordance_score,
            confidence: Math.max(
              response.data.facial_emotion?.confidence || 0,
              response.data.speech_emotion?.confidence || 0
            )
          });
          const videoCombinedEmotion = response.data.combined_emotion || 'unknown';
          const videoCombinedConfidence = videoCombinedEmotion === face
            ? Number(response.data.facial_emotion?.confidence || 0)
            : videoCombinedEmotion === speech
              ? Number(response.data.speech_emotion?.confidence || 0)
              : Math.max(
                Number(response.data.facial_emotion?.confidence || 0),
                Number(response.data.speech_emotion?.confidence || 0)
              );
          onResult({
            id: `video-${Date.now()}`,
            modality: 'multimodal',
            emotion: `${videoCombinedEmotion} (video)`,
            confidence: videoCombinedConfidence,
            probabilities: {
              facial: response.data.facial_emotion?.probabilities || {},
              speech: response.data.speech_emotion?.probabilities || {}
            },
            explainability: showExplainability ? 'requested' : 'none',
            concordance: videoConcordance,
            concordance_score: videoConcordancePercent,
            createdAt: new Date().toISOString(),
            note: '',
            pinned: false
          });
        }
      } else {
        setError(response.data.error || 'Prediction failed');
      }
    } catch (err) {
      setError('API Error: ' + getApiErrorMessage(err, 'Unable to analyze video'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
        <h3 className="text-slate-50 font-semibold mb-1">Combined Analysis</h3>
        <p className="text-slate-400 text-sm">
          Start here to analyze facial and speech emotion together. Separate image and audio inputs, or a single video, are supporting capture modes.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => switchMode('separate')}
          className={`ga-header-btn ${inputMode === 'separate' ? 'active' : ''} flex items-center gap-2`}
        >
          <span className="ga-nav-icon"><DashboardIcon name="multimodal" /></span>
          <span>Separate Inputs</span>
        </button>
        <button
          type="button"
          onClick={() => switchMode('video')}
          className={`ga-header-btn ${inputMode === 'video' ? 'active' : ''} flex items-center gap-2`}
        >
          <span className="ga-nav-icon"><DashboardIcon name="sessions" /></span>
          <span>Video Input</span>
        </button>
      </div>

      {/* Input Section */}
      {inputMode === 'separate' && (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Image Upload */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
          <div className="bg-blue-900 px-4 py-2 flex items-center gap-2">
            <span className="text-blue-300 text-sm font-medium">Image Input</span>
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
                  className="w-full bg-gradient-to-br from-blue-700 to-blue-900 hover:from-blue-600 hover:to-blue-800 text-white px-6 py-2 rounded-lg font-medium transition-all duration-200 hover:shadow-lg"
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
                  <button type="button" onClick={captureImage} className={`flex-1 ${BTN_SUCCESS}`}>Capture</button>
                  <button type="button" onClick={stopImageCamera} className={`flex-1 ${BTN_DANGER}`}>Cancel</button>
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
                <button
                  type="button"
                  onClick={startImageCamera}
                  className="w-full mt-3 bg-gradient-to-br from-blue-700 to-blue-900 hover:from-blue-600 hover:to-blue-800 text-white px-6 py-2 rounded-lg font-medium transition-all duration-200 hover:shadow-lg"
                >
                  Capture with Webcam
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Audio Upload */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
          <div className="bg-blue-900 px-4 py-2 flex items-center gap-2">
            <span className="text-blue-300 text-sm font-medium">Audio Input</span>
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
                  className="w-full bg-gradient-to-br from-blue-700 to-blue-900 hover:from-blue-600 hover:to-blue-800 text-white px-6 py-2 rounded-lg font-medium transition-all duration-200 hover:shadow-lg"
                >
                  Record Live with Mic
                </button>
                <div className="mt-3 rounded-lg border border-cyan-300/15 bg-cyan-400/5 px-3 py-2 text-xs text-cyan-100/80">
                  Tip: use audio of at least {RECOMMENDED_AUDIO_SECONDS} seconds for better feedback. Clips under {MIN_AUDIO_SECONDS} seconds are rejected.
                </div>
              </div>
            )}

            {isAudioRecording && (
              <div className="text-center">
                <div className="text-red-400 font-semibold mb-3">Recording...</div>
                <button type="button" onClick={stopAudioRecording} className={BTN_DANGER}>
                  Stop Recording
                </button>
              </div>
            )}

            {audioFile && !isAudioRecording && (
              <div className="text-center">
                <div className="bg-slate-700 rounded-lg p-8 mb-3">
                  <div className="text-cyan-300 font-semibold">Audio Loaded</div>
                </div>
                {audioPreviewUrl && (
                  <audio src={audioPreviewUrl} controls className="w-full mb-3" />
                )}
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
                <button
                  type="button"
                  onClick={startAudioRecording}
                  className="w-full mt-3 bg-gradient-to-br from-blue-700 to-blue-900 hover:from-blue-600 hover:to-blue-800 text-white px-6 py-2 rounded-lg font-medium transition-all duration-200 hover:shadow-lg"
                >
                  Record Live with Mic
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
      )}

      {inputMode === 'video' && (
        <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
          <div className="bg-blue-900 px-4 py-2 flex items-center gap-2">
            <span className="text-blue-300 text-sm font-medium">Video Input</span>
          </div>
          <div className="p-4 space-y-4">
            <p className="text-slate-400 text-sm">Upload a video with face + voice, or record one live using webcam and microphone.</p>
            {!isVideoCameraOn && !videoFile && (
              <>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-3" style={{ display: isVideoCameraOn ? 'none' : 'grid' }}>
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
                    className="w-full bg-gradient-to-br from-blue-700 to-blue-900 hover:from-blue-600 hover:to-blue-800 text-white px-6 py-2 rounded-lg font-medium transition-all duration-200 hover:shadow-lg"
                  >
                    Record Live Video (Cam + Mic)
                  </button>
                </div>
                <div className="rounded-lg border border-cyan-300/15 bg-cyan-400/5 px-3 py-2 text-xs text-cyan-100/80">
                  Tip: use video with at least {RECOMMENDED_AUDIO_SECONDS} seconds of usable audio for the most stable multimodal result.
                </div>
              </>
            )}

            <div style={{ display: (isVideoCameraOn || videoPreviewUrl) ? 'block' : 'none' }}>
              {isVideoCameraOn && (
                <div>
                  <video
                    ref={videoLiveRef}
                    autoPlay
                    playsInline
                    className="w-full rounded-lg mb-3"
                    style={{ maxHeight: '300px', width: '100%', display: isVideoCameraOn ? 'block' : 'none' }}
                    onLoadedMetadata={() => {
                      if (videoLiveRef.current) {
                        videoLiveRef.current.muted = true;
                      }
                    }}
                  />
                  <div className="flex flex-wrap gap-2">
                    {!isVideoRecording ? (
                      <button type="button" onClick={startVideoRecording} className={BTN_SUCCESS}>
                        Start Recording
                      </button>
                    ) : (
                      <button type="button" onClick={stopVideoRecording} className={BTN_DANGER}>
                        Stop Recording
                      </button>
                    )}
                    <button type="button" onClick={stopVideoCamera} className={BTN_NEUTRAL}>
                      Close Camera
                    </button>
                  </div>
                </div>
              )}

              {videoPreviewUrl && (
                <div>
                  <video
                    src={videoPreviewUrl}
                    controls
                    className="w-full rounded-lg"
                    style={{ maxHeight: '300px', objectFit: 'contain', width: '100%' }}
                  />
                  <div className="text-slate-400 text-xs mt-2">Video ready for multimodal analysis.</div>
                </div>
              )}
            </div>
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
              className="mt-1 w-5 h-5 rounded border-slate-600 bg-slate-700 text-blue-600 focus:ring-blue-500"
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
          className="w-full bg-gradient-to-br from-blue-700 to-blue-900 hover:from-blue-600 hover:to-blue-800 text-white px-6 py-3 rounded-lg font-medium text-lg transition-all duration-200 hover:shadow-lg disabled:opacity-50"
        >
          {loading ? 'Analyzing...' : 'Analyze Combined'}
        </button>
      )}

      {inputMode === 'video' && videoFile && (
        <button
          onClick={analyzeVideo}
          disabled={loading}
          className="w-full bg-gradient-to-br from-blue-700 to-blue-900 hover:from-blue-600 hover:to-blue-800 text-white px-6 py-3 rounded-lg font-medium text-lg transition-all duration-200 hover:shadow-lg disabled:opacity-50"
        >
          {loading ? 'Analyzing...' : 'Analyze Video'}
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
          <div className="rounded-2xl border border-slate-700 bg-gradient-to-br from-slate-800 via-slate-800 to-slate-900 p-5 shadow-lg">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-xl border border-cyan-300/10 bg-cyan-400/5 p-4">
                <div className="text-xs uppercase tracking-[0.3em] text-cyan-300/70 mb-2">Explainability</div>
                <div className="text-lg font-semibold text-slate-50 mb-1">
                  {showExplainability
                    ? (gradCam || saliency)
                      ? 'Explainability generated successfully.'
                      : explainabilityStatus?.errors?.length
                        ? 'Explainability requested, but map generation failed.'
                        : 'Explainability was requested but no maps were returned.'
                    : 'Explainability is off.'}
                </div>
                <div className="text-sm text-slate-300">
                  {showExplainability
                    ? (gradCam || saliency)
                      ? ''
                      : explainabilityStatus?.errors?.length
                        ? explainabilityStatus.errors.join(' | ')
                        : 'Enable Grad-CAM and saliency outputs to inspect model focus.'
                    : 'Enable the toggle to request Grad-CAM and saliency outputs.'}
                </div>
                <div className="mt-3 text-sm text-slate-200 leading-relaxed">
                  <div>{concordanceExplainabilityText.facialLine}</div>
                  <div>{concordanceExplainabilityText.speechLine}</div>
                  <div>{concordanceExplainabilityText.reasoningLine}</div>
                </div>
              </div>

              <div className={`rounded-xl border p-4 ${
                concordanceMetrics.categoryKey === 'MATCH'
                  ? 'bg-cyan-950/50 border border-cyan-700 text-cyan-100'
                  : concordanceMetrics.categoryKey === 'PARTIAL'
                    ? 'bg-amber-900/50 border border-amber-700 text-amber-200'
                    : concordanceMetrics.categoryKey === 'MISMATCH'
                      ? 'bg-red-900/50 border border-red-700 text-red-200'
                    : 'border-slate-600 bg-slate-900/40 text-slate-100'
              }`}>
                <div className="text-xs uppercase tracking-[0.3em] text-white/50 mb-2">Concordance</div>
                <div className="mt-2 text-3xl font-bold leading-none">
                  {formatConcordanceValue(concordanceMetrics.percent)}
                </div>
                <div className="mt-2 text-2xl font-semibold leading-tight">
                  {concordanceMetrics.categoryLabel || 'Unknown'}
                </div>
                <div className="mt-3 text-sm text-white/80">
                  Face: {facialEmotion || 'unknown'} | Speech: {speechEmotion || 'unknown'}
                </div>
              </div>
            </div>
          </div>

          {/* Results Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Facial Results */}
            <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
              <div className="bg-blue-900 px-4 py-2">
                <span className="text-blue-300 text-sm font-medium">Facial Emotion</span>
              </div>
              <div className="p-4">
                <div className="text-center mb-4">
                  <div className="text-xl font-bold text-slate-50">{formatEmotionLabel(facialEmotion)}</div>
                </div>
                {facialProbs && (
                  <div className="space-y-2">
                    {Object.entries(facialProbs).map(([emo, prob]) => (
                      <div key={emo} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-300">{formatEmotionLabel(emo)}</span>
                          <span className="text-slate-400">{(prob * 100).toFixed(0)}%</span>
                        </div>
                        <div className="w-full bg-slate-700 rounded-full h-1.5">
                          <div
                            className="bg-gradient-to-r from-blue-600 to-blue-800 h-1.5 rounded-full"
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
              <div className="bg-blue-900 px-4 py-2">
                <span className="text-blue-300 text-sm font-medium">Speech Emotion</span>
              </div>
              <div className="p-4">
                <div className="text-center mb-4">
                  <div className="text-xl font-bold text-slate-50">{formatEmotionLabel(speechEmotion)}</div>
                </div>
                {speechProbs && (
                  <div className="space-y-2">
                    {Object.entries(speechProbs).map(([emo, prob]) => (
                      <div key={emo} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-300">{formatEmotionLabel(emo)}</span>
                          <span className="text-slate-400">{(prob * 100).toFixed(0)}%</span>
                        </div>
                        <div className="w-full bg-slate-700 rounded-full h-1.5">
                          <div
                            className="bg-gradient-to-r from-blue-600 to-blue-800 h-1.5 rounded-full"
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
          {facialEmotion && inputMode === 'separate' && (
            <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
              <div className="bg-blue-900 px-4 py-2">
                <span className="text-blue-300 text-sm font-medium">Face Detection</span>
              </div>
              <div className="p-4">
                <p className="text-slate-400 text-sm mb-3">
                  {faceDetected
                    ? 'Face detected using MTCNN and boxed before explainability is computed.'
                    : 'Face box not found; full image was analyzed.'}
                </p>
                <img
                  src={annotatedFace ? `data:image/png;base64,${annotatedFace}` : imagePreview}
                  alt="Annotated"
                  className="w-full rounded-lg"
                  style={{ maxHeight: '300px', width: '100%', objectFit: 'contain', display: 'block', margin: '0 auto' }}
                />
              </div>
            </div>
          )}

          {/* Explainability Visualizations */}
          {(gradCam || saliency || waveform) && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4" style={{ alignItems: 'stretch' }}>
              {gradCam && (
                <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden lg:col-span-1" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                  <div className="bg-blue-900 px-4 py-2">
                    <span className="text-blue-300 text-sm font-medium">Facial Grad-CAM</span>
                  </div>
                  <div className="p-4" style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <img
                      src={`data:image/png;base64,${gradCam}`}
                      alt="Grad-CAM"
                      className="w-full rounded-lg"
                      style={{ width: '100%', objectFit: 'contain', maxHeight: '350px' }}
                    />
                  </div>
                </div>
              )}
              {waveform && (
                <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                  <div className="bg-blue-900 px-4 py-2">
                    <span className="text-blue-300 text-sm font-medium">Audio Spectrogram</span>
                  </div>
                  <div className="p-4" style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <img
                      src={`data:image/png;base64,${waveform}`}
                      alt="Audio Spectrogram"
                      className="w-full rounded-lg"
                      style={{ width: '100%', objectFit: 'contain', maxHeight: '350px' }}
                    />
                  </div>
                </div>
              )}
              {saliency && (
                <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                  <div className="bg-blue-900 px-4 py-2">
                    <span className="text-blue-300 text-sm font-medium">Audio Saliency</span>
                  </div>
                  <div className="p-4" style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <img
                      src={`data:image/png;base64,${saliency}`}
                      alt="Saliency"
                      className="w-full rounded-lg"
                      style={{ width: '100%', objectFit: 'contain', maxHeight: '350px' }}
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          {inputMode === 'video' && videoResult && (
            <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
              <div className="bg-blue-900 px-4 py-2">
                <span className="text-blue-300 text-sm font-medium">Video Analysis Details</span>
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
  const [modelStatus, setModelStatus] = useState(null);
  const [statusError, setStatusError] = useState('');

  useEffect(() => {
    let mounted = true;

    const loadStatus = async () => {
      try {
        const response = await axios.get(`${API_BASE}/api/models/status`);
        if (mounted) {
          setModelStatus(response.data || null);
          setStatusError('');
        }
      } catch (err) {
        if (mounted) {
          setStatusError(getApiErrorMessage(err, 'Failed to load model status'));
        }
      }
    };

    loadStatus();
    return () => {
      mounted = false;
    };
  }, []);

  const facialAccuracy = Number(modelStatus?.facial?.accuracy);
  const speechAccuracy = Number(modelStatus?.speech?.accuracy);

  return (
    <div className="space-y-4">
      {/* Model Details */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
        <div className="bg-blue-900 px-4 py-2">
          <span className="text-blue-300 text-sm font-medium">Model Details</span>
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
                  <span className="text-cyan-300 font-bold">
                    {Number.isFinite(facialAccuracy) ? `${(facialAccuracy * 100).toFixed(2)}%` : 'N/A'}
                  </span>
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
                  <span className="text-cyan-300 font-bold">
                    {Number.isFinite(speechAccuracy) ? `${(speechAccuracy * 100).toFixed(2)}%` : 'N/A'}
                  </span>
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
        <div className="bg-blue-900 px-4 py-2">
          <span className="text-blue-300 text-sm font-medium">System Info</span>
        </div>
        <div className="p-6">
          <div className="space-y-3 text-sm">
            <div className="flex justify-between items-center">
              <span className="text-slate-400">Device:</span>
              <span className="text-slate-200 font-medium">{modelStatus?.device || 'CPU/GPU (Auto-detected)'}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-400">Facial Model Status:</span>
              <span className={`${modelStatus?.facial?.loaded ? 'text-cyan-300' : 'text-amber-300'} font-medium`}>
                {modelStatus?.facial?.loaded ? 'Loaded' : 'Unavailable'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-400">Speech Model Status:</span>
              <span className={`${modelStatus?.speech?.loaded ? 'text-cyan-300' : 'text-amber-300'} font-medium`}>
                {modelStatus?.speech?.loaded ? 'Loaded' : 'Unavailable'}
              </span>
            </div>
            {statusError && (
              <div className="text-amber-300 text-xs pt-2">Live status unavailable: {statusError}</div>
            )}
          </div>
        </div>
      </div>

      {/* How to Use */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
        <div className="bg-blue-900 px-4 py-2">
          <span className="text-blue-300 text-sm font-medium">How to Use</span>
        </div>
        <div className="p-6">
          <div className="space-y-4 text-sm text-slate-300">
            <div>
              <h4 className="font-semibold text-slate-50 mb-2">Combined Analysis</h4>
              <p className="text-slate-400">
                Upload an image and audio file, or use a single video, to analyze facial and speech emotion together.
                This is the primary workflow and produces concordance plus explainability outputs.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-slate-50 mb-2">Facial Analysis</h4>
              <p className="text-slate-400">
                Use facial input alone when you want to inspect the image model independently.
                Grad-CAM helps you see which facial regions influenced the prediction.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-slate-50 mb-2">Speech Analysis</h4>
              <p className="text-slate-400">
                Use audio alone when you want to inspect the speech model independently.
                Audio saliency highlights which frequencies were most important.
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
  const location = useLocation();
  const redirectTo = location.state?.redirectTo || '/';

  const isSignup = mode === 'signup';

  const handleSubmit = async (e) => {
    // One submit handler supports both login and signup modes.
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
          navigate(redirectTo, { replace: true });
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
          navigate(redirectTo, { replace: true });
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
        <div className="ga-login-brand" aria-label="Project logo">
          <img src={logoImage} alt="Multi Modal Emotion Recognition logo" className="ga-login-logo" />
        </div>
        <div className="ga-auth-top">
          <button type="button" className="ga-auth-back" onClick={() => navigate('/')}>
            <span aria-hidden="true" className="ga-auth-back-chevron">&lt;</span>
            Back
          </button>
        </div>
        <div className="ga-login-title">{isSignup ? 'Create account' : 'Welcome back'}</div>
        <div className="ga-login-subtitle">
          {isSignup ? 'Join Multi Modal Emotion Recognition' : 'Sign in to Multi Modal Emotion Recognition'}
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
            onClick={() => navigate(isSignup ? '/login' : '/signup', { state: { redirectTo } })}
          >
            {isSignup ? 'Login' : 'Get Started'}
          </button>
        </div>
      </form>
    </div>
  );
}

function MarketingPage({ authUser, onLogout }) {
  const navigate = useNavigate();
  // Open profile links in a secure new tab from marketing/footer actions.
  const openExternal = (url) => window.open(url, '_blank', 'noopener,noreferrer');
  const linkedinUrl = 'https://www.linkedin.com/in/nishvaraj-k/';
  const githubUrl = 'https://github.com/Nishvaraj';
  const dashboardScreenshots = [
    '/screenshots/dashboard-1.png',
    '/screenshots/dashboard-2.png',
    '/screenshots/dashboard-3.png',
    '/screenshots/dashboard-4.png',
    '/screenshots/dashboard-5.png',
    '/screenshots/dashboard-6.png',
    '/screenshots/dashboard-7.png'
  ];
  const subtitle = 'Detecting emotions. Understanding humans.';
  const scenarios = useMemo(() => ([
    {
      label: 'Aligned Joy',
      face: 'Happy',
      voice: 'Happy',
      score: 96,
      status: 'Match',
      color: '#22c55e',
      desc: 'Face and voice align with authentic positive affect.'
    },
    {
      label: 'Suppressed Stress',
      face: 'Neutral',
      voice: 'Fearful',
      score: 22,
      status: 'Mismatch',
      color: '#ef4444',
      desc: 'Voice suggests distress while facial expression remains masked.'
    },
    {
      label: 'Polite Smile',
      face: 'Happy',
      voice: 'Neutral',
      score: 48,
      status: 'Partial Match',
      color: '#f59e0b',
      desc: 'Mild mismatch between expression and vocal energy.'
    },
    {
      label: 'Engaged Focus',
      face: 'Surprised',
      voice: 'Surprised',
      score: 88,
      status: 'Match',
      color: '#22c55e',
      desc: 'Both modalities indicate elevated attention and engagement.'
    }
  ]), []);
  const navItems = ['Architecture', 'Explainability', 'Concordance', 'Performance', 'Applications'];
  const sectionOrder = useMemo(() => ['architecture', 'explainability', 'concordance', 'performance', 'applications'], []);
  const pipelineSteps = [
    { id: '01', title: 'Input Capture', desc: 'Synchronized camera + microphone streams at real-time cadence.' },
    { id: '02', title: 'Vision Transformer', desc: 'Facial patches are encoded to predict affective states.' },
    { id: '03', title: 'HuBERT Audio', desc: 'Speech prosody and vocal dynamics classify emotion signatures.' },
    { id: '04', title: 'Concordance Engine', desc: 'Cross-modal agreement score identifies match, partial match, or mismatch.' },
    { id: '05', title: 'Explainability', desc: 'Grad-CAM and attention maps reveal model decision traces.' },
    { id: '06', title: 'Live Insight', desc: 'Unified result panel streams confidence and session summary.' }
  ];
  const useCases = [
    {
      title: 'Mental Health Tracking',
      desc: 'Observe emotional consistency trends across sessions to support therapeutic context.'
    },
    {
      title: 'Communication Coaching',
      desc: 'Identify tone-expression mismatch in interviews, public speaking, and leadership training.'
    },
    {
      title: 'Behavioral Research',
      desc: 'Run reproducible multimodal studies with explainable outputs for publication.'
    },
    {
      title: 'Accessible Feedback',
      desc: 'Help users better understand emotional signals through interpretable visual cues.'
    },
    {
      title: 'Client Fine-Tuning',
      desc: 'The workflow can be fine-tuned further for client-specific needs, domains, and emotional taxonomies.'
    }
  ];

  const [typedSubtitle, setTypedSubtitle] = useState('');
  const [activeSlide, setActiveSlide] = useState(0);
  const [attentionBars, setAttentionBars] = useState(() => Array.from({ length: 40 }, () => 0.2 + Math.random() * 0.75));
  const [activeScenario, setActiveScenario] = useState(0);
  const [neuralTick, setNeuralTick] = useState(0);
  const [hotspots, setHotspots] = useState([
    { x: 40, y: 34, w: 0.78 },
    { x: 60, y: 34, w: 0.74 },
    { x: 50, y: 58, w: 0.9 }
  ]);

  const pipelineRef = useRef(null);
  const xaiRef = useRef(null);
  const concordanceRef = useRef(null);
  const performanceRef = useRef(null);

  const [pipelineVisible, setPipelineVisible] = useState(false);
  const [xaiVisible, setXaiVisible] = useState(false);
  const [concordanceVisible, setConcordanceVisible] = useState(false);
  const [performanceVisible, setPerformanceVisible] = useState(false);
  const [activeSection, setActiveSection] = useState('architecture');
  const [showUserMenu, setShowUserMenu] = useState(false);

  // Build avatar initials from name/email for the navbar profile chip.
  const profileInitials = (() => {
    const base = authUser?.name || authUser?.email || 'User';
    const pieces = String(base).trim().split(/\s+/).filter(Boolean);
    if (pieces.length >= 2) return `${pieces[0][0]}${pieces[1][0]}`.toUpperCase();
    return String(base).slice(0, 2).toUpperCase();
  })();

  useEffect(() => {
    // Typewriter effect for the landing subtitle.
    let i = 0;
    const interval = setInterval(() => {
      if (i <= subtitle.length) {
        setTypedSubtitle(subtitle.slice(0, i));
        i += 1;
      } else {
        clearInterval(interval);
      }
    }, 45);
    return () => clearInterval(interval);
  }, [subtitle]);

  useEffect(() => {
    // Keep navbar section highlight synced with current scroll position.
    const offset = 64;
    const updateActiveSection = () => {
      const scrollTop = window.scrollY + offset;
      let current = sectionOrder[0];
      sectionOrder.forEach((id) => {
        const section = document.getElementById(id);
        if (section && section.offsetTop <= scrollTop) {
          current = id;
        }
      });
      setActiveSection(current);
    };

    updateActiveSection();
    window.addEventListener('scroll', updateActiveSection, { passive: true });
    return () => window.removeEventListener('scroll', updateActiveSection);
  }, [sectionOrder]);

  const scrollToSection = (id) => {
    // Smooth-scroll helper used by header and CTA buttons.
    const section = document.getElementById(id);
    if (section) {
      const top = section.getBoundingClientRect().top + window.scrollY - 18;
      window.scrollTo({ top, behavior: 'smooth' });
    }
  };

  useEffect(() => {
    // Animate synthetic bars/hotspots used in explainability and network visuals.
    const interval = setInterval(() => {
      setAttentionBars((prev) => prev.map((value) => {
        const target = 0.12 + Math.random() * 0.88;
        return value + (target - value) * 0.45;
      }));
      setHotspots((prev) => prev.map((point) => {
        const target = 0.3 + Math.random() * 0.7;
        return { ...point, w: point.w + (target - point.w) * 0.4 };
      }));
      setNeuralTick((tick) => tick + 1);
    }, 180);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // Rotate concordance demo scenarios on a timer.
    const scenarioTimer = setInterval(() => {
      setActiveScenario((index) => (index + 1) % scenarios.length);
    }, 4000);
    return () => {
      clearInterval(scenarioTimer);
    };
  }, [scenarios.length]);

  useEffect(() => {
    // Auto-advance screenshot slides while keeping manual controls available.
    const sliderTimer = setInterval(() => {
      setActiveSlide((index) => (index + 1) % dashboardScreenshots.length);
    }, 3500);
    return () => clearInterval(sliderTimer);
  }, [dashboardScreenshots.length]);

  useEffect(() => {
    // Trigger one-time reveal animation when sections enter the viewport.
    const sections = [
      { ref: pipelineRef, setVisible: setPipelineVisible },
      { ref: xaiRef, setVisible: setXaiVisible },
      { ref: concordanceRef, setVisible: setConcordanceVisible },
      { ref: performanceRef, setVisible: setPerformanceVisible }
    ];

    const observers = sections.map(({ ref, setVisible }) => {
      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              setVisible(true);
              observer.disconnect();
            }
          });
        },
        { threshold: 0.22 }
      );
      if (ref.current) {
        observer.observe(ref.current);
      }
      return observer;
    });

    return () => {
      observers.forEach((observer) => observer.disconnect());
    };
  }, []);

  const currentScenario = scenarios[activeScenario];
  const overlapGap = 40 - (currentScenario.score / 100) * 26;
  const navOverlay = '#0a0f1e';

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', 'dark');
    window.localStorage.setItem(THEME_STORAGE_KEY, 'dark');
  }, []);

  return (
    <div className="mmer-landing mmer-landing-dark min-h-screen overflow-x-hidden">
      <a
        href="#main-content"
        className="absolute left-4 top-4 z-[60] -translate-y-20 rounded-md bg-cyan-300 px-4 py-2 text-sm font-semibold text-[#041223] transition-transform focus:translate-y-0"
      >
        Skip to main content
      </a>
      <nav
        aria-label="Primary"
        className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-5 py-0 border-b border-white/10"
        style={{ background: navOverlay }}
      >
        <button
          type="button"
          className="flex items-center gap-2"
          aria-label="Project logo"
          onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
        >
          <img src={logoImage} alt="Multi Modal Emotion Recognition logo" className="mmer-nav-logo" />
        </button>
        <div className="hidden md:flex items-center gap-2 text-xs font-mono text-white/35 absolute left-1/2 -translate-x-1/2">
          {navItems.map((item) => (
            <button
              key={item}
              type="button"
              onClick={() => scrollToSection(item.toLowerCase())}
              aria-current={activeSection === item.toLowerCase() ? 'page' : undefined}
              className={`rounded-full px-3 py-1.5 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300 ${activeSection === item.toLowerCase() ? 'bg-cyan-400/15 text-cyan-200' : 'text-white/45 hover:text-cyan-200'}`}
            >
              {item}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          {!authUser && (
            <>
              <button className="ml-2 px-3 py-1.5 rounded-full text-sm border border-cyan-300/30 hover:bg-cyan-400/10 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300" onClick={() => navigate('/login', { state: { redirectTo: '/' } })}>Login</button>
              <button className="px-3 py-1.5 rounded-full text-sm bg-cyan-400/20 border border-cyan-300/40 hover:bg-cyan-400/30 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300" onClick={() => navigate('/signup', { state: { redirectTo: '/' } })}>Get Started</button>
            </>
          )}
          {authUser && (
            <div className="relative flex items-center gap-2">
              <button
                type="button"
                className="px-3 py-1.5 rounded-full text-sm bg-cyan-400/20 border border-cyan-300/40 hover:bg-cyan-400/30 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300"
                onClick={() => {
                  setShowUserMenu(false);
                  navigate('/app/dashboard');
                }}
              >
                Go to Dashboard
              </button>
              <button
                type="button"
                className="ga-profile"
                onClick={() => setShowUserMenu((prev) => !prev)}
                aria-haspopup="menu"
                aria-expanded={showUserMenu}
                title={authUser.name || authUser.email}
              >
                {profileInitials}
              </button>
              {showUserMenu && (
                <div className="absolute right-0 top-full mt-2 w-44 rounded-xl border border-cyan-300/30 bg-[#081525] shadow-lg p-1 z-[70]">
                  <button
                    type="button"
                    className="w-full text-left px-3 py-2 text-sm text-cyan-100 hover:bg-cyan-400/15 rounded-lg"
                    onClick={async () => {
                      setShowUserMenu(false);
                      await onLogout();
                    }}
                  >
                    Logout
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </nav>

      <main id="main-content" className="pt-10 relative">
        <div
          className="absolute inset-0 pointer-events-none opacity-20"
          aria-hidden="true"
          style={{
            backgroundImage: 'linear-gradient(rgba(148,163,184,0.25) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.25) 1px, transparent 1px)',
            backgroundSize: '32px 32px'
          }}
        />
        <section className="relative min-h-screen flex items-center px-4 py-20">
          <div className="relative z-10 max-w-6xl mx-auto w-full">
            <div className="grid lg:grid-cols-[1.1fr_0.9fr] gap-8 lg:gap-10 items-center">
              <div className="space-y-5 text-center lg:text-left">
                <p className="text-sm font-mono text-cyan-200/80 tracking-[0.3em] uppercase">Real-Time • Multi-Modal • Explainable</p>
                <h1 className="text-5xl md:text-7xl font-black text-white leading-[0.95]">Multi Modal Emotion Recognition</h1>
                <p aria-live="polite" className="text-xl md:text-3xl text-white/80 font-mono">{typedSubtitle}<span className="border-r-2 border-cyan-400 animate-pulse">&nbsp;</span></p>
                <p className="text-base md:text-lg text-white/70 max-w-2xl lg:max-w-none leading-relaxed">
                  Analyze face and voice together with transformer models, then explain predictions through visual evidence and concordance scoring.
                </p>
                <div className="pt-3 flex flex-wrap items-center justify-center lg:justify-start gap-3">
                  <button
                    type="button"
                    className="rounded-full bg-cyan-400/20 border border-cyan-300/45 px-5 py-2 text-base hover:bg-cyan-400/30 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300"
                    onClick={() => {
                      if (authUser) {
                        navigate('/app/dashboard');
                        return;
                      }
                      navigate('/login', { state: { redirectTo: '/app/dashboard' } });
                    }}
                  >
                    Go to Dashboard
                  </button>
                  <button
                    type="button"
                    className="rounded-full border border-cyan-300/35 px-5 py-2 text-base hover:bg-cyan-400/10 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300"
                    onClick={() => scrollToSection('architecture')}
                  >
                    Explore Architecture
                  </button>
                </div>
              </div>

              <div className="relative card-glass rounded-[28px] border border-cyan-300/20 p-3 sm:p-4">
                <div className="rounded-[22px] overflow-hidden bg-[#030914] aspect-[16/12] sm:aspect-[16/11] lg:aspect-[20/22] max-h-[640px] flex items-center justify-center">
                  <video
                    src="/landing-hero.mp4"
                    autoPlay
                    muted
                    loop
                    playsInline
                    preload="metadata"
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="absolute left-4 right-4 sm:left-6 sm:right-6 bottom-7 sm:bottom-7 rounded-2xl border border-cyan-300/30 bg-[#03101e]/85 backdrop-blur-md px-4 py-3 text-center">
                  <div className="flex items-center justify-center gap-2 mb-1">
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                    <p className="text-[10px] uppercase tracking-[0.25em] text-cyan-300/80 font-mono">Live Multimodal Inference</p>
                  </div>
                  <p className="text-xs text-white/75 leading-relaxed">ViT + HuBERT fusion with concordance scoring and explainable outputs.</p>
                </div>
              </div>
            </div>

            <div className="mt-14 flex flex-wrap justify-center gap-8 text-sm font-mono">
              {[
                { value: '71.29%', sub: 'ViT (FER2013)' },
                { value: '87.50%', sub: 'HuBERT (RAVDESS)' },
                { value: '35,887', sub: 'FER2013 Images' },
                { value: '1,440', sub: 'RAVDESS Files' }
              ].map((stat) => (
                <div key={stat.sub} className="text-center">
                  <div className="text-2xl font-bold shimmer-text">{stat.value}</div>
                  <div className="text-white/55 uppercase tracking-widest text-xs">{stat.sub}</div>
                </div>
              ))}
            </div>

            <div className="mt-6 overflow-x-auto">
              <p className="text-center whitespace-nowrap text-xs sm:text-sm font-mono uppercase tracking-[0.22em] text-white/55 px-3">
                Angry • Disgust • Fear • Happy • Neutral • Sad • Surprise • Calm • Fearful • Surprised
              </p>
            </div>
          </div>
        </section>

        <section id="architecture" ref={pipelineRef} className="pt-4 pb-12 sm:pt-4 sm:pb-20 lg:pb-20 px-4 scroll-mt-0">
          <div className="max-w-6xl mx-auto space-y-12">
            <div className="card-glass rounded-2xl p-4 sm:p-6">
              <div className="flex items-center justify-between gap-3 mb-4">
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-white/45">Dashboard Preview</p>
                  <h3 className="text-lg sm:text-2xl font-semibold text-white/90">Project Dashboard Screens</h3>
                </div>
                <div className="hidden sm:flex items-center gap-2">
                  <button
                    type="button"
                    aria-label="Previous screenshot"
                    onClick={() => setActiveSlide((index) => (index - 1 + dashboardScreenshots.length) % dashboardScreenshots.length)}
                    className="h-9 w-9 rounded-full border border-cyan-300/30 text-cyan-200 hover:bg-cyan-400/10"
                  >
                    ‹
                  </button>
                  <button
                    type="button"
                    aria-label="Next screenshot"
                    onClick={() => setActiveSlide((index) => (index + 1) % dashboardScreenshots.length)}
                    className="h-9 w-9 rounded-full border border-cyan-300/30 text-cyan-200 hover:bg-cyan-400/10"
                  >
                    ›
                  </button>
                </div>
              </div>

              <div className="relative overflow-hidden rounded-xl border border-cyan-300/20 bg-[#091524]">
                <div
                  className="flex transition-transform duration-500 ease-out"
                  style={{ transform: `translateX(-${activeSlide * 100}%)` }}
                >
                  {dashboardScreenshots.map((shot, idx) => (
                    <div key={shot} className="min-w-full">
                      <img
                        src={shot}
                        alt={`Dashboard screenshot ${idx + 1}`}
                        className="w-full h-[200px] sm:h-[280px] lg:h-[480px] object-contain bg-[#02060d]"
                        style={{
                          maxHeight: window.innerHeight < 600 && window.innerWidth > window.innerHeight ? '180px' : undefined
                        }}
                        loading={idx === 0 ? 'eager' : 'lazy'}
                      />
                    </div>
                  ))}
                </div>

                <div className="absolute inset-x-0 bottom-3 flex justify-center gap-2 px-3">
                  {dashboardScreenshots.map((_, idx) => (
                    <button
                      key={`dot-${idx}`}
                      type="button"
                      aria-label={`Go to screenshot ${idx + 1}`}
                      onClick={() => setActiveSlide(idx)}
                      className={`h-2.5 rounded-full transition-all ${activeSlide === idx ? 'w-7 bg-cyan-300' : 'w-2.5 bg-white/45 hover:bg-white/70'}`}
                    />
                  ))}
                </div>
              </div>

              <div className="mt-3 sm:hidden flex justify-center gap-2">
                <button
                  type="button"
                  aria-label="Previous screenshot"
                  onClick={() => setActiveSlide((index) => (index - 1 + dashboardScreenshots.length) % dashboardScreenshots.length)}
                  className="px-3 py-1.5 rounded-full border border-cyan-300/30 text-cyan-200 text-sm"
                >
                  Previous
                </button>
                <button
                  type="button"
                  aria-label="Next screenshot"
                  onClick={() => setActiveSlide((index) => (index + 1) % dashboardScreenshots.length)}
                  className="px-3 py-1.5 rounded-full border border-cyan-300/30 text-cyan-200 text-sm"
                >
                  Next
                </button>
              </div>
            </div>

            <div className="text-center space-y-3">
              <p className="inline-block px-3 py-1 text-xs font-mono tracking-[0.3em] uppercase border border-cyan-400/20 text-cyan-300/75 rounded-full">Architecture</p>
              <h2 className="text-4xl md:text-6xl font-bold">How the <span className="shimmer-text">pipeline</span> works</h2>
              <p className="text-white/70 max-w-2xl mx-auto text-base">Six stages transform raw camera and audio into actionable emotional intelligence.</p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
              {pipelineSteps.map((step, idx) => (
                <article
                  key={step.id}
                  className="card-glass rounded-2xl p-4 space-y-3 border border-cyan-300/10"
                  style={{
                    opacity: pipelineVisible ? 1 : 0,
                    transform: pipelineVisible ? 'translateY(0px)' : 'translateY(26px)',
                    transition: `opacity 0.6s ease ${idx * 120}ms, transform 0.6s ease ${idx * 120}ms`
                  }}
                >
                  <p className="text-2xl font-mono font-bold text-cyan-200">{step.id}</p>
                  <h3 className="text-base font-semibold text-white/90">{step.title}</h3>
                  <p className="text-sm text-white/65 leading-relaxed">{step.desc}</p>
                </article>
              ))}
            </div>
            <div className="card-glass rounded-2xl p-6">
              <h3 className="text-lg font-semibold text-white/90 mb-3">Pipeline Notes</h3>
              <ul className="list-disc list-inside space-y-2 text-sm text-white/70">
                <li>Input streams are synchronized before inference to preserve timing consistency.</li>
                <li>Each modality model contributes confidence and explainability artifacts.</li>
                <li>Concordance combines both outputs into one interpretable session summary.</li>
              </ul>
            </div>
          </div>
        </section>

        <section id="explainability" ref={xaiRef} className="pt-4 pb-12 sm:pt-4 sm:pb-20 lg:pb-20 px-4 scroll-mt-0">
          <div className="max-w-6xl mx-auto space-y-10">
            <div className="text-center space-y-3">
              <p className="inline-block px-3 py-1 text-xs font-mono tracking-[0.3em] uppercase border border-cyan-400/20 text-cyan-300/75 rounded-full">Explainability</p>
              <h2 className="text-4xl md:text-6xl font-bold">Transparent by <span className="shimmer-text">design</span></h2>
              <p className="text-white/70 max-w-2xl mx-auto text-base">See which facial regions and audio segments drove every prediction.</p>
            </div>
            <div className="grid md:grid-cols-2 gap-6">
              <article className="card-glass rounded-2xl p-6">
                <h3 className="text-base font-semibold text-white/90 mb-3">Grad-CAM Visual Heatmap</h3>
                <div className="h-56 rounded-xl bg-[#0a1626] relative overflow-hidden">
                  {hotspots.map((point, idx) => (
                    <span
                      key={idx}
                      className="absolute rounded-full mmer-hotspot"
                      style={{
                        left: `${point.x}%`,
                        top: `${point.y}%`,
                        width: `${26 + point.w * 42}px`,
                        height: `${22 + point.w * 36}px`,
                        transform: 'translate(-50%, -50%)',
                        opacity: xaiVisible ? 0.25 + point.w * 0.6 : 0.12,
                        animationDelay: `${idx * 0.2}s`
                      }}
                    />
                  ))}
                </div>
              </article>
              <article className="card-glass rounded-2xl p-6">
                <h3 className="text-base font-semibold text-white/90 mb-3">Audio Attention Map</h3>
                <div className="h-56 rounded-xl bg-[#0b1526] p-4 flex items-end gap-1">
                  {attentionBars.map((value, idx) => {
                    let color = '#4ade80';
                    if (value > 0.72) color = '#ef4444';
                    else if (value > 0.48) color = '#f59e0b';
                    return (
                      <span key={idx} className="flex-1 rounded-sm transition-all duration-150" style={{ height: `${15 + value * 85}%`, background: color, opacity: 0.45 + value * 0.55 }} />
                    );
                  })}
                </div>
              </article>
            </div>
            <div className="grid sm:grid-cols-3 gap-4">
              <div className="card-glass rounded-xl p-4">
                <h4 className="text-sm font-semibold text-cyan-200">Visual Saliency</h4>
                <p className="text-xs text-white/70 mt-2">Highlights eyes, mouth, and facial contour regions that influenced prediction.</p>
              </div>
              <div className="card-glass rounded-xl p-4">
                <h4 className="text-sm font-semibold text-cyan-200">Audio Saliency</h4>
                <p className="text-xs text-white/70 mt-2">Shows high-impact time-frequency zones in the input waveform.</p>
              </div>
              <div className="card-glass rounded-xl p-4">
                <h4 className="text-sm font-semibold text-cyan-200">Reason Trace</h4>
                <p className="text-xs text-white/70 mt-2">Provides interpretable evidence so model outcomes are auditable.</p>
              </div>
            </div>
          </div>
        </section>

        <section id="concordance" ref={concordanceRef} className="pt-4 pb-12 sm:pt-4 sm:pb-20 lg:pb-20 px-4 scroll-mt-0">
          <div className="max-w-6xl mx-auto space-y-10">
            <div className="text-center space-y-3">
              <p className="inline-block px-3 py-1 text-xs font-mono tracking-[0.3em] uppercase border border-cyan-400/20 text-cyan-300/75 rounded-full">Concordance</p>
              <h2 className="text-4xl md:text-6xl font-bold">Match, Partial, or <span className="shimmer-text">Mismatch?</span></h2>
              <p className="text-white/70 max-w-2xl mx-auto text-base">Cross-modal score reveals whether face and voice tell the same emotional story.</p>
            </div>
            <div className="flex justify-center">
              <article className="card-glass rounded-2xl p-3 w-full max-w-xl">
                <h3 className="text-base font-semibold text-white/90 mb-3">Live Concordance Snapshot</h3>
                <div className="h-[250px] rounded-xl border border-cyan-300/20 flex items-center justify-center bg-[#0a1627] relative overflow-hidden">
                  <svg viewBox="0 0 320 220" className="w-80 h-56">
                    <circle cx={160 - overlapGap} cy="88" r="72" fill="rgba(34,211,238,0.16)" stroke="#22d3ee" strokeOpacity="0.4" />
                    <circle cx={160 + overlapGap} cy="88" r="72" fill="rgba(59,130,246,0.16)" stroke="#3b82f6" strokeOpacity="0.4" />
                    <text x={160 - overlapGap - 20} y="84" fill="#22d3ee" fontSize="10" fontFamily="monospace">Face</text>
                    <text x={160 - overlapGap - 28} y="98" fill="#22d3ee" fontSize="10" fontFamily="monospace">{currentScenario.face}</text>
                    <text x={160 + overlapGap + 12} y="84" fill="#3b82f6" fontSize="10" fontFamily="monospace">Voice</text>
                    <text x={160 + overlapGap + 8} y="98" fill="#3b82f6" fontSize="10" fontFamily="monospace">{currentScenario.voice}</text>
                    <text x="160" y="188" textAnchor="middle" fill={currentScenario.color} fontSize="24" fontWeight="700" fontFamily="monospace">{currentScenario.score}%</text>
                  </svg>
                </div>
                <p className="text-sm text-white/70 mt-3 leading-relaxed">{currentScenario.status}: {currentScenario.desc}</p>
              </article>
            </div>
            <div className="grid sm:grid-cols-3 gap-4">
              <div className="card-glass rounded-xl p-4">
                <h4 className="text-sm font-semibold text-cyan-200">MATCH</h4>
                <p className="text-xs text-white/70 mt-2">Face and voice agree strongly, indicating emotionally consistent expression.</p>
              </div>
              <div className="card-glass rounded-xl p-4">
                <h4 className="text-sm font-semibold text-cyan-200">PARTIAL</h4>
                <p className="text-xs text-white/70 mt-2">Signals overlap but not fully; useful for nuanced emotional states.</p>
              </div>
              <div className="card-glass rounded-xl p-4">
                <h4 className="text-sm font-semibold text-cyan-200">MISMATCH</h4>
                <p className="text-xs text-white/70 mt-2">Face and voice diverge, often indicating masked or conflicted emotion.</p>
              </div>
            </div>
          </div>
        </section>

        <section id="performance" ref={performanceRef} className="pt-4 pb-12 sm:pt-4 sm:pb-20 lg:pb-20 px-4 scroll-mt-0">
          <div className="max-w-6xl mx-auto space-y-10">
            <div className="text-center space-y-3">
              <p className="inline-block px-3 py-1 text-xs font-mono tracking-[0.3em] uppercase border border-blue-400/20 text-blue-300/75 rounded-full">Performance</p>
              <h2 className="text-4xl md:text-6xl font-bold">Built on <span className="shimmer-text">state-of-the-art</span> models</h2>
              <p className="text-white/70 max-w-2xl mx-auto text-base">Fine-tuned transformers validated with practical latency and high confidence accuracy.</p>
            </div>
            <div className="grid md:grid-cols-2 gap-8 items-center">
              <div className="card-glass rounded-2xl p-6">
                <h3 className="text-base font-semibold text-white/90 mb-4">Fusion Network Activity</h3>
                <div className="aspect-[16/10] rounded-xl bg-[#0b1628] border border-blue-300/20 flex items-center justify-center">
                  <svg viewBox="0 0 340 220" className="w-full h-full p-4">
                    {Array.from({ length: 4 }, (_, i) => (
                      <circle key={`l-${i}`} cx={38} cy={32 + i * 45} r={4} fill="#22d3ee" opacity="0.85" />
                    ))}
                    {Array.from({ length: 6 }, (_, i) => (
                      <circle key={`m-${i}`} cx={170} cy={22 + i * 32} r={4} fill="#3b82f6" opacity="0.85" />
                    ))}
                    {Array.from({ length: 3 }, (_, i) => (
                      <circle key={`r-${i}`} cx={302} cy={60 + i * 46} r={4} fill="#34d399" opacity="0.9" />
                    ))}
                    {Array.from({ length: 4 }, (_, li) => (
                      Array.from({ length: 6 }, (_, mi) => {
                        const pulse = Math.sin((li * 5 + mi * 7 + neuralTick) * 0.15);
                        return (
                          <line
                            key={`lm-${li}-${mi}`}
                            x1={38}
                            y1={32 + li * 45}
                            x2={170}
                            y2={22 + mi * 32}
                            stroke="#60a5fa"
                            strokeWidth={pulse > 0.25 ? 1.2 : 0.5}
                            strokeOpacity={pulse > 0.25 ? 0.38 : 0.12}
                          />
                        );
                      })
                    ))}
                    {Array.from({ length: 6 }, (_, mi) => (
                      Array.from({ length: 3 }, (_, ri) => {
                        const pulse = Math.sin((mi * 3 + ri * 5 + neuralTick) * 0.17);
                        return (
                          <line
                            key={`mr-${mi}-${ri}`}
                            x1={170}
                            y1={22 + mi * 32}
                            x2={302}
                            y2={60 + ri * 46}
                            stroke="#34d399"
                            strokeWidth={pulse > 0.2 ? 1.15 : 0.45}
                            strokeOpacity={pulse > 0.2 ? 0.42 : 0.1}
                          />
                        );
                      })
                    ))}
                  </svg>
                </div>
              </div>
              <div className="space-y-4">
                {[
                  { label: 'ViT Accuracy (FER2013)', value: '71.29%', width: 71.29, color: '#22d3ee' },
                  { label: 'HuBERT Accuracy (RAVDESS)', value: '87.50%', width: 87.5, color: '#3b82f6' }
                ].map((metric, idx) => (
                  <div key={metric.label} className="card-glass rounded-xl p-4">
                    <div className="flex items-center justify-between text-sm mb-2">
                      <span className="text-white/80">{metric.label}</span>
                      <span style={{ color: metric.color }}>{metric.value}</span>
                    </div>
                    <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: performanceVisible ? `${metric.width}%` : '0%',
                          background: metric.color,
                          transition: `width 1.1s ease ${idx * 140}ms`
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="card-glass rounded-2xl p-6">
              <h3 className="text-lg font-semibold text-white/90 mb-3">Validation Summary</h3>
              <ul className="list-disc list-inside space-y-2 text-sm text-white/70">
                <li>FER2013 and RAVDESS are used as baseline evaluation datasets.</li>
                <li>Confidence calibration is tuned to improve session-level reliability.</li>
                <li>Latency profile targets interactive feedback under 500ms budget.</li>
              </ul>
            </div>
          </div>
        </section>

        <section id="applications" className="pt-4 pb-12 sm:pt-4 sm:pb-20 lg:pb-20 px-4 scroll-mt-0">
          <div className="max-w-6xl mx-auto space-y-10">
            <div className="text-center space-y-3">
              <p className="inline-block px-3 py-1 text-xs font-mono tracking-[0.3em] uppercase border border-cyan-400/20 text-cyan-300/75 rounded-full">Applications</p>
              <h2 className="text-4xl md:text-6xl font-bold">Built for <span className="shimmer-text">impact</span></h2>
              <p className="text-white/70 max-w-2xl mx-auto text-base">From clinical research to personal development, analyze emotional authenticity with confidence.</p>
            </div>
            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-5">
              {useCases.map((uc) => (
                <article
                  key={uc.title}
                  className={`card-glass rounded-2xl p-5 space-y-3 border border-cyan-300/10 ${uc.title === 'Client Fine-Tuning' ? 'sm:col-span-2 lg:col-span-2 lg:col-start-2' : ''}`}
                >
                  <h3 className="text-base font-semibold text-white/90">{uc.title}</h3>
                  <p className="text-sm text-white/70 leading-relaxed">{uc.desc}</p>
                </article>
              ))}
            </div>
            <div className="card-glass rounded-2xl p-8 text-center space-y-4 border border-cyan-300/20">
              <p className="text-sm font-mono text-white/45 uppercase tracking-[0.2em]">Privacy-first architecture</p>
              <h3 className="text-3xl md:text-4xl font-bold">Secure <span className="shimmer-text">cloud inference</span></h3>
              <p className="text-base text-white/75 max-w-2xl mx-auto">Emotion analysis runs on secure cloud infrastructure. No raw images or audio are stored permanently. Session data is protected with Row-Level Security.</p>
              <div className="flex flex-wrap justify-center gap-3 text-sm text-white/70">
                <span>Secure cloud processing</span>
                <span>•</span>
                <span>No permanent media storage</span>
                <span>•</span>
                <span>Row-Level Security (Supabase)</span>
              </div>
            </div>
            <div className="card-glass rounded-2xl p-6">
              <h3 className="text-lg font-semibold text-white/90 mb-3">Deployment Scenarios</h3>
              <ul className="list-disc list-inside space-y-2 text-sm text-white/70">
                <li>Clinical support dashboards with private on-device inference.</li>
                <li>Interview and coaching analysis for communication training.</li>
                <li>Academic and product research workflows with explainable evidence.</li>
              </ul>
            </div>
          </div>
        </section>

        <footer className="relative z-10 border-t border-white/10 py-12 px-4" style={{ background: navOverlay }}>
          <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8 items-start">
            <div className="space-y-2 text-center md:text-left">
              <div className="text-2xl font-bold text-white">Multi Modal Emotion Recognition</div>
              <p className="text-sm text-white/65 font-mono">Dual-modality emotion analysis with explainable inference.</p>
              <a
                href={linkedinUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-sm font-semibold text-cyan-300 hover:text-cyan-200"
              >
                LinkedIn Profile
              </a>
              <p className="text-xs text-white/45 mt-2">
                BSc Computer Science (Hons) Final Year Project · University of Westminster · 2026
              </p>
              <a
                href={githubUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-sm font-semibold text-cyan-300 hover:text-cyan-200"
              >
                GitHub Profile
              </a>
            </div>
            <div className="space-y-2 text-center">
              <p className="text-xs uppercase tracking-[0.2em] text-white/45">Technology Stack</p>
              <div className="flex flex-wrap justify-center gap-3 text-sm text-white/70">
                <span>Vision Transformer</span>
                <span>HuBERT</span>
                <span>Grad-CAM</span>
                <span>FastAPI</span>
                <span>React</span>
              </div>
            </div>
            <div className="space-y-2 text-center">
              <p className="text-xs uppercase tracking-[0.2em] text-white/45">System Status</p>
              <div className="flex items-center justify-center gap-2 text-sm text-white/70">
                <div className="w-2 h-2 rounded-full bg-cyan-400" />
                <span>Operational</span>
              </div>
              <p className="text-xs text-white/45">ViT + HuBERT · 30 FPS · &lt;500ms latency · WCAG 2.1 AA</p>
            </div>
          </div>
        </footer>

        <button
          type="button"
          aria-label="Scroll to top"
          onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
          className="fixed right-5 bottom-5 z-50 h-11 w-11 rounded-full border border-cyan-300/25 bg-[#071424] text-cyan-100 text-lg font-bold shadow-none hover:bg-[#0b1b31] focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300"
        >
          ↑
        </button>
      </main>
    </div>
  );
}

function DashboardIcon({ name, className = '' }) {
  // Central icon switch keeps sidebar and cards visually consistent.
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
  // Reusable KPI tile used across overview metrics.
  return (
    <article className="ga-kpi-card ga-kpi-summary-card">
      <div className="ga-kpi-title">{title}</div>
      <div className="ga-kpi-value">{value}</div>
      <div className={`ga-kpi-detail ${detailClass}`}>{detail}</div>
    </article>
  );
}

function ActivityHeatmapCard({ heatmapData }) {
  // GitHub-style heatmap summarizing analysis activity by day/week.
  return (
    <article className="ga-card ga-heatmap-card w-full max-w-none">
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

function ActivityTrendCard({ history }) {
  const weeklyVolumes = useMemo(
    () => buildWeeklySeries(history, { metric: 'volume', modality: 'all' }),
    [history]
  );
  const maxVolume = Math.max(1, ...weeklyVolumes);

  return (
    // Weekly bars provide a quick volume trend snapshot for recent sessions.
    <article className="ga-card ga-heatmap-card ga-activity-chart-card w-full max-w-none">
      <h3 className="ga-section-title ga-heatmap-title">Session Activity · Last 12 Weeks</h3>
      <div className="ga-activity-bars" role="img" aria-label="Weekly activity bar chart for last 12 weeks">
        {weeklyVolumes.map((value, idx) => {
          const height = value === 0 ? 0 : Math.max(2, Math.round((value / maxVolume) * 100));
          return (
            <div key={`wk-${idx}`} className="ga-activity-col" title={`Week ${idx + 1}: ${value} sessions`}>
              <div className="ga-activity-bar-wrap">
                <div className="ga-activity-bar" style={{ height: `${height}%` }} />
              </div>
              <div className="ga-activity-value">{value}</div>
            </div>
          );
        })}
      </div>
      <div className="ga-heatmap-scale">
        <span>12 weeks ago</span>
        <span>This week</span>
      </div>
    </article>
  );
}

function InsightCard({ icon, title, description }) {
  // Small narrative card for generated coaching/research insights.
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
  const avgConcordanceMetrics = getConcordanceMetrics(avgConcordance);
  const bestSessionMetrics = getConcordanceMetrics(bestSessionScore);

  return (
    <>
      <section className="ga-kpi-grid ga-kpi-summary-grid">
        <KpiCard
          title="Total Sessions"
          value={String(totalSessions)}
          detail={`${monthDelta >= 0 ? '↑' : '↓'} ${Math.abs(monthDelta)}% this month`}
          detailClass={monthDelta >= 0 ? 'up' : ''}
        />
        <KpiCard title="Avg Concordance" value={formatConcordanceValue(avgConcordance)} detail={`${avgConcordanceMetrics.categoryLabel} · ${weekDeltaText}`} detailClass={weeklyConcordanceDelta >= 0 ? 'up' : ''} />
        <KpiCard title="Best Session" value={formatConcordanceValue(bestSessionScore)} detail={`${bestSessionMetrics.categoryLabel} · ${bestSessionDate || '-'}`} />
        <KpiCard title="Streak" value={String(streakDays)} detail="days in a row" />
      </section>

      <section className="ga-overview-top-grid">
        <ActivityHeatmapCard heatmapData={heatmapData} />
        <ActivityTrendCard history={history} />
      </section>

      <section className="ga-insights-grid grid grid-cols-1 md:grid-cols-3 gap-4">
        <InsightCard
          icon="↗"
          title="Concordance improving"
          description={`Your average score ${weeklyConcordanceDelta >= 0 ? 'rose' : 'dropped'} ${Math.abs(weeklyConcordanceDelta)}% this week. ${history.length ? 'Keep practising daily sessions.' : 'Run a few sessions to unlock trend insights.'}`}
        />
        <InsightCard
          icon="◎"
          title={`${bestEmotionInsight.emotion} most consistent`}
          description={`${bestEmotionInsight.emotion} has the highest facial-speech alignment at ${formatConcordanceValue(bestEmotionInsight.rate)}.`}
        />
        <InsightCard
          icon="⚑"
          title={`${worstEmotionInsight.emotion} needs attention`}
          description={`${worstEmotionInsight.emotion} shows lowest concordance (${formatConcordanceValue(worstEmotionInsight.rate)}). Your face and voice diverge on this emotion.`}
        />
      </section>
    </>
  );
}

function EmotionTrendsTab({ analytics, history }) {
  const [selectedMetric, setSelectedMetric] = useState('concordance');
  const [selectedModality, setSelectedModality] = useState('all');

  const trendPoints = useMemo(
    () => buildWeeklySeries(history, { metric: selectedMetric, modality: selectedModality }),
    [history, selectedMetric, selectedModality]
  );

  // Keep true weekly counts for volume, but normalize only for chart coordinates.
  const trendChartPoints = useMemo(() => {
    if (selectedMetric !== 'volume') return trendPoints;
    const max = Math.max(...trendPoints, 0);
    if (max === 0) return trendPoints;
    return trendPoints.map((count) => Math.round((count / max) * 100));
  }, [trendPoints, selectedMetric]);

  const trendTitle = selectedMetric === 'confidence'
    ? 'Average confidence over time'
    : selectedMetric === 'volume'
      ? 'Session activity over time'
      : 'Concordance score over time';

  const trendTargetLine = selectedMetric === 'volume' ? 50 : 40;
  const xStep = 500 / Math.max(1, trendChartPoints.length - 1);
  const pointString = trendChartPoints
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
        <div className="ga-control-row">
          <div className="ga-section-title" style={{ marginBottom: 0 }}>{trendTitle}</div>
          <div className="ga-control-group">
            <select
              className="ga-select"
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
            >
              <option value="concordance">Concordance</option>
              <option value="confidence">Confidence</option>
              <option value="volume">Activity Volume</option>
            </select>
            <select
              className="ga-select"
              value={selectedModality}
              onChange={(e) => setSelectedModality(e.target.value)}
            >
              <option value="all">All Modalities</option>
              <option value="multimodal">Combined</option>
              <option value="facial">Facial</option>
              <option value="speech">Speech</option>
            </select>
          </div>
        </div>
        <div className="ga-mini-chart-wrap">
          <svg viewBox="0 0 500 100" className="ga-mini-chart" preserveAspectRatio="none" role="img" aria-label="Trend over last 12 weeks">
            <polyline points={pointString} className="ga-mini-line" />
            <polyline points={areaString} className="ga-mini-area" />
            <line x1="0" y1={trendTargetLine} x2="500" y2={trendTargetLine} className="ga-mini-target" />
          </svg>
        </div>
      </div>
      <div className="ga-design-two-col">
        <div className="ga-card">
          <div className="ga-section-title">Emotion Frequency</div>
          <div className="ga-bars">
            {analytics.emotionFrequency.map(([name, value]) => (
              <div key={name} className="ga-bar-row">
                <div className="ga-bar-label">{formatEmotionLabel(name)}</div>
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
                <div className="ga-bar-label">{formatEmotionLabel(name)}</div>
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
        <div key={card.title} className={`ga-card ga-feedback-card ${card.cls}`}>
          <div className="ga-feedback-title">{card.title}</div>
          <div className="ga-feedback-text">{card.text}</div>
        </div>
      ))}
    </div>
  );
}

function CompareSessionsTab({ analytics, history }) {
  const [modalityFilter, setModalityFilter] = useState('all');
  const comparableSessions = useMemo(
    () => [...history]
      .filter((row) => {
        if (modalityFilter === 'all') return true;
        if (modalityFilter === 'multimodal') {
          return row.modality === 'multimodal' || row.modality === 'combined';
        }
        return row.modality === modalityFilter;
      })
      .sort((a, b) => parseRecordTimestamp(b) - parseRecordTimestamp(a)),
    [history, modalityFilter]
  );
  const [selectedAId, setSelectedAId] = useState('');
  const [selectedBId, setSelectedBId] = useState('');
  const [selectedMetric, setSelectedMetric] = useState('concordance');

  useEffect(() => {
    if (comparableSessions.length < 2) {
      setSelectedAId('');
      setSelectedBId('');
      return;
    }

    const hasA = comparableSessions.some((row) => String(row.id) === String(selectedAId));
    const hasB = comparableSessions.some((row) => String(row.id) === String(selectedBId));

    if (!hasA) {
      setSelectedAId(String(comparableSessions[0].id));
    }
    if (!hasB || String(comparableSessions[0].id) === String(selectedBId)) {
      setSelectedBId(String(comparableSessions[1].id));
    }
  }, [comparableSessions, selectedAId, selectedBId]);

  const sessionA = comparableSessions.find((row) => String(row.id) === String(selectedAId)) || null;
  const sessionB = comparableSessions.find((row) => String(row.id) === String(selectedBId)) || null;

  if (!sessionA || !sessionB) {
    return (
      <div className="ga-design-stack">
        <div className="ga-card">
          <div className="ga-control-row">
            <div className="ga-section-title" style={{ marginBottom: 0 }}>Select sessions to compare</div>
            <div className="ga-control-group">
              <select className="ga-select" value={modalityFilter} onChange={(e) => setModalityFilter(e.target.value)}>
                <option value="all">All Session Types</option>
                <option value="multimodal">Combined/Multimodal</option>
                <option value="facial">Facial</option>
                <option value="speech">Speech</option>
              </select>
              <button className="ga-header-btn" onClick={() => setModalityFilter('all')}>Reset Filter</button>
            </div>
          </div>
          <p className="ga-empty">Need at least two sessions for the selected filter to compare. Adjust filter and try again.</p>
        </div>
      </div>
    );
  }

  const confidenceA = Math.round((sessionA.confidence || 0) * 100);
  const confidenceB = Math.round((sessionB.confidence || 0) * 100);
  const concordanceA = Math.round(getConcordanceScore(sessionA));
  const concordanceB = Math.round(getConcordanceScore(sessionB));
  const selectedValueA = selectedMetric === 'confidence' ? confidenceA : concordanceA;
  const selectedValueB = selectedMetric === 'confidence' ? confidenceB : concordanceB;
  const scoreDiff = selectedValueA - selectedValueB;
  const pairA = splitMultimodalEmotions(sessionA);
  const pairB = splitMultimodalEmotions(sessionB);
  const emotionPairA = pairA ? `${pairA.facial} | ${pairA.speech}` : sessionA.emotion;
  const emotionPairB = pairB ? `${pairB.facial} | ${pairB.speech}` : sessionB.emotion;
  const concordanceMetricsA = getConcordanceMetrics(sessionA);
  const concordanceMetricsB = getConcordanceMetrics(sessionB);
  const compareToneA = selectedMetric === 'concordance' ? getConcordanceToneClass(concordanceMetricsA.categoryKey) : '';
  const compareToneB = selectedMetric === 'concordance' ? getConcordanceToneClass(concordanceMetricsB.categoryKey) : '';

  return (
    <div className="ga-design-stack">
      <div className="ga-card">
        <div className="ga-control-row">
          <div className="ga-section-title" style={{ marginBottom: 0 }}>Select sessions to compare</div>
          <div className="ga-control-group">
            <select className="ga-select" value={modalityFilter} onChange={(e) => setModalityFilter(e.target.value)}>
              <option value="all">All Session Types</option>
              <option value="multimodal">Combined/Multimodal</option>
              <option value="facial">Facial</option>
              <option value="speech">Speech</option>
            </select>
            <select className="ga-select" value={selectedAId} onChange={(e) => setSelectedAId(e.target.value)}>
              {comparableSessions.map((row) => (
                <option key={`a-${row.id}`} value={row.id}>
                  A · {row.modality} · {new Date(row.createdAt).toLocaleString()}
                </option>
              ))}
            </select>
            <select className="ga-select" value={selectedBId} onChange={(e) => setSelectedBId(e.target.value)}>
              {comparableSessions.map((row) => (
                <option key={`b-${row.id}`} value={row.id}>
                  B · {row.modality} · {new Date(row.createdAt).toLocaleString()}
                </option>
              ))}
            </select>
            <button
              className="ga-header-btn"
              onClick={() => {
                setSelectedAId(String(sessionB.id));
                setSelectedBId(String(sessionA.id));
              }}
            >
              Swap
            </button>
            <select className="ga-select" value={selectedMetric} onChange={(e) => setSelectedMetric(e.target.value)}>
              <option value="concordance">Compare Concordance</option>
              <option value="confidence">Compare Confidence</option>
            </select>
          </div>
        </div>
      </div>
      <div className="ga-design-two-col">
        <div className="ga-card">
          <div className="ga-section-title">Session A · {formatDayMonth(new Date(parseRecordTimestamp(sessionA)))}</div>
          <div className={`ga-compare-score ${compareToneA}`}>
            {selectedMetric === 'concordance' ? formatConcordanceValue(concordanceMetricsA.percent) : formatPercent(selectedValueA)}
          </div>
          <div className="ga-compare-note">{selectedMetric === 'concordance' ? concordanceMetricsA.categoryLabel : (sessionA.concordance || 'Session')}</div>
          <div className="ga-compare-note">Emotion Pair: {emotionPairA}</div>
        </div>
        <div className="ga-card">
          <div className="ga-section-title">Session B · {formatDayMonth(new Date(parseRecordTimestamp(sessionB)))}</div>
          <div className={`ga-compare-score ${compareToneB}`}>
            {selectedMetric === 'concordance' ? formatConcordanceValue(concordanceMetricsB.percent) : formatPercent(selectedValueB)}
          </div>
          <div className="ga-compare-note">{selectedMetric === 'concordance' ? concordanceMetricsB.categoryLabel : (sessionB.concordance || 'Session')}</div>
          <div className="ga-compare-note">Emotion Pair: {emotionPairB}</div>
        </div>
      </div>
      <div className="ga-card">
        <div className="ga-section-title">What Changed</div>
        <div className="mt-4 space-y-6">
          <div className="rounded-2xl border border-cyan-300/10 bg-slate-950/40 p-4 sm:p-5">
            <div className="flex flex-wrap items-center gap-3 text-xs uppercase tracking-[0.25em] text-white/50 mb-4">
              <span className="inline-flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-cyan-400" /> Session A</span>
              <span className="inline-flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-blue-400" /> Session B</span>
              <span className="inline-flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-rose-400" /> Delta</span>
            </div>

            <div className="space-y-5">
              {[
                { label: 'Confidence', a: confidenceA, b: confidenceB },
                { label: 'Concordance', a: concordanceA, b: concordanceB }
              ].map((metric) => (
                <div key={metric.label} className="space-y-2">
                  <div className="flex items-center justify-between text-sm text-white/80">
                    <span>{metric.label}</span>
                    <span className="text-white/55">A {metric.a}% · B {metric.b}%</span>
                  </div>
                  <div className="grid grid-cols-[72px_1fr_72px_1fr] gap-3 items-center">
                    <div className="text-xs uppercase tracking-[0.2em] text-white/45 text-right">A</div>
                    <div className="h-3 rounded-full bg-white/10 overflow-hidden">
                      <div className="h-full rounded-full bg-cyan-400" style={{ width: `${metric.a}%` }} />
                    </div>
                    <div className="text-xs uppercase tracking-[0.2em] text-white/45 text-right">B</div>
                    <div className="h-3 rounded-full bg-white/10 overflow-hidden">
                      <div className="h-full rounded-full bg-blue-400" style={{ width: `${metric.b}%` }} />
                    </div>
                  </div>
                </div>
              ))}

              <div className="space-y-2 pt-1">
                <div className="flex items-center justify-between text-sm text-white/80">
                  <span>Concordance delta</span>
                  <span className={scoreDiff >= 0 ? 'text-cyan-300' : 'text-rose-300'}>{scoreDiff >= 0 ? '+' : ''}{scoreDiff}%</span>
                </div>
                <div className="h-4 rounded-full bg-white/10 overflow-hidden">
                  <div
                    className={`h-full rounded-full ${scoreDiff >= 0 ? 'bg-cyan-400' : 'bg-rose-400'}`}
                    style={{ width: `${Math.max(4, Math.min(100, Math.abs(scoreDiff)))}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ExportReportTab({ onExportSummary, analytics }) {
  const avgConcordanceMetrics = getConcordanceMetrics(analytics.avgConcordance);
  return (
    <div className="ga-design-stack">
      <div className="ga-card">
        <div className="ga-section-title">Summary · {new Date().toLocaleDateString(undefined, { month: 'long', year: 'numeric' })}</div>
        <div className="ga-report-grid">
          <div>Total sessions <strong>{analytics.totalSessions}</strong></div>
          <div>Average concordance <strong>{formatConcordanceValue(avgConcordanceMetrics.percent)}</strong></div>
          <div>Match sessions <strong>{analytics.matchCount}</strong></div>
          <div>Mismatch sessions <strong>{analytics.mismatchCount}</strong></div>
        </div>
      </div>
      <div className="ga-report-actions">
        <button className="ga-header-btn" onClick={onExportSummary}>Download PDF Report</button>
      </div>
    </div>
  );
}

function HistoryTab({ history, onTogglePin, onDelete, onUpdateNote }) {
  const [draftNotes, setDraftNotes] = useState({});

  useEffect(() => {
    const nextDrafts = {};
    history.forEach((row) => {
      nextDrafts[row.id] = row.note || '';
    });
    setDraftNotes(nextDrafts);
  }, [history]);

  return (
    <div className="ga-card">
      <div className="ga-section-title">Analysis History</div>
      {history.length === 0 ? (
        <p className="ga-empty">No analysis records yet. Run facial, speech, or multimodal analysis to populate history.</p>
      ) : (
        <div className="ga-table-wrap">
          <table className="ga-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Modality</th>
                <th>Result</th>
                <th>Confidence</th>
                <th>Explainability</th>
                <th>Concordance</th>
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
                  <td>{row.concordance ? `${getConcordanceMetrics(row).categoryLabel} · ${formatConcordanceValue(row)}` : '-'}</td>
                  <td>
                    <input
                      className="ga-note-input"
                      value={draftNotes[row.id] ?? ''}
                      placeholder="Add note"
                      onChange={(e) => {
                        const value = e.target.value;
                        setDraftNotes((prev) => ({ ...prev, [row.id]: value }));
                      }}
                    />
                  </td>
                  <td>
                    <div className="ga-row-actions">
                      <button className="ga-text-btn" onClick={() => onUpdateNote(row.id, draftNotes[row.id] ?? '')}>
                        Save
                      </button>
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
        </div>
      )}
    </div>
  );
}

const THEME_STORAGE_KEY = 'mmer_theme';

function DashboardConsole({ authUser, onLogout }) {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState(3);
  const [clearFacialSignal, setClearFacialSignal] = useState(0);
  const [clearSpeechSignal, setClearSpeechSignal] = useState(0);
  const [clearCombinedSignal, setClearCombinedSignal] = useState(0);
  const [search, setSearch] = useState('');
  const [dateValue, setDateValue] = useState('');
  const [savedOnly, setSavedOnly] = useState(false);
  const [history, setHistory] = useState([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [toasts, setToasts] = useState([]);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const logoDataUrlRef = useRef(null);
  const whiteLogoDataUrlRef = useRef(null);
  const toastCounterRef = useRef(0);

  const tabs = [
    { id: 0, icon: 'overview', label: 'Dashboard' },
    { id: 3, icon: 'multimodal', label: 'Combined Analysis' },
    { id: 1, icon: 'facial', label: 'Facial Upload' },
    { id: 2, icon: 'speech', label: 'Speech Upload' },
    { id: 8, icon: 'multimodal', label: 'Compare Sessions' },
    { id: 5, icon: 'history', label: 'Session History' },
    { id: 9, icon: 'conversion', label: 'Export Report' },
    { id: 4, icon: 'model', label: 'Model Info' }
  ];

  const navSections = [
    { label: 'Overview', tabIds: [0] },
    { label: 'Analysis', tabIds: [3, 1, 2] },
    { label: 'Insights', tabIds: [8] },
    { label: 'Records', tabIds: [5, 9, 4] }
  ];

  const tabDescriptions = {
    0: 'Overview and key session metrics.',
    1: 'Facial-only analysis with upload or webcam capture.',
    2: 'Speech-only analysis with upload or microphone recording.',
    3: 'Primary combined face + voice analysis using separate inputs or a single video.',
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
    document.documentElement.setAttribute('data-theme', 'dark');
    window.localStorage.setItem(THEME_STORAGE_KEY, 'dark');
  }, []);

  useEffect(() => {
    if (!toasts.length) return undefined;

    const timers = toasts.map((toast) => setTimeout(() => {
      setToasts((prev) => prev.filter((item) => item.id !== toast.id));
    }, 2800));

    return () => {
      timers.forEach((timerId) => clearTimeout(timerId));
    };
  }, [toasts]);

  const addToast = (message, type = 'info') => {
    toastCounterRef.current += 1;
    const id = `${Date.now()}-${toastCounterRef.current}`;
    setToasts((prev) => [...prev, { id, message, type }]);
  };

  const addHistoryRecord = async (record) => {
    // Save to Supabase
    const saved = await saveAnalysisToSupabase(record);
    if (saved) {
      // Reload history from Supabase
      const updated = await loadAnalysisHistoryFromSupabase();
      setHistory(updated);
      addToast('Session saved to history.', 'success');
    } else {
      addToast('Could not save session to history.', 'error');
    }
  };

  const togglePinHistory = async (id) => {
    const record = history.find((item) => item.id === id);
    if (record) {
      const toggled = await toggleAnalysisPin(id, record.pinned);
      if (toggled) {
        const updated = await loadAnalysisHistoryFromSupabase();
        setHistory(updated);
        addToast(record.pinned ? 'Session unpinned.' : 'Session pinned.', 'success');
      } else {
        addToast('Unable to update pin status.', 'error');
      }
    }
  };

  const deleteHistory = async (id) => {
    const deleted = await deleteAnalysisRecord(id);
    if (deleted) {
      const updated = await loadAnalysisHistoryFromSupabase();
      setHistory(updated);
      addToast('Session deleted.', 'success');
    } else {
      addToast('Failed to delete session.', 'error');
    }
  };

  const updateHistoryNote = async (id, note) => {
    const updated = await updateAnalysisNote(id, note);
    if (updated) {
      setHistory((prev) => prev.map((item) => (item.id === id ? { ...item, note } : item)));
      addToast('Note saved.', 'success');
      return;
    }
    addToast('Failed to save note.', 'error');
  };

  const filteredHistory = history
    .filter((row) => (dateValue ? row.createdAt.startsWith(dateValue) : true))
    .filter((row) => (savedOnly ? Boolean(row.pinned) : true))
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

  const ensureWhiteLogoDataUrl = async () => {
    if (whiteLogoDataUrlRef.current) return whiteLogoDataUrlRef.current;
    const logoDataUrl = await ensureLogoDataUrl();
    const image = new Image();
    image.src = logoDataUrl;
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = reject;
    });

    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const { data } = imageData;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i + 3] > 0) {
        data[i] = 255;
        data[i + 1] = 255;
        data[i + 2] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);
    whiteLogoDataUrlRef.current = canvas.toDataURL('image/png');
    return whiteLogoDataUrlRef.current;
  };

  const exportSummaryReport = async () => {
    if (!filteredHistory.length) {
      addToast('No records available to export.', 'info');
      return;
    }
    const byModality = filteredHistory.reduce((acc, cur) => {
      acc[cur.modality] = (acc[cur.modality] || 0) + 1;
      return acc;
    }, {});
    const totalRecords = filteredHistory.length;
    const avgConfidence = totalRecords
      ? Math.round((filteredHistory.reduce((sum, row) => sum + Number(row.confidence || 0), 0) / totalRecords) * 100)
      : 0;
    const avgConcordanceMetrics = getConcordanceMetrics(
      totalRecords ? filteredHistory.reduce((sum, row) => sum + getConcordanceScore(row), 0) / totalRecords : 0
    );
    const avgConcordance = totalRecords
      ? Math.round(filteredHistory.reduce((sum, row) => sum + getConcordanceScore(row), 0) / totalRecords)
      : 0;
    const matchCount = filteredHistory.filter((row) => getConcordanceMetrics(row).categoryKey === 'MATCH').length;
    const matchRate = totalRecords ? Math.round((matchCount / totalRecords) * 100) : 0;

    const doc = new jsPDF({ unit: 'pt', format: 'a4' });
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 34;
    const contentWidth = pageWidth - (margin * 2);

    const colors = {
      deep: [13, 34, 68],
      accent: [35, 92, 188],
      card: [246, 249, 254],
      stroke: [221, 230, 242],
      text: [31, 45, 65],
      muted: [103, 120, 144],
      white: [255, 255, 255],
      rowAlt: [250, 252, 255]
    };

    const toTitleCase = (value) => String(value || '')
      .replace(/[_-]+/g, ' ')
      .split(' ')
      .filter(Boolean)
      .map((part) => part[0].toUpperCase() + part.slice(1))
      .join(' ');

    const fitText = (text, maxWidth, suffix = '...') => {
      const raw = String(text || '');
      if (doc.getTextWidth(raw) <= maxWidth) return raw;
      let out = raw;
      while (out.length > 0 && doc.getTextWidth(`${out}${suffix}`) > maxWidth) {
        out = out.slice(0, -1);
      }
      return out.length ? `${out}${suffix}` : '';
    };

    doc.setFillColor(...colors.deep);
    doc.rect(0, 0, pageWidth, 108, 'F');
    doc.setFillColor(...colors.accent);
    doc.rect(0, 98, pageWidth, 10, 'F');

    try {
      const logoDataUrl = await ensureWhiteLogoDataUrl();
      const props = doc.getImageProperties(logoDataUrl);
      const maxW = 70;
      const maxH = 54;
      const ratio = props.width / props.height;
      let drawW = maxW;
      let drawH = maxH;
      if (ratio >= 1) {
        drawH = maxW / ratio;
      } else {
        drawW = maxH * ratio;
      }
      doc.addImage(logoDataUrl, 'PNG', margin, 26 + ((maxH - drawH) / 2), drawW, drawH);
    } catch (err) {
      // Continue if logo is not available.
    }

    doc.setTextColor(...colors.white);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(25);
    doc.text('Performance Summary', pageWidth / 2, 50, { align: 'center' });
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10);
    doc.text(`User: ${authUser?.email || 'N/A'}`, pageWidth / 2, 70, { align: 'center' });
    doc.text(`Range: ${dateValue || 'All records'}`, pageWidth / 2, 84, { align: 'center' });
    doc.text(new Date().toLocaleString(), pageWidth - margin, 92, { align: 'right' });

    let y = 124;
    doc.setTextColor(...colors.text);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(13);
    doc.text('Executive Snapshot', margin, y);
    y += 10;

    const cardGap = 8;
    const cardW = (contentWidth - (cardGap * 3)) / 4;
    const cardH = 58;
    const drawCard = (x, yPos, label, value, caption) => {
      doc.setFillColor(...colors.card);
      doc.setDrawColor(...colors.stroke);
      doc.roundedRect(x, yPos, cardW, cardH, 8, 8, 'FD');
      doc.setTextColor(...colors.muted);
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(7);
      doc.text(label.toUpperCase(), x + 10, yPos + 16);
      doc.setTextColor(...colors.text);
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(16);
      doc.text(String(value), x + 10, yPos + 34);
      doc.setTextColor(...colors.muted);
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(7);
      doc.text(fitText(caption, cardW - 20), x + 10, yPos + 48);
    };

    drawCard(margin, y, 'Total Sessions', totalRecords, 'Records in selected range');
    drawCard(margin + cardW + cardGap, y, 'Avg Confidence', `${avgConfidence}%`, 'Across all sessions');
    drawCard(margin + ((cardW + cardGap) * 2), y, 'Avg Concordance', `${avgConcordanceMetrics.percent}%`, 'Match quality percentage');
    drawCard(margin + ((cardW + cardGap) * 3), y, 'Match Rate', `${matchRate}%`, `${matchCount}/${totalRecords} sessions match`);

    y += cardH + 16;
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(12);
    doc.setTextColor(...colors.text);
    doc.text('Modality Distribution', margin, y);
    y += 10;

    const entries = Object.entries(byModality).sort((a, b) => b[1] - a[1]);
    entries.forEach(([modality, count], idx) => {
      const rowY = y + (idx * 18);
      const ratio = totalRecords ? count / totalRecords : 0;
      const labelW = 105;
      const valueW = 75;
      const barX = margin + labelW;
      const barW = contentWidth - labelW - valueW;
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(9);
      doc.setTextColor(...colors.text);
      doc.text(fitText(toTitleCase(modality), labelW - 6), margin, rowY + 9);
      doc.setFillColor(232, 238, 247);
      doc.roundedRect(barX, rowY, barW, 8, 4, 4, 'F');
      doc.setFillColor(...colors.accent);
      doc.roundedRect(barX, rowY, Math.max(4, barW * ratio), 8, 4, 4, 'F');
      doc.setTextColor(...colors.muted);
      doc.text(`${count} (${Math.round(ratio * 100)}%)`, margin + contentWidth, rowY + 9, { align: 'right' });
    });

    y += Math.max(1, entries.length) * 18 + 14;
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(12);
    doc.setTextColor(...colors.text);
    doc.text('Recent Sessions', margin, y);
    y += 8;

    const rowH = 20;
    const tableBottomPadding = 26;
    const cols = [
      { label: 'Timestamp', ratio: 0.27 },
      { label: 'Modality', ratio: 0.14 },
      { label: 'Emotion', ratio: 0.32 },
      { label: 'Conf.', ratio: 0.11 },
      { label: 'Conc.', ratio: 0.16 }
    ];
    const colsW = cols.map((col) => ({ ...col, width: Math.floor(contentWidth * col.ratio) }));
    const widthDelta = contentWidth - colsW.reduce((sum, col) => sum + col.width, 0);
    colsW[colsW.length - 1].width += widthDelta;

    const drawHeaderRow = (yPos) => {
      doc.setFillColor(236, 242, 250);
      doc.setDrawColor(...colors.stroke);
      doc.rect(margin, yPos, contentWidth, rowH, 'FD');
      let x = margin;
      doc.setTextColor(...colors.muted);
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(8);
      colsW.forEach((col) => {
        doc.text(col.label, x + 7, yPos + 13);
        x += col.width;
      });
    };

    const firstHeaderY = y;
    const firstDataStartY = firstHeaderY + rowH;
    const firstRowsPerPage = Math.max(1, Math.floor((pageHeight - tableBottomPadding - firstDataStartY) / rowH));

    const continuedHeaderY = margin + 34;
    const continuedDataStartY = continuedHeaderY + rowH;
    const nextRowsPerPage = Math.max(1, Math.floor((pageHeight - tableBottomPadding - continuedDataStartY) / rowH));

    const totalPages = filteredHistory.length <= firstRowsPerPage
      ? 1
      : 1 + Math.ceil((filteredHistory.length - firstRowsPerPage) / nextRowsPerPage);

    const drawDataRow = (row, rowY, rowIndex) => {
      if (rowIndex % 2 === 0) {
        doc.setFillColor(...colors.rowAlt);
        doc.rect(margin, rowY, contentWidth, rowH, 'F');
      }
      doc.setDrawColor(...colors.stroke);
      doc.rect(margin, rowY, contentWidth, rowH);

      const rowValues = [
        new Date(row.createdAt).toLocaleString(undefined, {
          month: 'short', day: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit'
        }),
        toTitleCase(row.modality),
        toTitleCase((row.emotion || '').replace('|', ' / ')),
        `${Math.round((row.confidence || 0) * 100)}%`,
        row.concordance || '-'
      ];

      let x = margin;
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(8.5);
      doc.setTextColor(...colors.text);
      rowValues.forEach((value, valueIdx) => {
        const colWidth = colsW[valueIdx].width;
        doc.text(fitText(value, colWidth - 12), x + 7, rowY + 13);
        x += colWidth;
      });
    };

    let dataIndex = 0;
    let pageNumber = 1;

    drawHeaderRow(firstHeaderY);
    let rowY = firstDataStartY;
    for (let i = 0; i < firstRowsPerPage && dataIndex < filteredHistory.length; i += 1) {
      drawDataRow(filteredHistory[dataIndex], rowY, dataIndex);
      rowY += rowH;
      dataIndex += 1;
    }

    while (dataIndex < filteredHistory.length) {
      doc.addPage();
      pageNumber += 1;

      doc.setFont('helvetica', 'bold');
      doc.setFontSize(11);
      doc.setTextColor(...colors.text);
      doc.text('Recent Sessions (continued)', margin, continuedHeaderY - 10);

      drawHeaderRow(continuedHeaderY);
      rowY = continuedDataStartY;
      for (let i = 0; i < nextRowsPerPage && dataIndex < filteredHistory.length; i += 1) {
        drawDataRow(filteredHistory[dataIndex], rowY, dataIndex);
        rowY += rowH;
        dataIndex += 1;
      }
    }

    doc.setTextColor(...colors.muted);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8);
    for (let page = 1; page <= pageNumber; page += 1) {
      doc.setPage(page);
      doc.text(`Page ${page} of ${pageNumber}`, pageWidth - margin, pageHeight - 16, { align: 'right' });
    }

    doc.save(`performance-summary-${dateValue || 'all'}.pdf`);
    addToast('PDF export started.', 'success');
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

    const emotionCounts = {};
    const emotionQuality = {};
    safeHistory.forEach((row) => {
      const labels = extractEmotionLabels(row);
      if (!labels.length) return;

      labels.forEach((emotion) => {
        emotionCounts[emotion] = (emotionCounts[emotion] || 0) + 1;
        emotionQuality[emotion] = emotionQuality[emotion] || { sum: 0, count: 0 };
        emotionQuality[emotion].sum += getConcordanceScore(row);
        emotionQuality[emotion].count += 1;
      });
    });

    const frequencyPairs = Object.entries(emotionCounts)
      .map(([emotion, count]) => [emotion, Math.round((count / Math.max(1, Object.values(emotionCounts).reduce((a, b) => a + b, 0))) * 100)])
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);

    const concordancePairs = Object.entries(emotionQuality)
      .map(([emotion, stats]) => [emotion, Math.round(stats.sum / Math.max(1, stats.count))])
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);

    const bestEmotionInsight = concordancePairs[0] || ['No Data', 0];
    const worstEmotionInsight = concordancePairs[concordancePairs.length - 1] || ['No Data', 0];

    const matchCount = safeHistory.filter((row) => getConcordanceMetrics(row).categoryKey === 'MATCH').length;
    const mismatchCount = safeHistory.filter((row) => getConcordanceMetrics(row).categoryKey === 'MISMATCH').length;

    const feedbackCards = [
      {
        title: `${formatEmotionLabel(bestEmotionInsight[0])} is your strongest pattern`,
        text: `${formatEmotionLabel(bestEmotionInsight[0])} currently leads with an average quality score of ${bestEmotionInsight[1]}%.`,
        cls: 'positive'
      },
      {
        title: `${formatEmotionLabel(worstEmotionInsight[0])} needs focused improvement`,
        text: `${formatEmotionLabel(worstEmotionInsight[0])} is at ${worstEmotionInsight[1]}%. Try short daily sessions and compare outcomes in the Compare Sessions tab.`,
        cls: 'warning'
      },
      {
        title: totalSessions < 3 ? 'Suggestion · run at least 3 sessions' : 'Suggestion · keep trend consistency',
        text: totalSessions < 3
          ? 'You have limited data right now. Add more facial, speech, or combined sessions to unlock stronger trend insights.'
          : `Your weekly trend is ${weeklyConcordanceDelta >= 0 ? 'improving' : 'declining'} by ${Math.abs(weeklyConcordanceDelta)}%. Keep sessions consistent for better signal quality.`,
        cls: 'suggestion'
      }
    ];

    const latestComparableSessions = [...safeHistory]
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
      emotionFrequency: frequencyPairs.length ? frequencyPairs : [['No Data', 0]],
      emotionConcordance: concordancePairs.length ? concordancePairs : [['No Data', 0]],
      bestEmotionInsight: { emotion: formatEmotionLabel(bestEmotionInsight[0]), rate: bestEmotionInsight[1] },
      worstEmotionInsight: { emotion: formatEmotionLabel(worstEmotionInsight[0]), rate: worstEmotionInsight[1] },
      feedbackCards,
      latestComparableSessions,
      matchCount,
      mismatchCount
    };
  }, [history]);

  return (
    <div className="ga-layout" style={{ display: 'flex', height: '100vh', gap: 0, margin: 0, padding: 0, marginTop: 0, paddingTop: 0 }}>
      <aside className="ga-sidebar" style={{ width: '260px', minWidth: '260px', maxWidth: '260px', flexShrink: 0, margin: 0, padding: 0, marginTop: 0, paddingTop: 0 }}>
        <div className="ga-brand-wrap">
          <div className="ga-brand-top">
            <button
              type="button"
              onClick={() => navigate('/')}
              aria-label="Go to landing page"
              style={{ background: 'transparent', border: 'none', padding: 0, cursor: 'pointer' }}
            >
              <img src={logoImage} alt="Multi Modal Emotion Recognition logo" className="ga-brand-logo" />
            </button>
          </div>
          <div className="ga-brand-subtitle">Multi Modal Emotion Recognition</div>
        </div>
        <nav className="ga-nav">
          {navSections.map((section) => (
            <div key={section.label} className="ga-nav-section" style={{ marginBottom: '4px' }}>
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
                    <span className="ga-nav-item-text" style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{tab.label}</span>
                    {tab.badge && <span className="ga-nav-badge">{tab.badge}</span>}
                  </button>
                );
              })}
            </div>
          ))}
        </nav>
        <div className="ga-sidebar-footer" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px' }}>
          <div className="ga-user-chip" style={{ flex: 1 }}>
            <div className="ga-user-avatar">{profileInitials}</div>
            <div className="ga-user-email">{authUser.email}</div>
          </div>
          <button className="ga-text-btn" onClick={async () => onLogout('/')}>Logout</button>
        </div>
      </aside>

      <div className="ga-content-wrap" style={{ flex: 1, overflow: 'auto', margin: 0, padding: 0 }}>
        <header className="ga-header">
          <div className="ga-top-title">{activeTabLabel}</div>
          <div className="ga-header-actions">
            <button className="ga-live-session-btn" onClick={() => setActiveTab(3)}>+ Start Combined Analysis</button>
            <div className="relative">
              <button
                className="ga-profile"
                title={authUser?.name || authUser?.email || 'Account'}
                onClick={() => setShowUserMenu((prev) => !prev)}
                aria-haspopup="menu"
                aria-expanded={showUserMenu}
              >
                {profileInitials}
              </button>
              {showUserMenu && (
                <div className="absolute right-0 mt-2 w-40 rounded-xl border border-cyan-300/30 bg-[#081525] shadow-lg p-1 z-[80]">
                  <button
                    type="button"
                    className="w-full text-left px-3 py-2 text-sm text-cyan-100 hover:bg-cyan-400/15 rounded-lg"
                    onClick={async () => {
                      setShowUserMenu(false);
                      await onLogout();
                    }}
                  >
                    Logout
                  </button>
                </div>
              )}
            </div>
          </div>
        </header>

        <main className="ga-main">
          {toasts.length > 0 && (
            <div className="ga-toast-stack" role="status" aria-live="polite">
              {toasts.map((toast) => (
                <div key={toast.id} className={`ga-toast ${toast.type}`}>
                  {toast.message}
                </div>
              ))}
            </div>
          )}

          {activeTab === 5 && (
            <div className="ga-record-toolbar">
              <input
                className="ga-search"
                placeholder="Search emotions, notes, explainability..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
              <input type="date" className="ga-date" value={dateValue} onChange={(e) => setDateValue(e.target.value)} />
              <button
                className={`ga-header-btn ${savedOnly ? 'active' : ''}`}
                onClick={() => setSavedOnly((prev) => !prev)}
                title="Show only saved sessions"
              >
                <span className="inline-flex items-center gap-2">
                  <svg viewBox="0 0 24 24" className="ga-glyph" aria-hidden="true">
                    <path d="M7 4h10a1 1 0 011 1v15l-6-3-6 3V5a1 1 0 011-1z" />
                  </svg>
                  <span>{savedOnly ? 'All Sessions' : 'Saved'}</span>
                </span>
              </button>
              <button className="ga-header-btn" onClick={exportSummaryReport} disabled={!filteredHistory.length}>Export PDF</button>
            </div>
          )}

          {activeTab !== 5 && activeTab !== 9 && (
            <div className="ga-page-title-row">
              <div>
                <h1 className="ga-page-title">{activeTabLabel}</h1>
                <p className="ga-page-subtitle">{activeDescription}</p>
              </div>
              <div className="ga-context-panel">
                <div className="ga-context-chips">
                  <span className="ga-chip">Today: {todayRecords} records</span>
                  <span className="ga-chip">Total: {history.length} records</span>
                </div>
                {(activeTab === 1 || activeTab === 2 || activeTab === 3) && (
                  <button
                    type="button"
                    className="ga-next-analysis-btn"
                    onClick={() => {
                      if (activeTab === 1) {
                        setClearFacialSignal((prev) => prev + 1);
                        return;
                      }
                      if (activeTab === 2) {
                        setClearSpeechSignal((prev) => prev + 1);
                        return;
                      }
                      setClearCombinedSignal((prev) => prev + 1);
                    }}
                  >
                    Clear Current Analysis
                  </button>
                )}
              </div>
            </div>
          )}

          {activeTab === 0 && <OverviewTab history={history} analytics={analytics} />}
          <div className="ga-card ga-tab-shell" style={{ display: activeTab === 1 ? 'block' : 'none' }}>
            <FacialTab onResult={addHistoryRecord} clearSignal={clearFacialSignal} />
          </div>
          <div className="ga-card ga-tab-shell" style={{ display: activeTab === 2 ? 'block' : 'none' }}>
            <SpeechTab onResult={addHistoryRecord} clearSignal={clearSpeechSignal} />
          </div>
          <div className="ga-card ga-tab-shell" style={{ display: activeTab === 3 ? 'block' : 'none' }}>
            <CombinedTab onResult={addHistoryRecord} clearSignal={clearCombinedSignal} />
          </div>
          {activeTab === 8 && <div className="ga-card ga-tab-shell"><CompareSessionsTab analytics={analytics} history={history} /></div>}
          {activeTab === 4 && <div className="ga-card ga-tab-shell"><ModelInfoTab /></div>}
          {activeTab === 9 && <div className="ga-card ga-tab-shell"><ExportReportTab onExportSummary={exportSummaryReport} analytics={analytics} /></div>}
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
  // Prevent logged-in users from reopening login/signup routes.
  if (isAuthenticated) {
    return <Navigate to="/" replace state={{ from: location }} />;
  }
  return children;
}

function ProtectedRoute({ isAuthenticated, children }) {
  const location = useLocation();
  // Block private routes unless a valid auth session exists.
  if (!isAuthenticated) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }
  return children;
}

function AppRouter() {
  const navigate = useNavigate();
  const [authUser, setAuthUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Force dark theme as the single supported visual mode.
    document.documentElement.setAttribute('data-theme', 'dark');
    window.localStorage.setItem(THEME_STORAGE_KEY, 'dark');
  }, []);

  useEffect(() => {
    // Initialize auth state and subscribe to Supabase auth changes.
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
    // Shared callback used by login/signup to populate app auth context.
    setAuthUser(user);
  };

  const handleLogout = async (redirectTo = '/') => {
    // Clear Supabase session and local caches before redirect.
    try {
      await supabase.auth.signOut();
    } finally {
      setAuthUser(null);
      window.localStorage.removeItem(AUTH_STORAGE_KEY);
      window.sessionStorage.removeItem(AUTH_STORAGE_KEY);
      navigate(redirectTo, { replace: true });
    }
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
    // Route map for marketing, auth, and protected dashboard areas.
    <Routes>
      <Route
        path="/"
        element={
          <MarketingPage authUser={authUser} onLogout={handleLogout} />
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
  // Top-level router wrapper.
  return (
    <BrowserRouter>
      <AppRouter />
    </BrowserRouter>
  );
}

export default App;
