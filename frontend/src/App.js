import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

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
};

const EMOTIONS_FACIAL = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];
const EMOTIONS_SPEECH = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'];
const API_BASE = 'http://127.0.0.1:8000';

// ============== NEW TAB COMPONENTS ==============

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
  const [showExplainability, setShowExplainability] = useState(false);
  const [gradCam, setGradCam] = useState(null);

  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = (evt) => {
        setImagePreview(evt.target.result);
        setEmotion(null);
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
      
      // Add explain parameter if explainability is enabled
      const explainParam = showExplainability ? '?explain=true' : '';
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
    <div className="tab-content">
      <h2>📸 Facial Emotion Recognition</h2>
      <div className="two-column">
        <div className="column">
          <h3>Capture Photo</h3>
          {!isCameraOn ? (
            <button className="btn btn-primary" onClick={startCamera}>📷 Start Webcam</button>
          ) : (
            <>
              <video ref={videoRef} autoPlay playsInline style={{ width: '100%', borderRadius: '8px', marginBottom: '10px', maxHeight: '300px' }} />
              <canvas ref={canvasRef} width="320" height="240" style={{ display: 'none' }} />
              <div className="button-group">
                <button className="btn btn-success" onClick={capturePhoto}>📸 Capture</button>
                <button className="btn btn-danger" onClick={stopCamera}>❌ Stop</button>
              </div>
            </>
          )}
        </div>
        <div className="column">
          <h3>Upload Image</h3>
          <input type="file" accept="image/*" onChange={handleImageSelect} className="file-input" />
        </div>
      </div>
      {imagePreview && (
        <>
          <img src={imagePreview} alt="Preview" style={{ width: '100%', maxHeight: '200px', borderRadius: '8px', marginBottom: '10px' }} />
          
          {/* Explainability Toggle */}
          <div style={{ marginBottom: '15px', padding: '10px', backgroundColor: '#f0f0f0', borderRadius: '8px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
              <input 
                type="checkbox" 
                checked={showExplainability} 
                onChange={(e) => setShowExplainability(e.target.checked)}
                style={{ cursor: 'pointer', width: '18px', height: '18px' }}
              />
              <span style={{ fontWeight: 'bold', fontSize: '14px' }}>
                🔍 Show Grad-CAM Visualization
              </span>
            </label>
            <p style={{ fontSize: '12px', color: '#666', margin: '5px 0 0 28px' }}>
              See which facial regions influenced the prediction
            </p>
          </div>
          
          <button className="btn btn-primary" onClick={analyzeFacial} disabled={loading} style={{ width: '100%' }}>
            {loading ? '⏳ Analyzing...' : '🔮 Analyze Face'}
          </button>
        </>
      )}
      {error && <div className="error-message">{error}</div>}
      {emotion && (
        <div className="results-box">
          <h3>{EMOTION_EMOJIS[emotion]} {emotion.toUpperCase()} ({(confidence * 100).toFixed(1)}%)</h3>
          {probabilities && (
            <div className="probabilities">
              {EMOTIONS_FACIAL.map((emo) => (
                <div key={emo} className="prob-item">
                  <span>{EMOTION_EMOJIS[emo]} {emo}</span>
                  <div className="prob-bar">
                    <div className="prob-fill" style={{ width: `${(probabilities[emo] || 0) * 100}%` }} />
                  </div>
                  <span>{((probabilities[emo] || 0) * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          )}
          
          {/* Grad-CAM Visualization */}
          {gradCam && (
            <div style={{ marginTop: '20px', padding: '15px', backgroundColor: '#f9f9f9', borderRadius: '8px', border: '2px solid #4CAF50' }}>
              <h4 style={{ marginTop: 0 }}>🔥 Grad-CAM Heatmap</h4>
              <p style={{ fontSize: '12px', color: '#666', marginBottom: '10px' }}>
                Red regions = areas the model focused on | Blue regions = less important
              </p>
              <img 
                src={`data:image/png;base64,${gradCam}`} 
                alt="Grad-CAM Heatmap"
                style={{ width: '100%', maxHeight: '300px', borderRadius: '8px', border: '1px solid #ddd' }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

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
  const analyzerRef = useRef(null);
  const canvasAnalyzerRef = useRef(null);
  const [showExplainability, setShowExplainability] = useState(false);
  const [saliency, setSaliency] = useState(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];
      
      // Setup audio visualization
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const analyser = audioContext.createAnalyser();
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      analyzerRef.current = analyser;
      
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
      
      // Start visualization
      visualizeAudio(analyser);
    } catch (err) {
      setError('Cannot access microphone');
    }
  };
  
  const visualizeAudio = (analyser) => {
    const canvas = canvasAnalyzerRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const draw = () => {
      if (!isRecording) return;
      
      requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArray);
      
      ctx.fillStyle = '#f0f0f0';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = '#667eea';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      const sliceWidth = canvas.width / bufferLength;
      let x = 0;
      
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;
        
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        
        x += sliceWidth;
      }
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    };
    draw();
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
      
      // Add explain parameter if explainability is enabled
      const explainParam = showExplainability ? '?explain=true' : '';
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
    <div className="tab-content">
      <h2>🎤 Speech Emotion Recognition</h2>
      <div className="two-column">
        <div className="column">
          <h3>Record Audio</h3>
          {!isRecording ? (
            <button className="btn btn-primary" onClick={startRecording}>🎙️ Start Recording</button>
          ) : (
            <>
              <p style={{ color: '#e74c3c', fontWeight: '600' }}>🔴 Recording...</p>
              <canvas 
                ref={canvasAnalyzerRef} 
                width="300" 
                height="80" 
                style={{ 
                  width: '100%', 
                  border: '2px solid #667eea', 
                  borderRadius: '4px', 
                  marginBottom: '10px', 
                  backgroundColor: '#f0f0f0' 
                }} 
              />
              <button className="btn btn-danger" onClick={stopRecording}>⏹️ Stop Recording</button>
            </>
          )}
          {audioFile && !isRecording && <p style={{ color: '#667eea', fontWeight: '600' }}>✅ Audio ready</p>}
        </div>
        <div className="column">
          <h3>Upload Audio</h3>
          <input type="file" accept="audio/*" onChange={handleAudioSelect} className="file-input" />
        </div>
      </div>
      {audioFile && (
        <>
          {/* Explainability Toggle */}
          <div style={{ marginBottom: '15px', marginTop: '15px', padding: '10px', backgroundColor: '#f0f0f0', borderRadius: '8px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
              <input 
                type="checkbox" 
                checked={showExplainability} 
                onChange={(e) => setShowExplainability(e.target.checked)}
                style={{ cursor: 'pointer', width: '18px', height: '18px' }}
              />
              <span style={{ fontWeight: 'bold', fontSize: '14px' }}>
                📊 Show Audio Saliency Map
              </span>
            </label>
            <p style={{ fontSize: '12px', color: '#666', margin: '5px 0 0 28px' }}>
              See which frequencies influenced the prediction
            </p>
          </div>
          
          <button className="btn btn-primary" onClick={analyzeSpeech} disabled={loading} style={{ width: '100%' }}>
            {loading ? '⏳ Analyzing...' : '🔮 Analyze Audio'}
          </button>
        </>
      )}
      {error && <div className="error-message">{error}</div>}
      {emotion && (
        <div className="results-box">
          <h3>{EMOTION_EMOJIS[emotion]} {emotion.toUpperCase()} ({(confidence * 100).toFixed(1)}%)</h3>
          {probabilities && (
            <div className="probabilities">
              {EMOTIONS_SPEECH.map((emo) => (
                <div key={emo} className="prob-item">
                  <span>{EMOTION_EMOJIS[emo]} {emo}</span>
                  <div className="prob-bar">
                    <div className="prob-fill" style={{ width: `${(probabilities[emo] || 0) * 100}%` }} />
                  </div>
                  <span>{((probabilities[emo] || 0) * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          )}
          
          {/* Audio Saliency Visualization */}
          {saliency && (
            <div style={{ marginTop: '20px', padding: '15px', backgroundColor: '#f9f9f9', borderRadius: '8px', border: '2px solid #FF6B6B' }}>
              <h4 style={{ marginTop: 0 }}>📊 Audio Saliency Map</h4>
              <p style={{ fontSize: '12px', color: '#666', marginBottom: '10px' }}>
                Red frequencies = important for prediction | Blue frequencies = less important
              </p>
              <img 
                src={`data:image/png;base64,${saliency}`} 
                alt="Audio Saliency"
                style={{ width: '100%', maxHeight: '300px', borderRadius: '8px', border: '1px solid #ddd' }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

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
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isRecordingVideo, setIsRecordingVideo] = useState(false);
  const [showExplainability, setShowExplainability] = useState(false);
  const [gradCam, setGradCam] = useState(null);
  const [saliency, setSaliency] = useState(null);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const streamRef = useRef(null);
  const analyzerRef = useRef(null);
  const canvasAnalyzerRef = useRef(null);

  // Ensure video plays when camera is on
  useEffect(() => {
    if (isCameraOn && videoRef.current && videoRef.current.srcObject) {
      videoRef.current.play().catch(err => console.log('Video play error:', err));
    }
  }, [isCameraOn]);

  const startVideoRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 1280, height: 720 },
        audio: true 
      });
      
      // Store stream first
      streamRef.current = stream;
      
      // Set up video element
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        
        // Wait a moment for the video element to be ready, then play it
        setTimeout(() => {
          if (videoRef.current) {
            videoRef.current.play().catch(err => {
              console.error('Video play failed:', err);
              setError('Video playback error: ' + err.message);
            });
          }
        }, 100);
      }
      
      chunksRef.current = [];
      setIsCameraOn(true);
      setError(null);
      
      // Setup audio visualization
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const analyser = audioContext.createAnalyser();
      const audioSource = audioContext.createMediaStreamSource(stream);
      audioSource.connect(analyser);
      analyzerRef.current = analyser;
      
      // Start video + audio recording
      const mediaRecorder = new MediaRecorder(stream, { 
        mimeType: 'video/webm;codecs=vp8,opus' 
      });
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.ondataavailable = (e) => chunksRef.current.push(e.data);
      mediaRecorder.start();
      setIsRecordingVideo(true);
      
      // Visualize audio during recording
      visualizeAudio(analyser);
    } catch (err) {
      setError('Cannot access camera/microphone: ' + err.message);
      console.error('Camera/microphone error:', err);
    }
  };

  const visualizeAudio = (analyser) => {
    const canvas = canvasAnalyzerRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const draw = () => {
      if (!isRecordingVideo) return;
      
      requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArray);
      
      ctx.fillStyle = '#f0f0f0';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = '#667eea';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      const sliceWidth = canvas.width * 1.0 / bufferLength;
      let x = 0;
      
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;
        
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        
        x += sliceWidth;
      }
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    };
    draw();
  };

  const stopVideoRecording = () => {
    if (mediaRecorderRef.current && isRecordingVideo) {
      mediaRecorderRef.current.onstop = () => {
        const videoBlob = new Blob(chunksRef.current, { type: 'video/webm' });
        
        // Extract frame from video
        const video = document.createElement('video');
        video.src = URL.createObjectURL(videoBlob);
        video.muted = true;
        
        video.onloadedmetadata = () => {
          // Extract frame from middle of video for better quality
          const canvas = document.createElement('canvas');
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const ctx = canvas.getContext('2d');
          
          // Seek to middle of video
          video.currentTime = Math.min(video.duration * 0.5, 2);
          
          video.onseeked = () => {
            ctx.drawImage(video, 0, 0);
            canvas.toBlob((imageBlob) => {
              if (imageBlob) {
                setImageFile(imageBlob);
                const reader = new FileReader();
                reader.onload = (evt) => setImagePreview(evt.target.result);
                reader.readAsDataURL(imageBlob);
              }
            });
          };
        };
        
        // Note: Audio extraction from webm is complex, user must upload audio separately
        setError('Video captured! Now upload audio file to complete the analysis.');
      };
      
      mediaRecorderRef.current.stop();
      setIsRecordingVideo(false);
      
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    }
  };

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
      
      // Add explain parameter if explainability is enabled
      const explainParam = showExplainability ? '?explain=true' : '';
      const response = await axios.post(`${API_BASE}/api/predict/combined${explainParam}`, formData);
      if (response.data.success) {
        setFacialEmotion(response.data.facial_emotion.emotion);
        setSpeechEmotion(response.data.speech_emotion.emotion);
        setConcordance(response.data.concordance);
        setFacialProbs(response.data.facial_emotion.probabilities);
        setSpeechProbs(response.data.speech_emotion.probabilities);
        
        // Get explainability visualizations if available
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
    <div className="tab-content">
      <h2>🎥 Combined Video + Analysis</h2>
      
      {/* VIDEO RECORDING SECTION */}
      <div style={{ marginBottom: '20px', paddingBottom: '20px', borderBottom: '1px solid #ddd' }}>
        <h3>🎬 Record Video (webcam + voice)</h3>
        <p style={{ color: '#666', fontSize: '0.9em', marginBottom: '10px' }}>Record a video with your face and voice together - it will automatically extract image and audio for analysis</p>
        
        {!isCameraOn ? (
          <button className="btn btn-primary" onClick={startVideoRecording} style={{ padding: '12px 20px', fontSize: '16px' }}>
            🎥 Start Video Recording
          </button>
        ) : (
          <>
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              muted
              style={{ 
                width: '100%', 
                height: 'auto',
                minHeight: '300px',
                borderRadius: '8px', 
                marginBottom: '10px', 
                backgroundColor: '#000',
                border: '3px solid #667eea',
                display: 'block',
                objectFit: 'cover'
              }} 
            />
            <canvas ref={canvasRef} width="320" height="240" style={{ display: 'none' }} />
            
            {isRecordingVideo && (
              <>
                <p style={{ color: '#e74c3c', fontWeight: '600', marginBottom: '10px' }}>
                  🔴 RECORDING... (Video + Audio)
                </p>
                <canvas 
                  ref={canvasAnalyzerRef} 
                  width="600" 
                  height="100" 
                  style={{ 
                    width: '100%', 
                    border: '2px solid #667eea', 
                    borderRadius: '4px', 
                    marginBottom: '10px', 
                    backgroundColor: '#f0f0f0' 
                  }} 
                />
              </>
            )}
            
            <div className="button-group">
              <button className="btn btn-danger" onClick={stopVideoRecording} style={{ padding: '10px 20px' }}>
                ⏹️ Stop & Extract
              </button>
            </div>
          </>
        )}
        
        {imageFile && audioFile && (
          <p style={{ color: '#27ae60', fontWeight: '600', marginTop: '10px' }}>
            ✅ Image & Audio extracted from video
          </p>
        )}
      </div>
      
      {/* MANUAL UPLOAD SECTION */}
      <div style={{ marginBottom: '20px', paddingBottom: '20px', borderBottom: '1px solid #ddd' }}>
        <h3>📤 Or Upload Separately</h3>
        <div className="two-column">
          <div className="column">
            <h4>📸 Image File</h4>
            <input type="file" accept="image/*" onChange={handleImageSelect} className="file-input" />
            {imagePreview && (
              <img 
                src={imagePreview} 
                alt="Preview" 
                style={{ width: '100%', maxHeight: '150px', marginTop: '10px', borderRadius: '8px' }} 
              />
            )}
            {imageFile && <p style={{ color: '#667eea', fontWeight: '600' }}>✅ Ready</p>}
          </div>
          <div className="column">
            <h4>🎤 Audio File</h4>
            <input type="file" accept="audio/*" onChange={handleAudioSelect} className="file-input" />
            {audioFile && <p style={{ color: '#667eea', fontWeight: '600' }}>✅ Ready</p>}
          </div>
        </div>
      </div>
      
      {/* ANALYZE SECTION */}
      {imageFile && audioFile && (
        <>
          {/* Explainability Toggle */}
          <div style={{ marginBottom: '15px', padding: '10px', backgroundColor: '#f0f0f0', borderRadius: '8px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
              <input 
                type="checkbox" 
                checked={showExplainability} 
                onChange={(e) => setShowExplainability(e.target.checked)}
                style={{ cursor: 'pointer', width: '18px', height: '18px' }}
              />
              <span style={{ fontWeight: 'bold', fontSize: '14px' }}>
                🔍 Show Explainability (Grad-CAM + Saliency)
              </span>
            </label>
            <p style={{ fontSize: '12px', color: '#666', margin: '5px 0 0 28px' }}>
              See visualizations of what the models focused on
            </p>
          </div>
          
          <button 
            className="btn btn-primary" 
            onClick={analyzeCombined} 
            disabled={loading} 
            style={{ width: '100%', padding: '12px', fontSize: '16px', marginBottom: '20px' }}
          >
            {loading ? '⏳ Analyzing...' : '🚀 Analyze Both'}
          </button>
        </>
      )}
      
      {error && <div className="error-message">{error}</div>}
      
      {/* RESULTS SECTION */}
      {facialEmotion && speechEmotion && (
        <div className="results-box">
          <h3 style={{ marginBottom: '15px' }}>
            📊 Results - {concordance === 'MATCH' ? '✅ MATCH!' : '⚠️ MISMATCH'}
          </h3>
          <div className="two-column">
            <div className="column results-col">
              <h4>📸 Facial: {EMOTION_EMOJIS[facialEmotion]} {facialEmotion.toUpperCase()}</h4>
              {facialProbs && (
                <div className="probabilities">
                  {Object.entries(facialProbs).slice(0, 4).map(([emo, prob]) => (
                    <div key={emo} className="prob-item" style={{ fontSize: '0.85em' }}>
                      <span>{EMOTION_EMOJIS[emo]} {emo}</span>
                      <div className="prob-bar"><div className="prob-fill" style={{ width: `${prob * 100}%` }} /></div>
                      <span>{(prob * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div className="column results-col">
              <h4>🎤 Speech: {EMOTION_EMOJIS[speechEmotion]} {speechEmotion.toUpperCase()}</h4>
              {speechProbs && (
                <div className="probabilities">
                  {Object.entries(speechProbs).slice(0, 4).map(([emo, prob]) => (
                    <div key={emo} className="prob-item" style={{ fontSize: '0.85em' }}>
                      <span>{EMOTION_EMOJIS[emo]} {emo}</span>
                      <div className="prob-bar"><div className="prob-fill" style={{ width: `${prob * 100}%` }} /></div>
                      <span>{(prob * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
          
          {/* Explainability Visualizations */}
          {(gradCam || saliency) && (
            <div style={{ marginTop: '20px', padding: '15px', backgroundColor: '#f0f0f0', borderRadius: '8px' }}>
              <h4 style={{ marginTop: 0 }}>🔍 Model Explainability</h4>
              <div className="two-column" style={{ gap: '15px' }}>
                {gradCam && (
                  <div style={{ padding: '10px', backgroundColor: 'white', borderRadius: '8px', border: '2px solid #4CAF50' }}>
                    <h5 style={{ marginTop: 0, marginBottom: '10px' }}>🔥 Facial Grad-CAM</h5>
                    <p style={{ fontSize: '11px', color: '#666', marginBottom: '10px' }}>
                      Red = important facial regions | Blue = less important
                    </p>
                    <img 
                      src={`data:image/png;base64,${gradCam}`} 
                      alt="Grad-CAM"
                      style={{ width: '100%', borderRadius: '8px', border: '1px solid #ddd' }}
                    />
                  </div>
                )}
                {saliency && (
                  <div style={{ padding: '10px', backgroundColor: 'white', borderRadius: '8px', border: '2px solid #FF6B6B' }}>
                    <h5 style={{ marginTop: 0, marginBottom: '10px' }}>📊 Audio Saliency</h5>
                    <p style={{ fontSize: '11px', color: '#666', marginBottom: '10px' }}>
                      Red = important frequencies | Blue = less important
                    </p>
                    <img 
                      src={`data:image/png;base64,${saliency}`} 
                      alt="Saliency"
                      style={{ width: '100%', borderRadius: '8px', border: '1px solid #ddd' }}
                    />
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ModelInfoTab() {
  return (
    <div className="tab-content">
      <h2>ℹ️ Model Information</h2>
      <div className="two-column">
        <div className="column">
          <h3>📸 Facial Emotion (ViT)</h3>
          <ul>
            <li><strong>Architecture:</strong> Vision Transformer</li>
            <li><strong>Dataset:</strong> FER2013 (35,887 images)</li>
            <li><strong>Emotions:</strong> 7 classes</li>
            <li><strong>Accuracy:</strong> 71.29%</li>
            <li><strong>Size:</strong> 327MB</li>
            <li><strong>Input:</strong> 224×224 RGB</li>
          </ul>
        </div>
        <div className="column">
          <h3>🎤 Speech Emotion (HuBERT)</h3>
          <ul>
            <li><strong>Architecture:</strong> HuBERT Large</li>
            <li><strong>Dataset:</strong> RAVDESS (1,440 files)</li>
            <li><strong>Emotions:</strong> 8 classes</li>
            <li><strong>Accuracy:</strong> 87.50%</li>
            <li><strong>Size:</strong> ~360MB</li>
            <li><strong>Input:</strong> 16kHz Mono</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState(0);
  
  // Separate Testing - Facial
  const [facialImage, setFacialImage] = useState(null);
  const [facialImageUrl, setFacialImageUrl] = useState(null);
  const [facialAnnotatedUrl, setFacialAnnotatedUrl] = useState(null);
  const [facialEmotion, setFacialEmotion] = useState(null);
  const [facialConfidence, setFacialConfidence] = useState(null);
  const [facialProbs, setFacialProbs] = useState(null);
  const [isProcessingFacial, setIsProcessingFacial] = useState(false);
  const facialVideoRef = useRef(null);
  const facialCanvasRef = useRef(null);
  
  // Separate Testing - Speech
  const [speechAudio, setSpeechAudio] = useState(null);
  const [speechEmotion, setSpeechEmotion] = useState(null);
  const [speechConfidence, setSpeechConfidence] = useState(null);
  const [speechProbs, setSpeechProbs] = useState(null);
  const [isProcessingSpeech, setIsProcessingSpeech] = useState(false);
  const [isRecordingSpeech, setIsRecordingSpeech] = useState(false);
  const speechAudioRef = useRef(null);
  const speechMediaRecorderRef = useRef(null);
  const speechChunksRef = useRef([]);
  const speechStreamRef = useRef(null);
  
  // Combined Testing
  const [combinedImage, setCombinedImage] = useState(null);
  const [combinedImageUrl, setCombinedImageUrl] = useState(null);
  const [combinedAnnotatedUrl, setCombinedAnnotatedUrl] = useState(null);
  const [combinedAudio, setCombinedAudio] = useState(null);
  const [combinedFacialEmotion, setCombinedFacialEmotion] = useState(null);
  const [combinedSpeechEmotion, setCombinedSpeechEmotion] = useState(null);
  const [combinedComparison, setCombinedComparison] = useState(null);
  const [combinedFacialProbs, setCombinedFacialProbs] = useState(null);
  const [combinedSpeechProbs, setCombinedSpeechProbs] = useState(null);
  const [isProcessingCombined, setIsProcessingCombined] = useState(false);
  const combinedVideoRef = useRef(null);
  const combinedCanvasRef = useRef(null);
  
  // Video Analysis
  const [videoFile, setVideoFile] = useState(null);
  const [videoResult, setVideoResult] = useState(null);
  const [isProcessingVideo, setIsProcessingVideo] = useState(false);

  const API_URL = 'http://127.0.0.1:8000';

  // ============== FACIAL FUNCTIONS ==============
  
  const handleFacialFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setFacialImage(file);
      setFacialImageUrl(URL.createObjectURL(file));
      setFacialEmotion(null);
      setFacialAnnotatedUrl(null);
      setFacialProbs(null);
    }
  };

  const startFacialWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 224, height: 224 } });
      facialVideoRef.current.srcObject = stream;
      facialVideoRef.current.style.display = 'block';
    } catch (error) {
      alert('Cannot access webcam: ' + error.message);
    }
  };

  const captureFacialImage = () => {
    const video = facialVideoRef.current;
    const canvas = facialCanvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    canvas.toBlob((blob) => {
      const file = new File([blob], 'facial_capture.jpg', { type: 'image/jpeg' });
      setFacialImage(file);
      setFacialImageUrl(canvas.toDataURL());
      setFacialEmotion(null);
      setFacialAnnotatedUrl(null);
      setFacialProbs(null);
      video.srcObject.getTracks().forEach(track => track.stop());
      facialVideoRef.current.style.display = 'none';
    });
  };

  const predictFacialEmotion = async () => {
    if (!facialImage) {
      alert('Please select or capture an image');
      return;
    }

    setIsProcessingFacial(true);
    const formData = new FormData();
    formData.append('file', facialImage);

    try {
      const response = await axios.post(`${API_URL}/api/predict/facial`, formData);
      const emotion = response.data.emotion || response.data.prediction || 'Unknown';
      const confidence = response.data.confidence || response.data.max_probability || 0;
      
      setFacialEmotion(emotion);
      setFacialConfidence(confidence);
      
      if (response.data.annotated_image) {
        setFacialAnnotatedUrl('data:image/jpeg;base64,' + response.data.annotated_image);
      }
      
      if (response.data.probabilities) {
        setFacialProbs(response.data.probabilities);
      } else if (response.data.all_scores) {
        setFacialProbs(response.data.all_scores);
      }
    } catch (error) {
      alert('Error: ' + error.message);
    } finally {
      setIsProcessingFacial(false);
    }
  };

  // ============== SPEECH FUNCTIONS ==============
  
  const handleSpeechFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSpeechAudio(file);
      speechAudioRef.current.src = URL.createObjectURL(file);
      setSpeechEmotion(null);
      setSpeechProbs(null);
    }
  };

  const startSpeechRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      speechStreamRef.current = stream;
      const mediaRecorder = new MediaRecorder(stream);
      speechMediaRecorderRef.current = mediaRecorder;
      speechChunksRef.current = [];
      setIsRecordingSpeech(true);

      mediaRecorder.ondataavailable = (event) => {
        speechChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(speechChunksRef.current, { type: 'audio/wav' });
        const audioFile = new File([audioBlob], 'speech_recording.wav', { type: 'audio/wav' });
        setSpeechAudio(audioFile);
        speechAudioRef.current.src = URL.createObjectURL(audioBlob);
        stream.getTracks().forEach(track => track.stop());
        setIsRecordingSpeech(false);
        setSpeechEmotion(null);
        setSpeechProbs(null);
      };

      mediaRecorder.start();
    } catch (error) {
      alert('Cannot access microphone: ' + error.message);
    }
  };

  const stopSpeechRecording = () => {
    if (speechMediaRecorderRef.current && isRecordingSpeech) {
      speechMediaRecorderRef.current.stop();
    }
  };

  const predictSpeechEmotion = async () => {
    if (!speechAudio) {
      alert('Please record or upload audio');
      return;
    }

    setIsProcessingSpeech(true);
    const formData = new FormData();
    formData.append('file', speechAudio);

    try {
      const response = await axios.post(`${API_URL}/api/predict/speech`, formData);
      const emotion = response.data.emotion || response.data.prediction || 'Unknown';
      const confidence = response.data.confidence || response.data.max_probability || 0;
      
      setSpeechEmotion(emotion);
      setSpeechConfidence(confidence);
      
      if (response.data.probabilities) {
        setSpeechProbs(response.data.probabilities);
      } else if (response.data.all_scores) {
        setSpeechProbs(response.data.all_scores);
      }
    } catch (error) {
      alert('Error: ' + error.message);
    } finally {
      setIsProcessingSpeech(false);
    }
  };

  // ============== COMBINED FUNCTIONS ==============
  
  const handleCombinedImageSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setCombinedImage(file);
      setCombinedImageUrl(URL.createObjectURL(file));
      setCombinedFacialEmotion(null);
      setCombinedAnnotatedUrl(null);
      setCombinedComparison(null);
      setCombinedFacialProbs(null);
    }
  };

  const startCombinedWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 224, height: 224 } });
      combinedVideoRef.current.srcObject = stream;
      combinedVideoRef.current.style.display = 'block';
    } catch (error) {
      alert('Cannot access webcam: ' + error.message);
    }
  };

  const captureCombinedImage = () => {
    const video = combinedVideoRef.current;
    const canvas = combinedCanvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    canvas.toBlob((blob) => {
      const file = new File([blob], 'combined_capture.jpg', { type: 'image/jpeg' });
      setCombinedImage(file);
      setCombinedImageUrl(canvas.toDataURL());
      setCombinedFacialEmotion(null);
      setCombinedAnnotatedUrl(null);
      setCombinedComparison(null);
      setCombinedFacialProbs(null);
      video.srcObject.getTracks().forEach(track => track.stop());
      combinedVideoRef.current.style.display = 'none';
    });
  };

  const handleCombinedAudioSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setCombinedAudio(file);
      setCombinedSpeechEmotion(null);
      setCombinedSpeechProbs(null);
    }
  };

  const startCombinedAudioRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      const chunks = [];

      mediaRecorder.ondataavailable = (event) => {
        chunks.push(event.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(chunks, { type: 'audio/wav' });
        const audioFile = new File([audioBlob], 'combined_recording.wav', { type: 'audio/wav' });
        setCombinedAudio(audioFile);
        stream.getTracks().forEach(track => track.stop());
        setCombinedSpeechEmotion(null);
        setCombinedSpeechProbs(null);
      };

      mediaRecorder.start();
      setTimeout(() => mediaRecorder.stop(), 5000);
    } catch (error) {
      alert('Cannot access microphone: ' + error.message);
    }
  };

  const predictCombinedEmotion = async () => {
    if (!combinedImage || !combinedAudio) {
      alert('Please provide both image and audio');
      return;
    }

    setIsProcessingCombined(true);
    const formData = new FormData();
    formData.append('image', combinedImage);
    formData.append('audio', combinedAudio);

    try {
      const response = await axios.post(`${API_URL}/api/predict/combined`, formData);
      
      const facialEmotion = response.data.facial?.emotion || response.data.facial || 'Unknown';
      const speechEmotion = response.data.speech?.emotion || response.data.speech || 'Unknown';
      
      setCombinedFacialEmotion(facialEmotion);
      setCombinedSpeechEmotion(speechEmotion);
      
      if (response.data.comparison) {
        setCombinedComparison(response.data.comparison);
      } else {
        setCombinedComparison(facialEmotion === speechEmotion ? 
          `✅ MATCH! Both indicate ${facialEmotion}` : 
          `⚠️ MISMATCH - Face: ${facialEmotion} | Voice: ${speechEmotion}`);
      }
      
      if (response.data.annotated_image) {
        setCombinedAnnotatedUrl('data:image/jpeg;base64,' + response.data.annotated_image);
      }
      
      if (response.data.facial?.probabilities) {
        setCombinedFacialProbs(response.data.facial.probabilities);
      }
      
      if (response.data.speech?.probabilities) {
        setCombinedSpeechProbs(response.data.speech.probabilities);
      }
    } catch (error) {
      alert('Error: ' + error.message);
    } finally {
      setIsProcessingCombined(false);
    }
  };

  // ============== VIDEO ANALYSIS FUNCTIONS ==============
  
  const handleVideoSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setVideoFile(file);
      setVideoResult(null);
    }
  };

  const predictVideoEmotion = async () => {
    if (!videoFile) {
      alert('Please select a video file');
      return;
    }

    setIsProcessingVideo(true);
    const formData = new FormData();
    formData.append('file', videoFile);

    try {
      const response = await axios.post(`${API_URL}/api/predict/video`, formData);
      setVideoResult(response.data);
    } catch (error) {
      alert('Error: ' + error.message);
    } finally {
      setIsProcessingVideo(false);
    }
  };

  // Helper to render confidence scores
  const renderConfidenceScores = (probs) => {
    if (!probs) return null;
    
    const entries = typeof probs === 'object' ? Object.entries(probs) : [];
    return (
      <div className="emotion-scores">
        {entries.map(([emotion, score]) => (
          <div key={emotion} className="score-bar">
            <span>{emotion.replace(/^[😠🤢😨😊😐😢😲😌]*\s*/, '')}</span>
            <div className="bar">
              <div 
                className="bar-fill" 
                style={{ width: `${(score * 100).toFixed(1)}%` }}
              />
            </div>
            <span>{(score * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="gradio-header">
        <h1>🎭 Unified Facial + Voice Emotion Recognition</h1>
        <p><strong>Test both facial expressions and voice tone simultaneously!</strong></p>
        <p>This demo combines two state-of-the-art emotion recognition models:</p>
        <ul style={{ textAlign: 'left', display: 'inline-block' }}>
          <li>📸 <strong>Vision Transformer (ViT)</strong> for facial emotion (71.29% accuracy)</li>
          <li>🎤 <strong>HuBERT</strong> for speech emotion (87.50% accuracy)</li>
        </ul>
        <hr />
      </header>

      {/* Tabs */}
      <div className="tabs">
        <button 
          className={`tab-btn ${activeTab === 0 ? 'active' : ''}`}
          onClick={() => setActiveTab(0)}
        >
          📸 Facial Emotion
        </button>
        <button 
          className={`tab-btn ${activeTab === 1 ? 'active' : ''}`}
          onClick={() => setActiveTab(1)}
        >
          🎤 Speech Emotion
        </button>
        <button 
          className={`tab-btn ${activeTab === 2 ? 'active' : ''}`}
          onClick={() => setActiveTab(2)}
        >
          🔗 Combined Analysis
        </button>
        <button 
          className={`tab-btn ${activeTab === 3 ? 'active' : ''}`}
          onClick={() => setActiveTab(3)}
        >
          ℹ️ Model Information
        </button>
      </div>

      {/* Tab Rendering */}
      {activeTab === 0 && <FacialTab />}
      {activeTab === 1 && <SpeechTab />}
      {activeTab === 2 && <CombinedTab />}
      {activeTab === 3 && <ModelInfoTab />}
      <footer className="footer">
        <p>© 2026 Multi-Modal Emotion Recognition System</p>
      </footer>
    </div>
  );
}

export default App;
