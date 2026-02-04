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
  'surprised': '😲',
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
  const [gradCamImage, setGradCamImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isCameraOn, setIsCameraOn] = useState(false);

  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = (evt) => {
        setImagePreview(evt.target.result);
        setEmotion(null);
        setGradCamImage(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraOn(true);
        setError(null);
        
        // Ensure video plays
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play().catch(err => console.log('Play error:', err));
        };
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
    try {
      const formData = new FormData();
      formData.append('file', imageFile);
      const response = await axios.post(`${API_BASE}/api/predict/facial`, formData);
      if (response.data.success) {
        setEmotion(response.data.emotion);
        setConfidence(response.data.confidence);
        setProbabilities(response.data.probabilities);
        if (response.data.grad_cam_image) {
          setGradCamImage(response.data.grad_cam_image);
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
              <video 
                ref={videoRef} 
                autoPlay
                playsInline 
                style={{ 
                  width: '100%', 
                  height: 'auto',
                  minHeight: '300px',
                  maxHeight: '400px',
                  borderRadius: '8px', 
                  marginBottom: '10px', 
                  backgroundColor: '#000',
                  border: '2px solid #667eea',
                  objectFit: 'cover',
                  display: 'block'
                }} 
              />
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
          <button className="btn btn-primary" onClick={analyzeFacial} disabled={loading} style={{ width: '100%' }}>
            {loading ? '⏳ Analyzing...' : '🔮 Analyze Face'}
          </button>
        </>
      )}
      {error && <div className="error-message">{error}</div>}
      {emotion && (
        <div className="results-box">
          <h3>{EMOTION_EMOJIS[emotion]} {emotion.toUpperCase()} ({(confidence * 100).toFixed(1)}%)</h3>
          {gradCamImage && (
            <div style={{ marginBottom: '15px', padding: '10px', backgroundColor: '#f9f9f9', borderRadius: '8px' }}>
              <p style={{ fontSize: '0.9em', color: '#666', marginBottom: '10px' }}>🔍 Grad-CAM Heatmap (shows which facial regions influenced the prediction):</p>
              <img src={gradCamImage.startsWith('data:') ? gradCamImage : `data:image/png;base64,${gradCamImage}`} alt="Grad-CAM" style={{ width: '100%', borderRadius: '6px' }} />
            </div>
          )}
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
  const [saliencyMap, setSaliencyMap] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const streamRef = useRef(null);
  const analyzerRef = useRef(null);
  const canvasAnalyzerRef = useRef(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];
      
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
      setSaliencyMap(null);
    }
  };

  const analyzeSpeech = async () => {
    if (!audioFile) {
      setError('Please record or upload audio');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', audioFile);
      const response = await axios.post(`${API_BASE}/api/predict/speech`, formData);
      if (response.data.success) {
        setEmotion(response.data.emotion);
        setConfidence(response.data.confidence);
        setProbabilities(response.data.probabilities);
        if (response.data.saliency_map) {
          setSaliencyMap(response.data.saliency_map);
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
                style={{ width: '100%', border: '2px solid #667eea', borderRadius: '4px', marginBottom: '10px', backgroundColor: '#f0f0f0' }} 
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
        <button className="btn btn-primary" onClick={analyzeSpeech} disabled={loading} style={{ width: '100%' }}>
          {loading ? '⏳ Analyzing...' : '🔮 Analyze Audio'}
        </button>
      )}
      {error && <div className="error-message">{error}</div>}
      {emotion && (
        <div className="results-box">
          <h3>{EMOTION_EMOJIS[emotion]} {emotion.toUpperCase()} ({(confidence * 100).toFixed(1)}%)</h3>
          {saliencyMap && (
            <div style={{ marginBottom: '15px', padding: '10px', backgroundColor: '#f9f9f9', borderRadius: '8px' }}>
              <p style={{ fontSize: '0.9em', color: '#666', marginBottom: '10px' }}>🎵 Frequency Saliency Map (shows which frequencies influenced the prediction):</p>
              <img src={saliencyMap.startsWith('data:') ? saliencyMap : `data:image/png;base64,${saliencyMap}`} alt="Saliency Map" style={{ width: '100%', borderRadius: '6px' }} />
            </div>
          )}
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
        </div>
      )}
    </div>
  );
}

function CombinedTab() {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [videoFile, setVideoFile] = useState(null);
  const [facialEmotion, setFacialEmotion] = useState(null);
  const [speechEmotion, setSpeechEmotion] = useState(null);
  const [concordance, setConcordance] = useState(null);
  const [facialProbs, setFacialProbs] = useState(null);
  const [speechProbs, setSpeechProbs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isRecordingVideo, setIsRecordingVideo] = useState(false);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const streamRef = useRef(null);

  const startVideoRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 1280, height: 720 },
        audio: true 
      });
      
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play().catch(err => console.log('Play error:', err));
        };
      }
      
      chunksRef.current = [];
      setIsCameraOn(true);
      setError(null);
      
      // Start recording with audio
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.ondataavailable = (e) => chunksRef.current.push(e.data);
      mediaRecorder.start();
      setIsRecordingVideo(true);
    } catch (err) {
      setError('Cannot access camera/microphone: ' + err.message);
      console.error('Camera/microphone error:', err);
    }
  };

  const stopVideoRecording = async () => {
    if (mediaRecorderRef.current && isRecordingVideo) {
      return new Promise((resolve) => {
        mediaRecorderRef.current.onstop = async () => {
          const videoBlob = new Blob(chunksRef.current, { type: 'video/webm' });
          
          // Extract both frame and audio from video
          try {
            const videoElement = document.createElement('video');
            videoElement.src = URL.createObjectURL(videoBlob);
            videoElement.muted = true;
            
            videoElement.onloadedmetadata = async () => {
              // Extract frame
              const canvas = document.createElement('canvas');
              canvas.width = videoElement.videoWidth;
              canvas.height = videoElement.videoHeight;
              const ctx = canvas.getContext('2d');
              
              videoElement.currentTime = Math.min(videoElement.duration * 0.5, 2);
              
              videoElement.onseeked = async () => {
                ctx.drawImage(videoElement, 0, 0);
                canvas.toBlob((imageBlob) => {
                  if (imageBlob) {
                    setImageFile(imageBlob);
                    const reader = new FileReader();
                    reader.onload = (evt) => setImagePreview(evt.target.result);
                    reader.readAsDataURL(imageBlob);
                  }
                });
                
                // Extract audio using Web Audio API
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const mediaSource = audioContext.createMediaElementAudioSource(videoElement);
                const destination = audioContext.createMediaStreamAudioDestination();
                mediaSource.connect(destination);
                
                const audioRecorder = new MediaRecorder(destination.stream);
                const audioChunks = [];
                
                audioRecorder.ondataavailable = (e) => audioChunks.push(e.data);
                audioRecorder.onstop = () => {
                  const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                  const audioFile = new File([audioBlob], 'extracted_audio.wav', { type: 'audio/wav' });
                  setAudioFile(audioFile);
                };
                
                audioRecorder.start();
                videoElement.play().catch(err => console.log('Play error:', err));
                
                setTimeout(() => {
                  audioRecorder.stop();
                  videoElement.pause();
                }, videoElement.duration * 1000 + 500);
              };
            };
          } catch (err) {
            console.error('Audio extraction error:', err);
            setError('Audio extraction failed, please try again');
          }
          
          resolve();
        };
        
        mediaRecorderRef.current.stop();
        setIsRecordingVideo(false);
        
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
        }
      });
    }
  };

  const handleVideoUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const videoElement = document.createElement('video');
      videoElement.src = URL.createObjectURL(file);
      videoElement.muted = true;
      
      videoElement.onloadedmetadata = async () => {
        // Extract frame
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        const ctx = canvas.getContext('2d');
        
        videoElement.currentTime = Math.min(videoElement.duration * 0.5, 2);
        
        videoElement.onseeked = async () => {
          ctx.drawImage(videoElement, 0, 0);
          canvas.toBlob((imageBlob) => {
            if (imageBlob) {
              setImageFile(imageBlob);
              const reader = new FileReader();
              reader.onload = (evt) => setImagePreview(evt.target.result);
              reader.readAsDataURL(imageBlob);
            }
          });
          
          // Extract audio using Web Audio API
          const audioContext = new (window.AudioContext || window.webkitAudioContext)();
          const mediaSource = audioContext.createMediaElementAudioSource(videoElement);
          const destination = audioContext.createMediaStreamAudioDestination();
          mediaSource.connect(destination);
          
          const audioRecorder = new MediaRecorder(destination.stream);
          const audioChunks = [];
          
          audioRecorder.ondataavailable = (e) => audioChunks.push(e.data);
          audioRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const audioFile = new File([audioBlob], 'extracted_audio.wav', { type: 'audio/wav' });
            setAudioFile(audioFile);
            setLoading(false);
            setError('✅ Video processed! Image and audio extracted. Ready to analyze.');
          };
          
          audioRecorder.start();
          videoElement.play().catch(err => console.log('Play error:', err));
          
          setTimeout(() => {
            audioRecorder.stop();
            videoElement.pause();
          }, videoElement.duration * 1000 + 500);
        };
      };
    } catch (err) {
      setError('Error processing video: ' + err.message);
      setLoading(false);
    }
  };

  const analyzeCombined = async () => {
    if (!imageFile || !audioFile) {
      setError('Please provide both image and audio');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('image_file', imageFile);
      formData.append('audio_file', audioFile);
      const response = await axios.post(`${API_BASE}/api/predict/combined`, formData);
      if (response.data.success) {
        setFacialEmotion(response.data.facial_emotion.emotion);
        setSpeechEmotion(response.data.speech_emotion.emotion);
        setConcordance(response.data.concordance);
        setFacialProbs(response.data.facial_emotion.probabilities);
        setSpeechProbs(response.data.speech_emotion.probabilities);
      }
    } catch (err) {
      setError('API Error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="tab-content">
      <h2>🎥 Combined Video Analysis</h2>
      <p style={{ color: '#666', marginBottom: '20px' }}>Record a video with your face and voice, or upload a video file to analyze both emotions simultaneously.</p>
      
      {/* OPTION 1: RECORD VIDEO */}
      <div style={{ marginBottom: '30px', paddingBottom: '25px', borderBottom: '2px solid #eee' }}>
        <h3 style={{ color: '#667eea', marginBottom: '15px' }}>🎥 Option 1: Record Video</h3>
        <p style={{ color: '#666', fontSize: '0.95em', marginBottom: '15px' }}>Record yourself speaking with your face visible. We'll extract the face and voice automatically.</p>
        
        {!isCameraOn ? (
          <button className="btn btn-primary" onClick={startVideoRecording} style={{ padding: '12px 24px', fontSize: '16px' }}>
            🎬 Start Recording
          </button>
        ) : (
          <div>
            <video 
              ref={videoRef} 
              autoPlay
              playsInline
              style={{ 
                width: '100%',
                height: 'auto',
                minHeight: '300px',
                maxHeight: '400px',
                borderRadius: '8px', 
                marginBottom: '15px', 
                backgroundColor: '#000',
                border: '3px solid #667eea',
                objectFit: 'cover',
                display: 'block'
              }} 
            />
            <canvas ref={canvasRef} width="320" height="240" style={{ display: 'none' }} />
            
            <p style={{ color: '#e74c3c', fontWeight: '600', marginBottom: '10px' }}>
              🔴 RECORDING...
            </p>
            
            <button 
              className="btn btn-danger" 
              onClick={stopVideoRecording} 
              style={{ padding: '10px 20px', fontSize: '15px' }}
            >
              ⏹️ Stop Recording
            </button>
          </div>
        )}
        
        {imageFile && audioFile && (
          <p style={{ color: '#27ae60', fontWeight: '600', marginTop: '10px' }}>
            ✅ Video processed! Face and voice extracted.
          </p>
        )}
      </div>
      
      {/* OPTION 2: UPLOAD VIDEO */}
      <div style={{ marginBottom: '25px' }}>
        <h3 style={{ color: '#667eea', marginBottom: '15px' }}>📤 Option 2: Upload Video File</h3>
        <p style={{ color: '#666', fontSize: '0.95em', marginBottom: '15px' }}>Upload an MP4 file with both audio and video. We'll extract the face and voice.</p>
        
        <input 
          type="file" 
          accept="video/mp4,video/webm,video/*" 
          onChange={handleVideoUpload} 
          className="file-input" 
          disabled={loading}
        />
        
        {imageFile && audioFile && (
          <p style={{ color: '#27ae60', fontWeight: '600', marginTop: '10px' }}>
            ✅ Video processed! Face and voice extracted.
          </p>
        )}
      </div>
      
      {error && !error.includes('✅') && <div className="error-message">{error}</div>}
      {error && error.includes('✅') && <div style={{ backgroundColor: '#d4edda', color: '#155724', padding: '12px', borderRadius: '6px', marginBottom: '15px', border: '1px solid #c3e6cb' }}>{error}</div>}
      
      {/* ANALYZE BUTTON */}
      {imageFile && audioFile && !facialEmotion && (
        <button 
          className="btn btn-primary" 
          onClick={analyzeCombined} 
          disabled={loading} 
          style={{ width: '100%', padding: '14px', marginBottom: '20px', fontSize: '16px', fontWeight: '600' }}
        >
          {loading ? '⏳ Analyzing...' : '🚀 Analyze Both Face & Voice'}
        </button>
      )}
      
      {/* RESULTS */}
      {facialEmotion && speechEmotion && (
        <div className="results-box">
          <div style={{ 
            padding: '15px', 
            backgroundColor: concordance === 'MATCH' ? '#d4edda' : '#fff3cd', 
            borderRadius: '8px', 
            marginBottom: '15px',
            border: `2px solid ${concordance === 'MATCH' ? '#28a745' : '#ffc107'}`
          }}>
            <h3 style={{ margin: '0 0 10px 0' }}>
              {concordance === 'MATCH' ? '✅ MATCH!' : '⚠️ MISMATCH'}
            </h3>
            <p style={{ margin: '0', fontSize: '1.1em' }}>
              Face: <strong>{EMOTION_EMOJIS[facialEmotion]} {facialEmotion}</strong> | 
              Voice: <strong>{EMOTION_EMOJIS[speechEmotion]} {speechEmotion}</strong>
            </p>
          </div>
          
          {imagePreview && (
            <div style={{ marginBottom: '15px' }}>
              <p style={{ fontSize: '0.95em', color: '#666', marginBottom: '8px' }}>📸 Extracted Frame:</p>
              <img src={imagePreview} alt="Extracted Frame" style={{ width: '100%', maxHeight: '200px', borderRadius: '6px', border: '2px solid #667eea' }} />
            </div>
          )}
          
          <div className="two-column">
            <div className="column results-col">
              <h4>📸 Facial: {EMOTION_EMOJIS[facialEmotion]} {facialEmotion.toUpperCase()}</h4>
              {facialProbs && (
                <div className="probabilities">
                  {Object.entries(facialProbs).slice(0, 5).map(([emo, prob]) => (
                    <div key={emo} className="prob-item" style={{ fontSize: '0.85em' }}>
                      <span>{EMOTION_EMOJIS[emo] || '😐'} {emo}</span>
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
                  {Object.entries(speechProbs).slice(0, 5).map(([emo, prob]) => (
                    <div key={emo} className="prob-item" style={{ fontSize: '0.85em' }}>
                      <span>{EMOTION_EMOJIS[emo] || '😐'} {emo}</span>
                      <div className="prob-bar"><div className="prob-fill" style={{ width: `${prob * 100}%` }} /></div>
                      <span>{(prob * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
          
          <button 
            className="btn btn-primary" 
            onClick={() => {
              setImageFile(null);
              setAudioFile(null);
              setVideoFile(null);
              setImagePreview(null);
              setFacialEmotion(null);
              setSpeechEmotion(null);
              setConcordance(null);
              setError(null);
            }} 
            style={{ width: '100%', marginTop: '15px' }}
          >
            🔄 Analyze Another Video
          </button>
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

function SessionsTab() {
  const [sessions, setSessions] = useState([]);
  const [selectedSession, setSelectedSession] = useState(null);
  const [sessionDetails, setSessionDetails] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchSessions = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_BASE}/api/sessions`);
      if (response.data.success) {
        setSessions(response.data.sessions || []);
      }
    } catch (err) {
      setError('Failed to fetch sessions: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchSessionDetails = async (sessionId) => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_BASE}/api/sessions/${sessionId}`);
      if (response.data.success) {
        setSelectedSession(sessionId);
        setSessionDetails(response.data);
      }
    } catch (err) {
      setError('Failed to fetch session details: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const exportSession = async (sessionId, format) => {
    try {
      const response = await axios.get(`${API_BASE}/api/sessions/${sessionId}/export/${format}`);
      if (response.data.success) {
        const data = format === 'csv' ? response.data.data : JSON.stringify(response.data.data, null, 2);
        const blob = new Blob([data], { type: format === 'csv' ? 'text/csv' : 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = response.data.filename;
        link.click();
        URL.revokeObjectURL(url);
      }
    } catch (err) {
      alert('Export failed: ' + err.message);
    }
  };

  const deleteSession = async (sessionId) => {
    if (!window.confirm('Are you sure you want to delete this session?')) return;
    
    try {
      const response = await axios.delete(`${API_BASE}/api/sessions/${sessionId}`);
      if (response.data.success) {
        setSessions(sessions.filter(s => s.id !== sessionId));
        setSelectedSession(null);
        setSessionDetails(null);
        alert('Session deleted successfully');
      }
    } catch (err) {
      alert('Delete failed: ' + err.message);
    }
  };

  return (
    <div className="tab-content">
      <h2>💾 Session History</h2>
      
      <button className="btn btn-primary" onClick={fetchSessions} disabled={loading} style={{ marginBottom: '20px' }}>
        {loading ? '⏳ Loading...' : '🔄 Load Sessions'}
      </button>

      {error && <div className="error-message">{error}</div>}

      {sessions.length === 0 ? (
        <p style={{ color: '#999', textAlign: 'center', padding: '40px 20px' }}>📭 No sessions found. Click "Load Sessions" to refresh.</p>
      ) : (
        <div className="two-column" style={{ marginBottom: '20px' }}>
          <div className="column">
            <h3>📋 Sessions ({sessions.length})</h3>
            <div style={{ 
              border: '1px solid #ddd', 
              borderRadius: '8px', 
              maxHeight: '500px', 
              overflowY: 'auto' 
            }}>
              {sessions.map(session => (
                <div
                  key={session.id}
                  onClick={() => fetchSessionDetails(session.id)}
                  style={{
                    padding: '12px',
                    borderBottom: '1px solid #eee',
                    cursor: 'pointer',
                    backgroundColor: selectedSession === session.id ? '#e8f4f8' : '#fff',
                    transition: 'background-color 0.2s',
                    userSelect: 'none'
                  }}
                  onMouseEnter={(e) => e.target.style.backgroundColor = selectedSession === session.id ? '#e8f4f8' : '#f5f5f5'}
                  onMouseLeave={(e) => e.target.style.backgroundColor = selectedSession === session.id ? '#e8f4f8' : '#fff'}
                >
                  <div style={{ fontWeight: '600', color: '#667eea' }}>
                    👤 {session.user_name || 'Anonymous'} 
                  </div>
                  <div style={{ fontSize: '0.85em', color: '#999', marginTop: '4px' }}>
                    {session.total_predictions} predictions
                  </div>
                  <div style={{ fontSize: '0.8em', color: '#bbb' }}>
                    {new Date(session.created_at).toLocaleString()}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="column">
            <h3>📊 Session Details</h3>
            {selectedSession && sessionDetails ? (
              <div style={{ border: '1px solid #ddd', borderRadius: '8px', padding: '15px', maxHeight: '500px', overflowY: 'auto' }}>
                <div style={{ marginBottom: '15px', paddingBottom: '15px', borderBottom: '1px solid #eee' }}>
                  <p style={{ margin: '0 0 8px 0' }}><strong>👤 User:</strong> {sessionDetails.session.user_name || 'Anonymous'}</p>
                  <p style={{ margin: '0 0 8px 0' }}><strong>📅 Created:</strong> {new Date(sessionDetails.session.created_at).toLocaleString()}</p>
                  <p style={{ margin: '0' }}><strong>📈 Predictions:</strong> {sessionDetails.session.total_predictions}</p>
                </div>

                {sessionDetails.statistics && (
                  <div style={{ marginBottom: '15px', padding: '12px', backgroundColor: '#f0f7ff', borderRadius: '6px', border: '1px solid #e0f0ff' }}>
                    <strong style={{ color: '#667eea' }}>📊 Statistics</strong>
                    <div style={{ fontSize: '0.9em', marginTop: '8px', lineHeight: '1.6' }}>
                      <div>✅ Matches: <strong>{sessionDetails.statistics.concordance_matches}</strong></div>
                      <div>⚠️ Mismatches: <strong>{sessionDetails.statistics.concordance_mismatches}</strong></div>
                      <div>🎯 Avg Confidence: <strong>{(sessionDetails.statistics.average_confidence * 100).toFixed(1)}%</strong></div>
                    </div>
                  </div>
                )}

                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '15px' }}>
                  <button 
                    className="btn btn-primary" 
                    onClick={() => exportSession(selectedSession, 'csv')}
                    style={{ flex: '1', minWidth: '100px', fontSize: '0.9em', padding: '8px' }}
                  >
                    📥 CSV
                  </button>
                  <button 
                    className="btn btn-primary" 
                    onClick={() => exportSession(selectedSession, 'json')}
                    style={{ flex: '1', minWidth: '100px', fontSize: '0.9em', padding: '8px' }}
                  >
                    📥 JSON
                  </button>
                  <button 
                    className="btn btn-danger" 
                    onClick={() => deleteSession(selectedSession)}
                    style={{ flex: '1', minWidth: '100px', fontSize: '0.9em', padding: '8px' }}
                  >
                    🗑️ Delete
                  </button>
                </div>

                {sessionDetails.predictions && sessionDetails.predictions.length > 0 && (
                  <div style={{ marginTop: '15px', paddingTop: '15px', borderTop: '1px solid #eee' }}>
                    <strong style={{ color: '#667eea' }}>🔍 Recent Predictions ({sessionDetails.predictions.length})</strong>
                    <div style={{ marginTop: '10px', fontSize: '0.85em' }}>
                      {sessionDetails.predictions.slice(0, 5).map((pred, idx) => (
                        <div key={idx} style={{ 
                          padding: '8px', 
                          backgroundColor: '#f9f9f9', 
                          borderRadius: '4px', 
                          marginBottom: '6px',
                          borderLeft: '3px solid #667eea'
                        }}>
                          <div><strong>{pred.modality}:</strong> {EMOTION_EMOJIS[pred.emotion] || '😐'} {pred.emotion}</div>
                          <div style={{ color: '#999', marginTop: '2px' }}>Confidence: {(pred.confidence * 100).toFixed(1)}%</div>
                        </div>
                      ))}
                      {sessionDetails.predictions.length > 5 && (
                        <div style={{ color: '#999', fontStyle: 'italic', marginTop: '8px', textAlign: 'center', padding: '8px' }}>
                          ... and {sessionDetails.predictions.length - 5} more
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p style={{ color: '#999', textAlign: 'center', padding: '40px 20px' }}>👈 Select a session to view details</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState(0);

  return (
    <div className="app-container">
      {/* Header */}
      <header className="gradio-header">
        <h1>🎭 Multimodal Emotion Recognition</h1>
        <p><strong>Analyze emotions from facial expressions and voice tone simultaneously!</strong></p>
        <hr style={{ marginTop: '15px', marginBottom: '15px' }} />
      </header>

      {/* Tabs */}
      <div className="tabs">
        <button 
          className={`tab-btn ${activeTab === 0 ? 'active' : ''}`}
          onClick={() => setActiveTab(0)}
          title="Analyze facial emotions from images or webcam"
        >
          📸 Facial
        </button>
        <button 
          className={`tab-btn ${activeTab === 1 ? 'active' : ''}`}
          onClick={() => setActiveTab(1)}
          title="Analyze speech emotions from audio or microphone"
        >
          🎤 Speech
        </button>
        <button 
          className={`tab-btn ${activeTab === 2 ? 'active' : ''}`}
          onClick={() => setActiveTab(2)}
          title="Compare facial and speech emotions simultaneously"
        >
          🔗 Combined
        </button>
        <button 
          className={`tab-btn ${activeTab === 3 ? 'active' : ''}`}
          onClick={() => setActiveTab(3)}
          title="View model information and technical details"
        >
          ℹ️ Model Info
        </button>
        <button 
          className={`tab-btn ${activeTab === 4 ? 'active' : ''}`}
          onClick={() => setActiveTab(4)}
          title="View and manage analysis sessions"
        >
          💾 Sessions
        </button>
      </div>

      {/* Tab Content */}
      {activeTab === 0 && <FacialTab />}
      {activeTab === 1 && <SpeechTab />}
      {activeTab === 2 && <CombinedTab />}
      {activeTab === 3 && <ModelInfoTab />}
      {activeTab === 4 && <SessionsTab />}

      {/* Footer */}
      <footer className="footer">
        <p>© 2026 Multimodal Emotion Recognition System | ViT (71.29%) + HuBERT (87.50%)</p>
      </footer>
    </div>
  );
}

export default App;
