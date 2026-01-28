import React, { useState, useRef } from 'react';
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

function App() {
  const [activeTab, setActiveTab] = useState('separate');
  
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
          className={`tab-btn ${activeTab === 'separate' ? 'active' : ''}`}
          onClick={() => setActiveTab('separate')}
        >
          🔀 Separate Testing
        </button>
        <button 
          className={`tab-btn ${activeTab === 'combined' ? 'active' : ''}`}
          onClick={() => setActiveTab('combined')}
        >
          🔗 Combined Analysis
        </button>
        <button 
          className={`tab-btn ${activeTab === 'video' ? 'active' : ''}`}
          onClick={() => setActiveTab('video')}
        >
          🎬 Video Analysis
        </button>
        <button 
          className={`tab-btn ${activeTab === 'info' ? 'active' : ''}`}
          onClick={() => setActiveTab('info')}
        >
          ℹ️ Model Information
        </button>
      </div>

      {/* Tab 1: Separate Testing */}
      {activeTab === 'separate' && (
        <div className="tab-content">
          <h2>### Test facial and voice emotions separately</h2>
          
          <div className="two-column">
            {/* FACIAL */}
            <div className="column">
              <h3>📸 Facial Emotion</h3>
              
              <label className="input-label">Capture or Upload Image</label>
              <video 
                ref={facialVideoRef} 
                autoPlay 
                playsInline
                style={{ display: 'none', width: '100%', borderRadius: '8px', marginBottom: '10px', maxHeight: '300px' }}
              />
              <canvas ref={facialCanvasRef} style={{ display: 'none' }} />
              
              <div className="button-group">
                <button onClick={startFacialWebcam} className="btn btn-secondary">📷 Start Webcam</button>
              </div>
              
              {facialVideoRef.current?.srcObject && (
                <button onClick={captureFacialImage} className="btn btn-primary" style={{ marginTop: '10px', width: '100%' }}>
                  📸 Capture Image
                </button>
              )}
              
              <input 
                type="file" 
                accept="image/*"
                onChange={handleFacialFileSelect}
                className="file-input"
                style={{ marginTop: '10px' }}
              />
              
              {facialImageUrl && (
                <img src={facialImageUrl} alt="Facial" style={{ width: '100%', borderRadius: '8px', marginTop: '10px', maxHeight: '300px' }} />
              )}
              
              <button 
                onClick={predictFacialEmotion} 
                disabled={isProcessingFacial || !facialImage}
                className="btn btn-primary"
                style={{ marginTop: '15px', width: '100%' }}
              >
                {isProcessingFacial ? '⏳ Analyzing...' : '🔮 Analyze Face'}
              </button>
              
              {facialEmotion && (
                <div className="result-box">
                  <h4>😊 Facial Emotion: <strong>{facialEmotion.toUpperCase()}</strong> ({(facialConfidence * 100).toFixed(1)}%)</h4>
                  {facialAnnotatedUrl && (
                    <img src={facialAnnotatedUrl} alt="Annotated" style={{ width: '100%', borderRadius: '8px', marginTop: '10px' }} />
                  )}
                  {facialProbs && (
                    <>
                      <h4 style={{ marginTop: '15px' }}>📊 Confidence Scores</h4>
                      {renderConfidenceScores(facialProbs)}
                    </>
                  )}
                </div>
              )}
            </div>

            {/* SPEECH */}
            <div className="column">
              <h3>🎤 Voice Emotion</h3>
              
              <label className="input-label">Record or Upload Audio</label>
              <audio 
                ref={speechAudioRef}
                controls
                style={{ width: '100%', marginBottom: '10px' }}
              />
              
              <div className="button-group">
                {!isRecordingSpeech ? (
                  <button 
                    onClick={startSpeechRecording} 
                    className="btn btn-secondary"
                  >
                    🎤 Record Audio
                  </button>
                ) : (
                  <button 
                    onClick={stopSpeechRecording} 
                    className="btn btn-secondary"
                  >
                    ⏹️ Stop Recording
                  </button>
                )}
              </div>
              
              <input 
                type="file" 
                accept="audio/*"
                onChange={handleSpeechFileSelect}
                className="file-input"
                style={{ marginTop: '10px' }}
              />
              
              <button 
                onClick={predictSpeechEmotion} 
                disabled={isProcessingSpeech || !speechAudio}
                className="btn btn-primary"
                style={{ marginTop: '15px', width: '100%' }}
              >
                {isProcessingSpeech ? '⏳ Analyzing...' : '🔮 Analyze Voice'}
              </button>
              
              {speechEmotion && (
                <div className="result-box">
                  <h4>🎤 Voice Emotion: <strong>{speechEmotion.toUpperCase()}</strong> ({(speechConfidence * 100).toFixed(1)}%)</h4>
                  {speechProbs && (
                    <>
                      <h4 style={{ marginTop: '15px' }}>📊 Confidence Scores</h4>
                      {renderConfidenceScores(speechProbs)}
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Combined Analysis */}
      {activeTab === 'combined' && (
        <div className="tab-content">
          <h2>### Test facial and voice together for multimodal comparison</h2>
          
          <div className="two-column">
            <div className="column">
              <label className="input-label">📸 Capture or Upload Image</label>
              
              <video 
                ref={combinedVideoRef} 
                autoPlay 
                playsInline
                style={{ display: 'none', width: '100%', borderRadius: '8px', marginBottom: '10px', maxHeight: '300px' }}
              />
              <canvas ref={combinedCanvasRef} style={{ display: 'none' }} />
              
              <div className="button-group">
                <button onClick={startCombinedWebcam} className="btn btn-secondary">📷 Webcam</button>
              </div>
              
              {combinedVideoRef.current?.srcObject && (
                <button onClick={captureCombinedImage} className="btn btn-primary" style={{ marginTop: '10px', width: '100%' }}>
                  📸 Capture
                </button>
              )}
              
              <input 
                type="file" 
                accept="image/*"
                onChange={handleCombinedImageSelect}
                className="file-input"
                style={{ marginTop: '10px' }}
              />
              
              {combinedImageUrl && (
                <img src={combinedImageUrl} alt="Combined" style={{ width: '100%', borderRadius: '8px', marginTop: '10px', maxHeight: '300px' }} />
              )}
            </div>

            <div className="column">
              <label className="input-label">🎤 Record or Upload Audio</label>
              
              <div className="button-group">
                <button onClick={startCombinedAudioRecording} className="btn btn-secondary">🎤 Record</button>
              </div>
              
              <input 
                type="file" 
                accept="audio/*"
                onChange={handleCombinedAudioSelect}
                className="file-input"
                style={{ marginTop: '10px' }}
              />
              
              {combinedAudio && (
                <p style={{ marginTop: '10px', color: '#667eea', fontWeight: '600' }}>✓ Audio selected</p>
              )}
            </div>
          </div>
          
          <button 
            onClick={predictCombinedEmotion} 
            disabled={isProcessingCombined || !combinedImage || !combinedAudio}
            className="btn btn-large"
            style={{ marginTop: '20px', width: '100%', padding: '15px', fontSize: '16px' }}
          >
            {isProcessingCombined ? '⏳ Analyzing Both...' : '🚀 Analyze Both'}
          </button>
          
          {combinedFacialEmotion && combinedSpeechEmotion && (
            <div className="combined-results">
              <h3>📊 Multimodal Analysis Results</h3>
              
              {combinedAnnotatedUrl && (
                <img src={combinedAnnotatedUrl} alt="Annotated" style={{ width: '100%', borderRadius: '8px', marginBottom: '20px' }} />
              )}
              
              <div className="results-grid">
                <div className="result-item">
                  <h4>Facial Emotion</h4>
                  <p className="emotion-name">{EMOTION_EMOJIS[combinedFacialEmotion.toLowerCase()] || '😐'} {combinedFacialEmotion.toUpperCase()}</p>
                  {combinedFacialProbs && renderConfidenceScores(combinedFacialProbs)}
                </div>
                <div className="result-item">
                  <h4>Voice Emotion</h4>
                  <p className="emotion-name">{EMOTION_EMOJIS[combinedSpeechEmotion.toLowerCase()] || '🎤'} {combinedSpeechEmotion.toUpperCase()}</p>
                  {combinedSpeechProbs && renderConfidenceScores(combinedSpeechProbs)}
                </div>
                <div className="result-item">
                  <h4>Concordance</h4>
                  <p className="emotion-name">{combinedComparison?.includes('MATCH') ? '✅ MATCH' : '⚠️ MISMATCH'}</p>
                  <p style={{ fontSize: '0.9em', marginTop: '10px' }}>{combinedComparison}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tab 3: Video Analysis */}
      {activeTab === 'video' && (
        <div className="tab-content">
          <h2>### Analyze emotions from video files</h2>
          <p style={{ color: '#666', marginBottom: '20px' }}>Upload a video file to automatically extract facial emotions from frames and speech emotion from audio</p>
          
          <div style={{ maxWidth: '600px', margin: '0 auto' }}>
            <label className="input-label">📹 Upload Video File</label>
            <input 
              type="file" 
              accept="video/*"
              onChange={handleVideoSelect}
              className="file-input"
            />
            
            {videoFile && (
              <video 
                src={URL.createObjectURL(videoFile)}
                controls
                style={{ width: '100%', borderRadius: '8px', marginTop: '15px', maxHeight: '400px' }}
              />
            )}
            
            <button 
              onClick={predictVideoEmotion}
              disabled={isProcessingVideo || !videoFile}
              className="btn btn-large"
              style={{ marginTop: '20px', width: '100%', padding: '15px', fontSize: '16px' }}
            >
              {isProcessingVideo ? '⏳ Analyzing Video...' : '🎬 Analyze Video'}
            </button>
          </div>
          
          {videoResult && (
            <div className="combined-results" style={{ marginTop: '30px' }}>
              <h3>📊 Video Analysis Results</h3>
              
              <div className="results-grid">
                <div className="result-item">
                  <h4>Facial Analysis</h4>
                  <p className="emotion-name">{EMOTION_EMOJIS[videoResult.facial_emotion?.emotion?.toLowerCase()] || '😐'} {videoResult.facial_emotion?.emotion?.toUpperCase()}</p>
                  <p style={{ fontSize: '0.9em', color: '#666', marginTop: '10px' }}>Confidence: {(videoResult.facial_emotion?.confidence * 100).toFixed(1)}%</p>
                  <p style={{ fontSize: '0.85em', color: '#999', marginTop: '5px' }}>Frames analyzed: {videoResult.facial_emotion?.frames_analyzed}</p>
                  {videoResult.facial_emotion?.probabilities && (
                    <>
                      <h5 style={{ marginTop: '15px', fontSize: '0.95em' }}>Emotion Breakdown:</h5>
                      {renderConfidenceScores(videoResult.facial_emotion?.probabilities)}
                    </>
                  )}
                </div>
                
                <div className="result-item">
                  <h4>Voice Analysis</h4>
                  <p className="emotion-name">{EMOTION_EMOJIS[videoResult.speech_emotion?.emotion?.toLowerCase()] || '🎤'} {videoResult.speech_emotion?.emotion?.toUpperCase()}</p>
                  <p style={{ fontSize: '0.9em', color: '#666', marginTop: '10px' }}>Confidence: {(videoResult.speech_emotion?.confidence * 100).toFixed(1)}%</p>
                  {videoResult.speech_emotion?.probabilities && (
                    <>
                      <h5 style={{ marginTop: '15px', fontSize: '0.95em' }}>Emotion Breakdown:</h5>
                      {renderConfidenceScores(videoResult.speech_emotion?.probabilities)}
                    </>
                  )}
                </div>
                
                <div className="result-item">
                  <h4>Concordance</h4>
                  <p className="emotion-name">{videoResult.facial_emotion?.emotion?.toLowerCase() === videoResult.speech_emotion?.emotion?.toLowerCase() ? '✅ MATCH' : '⚠️ MISMATCH'}</p>
                  <p style={{ fontSize: '0.9em', marginTop: '15px' }}>
                    {videoResult.facial_emotion?.emotion?.toLowerCase() === videoResult.speech_emotion?.emotion?.toLowerCase() ? 
                      `Both facial and voice indicate ${videoResult.facial_emotion?.emotion?.toUpperCase()}` : 
                      `Face: ${videoResult.facial_emotion?.emotion?.toUpperCase()} | Voice: ${videoResult.speech_emotion?.emotion?.toUpperCase()}`
                    }
                  </p>
                </div>
              </div>
              
              <div style={{ marginTop: '20px', padding: '15px', backgroundColor: '#f5f5f5', borderRadius: '8px', textAlign: 'center', color: '#666' }}>
                <p><strong>Video Duration:</strong> {(videoResult.video_duration).toFixed(2)}s | <strong>Total Frames:</strong> {videoResult.frames_processed}</p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tab 4: Model Information */}
      {activeTab === 'info' && (
        <div className="tab-content">
          <h2>## 📊 Model Details</h2>
          
          <h3>📸 Facial Emotion Recognition (ViT)</h3>
          <ul>
            <li><strong>Architecture:</strong> Vision Transformer (google/vit-base-patch16-224-in21k)</li>
            <li><strong>Training Data:</strong> FER2013 Dataset (35,887 images)</li>
            <li><strong>Emotions:</strong> 7 classes (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)</li>
            <li><strong>Accuracy:</strong> 71.29%</li>
            <li><strong>Model Size:</strong> 327MB</li>
            <li><strong>Input:</strong> RGB Images (224×224)</li>
          </ul>
          
          <h3>🎤 Speech Emotion Recognition (HuBERT)</h3>
          <ul>
            <li><strong>Architecture:</strong> HuBERT Large (facebook/hubert-large-ls960-ft)</li>
            <li><strong>Training Data:</strong> RAVDESS Dataset (1,440 audio files)</li>
            <li><strong>Emotions:</strong> 8 classes (Angry, Calm, Disgust, Fearful, Happy, Neutral, Sad, Surprised)</li>
            <li><strong>Accuracy:</strong> 87.50%</li>
            <li><strong>Model Size:</strong> ~360MB</li>
            <li><strong>Input:</strong> 16kHz Mono Audio</li>
          </ul>
          
          <h3>🎯 How to Use</h3>
          <strong>Separate Testing:</strong>
          <ol>
            <li>Use the 🔀 tab to test facial or voice separately</li>
            <li>Capture/upload image and click "🔮 Analyze Face"</li>
            <li>Record/upload audio and click "🔮 Analyze Voice"</li>
            <li>View confidence scores for each emotion</li>
          </ol>
          
          <strong>Combined Analysis:</strong>
          <ol>
            <li>Use the 🔗 tab for multimodal testing</li>
            <li>Capture/upload both image and audio</li>
            <li>Click "🚀 Analyze Both"</li>
            <li>Compare facial expression with voice tone</li>
            <li>Check for emotional concordance (match/mismatch)</li>
          </ol>
        </div>
      )}

      <footer className="footer">
        <p>© 2026 Multi-Modal Emotion Recognition System</p>
      </footer>
    </div>
  );
}

export default App;
