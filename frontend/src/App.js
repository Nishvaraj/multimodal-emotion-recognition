import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [useWebcam, setUseWebcam] = useState(false);
  const [recording, setRecording] = useState(false);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);

  const API_URL = 'http://127.0.0.1:8000';

  // Handle file selection
  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setResults(null);
    }
  };

  // Start webcam
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: true
      });
      videoRef.current.srcObject = stream;
      setUseWebcam(true);
      recordedChunksRef.current = [];
      
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'video/webm'
      });
      mediaRecorderRef.current = mediaRecorder;
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };
    } catch (error) {
      alert('Cannot access webcam: ' + error.message);
    }
  };

  // Stop webcam and start recording
  const stopWebcamAndRecord = () => {
    if (mediaRecorderRef.current && videoRef.current.srcObject) {
      setRecording(true);
      mediaRecorderRef.current.start();
      
      setTimeout(() => {
        mediaRecorderRef.current.stop();
        
        mediaRecorderRef.current.onstop = () => {
          const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
          setSelectedFile(blob);
          setRecording(false);
          
          videoRef.current.srcObject.getTracks().forEach(track => track.stop());
          setUseWebcam(false);
        };
      }, 5000);
    }
  };

  // Process video
  const processVideo = async () => {
    if (!selectedFile) {
      alert('Please select or record a video');
      return;
    }

    setIsProcessing(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_URL}/api/predict/video`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResults(response.data);
    } catch (error) {
      alert('Error processing video: ' + error.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const getEmotionEmoji = (emotion) => {
    const emojiMap = {
      'angry': '😠', 'disgust': '🤢', 'fear': '😨', 'happy': '😊',
      'neutral': '😐', 'sad': '😢', 'surprise': '😮', 'calm': '😌', 'fearful': '😨'
    };
    return emojiMap[emotion] || '❓';
  };

  return (
    <div className="App">
      <header className="header">
        <h1>🎬 Multi-Modal Emotion Recognition</h1>
        <p>Real-time emotion detection from video with facial and speech analysis</p>
      </header>

      <main className="main-content">
        <div className="input-section">
          <div className="option-group">
            <h2>Choose Input Method</h2>
            
            <div className="input-options">
              {/* Upload Video */}
              <div className="option">
                <h3>📁 Upload Video File</h3>
                <input 
                  type="file" 
                  accept="video/*"
                  onChange={handleFileSelect}
                  className="file-input"
                />
                {selectedFile && !useWebcam && <p className="success">✓ Selected: {selectedFile.name || 'Recorded video'}</p>}
              </div>

              {/* Webcam Recording */}
              <div className="option">
                <h3>📹 Use Webcam</h3>
                {!useWebcam ? (
                  <button onClick={startWebcam} className="btn btn-primary">
                    Start Webcam
                  </button>
                ) : (
                  <>
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      className="video-preview"
                    />
                    <button 
                      onClick={stopWebcamAndRecord} 
                      disabled={recording}
                      className="btn btn-success"
                    >
                      {recording ? '⏳ Recording 5s...' : '🔴 Record 5 Seconds'}
                    </button>
                  </>
                )}
              </div>
            </div>

            {selectedFile && !useWebcam && (
              <button 
                onClick={processVideo} 
                disabled={isProcessing}
                className="btn btn-process"
              >
                {isProcessing ? '⏳ Processing...' : '🔮 Analyze Emotion'}
              </button>
            )}
          </div>

          <canvas ref={canvasRef} style={{ display: 'none' }} />
        </div>

        {/* Results */}
        {results && (
          <div className="results-section">
            <h2>📊 Analysis Results</h2>
            
            <div className="results-grid">
              {/* Facial Results */}
              <div className="result-card facial">
                <h3>😊 Facial Emotion</h3>
                <div className="emotion-display">
                  <p className="emotion-emoji">{getEmotionEmoji(results.facial_emotion.emotion)}</p>
                  <p className="emotion-text">{results.facial_emotion.emotion.toUpperCase()}</p>
                  <p className="confidence">Confidence: {(results.facial_emotion.confidence * 100).toFixed(1)}%</p>
                  <p className="meta">Frames analyzed: {results.facial_emotion.frames_analyzed}</p>
                </div>
              </div>

              {/* Speech Results */}
              <div className="result-card speech">
                <h3>🎤 Speech Emotion</h3>
                <div className="emotion-display">
                  <p className="emotion-emoji">{getEmotionEmoji(results.speech_emotion.emotion)}</p>
                  <p className="emotion-text">{results.speech_emotion.emotion.toUpperCase()}</p>
                  <p className="confidence">Confidence: {(results.speech_emotion.confidence * 100).toFixed(1)}%</p>
                </div>
              </div>

              {/* Combined Results */}
              <div className="result-card combined">
                <h3>✨ Combined Emotion</h3>
                <div className="emotion-display">
                  <p className="emotion-emoji">{getEmotionEmoji(results.combined_emotion)}</p>
                  <p className="emotion-text">{results.combined_emotion.toUpperCase()}</p>
                  <p className="meta">Duration: {results.video_duration.toFixed(1)}s • FPS: {results.fps.toFixed(1)}</p>
                </div>
              </div>
            </div>

            {/* Probability Scores */}
            <div className="probabilities">
              <h3>📈 Detailed Probability Scores</h3>
              <div className="prob-grid">
                <div className="prob-column">
                  <h4>Facial Emotions</h4>
                  <div className="prob-list">
                    {Object.entries(results.facial_emotion.probabilities)
                      .sort((a, b) => b[1] - a[1])
                      .map(([emotion, prob]) => (
                        <div key={emotion} className="prob-bar">
                          <span className="emotion-label">{emotion}</span>
                          <div className="bar">
                            <div 
                              className="fill facial-fill" 
                              style={{ width: `${prob * 100}%` }}
                            />
                          </div>
                          <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                  </div>
                </div>
                
                <div className="prob-column">
                  <h4>Speech Emotions</h4>
                  <div className="prob-list">
                    {Object.entries(results.speech_emotion.probabilities)
                      .sort((a, b) => b[1] - a[1])
                      .map(([emotion, prob]) => (
                        <div key={emotion} className="prob-bar">
                          <span className="emotion-label">{emotion}</span>
                          <div className="bar">
                            <div 
                              className="fill speech-fill" 
                              style={{ width: `${prob * 100}%` }}
                            />
                          </div>
                          <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>Facial Emotion: ViT (71.29% acc) • Speech Emotion: HuBERT (87.50% acc)</p>
      </footer>
    </div>
  );
}

export default App;
