import os
import numpy as np
import librosa
import cv2
import pandas as pd

def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    features = []
    
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features.append(gray_frame.flatten())
    
    cap.release()
    return np.mean(features, axis=0)

def extract_text_features(text_path):
    with open(text_path, 'r') as file:
        text = file.read()
    # Simple feature: word count
    word_count = len(text.split())
    return np.array([word_count])

def extract_features(data_dir, output_file):
    audio_features = []
    video_features = []
    text_features = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.wav'):
            audio_path = os.path.join(data_dir, filename)
            audio_features.append(extract_audio_features(audio_path))
        elif filename.endswith('.mp4'):
            video_path = os.path.join(data_dir, filename)
            video_features.append(extract_video_features(video_path))
        elif filename.endswith('.txt'):
            text_path = os.path.join(data_dir, filename)
            text_features.append(extract_text_features(text_path))
    
    audio_features = np.array(audio_features)
    video_features = np.array(video_features)
    text_features = np.array(text_features)
    
    features_df = pd.DataFrame({
        'audio': list(audio_features),
        'video': list(video_features),
        'text': list(text_features)
    })
    
    features_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    data_directory = '../data/raw'  # Adjust path as necessary
    output_file_path = '../data/processed/features.csv'  # Adjust path as necessary
    extract_features(data_directory, output_file_path)