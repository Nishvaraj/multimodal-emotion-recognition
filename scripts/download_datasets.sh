#!/bin/bash

# Create directories for raw and processed datasets
mkdir -p ../data/raw/audio
mkdir -p ../data/raw/video
mkdir -p ../data/raw/text
mkdir -p ../data/processed/audio
mkdir -p ../data/processed/video
mkdir -p ../data/processed/text

# Download audio dataset
echo "Downloading audio dataset..."
wget -P ../data/raw/audio/ <audio_dataset_url>

# Download video dataset
echo "Downloading video dataset..."
wget -P ../data/raw/video/ <video_dataset_url>

# Download text dataset
echo "Downloading text dataset..."
wget -P ../data/raw/text/ <text_dataset_url>

echo "Datasets downloaded successfully."