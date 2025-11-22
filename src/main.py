import os
from data_loaders.audio_loader import AudioLoader
from data_loaders.video_loader import VideoLoader
from models.audio_model import AudioModel
from models.video_model import VideoModel
from models.fusion import FusionModel
from training.train import train_model

def main():
    # Load datasets (text modality removed — audio + video only)
    audio_loader = AudioLoader('data/processed/audio')
    video_loader = VideoLoader('data/processed/video')

    audio_data = audio_loader.load_data()
    video_data = video_loader.load_data()

    # Initialize models
    audio_model = AudioModel()
    video_model = VideoModel()
    fusion_model = FusionModel()

    # Train the models (audio & video only)
    train_model(audio_model, audio_data)
    train_model(video_model, video_data)

    # Combine audio and video (facial) models for multi-modal recognition
    fusion_model.combine_models(audio_model, video_model)

if __name__ == "__main__":
    main()