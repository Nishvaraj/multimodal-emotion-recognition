import os
import torch
from torch.utils.data import DataLoader
from models.audio_model import AudioModel
from models.video_model import VideoModel
from models.fusion import FusionModel
from data_loaders.audio_loader import AudioLoader
from data_loaders.video_loader import VideoLoader
from utils.metrics import calculate_metrics

def train_model(model, data_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader):.4f}')

def main():
    # Load datasets (text modality removed)
    audio_loader = AudioLoader('data/processed/audio')
    video_loader = VideoLoader('data/processed/video')

    audio_data = audio_loader.load_data()
    video_data = video_loader.load_data()

    # Create DataLoader for audio + video combined
    train_loader = DataLoader(audio_data + video_data, batch_size=32, shuffle=True)

    # Initialize models (audio + video)
    audio_model = AudioModel()
    video_model = VideoModel()
    fusion_model = FusionModel()

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=0.001)

    # Train the model
    train_model(fusion_model, train_loader, criterion, optimizer, num_epochs=10)

if __name__ == '__main__':
    main()