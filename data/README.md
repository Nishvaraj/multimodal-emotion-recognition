# Dataset Documentation for Multi-Modal Emotion Recognition Project

This directory contains the datasets used for the combined multimodal emotion recognition project. Facial and speech-only assets are still present as supporting data, and the datasets are organized into two main categories: raw and processed.

## Raw Datasets

The `raw` directory includes the original datasets that have not been altered or processed. It contains the following subdirectories:

- **ravdess**: RAVDESS audio-visual emotion dataset used for combined analysis workflows.
- **video**: Contains raw video files used for facial emotion recognition.
- **fer2013**: FER2013 facial expression dataset with train/test/val splits.
- **audio**: Contains raw audio files (RAVDESS) used for speech emotion recognition.

## Processed Datasets

The `processed` directory includes datasets that have been preprocessed and are ready for use in model training. It contains the following subdirectories:

- **video**: Contains processed video files, which may include resized frames, face crops, and extracted facial features.
- **audio**: Contains processed audio files, which may include MFCC features and spectrograms extracted from raw audio.

## Usage

To utilize the datasets in your training or evaluation processes, ensure that you load the appropriate files from the `processed` directory. The data loaders in the `src/data_loaders` directory are designed to facilitate this process.

## Acknowledgments

Please refer to the individual dataset documentation for any specific licensing or usage restrictions.
