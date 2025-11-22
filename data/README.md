# Dataset Documentation for Multi-Modal Emotion Recognition Project

This directory contains the datasets used for the multi-modal emotion recognition project. The datasets are organized into two main categories: raw and processed.

## Raw Datasets

The `raw` directory includes the original datasets that have not been altered or processed. It contains the following subdirectories:

- **audio**: Contains raw audio files used for emotion recognition.
- **video**: Contains raw video files used for emotion recognition.
- **text**: Contains raw text data (e.g., transcripts, captions) used for emotion recognition.

## Processed Datasets

The `processed` directory includes datasets that have been preprocessed and are ready for use in model training. It contains the following subdirectories:

- **audio**: Contains processed audio files, which may include features extracted from the raw audio.
- **video**: Contains processed video files, which may include resized frames or extracted features.
- **text**: Contains processed text data, which may include tokenized and vectorized representations.

## Usage

To utilize the datasets in your training or evaluation processes, ensure that you load the appropriate files from the `processed` directory. The data loaders in the `src/data_loaders` directory are designed to facilitate this process.

## Acknowledgments

Please refer to the individual dataset documentation for any specific licensing or usage restrictions.