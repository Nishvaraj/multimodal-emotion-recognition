# Dataset Notes

This folder stores dataset documentation and local data organization guidance for the multimodal emotion recognition project.

## Raw Data

Raw assets are the original source files used for training and experimentation.

- `ravdess`: audio-visual emotion samples used by the combined workflows.
- `video`: raw video clips used for facial emotion experiments.
- `fer2013`: facial expression dataset with train/test/validation splits.
- `audio`: raw speech samples used for speech emotion experiments.

## Processed Data

Processed assets are precomputed or cleaned versions of the raw data.

- `video`: resized frames, cropped faces, or extracted facial features.
- `audio`: MFCC features, mel-spectrograms, or other prepared audio features.

## Usage Guidance

Use the processed folders for model training and evaluation whenever possible. The Python dataset loaders in `backend/services/data_loader.py` match the structure described here.

## Licensing

Refer to the upstream dataset licenses and project documentation before reusing the data outside this repository.
