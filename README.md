# Multi-Modal Emotion Recognition with Concordance Analysis

Real-time emotion recognition system combining facial expressions and speech analysis with a novel concordance metric for measuring emotional authenticity.

## Quick Start

1. **Activate virtual environment**

   ```bash
   source venv/bin/activate  # Mac/Linux
   # or
   venv\Scripts\activate  # Windows
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download datasets**

   ```bash
   python scripts/download_datasets.py
   ```

   Then manually download FER2013 and RAVDESS from Kaggle (see script output)

4. **Run backend**

   ```bash
   python backend/main.py
   ```

5. **Run frontend** (in separate terminal)
   ```bash
   cd frontend && npm start
   ```

## Project Structure

- `/frontend` - React UI for real-time emotion analysis
- `/backend` - FastAPI server with emotion recognition APIs
- `/src` - Core ML models and training code
  - `/models` - Audio, video (facial), and fusion models
  - `/data_loaders` - Dataset loading utilities
  - `/preprocessing` - Audio and video preprocessing
  - `/training` - Model training pipelines
  - `/utils` - Metrics and utilities
- `/models` - Pre-trained model checkpoints
  - `/facial` - Facial emotion models (ViT, ResNet)
  - `/speech` - Speech emotion models (HuBERT)
  - `/fusion` - Multi-modal fusion models
- `/data` - Training data (raw and processed)
- `/notebooks` - Jupyter notebooks for EDA and experiments
- `/tests` - Unit and integration tests
- `/docs` - Project documentation
- `/logs` - Training and application logs

## Key Features

- **Real-time Processing**: 30 FPS facial emotion recognition
- **Multi-modal Fusion**: Combines facial + speech modalities
- **Concordance Metric**: Novel measure of emotional authenticity
- **Explainability**: Grad-CAM visualizations for transparency
- **Privacy-First**: 100% local processing, zero cloud transmission

## Technology Stack

- **Backend**: FastAPI, Python 3.10+, PyTorch
- **Frontend**: React 18, Socket.IO, Plotly.js
- **ML Models**: Vision Transformer (ViT), HuBERT, ResNet-18
- **Computer Vision**: OpenCV, MTCNN
- **Audio**: Librosa, Soundfile

## Development Setup

See full setup guide in Phase 0 implementation documentation.

```
conda env create -f environment.yml
```

3. **Download Datasets**:
   Run the following script to download the necessary datasets:

   ```
   bash scripts/download_datasets.sh
   ```

4. **Prepare Datasets**:
   After downloading, prepare the datasets for training by running:

   ```
   python scripts/prepare_datasets.py
   ```

5. **Extract Features**:
   Extract features from the raw data using:

   ```
   python scripts/extract_features.py
   ```

6. **Run the Application**:
   Start the training or evaluation process by executing:
   ```
   python src/main.py
   ```

## Usage

Follow the instructions in the `notebooks/EDA.ipynb` for exploratory data analysis and to understand the dataset better. Modify the configuration in `configs/config.yaml` as needed for your experiments.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
