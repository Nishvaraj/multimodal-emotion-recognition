# Multi-Modal Emotion Recognition

Emotion analytics platform centered on combined facial + speech analysis, with facial and speech-only modes kept as secondary workflows. It includes multimodal concordance analysis, explainability outputs, authentication, and user history tracking.

## Implemented Capabilities

### 1. Inference Modes
- Combined multimodal prediction:
  - Separate image + audio mode
  - Video mode (upload or live recording)
- Facial emotion prediction (upload + webcam)
- Speech emotion prediction (upload + live microphone)

### 2. Explainability
- Facial Grad-CAM heatmaps
- Audio saliency visualization
- Explainability status reporting when generation is partial or unavailable

### 3. User System
- Supabase authentication (signup/login/logout/session)
- Per-user analysis history with notes, pin/unpin, and delete
- Export tools:
  - CSV history export
  - Text summary report export

### 4. Backend API
- Root and health endpoints
- Combined, facial, speech, and video prediction endpoints
- Emotion list endpoints
- Model status endpoint

## Local Quick Start

1. Activate Python environment

```bash
source .venv/bin/activate
```

2. Install backend dependencies

```bash
pip install -r requirements.txt
```

3. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

4. Run backend

```bash
./.venv/bin/uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

5. Run frontend (new terminal)

```bash
cd frontend
npm start
```

6. Open app

- Frontend: http://localhost:3000
- Backend health: http://127.0.0.1:8000/health

## Environment Variables

### Frontend
- REACT_APP_API_BASE
- REACT_APP_SUPABASE_URL
- REACT_APP_SUPABASE_ANON_KEY

### Backend
- ENV
- USE_GPU
- PRELOAD_MODELS
- FRONTEND_URL
- CORS_ORIGINS

## Deployment (Railway + Vercel)

### 1. Deploy backend to Railway

This repo includes `railway.json` with a production start command for FastAPI.

1. Create a new Railway project from this repository.
2. Set these Railway environment variables:
  - `ENV=production`
  - `USE_GPU=false` (recommended unless you have a GPU-enabled setup)
  - `PRELOAD_MODELS=false` (recommended on Railway for faster boot)
  - `FRONTEND_URL=https://<your-vercel-domain>`
  - Optional: `CORS_ORIGINS=https://<your-vercel-domain>,https://www.<your-vercel-domain>`
3. Deploy and confirm health endpoint:
  - `https://<your-railway-domain>/health`

### 2. Deploy frontend to Vercel

This repo includes `frontend/vercel.json` for static React deployment.

1. Import this repository in Vercel.
2. Set **Root Directory** to `frontend`.
3. Set frontend environment variables in Vercel:
  - `REACT_APP_API_BASE=https://<your-railway-domain>`
  - `REACT_APP_SUPABASE_URL=<your-supabase-url>`
  - `REACT_APP_SUPABASE_ANON_KEY=<your-supabase-anon-key>`
4. Deploy and open the Vercel URL.

### 3. Final CORS check

After Vercel domain is live, ensure Railway backend allows it:
- Preferred: set `CORS_ORIGINS` to an explicit comma-separated allowlist.
- Fallback: set `FRONTEND_URL` to your primary Vercel domain.

## Project Structure

- `frontend/`: React app (auth, dashboard, analysis tabs, history, exports)
- `backend/`: FastAPI inference and explainability services
- `models/`: trained checkpoints
- `notebooks/`: training and experimentation notebooks
- `configs/`: configuration files

## Current Limitations

- Facial model accuracy is still below the target in project planning and remains an active improvement area.
- Explainability quality can vary with low-quality/noisy inputs.

## Notes

- First backend startup may take longer due to model initialization.
- Webcam/microphone features require browser permissions.
