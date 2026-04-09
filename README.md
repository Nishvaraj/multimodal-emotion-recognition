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
- REACT_APP_VERCEL_URL

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

## Deployment (Render + Vercel)

### 1. Deploy Backend on Render

1. Push this repository to GitHub.
2. In Render, click **New +** -> **Blueprint**.
3. Connect your GitHub repo and select this project.
4. Render will detect `render.yaml` and create the backend web service.
5. In Render service environment variables, set:
  - `REACT_APP_VERCEL_URL` = your Vercel production URL (example: `https://your-app.vercel.app`)
6. Deploy and verify backend health:
  - `https://<your-render-service>.onrender.com/health`

### 2. Deploy Frontend on Vercel

1. In Vercel, import this same GitHub repository.
2. Set **Root Directory** to `frontend`.
3. Configure environment variables in Vercel:
  - `REACT_APP_API_BASE` = your Render backend URL (example: `https://<your-render-service>.onrender.com`)
  - `REACT_APP_SUPABASE_URL` = your Supabase URL
  - `REACT_APP_SUPABASE_ANON_KEY` = your Supabase anon key
4. Deploy the frontend.

### 3. Final Wiring

1. Copy your final Vercel production URL.
2. Update Render env var `REACT_APP_VERCEL_URL` with that URL.
3. Redeploy Render once so CORS allows your frontend.

### 4. Common Issues

- CORS errors: ensure `REACT_APP_VERCEL_URL` in Render exactly matches the production Vercel domain with `https`.
- Frontend still calling localhost: confirm `REACT_APP_API_BASE` is set in Vercel and redeploy.
- Slow/cold starts on backend: this is expected for large ML model initialization.
