# Multi-Modal Emotion Recognition

Multimodal emotion analysis app that predicts facial and speech emotions, compares the two modalities, and stores results in Supabase. The repository includes the FastAPI backend, React frontend, training notebooks, and deployment files for Railway and Vercel.

## What The Project Does

### Prediction Workflows
- Facial emotion prediction from uploaded images or webcam frames.
- Speech emotion prediction from uploaded audio or live microphone input.
- Combined multimodal prediction from image plus audio.
- Video-based prediction that samples facial frames and extracts speech from the same upload.

### Explainability
- Facial Grad-CAM heatmaps.
- Audio saliency plots.
- Explainability status messages when a visualization cannot be generated.

### User Features
- Supabase authentication for sign in, sign up, and session persistence.
- Per-user analysis history with notes, pinning, deleting, and export options.
- Results summary views for recent analyses and concordance trends.

### Backend API
- Health and model status endpoints.
- Facial, speech, combined, and video prediction endpoints.
- Emotion-list endpoints used by the frontend.

## Repository Layout

- `backend/`: FastAPI inference server and explainability helpers.
- `frontend/`: React UI and Supabase history integration.
- `configs/`: runtime configuration files.
- `data/`: dataset notes and storage guidance.
- `notebooks/`: training notebooks for facial and speech models.
- `supabase_schema.sql` and `supabase_setup.sql`: database schema and setup scripts.

## Local Setup

1. Activate the Python environment.

```bash
source .venv/bin/activate
```

2. Install backend dependencies.

```bash
pip install -r requirements.txt
```

3. Install frontend dependencies.

```bash
cd frontend
npm install
cd ..
```

4. Start the backend.

```bash
./.venv/bin/uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

5. Start the frontend in a second terminal.

```bash
cd frontend
npm start
```

6. Open the application.

- Frontend: http://localhost:3000
- Backend health: http://127.0.0.1:8000/health

## Environment Variables

### Frontend
- `REACT_APP_API_BASE`
- `REACT_APP_SUPABASE_URL`
- `REACT_APP_SUPABASE_ANON_KEY`

### Backend
- `ENV`
- `USE_GPU`
- `PRELOAD_MODELS`
- `FRONTEND_URL`
- `CORS_ORIGINS`

## Deployment Notes

### Railway Backend
1. Create a Railway project from this repository.
2. Set `ENV=production`.
3. Set `USE_GPU=false` unless the deployment has GPU support.
4. Set `PRELOAD_MODELS=false` for faster startup on cold deployments.
5. Set `FRONTEND_URL` to the Vercel domain.
6. Optionally set `CORS_ORIGINS` to a comma-separated allowlist.
7. Confirm the health endpoint responds at `/health`.

### Vercel Frontend
1. Import the repository into Vercel.
2. Set the root directory to `frontend`.
3. Configure the frontend environment variables listed above.
4. Deploy and open the generated Vercel URL.

### CORS Check
- Prefer an explicit `CORS_ORIGINS` allowlist.
- Use `FRONTEND_URL` only as the fallback origin for a single primary domain.

## Notes For Review

- The backend uses lazy model loading by default to reduce startup cost.
- Webcam and microphone features require browser permissions.
- Generated artifacts such as cache folders, build outputs, and local environment files are intentionally excluded from version control.
