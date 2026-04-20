# Multi-Modal Emotion Recognition

Multimodal emotion analysis app that predicts facial and speech emotions, compares the two modalities, and stores results in Supabase. The current deployment architecture uses a FastAPI backend on Hugging Face Spaces (Docker) and a React frontend on Vercel.

## Documentation And Sustainability

This repository is documented for long-term maintainability using three layers:

- **Project documentation**: this README captures architecture, setup, operations, and contribution guidance.
- **API documentation**: FastAPI auto-generates OpenAPI docs at runtime (`/docs`, `/redoc`, `/openapi.json`).
- **Code documentation**: source files include module-level docstrings, section headers, and intent-driven inline comments.

### OpenAPI Documentation

When the backend is running locally, the API docs are available at:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

These endpoints are generated from `backend/main.py` route definitions and should be treated as the canonical API contract for frontend/backend integration.

Detailed documentation coverage across files is tracked in `docs/CODE_DOCUMENTATION.md`.

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

## File Documentation Policy

To keep the codebase sustainable for academic and production maintenance:

- All editable source files include a file purpose comment or module docstring.
- Complex logic blocks include concise rationale comments.
- API handlers include clear summaries and tags in FastAPI decorators.
- Non-commentable formats (JSON lock/config files, binaries, media assets) are documented here in README and by folder naming conventions.

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

### Hugging Face Spaces Backend (Docker)
1. Use a Docker Space and include `Dockerfile`, `requirements.txt`, and backend source files.
2. Keep the runtime command bound to the Space port (`7860`) in Docker runtime or startup script.
3. Set `ENV=production`.
4. Set `USE_GPU=false` unless the Space hardware includes GPU support.
5. Set `PRELOAD_MODELS=false` for faster cold starts (or `true` for faster first inference).
6. Set `FRONTEND_URL` to the Vercel domain.
7. Optionally set `CORS_ORIGINS` to a comma-separated allowlist.
8. Add `HF_TOKEN` in Space secrets for higher Hub rate limits and faster model downloads.
9. Confirm the health endpoint responds at `/health`.

### Vercel Frontend
1. Import the repository into Vercel.
2. Set the root directory to `frontend`.
3. Set `REACT_APP_API_BASE` to your Hugging Face Space backend URL.
4. Configure the remaining frontend environment variables listed above.
4. Deploy and open the generated Vercel URL.

### CORS Check
- Prefer an explicit `CORS_ORIGINS` allowlist.
- Use `FRONTEND_URL` only as the fallback origin for a single primary domain.

## Notes For Review

- The backend uses lazy model loading by default to reduce startup cost.
- First inference can be slower because facial and speech models are loaded on-demand.
- Webcam and microphone features require browser permissions.
- Generated artifacts such as cache folders, build outputs, and local environment files are intentionally excluded from version control.

## Maintenance Checklist

- Keep endpoint summaries/tags updated in `backend/main.py` whenever API routes change.
- Keep this README aligned with startup commands and deployment defaults.
- Add comments only where logic is non-obvious; avoid line-by-line noise comments.
- Run tests/lint checks before deployment to preserve documentation-code consistency.
