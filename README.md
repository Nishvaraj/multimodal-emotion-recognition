# Multi-Modal Emotion Recognition

Emotion analytics web app with facial, speech, and combined (image+audio or video) analysis.

## Current Capabilities

- Facial emotion prediction (upload + webcam capture)
- Speech emotion prediction (upload + live mic recording)
- Combined analysis:
  - Separate image + audio mode
  - Video upload/live recording mode
- Explainability outputs (Grad-CAM and audio saliency when available)
- Supabase auth (signup/login/logout/session)
- Supabase-backed history with notes, pin/unpin, delete, CSV and summary export

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

## Environment

- Frontend local env: `frontend/.env.local`
- Example shared vars: `.env.example`

## Project Structure

- `frontend/` React dashboard and auth flows
- `backend/` FastAPI inference endpoints
- `models/` trained checkpoints
- `notebooks/` training and experiments
- `configs/` project configuration files

## Notes

- First backend startup may take longer due to model initialization.
- Webcam/mic features require browser permissions.
