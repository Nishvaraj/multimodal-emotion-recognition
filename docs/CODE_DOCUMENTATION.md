# Code Documentation Coverage

This document tracks how documentation is provided across the MMER codebase from a sustainability and maintainability perspective.

## Coverage Strategy

- Source files that support comments include module docstrings, section headers, and inline rationale comments.
- Runtime API documentation is generated via FastAPI OpenAPI.
- Non-commentable file formats (for example JSON) are documented via this index and README.

## Backend

- `backend/main.py`
  - Coverage: module docstring, section headers, endpoint summaries/tags, inline rationale comments.
- `backend/__init__.py`
  - Coverage: module docstring.
- `backend/services/explainability.py`
  - Coverage: module docstring, section headers, algorithm rationale comments.
- `backend/services/data_loader.py`
  - Coverage: module docstring, dataset/transform comments.
- `backend/services/__init__.py`
  - Coverage: module docstring.

## Frontend

- `frontend/src/App.js`
  - Coverage: file-level purpose comment, section headers, targeted inline comments.
- `frontend/src/supabaseHistoryService.js`
  - Coverage: section headers and CRUD operation comments.
- `frontend/src/supabaseClient.js`
  - Coverage: configuration and client purpose comments.
- `frontend/src/index.js`
  - Coverage: app mount comment.
- `frontend/src/index.css`
  - Coverage: top-level styling scope comment and section markers.
- `frontend/src/App.css`
  - Coverage: top-level legacy-style scope comment and section markers.

## Deployment And Runtime

- `Dockerfile`
  - Coverage: stage and layer purpose comments.
- `start.sh`
  - Coverage: startup contract and runtime behavior comments.
- `.env.example`
  - Coverage: grouped environment variable comments and API docs route note.
- `requirements.txt`
  - Coverage: dependency group comments.

## Data And Database

- `supabase_schema.sql`
  - Coverage: step-by-step migration/setup comments.
- `supabase_setup.sql`
  - Coverage: schema/policy/function comments.
- `data/README.md`
  - Coverage: folder usage and licensing guidance.

## Configuration

- `configs/config.yaml`
  - Coverage: section-level comments for model/data/training/evaluation/logging.
- `frontend/tailwind.config.js`
  - Coverage: scan/theme comments.
- `frontend/postcss.config.js`
  - Coverage: pipeline comment.

## Non-Commentable Files

The following are documented by README and this index because the file format does not support comments reliably or because they are binary assets:

- JSON files:
  - `frontend/package.json`
  - `frontend/package-lock.json`
  - `frontend/vercel.json`
  - `.vscode/settings.json`
  - `.vscode/launch.json`
  - `.vscode/tasks.json`
- Binary/media assets:
  - `frontend/src/assets/logo.png`
  - `frontend/public/landing-hero.mp4`
  - `frontend/public/screenshots/*.png`
- Notebook files (`.ipynb`) are narrative-capable documents and should be documented inside notebook markdown cells.

## API Documentation Entrypoints

When backend is running:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`
