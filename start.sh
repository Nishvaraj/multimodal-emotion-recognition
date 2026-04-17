#!/bin/bash
# Start FastAPI via Gunicorn+Uvicorn and honor platform-provided PORT when present.
exec python -m gunicorn backend.main:app -w 1 -k uvicorn.workers.UvicornWorker --timeout 600 --bind 0.0.0.0:${PORT:-8080}