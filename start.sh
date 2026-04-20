#!/bin/bash
# Startup contract:
# - Use Gunicorn + Uvicorn worker for production ASGI serving
# - Bind to PORT from hosting platform when provided
# - Keep a generous timeout to tolerate first-request model warm-up
exec python -m gunicorn backend.main:app -w 1 -k uvicorn.workers.UvicornWorker --timeout 600 --bind 0.0.0.0:${PORT:-8080}