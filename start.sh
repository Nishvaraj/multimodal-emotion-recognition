#!/bin/bash
pip uninstall -y opencv-python || true
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
