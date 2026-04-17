FROM python:3.12-slim

# Install system dependencies including libxcb
RUN apt-get update && apt-get install -y \
    libxcb1 \
    libxcb-render0 \
    libxcb-shm0 \
    libxcb-xfixes0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install deps, then force-remove full opencv and keep only headless
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir opencv-python-headless>=4.10.0

COPY . .

CMD gunicorn backend.main:app -w 1 -k uvicorn.workers.UvicornWorker --timeout 600 --bind 0.0.0.0:$PORT