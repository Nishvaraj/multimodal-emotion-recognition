FROM python:3.12-slim

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

# Cache buster
ARG CACHEBUST=1

RUN pip install --no-cache-dir --timeout=300 --retries=5 -r requirements.txt && \
    pip uninstall -y opencv-python || true && \
    pip install --no-cache-dir --timeout=300 opencv-python-headless>=4.10.0

COPY . .