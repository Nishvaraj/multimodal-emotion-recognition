FROM python:3.12-slim

# Install native libraries required by OpenCV, audio decoding, and PyTorch runtime dependencies.
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

# Copy dependency list first so Docker can reuse cached layers when app code changes.
COPY requirements.txt .

ARG CACHEBUST=4
# Install Python dependencies with a higher timeout to reduce transient network failures on CI builds.
RUN pip install --no-cache-dir --timeout=300 --retries=5 -r requirements.txt
# Ensure only headless OpenCV is installed in container runtime.
RUN pip uninstall -y opencv-python || true
RUN pip install --no-cache-dir --timeout=300 --force-reinstall opencv-python-headless>=4.10.0

# Copy application source after dependency installation for better build caching.
COPY . .

RUN chmod +x start.sh

# Container entrypoint delegates startup flags to start.sh.
CMD ["./start.sh"]