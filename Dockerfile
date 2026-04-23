FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libxcb1 libxcb-render0 libxcb-shm0 libxcb-xfixes0 \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 ffmpeg gcc \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --timeout=300 --retries=5 -r requirements.txt
RUN pip uninstall -y opencv-python || true
RUN pip install --no-cache-dir --timeout=300 --force-reinstall opencv-python-headless>=4.10.0

COPY --chown=user . .

CMD ["python", "-m", "gunicorn", "backend.main:app", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "600", "--bind", "0.0.0.0:7860"]
