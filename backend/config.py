import os
from dotenv import load_dotenv

load_dotenv()

# Environment
ENV = os.getenv("ENV", "development")
PORT = int(os.getenv("PORT", 8000))
WORKERS = int(os.getenv("WORKERS", 1))

# API Configuration
FRONTEND_URL = os.getenv("REACT_APP_VERCEL_URL", "http://localhost:3001")
API_BASE = os.getenv("REACT_APP_API_BASE", "http://localhost:8000")

# Model Configuration
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"

print(f"[Config] Env: {ENV} | Port: {PORT} | Workers: {WORKERS} | GPU: {USE_GPU}")
