import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Whisper Configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # cpu, cuda, metal
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "4"))
WHISPER_NUM_WORKERS = int(os.getenv("WHISPER_NUM_WORKERS", "1"))
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
WHISPER_LOCAL_FILES_ONLY = (
    os.getenv("WHISPER_LOCAL_FILES_ONLY", "False").lower() == "true"
)

# File paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
TEMP_DIR = BASE_DIR / "temp"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Supported audio formats
SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
