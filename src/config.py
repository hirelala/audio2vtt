import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Whisper Configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # cpu, cuda, metal
WHISPER_DEVICE_INDEX = [
    int(i) for i in os.getenv("WHISPER_DEVICE_INDEX", "0").split(",")
]
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "4"))
WHISPER_NUM_WORKERS = int(os.getenv("WHISPER_NUM_WORKERS", "1"))
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
WHISPER_LOCAL_FILES_ONLY = (
    os.getenv("WHISPER_LOCAL_FILES_ONLY", "False").lower() == "true"
)

# API Authentication
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

# Queue Configuration
QUEUE_WORKERS = int(os.getenv("QUEUE_WORKERS", "2"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "100"))

# File paths
MODELS_DIR = Path(__file__).parent.parent.joinpath("models")
MODELS_DIR.mkdir(exist_ok=True)


# Supported audio formats
SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
