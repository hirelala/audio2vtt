from faster_whisper import WhisperModel
from src.config import WHISPER_MODEL, MODELS_DIR


download_root = MODELS_DIR.joinpath(WHISPER_MODEL)
download_root.mkdir(parents=True, exist_ok=True)

model = WhisperModel(
    model_size_or_path=WHISPER_MODEL,
    download_root=download_root.as_posix(),
    local_files_only=False,
)

print(f"Model {WHISPER_MODEL} downloaded successfully!")

