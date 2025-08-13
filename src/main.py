from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, Header
from fastapi.security import APIKeyHeader
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.whisper_utils import whisper_transcribe
from src.config import (
    DEBUG,
    SUPPORTED_AUDIO_FORMATS,
    ADMIN_API_KEY,
)

app = FastAPI(
    title="Audio to VTT API",
    description="Convert audio files to VTT subtitles using Fast Whisper",
    version="1.0.0",
)

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(api_key: Optional[str] = Depends(api_key_header)):
    """Get and validate API key"""
    # If no admin API key is set, skip authentication
    if not ADMIN_API_KEY:
        return None

    # If admin API key is set but no header provided
    if not api_key:
        raise HTTPException(
            status_code=401, detail="API key required. Please provide X-API-Key header"
        )

    # Verify the API key
    if api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return api_key


@app.get("/")
async def root():
    return 1


@app.post("/transcribe/vtt", dependencies=[Depends(get_api_key)])
async def transcribe_audio_vtt_only(
    file: UploadFile = File(...), language: Optional[str] = Form(None)
):
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}",
        )

    try:
        vtt_content, _ = whisper_transcribe(file.file, language)
        return PlainTextResponse(content=vtt_content, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=DEBUG)
