from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import uvicorn
import io

from src.whisper_utils import whisper_transcribe, detect_audio_format
from src.config import SUPPORTED_AUDIO_FORMATS, API_ADMIN_KEY

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
    if not API_ADMIN_KEY:
        return None

    # If admin API key is set but no header provided
    if not api_key:
        raise HTTPException(
            status_code=401, detail="API key required. Please provide X-API-Key header"
        )

    # Verify the API key
    if api_key != API_ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return api_key


@app.get("/")
async def root():
    return RedirectResponse("/docs")
    

@app.post("/vtt", dependencies=[Depends(get_api_key)])
async def transcribe(
    file: UploadFile = File(...), language: Optional[str] = Form(None)
):
    # Read audio data
    audio_data = await file.read()
    
    # Detect actual audio format from the data
    detected_format = detect_audio_format(audio_data)
    if not detected_format:
        raise HTTPException(
            status_code=400,
            detail="Could not detect audio format from the provided data. The file may be corrupted or in an unsupported format.",
        )
    
    if detected_format not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{detected_format}'. Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}",
        )
    
    audio_io = io.BytesIO(audio_data)
    result, word_count = whisper_transcribe(audio_io, language)
    return JSONResponse(
        content={
            "vtt": result,
            "word_count": word_count
        },
        status_code=200,
    )


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000)
