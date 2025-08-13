import os
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import PlainTextResponse
import uvicorn

from whisper_utils import whisper_transcribe
from config import (
    API_HOST,
    API_PORT,
    DEBUG,
    SUPPORTED_AUDIO_FORMATS,
    UPLOAD_DIR,
    TEMP_DIR,
)

app = FastAPI(
    title="Audio to VTT API",
    description="Convert audio files to VTT subtitles using Fast Whisper",
    version="1.0.0",
)


@app.get("/")
async def root():
    return 1


@app.post("/transcribe/vtt")
async def transcribe_audio_vtt_only(
    file: UploadFile = File(...), language: Optional[str] = Form(None)
):
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}",
        )

    # Create temporary file
    temp_file = TEMP_DIR / f"temp_{file.filename}"

    try:
        # Save uploaded file
        with open(temp_file, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Transcribe audio
        vtt_content, _ = whisper_transcribe(temp_file, language)

        return PlainTextResponse(
            content=vtt_content,
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename={file.filename}.vtt"
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    finally:
        # Clean up temporary file
        if temp_file.exists():
            temp_file.unlink()


if __name__ == "__main__":
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=DEBUG)
