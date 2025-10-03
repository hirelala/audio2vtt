from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.queue_manager import get_queue_manager
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


@app.on_event("startup")
async def startup_event():
    """Start the queue workers on application startup"""
    queue_manager = get_queue_manager()
    await queue_manager.start()


@app.on_event("shutdown")
async def shutdown_event():
    """Stop the queue workers on application shutdown"""
    queue_manager = get_queue_manager()
    await queue_manager.stop()


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
    """Submit a transcription job to the queue"""
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}",
        )

    try:
        # Read the audio file data
        audio_data = await file.read()
        
        # Submit job to queue
        queue_manager = get_queue_manager()
        job_id = await queue_manager.submit_job(audio_data, file.filename, language)
        
        return JSONResponse(
            content={
                "job_id": job_id,
                "status": "pending",
                "message": "Job submitted successfully. Use /transcribe/status/{job_id} to check status."
            },
            status_code=202
        )
    except Exception as e:
        if "Queue is full" in str(e):
            raise HTTPException(status_code=503, detail="Server is busy. Please try again later.")
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")


@app.get("/transcribe/status/{job_id}", dependencies=[Depends(get_api_key)])
async def get_job_status(job_id: str):
    """Get the status of a transcription job"""
    queue_manager = get_queue_manager()
    job_status = queue_manager.get_job_status(job_id)
    
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JSONResponse(content=job_status)


@app.get("/transcribe/result/{job_id}", dependencies=[Depends(get_api_key)])
async def get_job_result(job_id: str):
    """Get the result of a completed transcription job as VTT"""
    queue_manager = get_queue_manager()
    job_status = queue_manager.get_job_status(job_id)
    
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_status["status"] == "pending":
        raise HTTPException(status_code=202, detail="Job is still pending")
    elif job_status["status"] == "processing":
        raise HTTPException(status_code=202, detail="Job is still processing")
    elif job_status["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"Job failed: {job_status.get('error', 'Unknown error')}")
    elif job_status["status"] == "completed":
        return PlainTextResponse(content=job_status["result"], media_type="text/plain")
    
    raise HTTPException(status_code=500, detail="Unknown job status")


@app.get("/queue/info", dependencies=[Depends(get_api_key)])
async def get_queue_info():
    """Get information about the queue"""
    queue_manager = get_queue_manager()
    return JSONResponse(content=queue_manager.get_queue_info())


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=DEBUG)
