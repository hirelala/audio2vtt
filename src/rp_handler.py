import runpod
import io
import base64
from pathlib import Path
from typing import Optional

from src.whisper_utils import whisper_transcribe
from src.config import SUPPORTED_AUDIO_FORMATS


def handler(event):
    """
    RunPod handler function for audio to VTT transcription.
    
    Expected input format:
    {
        "input": {
            "audio": "<base64_encoded_audio_data>",
            "filename": "audio.mp3",
            "language": "en"  # optional
        }
    }
    
    Returns:
    {
        "vtt": "<vtt_content>",
        "text": "<plain_text_transcription>"
    }
    """
    try:
        # Extract input from event
        job_input = event.get("input", {})
        
        # Get audio data (base64 encoded)
        audio_base64 = job_input.get("audio")
        if not audio_base64:
            return {"error": "No audio data provided. Please provide 'audio' field with base64 encoded audio data."}
        
        # Get filename and validate extension
        filename = job_input.get("filename", "audio.mp3")
        file_extension = Path(filename).suffix.lower()
        if file_extension not in SUPPORTED_AUDIO_FORMATS:
            return {
                "error": f"Unsupported file format '{file_extension}'. Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
            }
        
        # Get optional language parameter
        language = job_input.get("language")
        
        # Decode base64 audio data
        try:
            audio_data = base64.b64decode(audio_base64)
        except Exception as e:
            return {"error": f"Failed to decode base64 audio data: {str(e)}"}
        
        # Create BytesIO object from audio data
        audio_io = io.BytesIO(audio_data)
        
        # Transcribe using whisper
        vtt_content, plain_text = whisper_transcribe(audio_io, language)
        
        # Return result
        return {
            "vtt": vtt_content,
            "text": plain_text
        }
        
    except Exception as e:
        return {"error": f"Transcription failed: {str(e)}"}


# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})