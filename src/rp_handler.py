import runpod
import io
import base64
from pathlib import Path
from typing import Optional

from src.whisper_utils import whisper_transcribe, detect_audio_format
from src.config import SUPPORTED_AUDIO_FORMATS


def handler(event):
    """
    RunPod handler function for audio to VTT transcription.
    
    Expected input format:
    {
        "input": {
            "audio": "<base64_encoded_audio_data>",
            "language": "en"  # optional
        }
    }
    
    Note: Audio format is automatically detected from the audio data itself.
    
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
        
        # Get optional language parameter
        language = job_input.get("language")
        
        # Decode base64 audio data
        try:
            audio_data = base64.b64decode(audio_base64)
        except Exception as e:
            return {"error": f"Failed to decode base64 audio data: {str(e)}"}
        
        # Detect actual audio format from the data
        detected_format = detect_audio_format(audio_data)
        if not detected_format:
            return {"error": "Could not detect audio format from the provided data. The file may be corrupted or in an unsupported format."}
        
        if detected_format not in SUPPORTED_AUDIO_FORMATS:
            return {
                "error": f"Unsupported audio format '{detected_format}'. Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
            }
        
        # Create BytesIO object from audio data
        audio_io = io.BytesIO(audio_data)
        
        # Transcribe using whisper
        vtt_content, word_count = whisper_transcribe(audio_io, language)
        
        # Return result
        return {
            "vtt": vtt_content,
            "word_count": word_count
        }
        
    except Exception as e:
        return {"error": f"Transcription failed: {str(e)}"}


# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})