import runpod
import io
import os
import base64

from src.audio_subtitler import AudioSubtitler

audio2vtt = AudioSubtitler(
    model_size_or_path=os.getenv("WHISPER_MODEL", "base"),
    device=os.getenv("WHISPER_DEVICE", "cpu"),
    device_index=[int(i) for i in os.getenv("WHISPER_DEVICE_INDEX", "0").split(",")],
    compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
    cpu_threads=int(os.getenv("WHISPER_CPU_THREADS", "4")),
    num_workers=int(os.getenv("WHISPER_NUM_WORKERS", "1")),
    download_root=os.getenv("DOWNLOAD_ROOT", "models"),
    local_files_only=os.getenv("LOCAL_FILES_ONLY", "true").lower() == "true",
)


def handler(event):
    job_input = event.get("input", {})
    audio_base64 = job_input.get("audio")
    format = job_input.get("format", "vtt")
    if not audio_base64:
        return {"error": "No audio data provided. Please provide 'audio' field with base64 encoded audio data."}
    
    try:
        audio_data = base64.b64decode(audio_base64)
    except Exception as e:
        return {"error": f"Failed to decode base64 audio data: {str(e)}"}

    try:    
        return audio2vtt.transcribe(
            io.BytesIO(audio_data),
            format=format,
            beam_size=os.getenv("WHISPER_BEAM_SIZE", "5"),
            vad_filter=os.getenv("WHISPER_VAD_FILTER", "true").lower() == "true",
        )
    except Exception as e:
        return {"error": f"Transcription failed: {str(e)}"}


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
