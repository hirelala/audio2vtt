# Audio to VTT API

A FastAPI-based REST API server that converts audio files to VTT subtitles using Fast Whisper.

## Features

- 🎵 Support for multiple audio formats (MP3, WAV, M4A, FLAC, OGG, AAC)
- 🚀 Fast transcription using Fast Whisper
- 📝 Output in VTT (WebVTT) format with timestamps
- 🔧 Configurable Whisper model and settings
- 🏥 Health check endpoint
- 📊 Interactive API documentation
- 🔄 Queue-based job processing with configurable workers
- 🔐 API key authentication support

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd audio-to-vtt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create environment file (optional):
```bash
cp env_example.txt .env
# Edit .env with your preferred settings
```

## Usage

### Starting the Server

```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`

### API Endpoints

#### 1. Root Endpoint
- **GET** `/` - API information and available endpoints

#### 2. Transcribe Audio (Queue-based)
- **POST** `/transcribe/vtt` - Submit audio file for transcription (returns job ID)
- **Parameters:**
  - `file`: Audio file (required)
  - `language`: Language code (optional, e.g., 'en', 'es', 'fr')
- **Headers:**
  - `X-API-Key`: API key (if ADMIN_API_KEY is set)
- **Returns:** Job ID and status (HTTP 202)

#### 3. Check Job Status
- **GET** `/transcribe/status/{job_id}` - Check the status of a transcription job
- **Headers:**
  - `X-API-Key`: API key (if ADMIN_API_KEY is set)
- **Returns:** Job status (pending, processing, completed, failed)

#### 4. Get Job Result
- **GET** `/transcribe/result/{job_id}` - Get the VTT result of a completed job
- **Headers:**
  - `X-API-Key`: API key (if ADMIN_API_KEY is set)
- **Returns:** VTT content (HTTP 200) or error (HTTP 202/500)

#### 5. Queue Information
- **GET** `/queue/info` - Get information about the queue and workers
- **Headers:**
  - `X-API-Key`: API key (if ADMIN_API_KEY is set)
- **Returns:** Queue statistics (workers, queue size, job counts)

### Example Usage

#### Using curl:

```bash
# Submit a transcription job (with API key if required)
curl -X POST "http://localhost:8000/transcribe/vtt" \
  -H "X-API-Key: your-api-key-here" \
  -F "file=@audio.mp3" \
  -F "language=en"

# Response: {"job_id": "uuid-here", "status": "pending", "message": "..."}

# Check job status
JOB_ID="uuid-here"
curl -X GET "http://localhost:8000/transcribe/status/$JOB_ID" \
  -H "X-API-Key: your-api-key-here"

# Get completed result
curl -X GET "http://localhost:8000/transcribe/result/$JOB_ID" \
  -H "X-API-Key: your-api-key-here" \
  --output subtitles.vtt

# Check queue information
curl -X GET "http://localhost:8000/queue/info" \
  -H "X-API-Key: your-api-key-here"
```

#### Using Python requests:

```python
import requests
import time

# API configuration (set API key if ADMIN_API_KEY is configured)
API_URL = "http://localhost:8000"
API_KEY = "your-api-key-here"  # Optional, only if ADMIN_API_KEY is set
headers = {"X-API-Key": API_KEY} if API_KEY else {}

# 1. Submit transcription job
with open('audio.mp3', 'rb') as f:
    files = {'file': f}
    data = {'language': 'en'}  # Optional
    response = requests.post(
        f"{API_URL}/transcribe/vtt",
        files=files,
        data=data,
        headers=headers
    )
    job_data = response.json()
    job_id = job_data['job_id']
    print(f"Job submitted: {job_id}")

# 2. Poll for job completion
while True:
    status_response = requests.get(
        f"{API_URL}/transcribe/status/{job_id}",
        headers=headers
    )
    status = status_response.json()
    print(f"Status: {status['status']}")
    
    if status['status'] == 'completed':
        break
    elif status['status'] == 'failed':
        print(f"Job failed: {status.get('error')}")
        exit(1)
    
    time.sleep(2)  # Wait 2 seconds before checking again

# 3. Get the result
result_response = requests.get(
    f"{API_URL}/transcribe/result/{job_id}",
    headers=headers
)

# Save VTT content to file
with open('subtitles.vtt', 'w', encoding='utf-8') as f:
    f.write(result_response.text)

print("Transcription complete!")
```

### API Documentation

Once the server is running, you can access:
- **Interactive API docs**: `http://localhost:8000/docs`
- **ReDoc documentation**: `http://localhost:8000/redoc`

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `False` | Enable debug mode |
| `ADMIN_API_KEY` | `` | API key for authentication (optional, leave empty to disable) |
| `QUEUE_WORKERS` | `2` | Number of worker threads for processing transcription jobs |
| `MAX_QUEUE_SIZE` | `100` | Maximum number of jobs that can be queued |
| `WHISPER_MODEL` | `base` | Whisper model size (tiny, base, small, medium, large, large-v2, large-v3) |
| `WHISPER_DEVICE` | `cpu` | Device to use (cpu, cuda, metal) |
| `WHISPER_COMPUTE_TYPE` | `int8` | Compute type (int8, float16, float32) |
| `WHISPER_CPU_THREADS` | `4` | Number of CPU threads |
| `WHISPER_NUM_WORKERS` | `1` | Number of Whisper model workers |
| `WHISPER_BEAM_SIZE` | `5` | Beam size for decoding |
| `WHISPER_LOCAL_FILES_ONLY` | `False` | Use only local model files |

### Whisper Models

Available model sizes (in order of speed vs accuracy):
- `tiny` - Fastest, least accurate
- `base` - Good balance (default)
- `small` - Better accuracy
- `medium` - Even better accuracy
- `large` - Best accuracy, slower
- `large-v2` - Improved large model
- `large-v3` - Latest large model

## Queue System

The API now uses a queue-based system for processing transcription jobs:

1. **Job Submission**: When you submit an audio file, it's added to a queue and you receive a job ID
2. **Background Processing**: Configurable worker threads process jobs from the queue
3. **Status Tracking**: Check job status anytime using the job ID
4. **Result Retrieval**: Download the VTT result once the job is completed

### Queue Configuration

- `QUEUE_WORKERS`: Number of concurrent workers processing transcription jobs (default: 2)
- `MAX_QUEUE_SIZE`: Maximum number of jobs in the queue (default: 100)

### Benefits

- **Non-blocking**: API responds immediately without waiting for transcription to complete
- **Concurrent Processing**: Multiple jobs can be processed simultaneously
- **Queue Management**: Jobs are processed in order, preventing server overload
- **Status Tracking**: Monitor job progress and retrieve results when ready

## Authentication

The API supports optional API key authentication:

1. Set `ADMIN_API_KEY` environment variable to enable authentication
2. Include `X-API-Key` header in all requests when authentication is enabled
3. Leave `ADMIN_API_KEY` empty to disable authentication (open API)

## Project Structure

```
audio-to-vtt/
├── src/
│   ├── main.py              # FastAPI application
│   ├── whisper_utils.py     # Whisper transcription utilities
│   ├── config.py            # Configuration settings
│   ├── queue_manager.py     # Queue and worker management
│   └── __init__.py
├── pyproject.toml          # Project dependencies
├── uv.lock                 # Lock file
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── README.md              # This file
└── models/                # Whisper models directory (auto-created)
```

## Supported Audio Formats

- MP3 (.mp3)
- WAV (.wav)
- M4A (.m4a)
- FLAC (.flac)
- OGG (.ogg)
- AAC (.aac)

## Supported Languages

The API supports language specification for better transcription accuracy. Common language codes include:

- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `ja` - Japanese
- `ko` - Korean
- `zh` - Chinese

If no language is specified, Whisper will auto-detect the language. Specifying the correct language can improve transcription accuracy.

## Output Format

The API returns VTT (WebVTT) format subtitles:

```
WEBVTT

00:00:00.000 --> 00:00:03.500
Hello, this is a test transcription.

00:00:03.500 --> 00:00:07.200
The audio has been converted to subtitles.
```

## Error Handling

The API includes comprehensive error handling:
- Invalid file formats
- File upload errors
- Transcription failures
- Server errors

All errors return appropriate HTTP status codes and error messages.

## Performance Tips

1. **Model Selection**: Use smaller models (tiny, base) for faster processing
2. **Device**: Use GPU (cuda/metal) if available for better performance
3. **CPU Threads**: Adjust based on your system's capabilities
4. **File Size**: Larger files take longer to process

## License

[Add your license here]
