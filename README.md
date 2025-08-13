# Audio to VTT API

A FastAPI-based REST API server that converts audio files to VTT subtitles using Fast Whisper.

## Features

- 🎵 Support for multiple audio formats (MP3, WAV, M4A, FLAC, OGG, AAC)
- 🚀 Fast transcription using Fast Whisper
- 📝 Output in VTT (WebVTT) format with timestamps
- 🔧 Configurable Whisper model and settings
- 🏥 Health check endpoint
- 📊 Interactive API documentation

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

#### 2. Health Check
- **GET** `/health` - Health check endpoint

#### 3. Transcribe Audio
- **POST** `/transcribe/vtt` - Upload audio file and get VTT file download
- **Parameters:**
  - `file`: Audio file (required)
  - `language`: Language code (optional, e.g., 'en', 'es', 'fr')

### Example Usage

#### Using curl:

```bash
# Transcribe and download VTT file (auto-detect language)
curl -X POST "http://localhost:8000/transcribe/vtt" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.mp3" \
  --output subtitles.vtt

# Transcribe and download VTT file with specific language
curl -X POST "http://localhost:8000/transcribe/vtt" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.mp3" \
  -F "language=en" \
  --output subtitles.vtt

# Transcribe with Spanish language
curl -X POST "http://localhost:8000/transcribe/vtt" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.mp3" \
  -F "language=es" \
  --output subtitles.vtt
```

#### Using Python requests:

```python
import requests

# Transcribe audio file (auto-detect language)
with open('audio.mp3', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/transcribe/vtt', files=files)
    
# Save VTT content to file
with open('subtitles.vtt', 'w', encoding='utf-8') as f:
    f.write(response.text)

# Transcribe audio file with specific language
with open('audio.mp3', 'rb') as f:
    files = {'file': f}
    data = {'language': 'en'}  # Specify English
    response = requests.post('http://localhost:8000/transcribe/vtt', files=files, data=data)
    
# Save VTT content to file
with open('subtitles_en.vtt', 'w', encoding='utf-8') as f:
    f.write(response.text)
```

### API Documentation

Once the server is running, you can access:
- **Interactive API docs**: `http://localhost:8000/docs`
- **ReDoc documentation**: `http://localhost:8000/redoc`

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |
| `DEBUG` | `False` | Enable debug mode |
| `WHISPER_MODEL` | `base` | Whisper model size (tiny, base, small, medium, large, large-v2, large-v3) |
| `WHISPER_DEVICE` | `cpu` | Device to use (cpu, cuda, metal) |
| `WHISPER_COMPUTE_TYPE` | `float32` | Compute type for GPU |
| `WHISPER_CPU_THREADS` | `4` | Number of CPU threads |
| `WHISPER_NUM_WORKERS` | `1` | Number of workers |
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

## Project Structure

```
audio-to-vtt/
├── main.py              # FastAPI application
├── whisper_utils.py     # Whisper transcription utilities
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
├── env_example.txt      # Example environment variables
├── README.md           # This file
├── models/             # Whisper models directory (auto-created)
├── uploads/            # Upload directory (auto-created)
└── temp/               # Temporary files directory (auto-created)
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
