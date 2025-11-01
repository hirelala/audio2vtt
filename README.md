# Audio Subtitler

Convert audio files to subtitles (VTT, SRT) using Faster-Whisper.

[![PyPI](https://img.shields.io/pypi/v/audio-subtitler.svg)](https://pypi.org/project/audio-subtitler/)
[![Python Versions](https://img.shields.io/pypi/pyversions/audio-subtitler.svg)](https://pypi.org/project/audio-subtitler/)

## Features

- 🎵 Multiple audio formats (MP3, WAV, M4A, FLAC, OGG, AAC)
- 🚀 Fast transcription with Faster-Whisper
- 📝 Multiple subtitle formats: VTT (WebVTT) and SRT
- 🔧 Configurable models and settings
- 🐳 Docker support (CPU/GPU)
- ☁️ RunPod serverless deployment ready
- 💻 Simple CLI and Python API

## Installation

```bash
pip install audio-subtitler
```

With RunPod support:

```bash
pip install audio-subtitler[runpod]
```

For development:

```bash
pip install audio-subtitler[dev]
```

## Quick Start

### Command Line Interface

After installation, you can use the `audiosubtitler` command (or the shorter `audiosub`):

```bash
# Basic usage - VTT output to stdout
audiosubtitler input.mp3 > output.vtt

# Generate SRT format
audiosubtitler input.mp3 --format srt > output.srt

# Specify output file directly
audiosubtitler input.mp3 -o output.vtt

# Use a different model
audiosubtitler input.mp3 --model large-v2 > output.vtt

# Specify language
audiosubtitler input.mp3 --language en > output.vtt

# Use GPU
audiosubtitler input.mp3 --device cuda > output.vtt

# Quiet mode (suppress progress messages)
audiosubtitler input.mp3 --quiet > output.vtt

# Using the shorter command
audiosub input.mp3 --format srt -o output.srt

# Show all options
audiosubtitler --help
```

### Python API

```python
from audio_subtitler import AudioSubtitler

# Initialize the converter
converter = AudioSubtitler(
    model_size_or_path="base",
    device="cpu",
    compute_type="int8"
)

# Generate VTT subtitles
result = converter.transcribe("audio.mp3", format="vtt", language="en")
print(result["vtt"])
print(f"Transcribed {result['word_count']} words")

# Generate SRT subtitles
result = converter.transcribe("audio.mp3", format="srt", language="en")
print(result["srt"])
```

### Transcribe from File Object

```python
with open("audio.mp3", "rb") as audio_file:
    # VTT format
    result = converter.transcribe(audio_file, format="vtt")
    print(result["vtt"])
    
    # SRT format
    result = converter.transcribe(audio_file, format="srt")
    print(result["srt"])
```

### Advanced Configuration

```python
converter = AudioSubtitler(
    model_size_or_path="large-v3",
    device="cuda",
    compute_type="float16",
    cpu_threads=8,
    num_workers=4,
    download_root="./models"
)

# Transcribe with custom parameters
result = converter.transcribe(
    "audio.mp3",
    format="srt",  # or "vtt"
    language="en",
    beam_size=5,
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 500}
)

print(f"Subtitle content: {result['srt']}")  # or result['vtt']
print(f"Word count: {result['word_count']}")
```

## RunPod Serverless

Deploy with Docker:

```bash
docker-compose up
```

For GPU:

```bash
docker-compose -f docker-compose-gpu.yml up
```

**Input:**
```json
{
  "input": {
    "audio": "<base64_encoded_audio>",
    "language": "en"
  }
}
```

**Output:**
```json
{
  "vtt": "WEBVTT\n\n00:00:00.000 --> ...",
  "word_count": 150
}
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `base` | Model size: tiny, base, small, medium, large, large-v3 |
| `WHISPER_DEVICE` | `cpu` | Device: cpu, cuda, metal |
| `WHISPER_COMPUTE_TYPE` | `int8` | Compute type: int8, float16, float32 |
| `WHISPER_BEAM_SIZE` | `5` | Beam size for decoding |

## Supported Languages

Auto-detects language or specify: `en`, `es`, `fr`, `de`, `it`, `pt`, `ru`, `ja`, `ko`, `zh`, etc.

## Output Formats

### VTT (WebVTT) Format

```
WEBVTT

00:00:00.000 --> 00:00:03.500
Hello, this is a test transcription.

00:00:03.500 --> 00:00:07.200
The audio is converted to text with timestamps.
```

### SRT Format

```
1
00:00:00,000 --> 00:00:03,500
Hello, this is a test transcription.

2
00:00:03,500 --> 00:00:07,200
The audio is converted to text with timestamps.
```

## Publishing to PyPI

```bash
pip install build twine
python -m build
python -m twine upload dist/*
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
