# Audio2VTT

Convert audio files to VTT subtitles using Faster-Whisper.

[![Runpod](https://api.runpod.io/badge/garylab/audio2vtt)](https://console.runpod.io/hub/garylab/audio2vtt)
[![PyPI](https://img.shields.io/pypi/v/audio2vtt.svg)](https://pypi.org/project/audio2vtt/)
[![Python Versions](https://img.shields.io/pypi/pyversions/audio2vtt.svg)](https://pypi.org/project/audio2vtt/)

## Features

- 🎵 Multiple audio formats (MP3, WAV, M4A, FLAC, OGG, AAC)
- 🚀 Fast transcription with Faster-Whisper
- 📝 VTT (WebVTT) format output with timestamps
- 🔧 Configurable models and settings
- 🐳 Docker support (CPU/GPU)
- ☁️ RunPod serverless deployment ready

## Installation

```bash
pip install audio2vtt
```

With RunPod support:

```bash
pip install audio2vtt[runpod]
```

For development:

```bash
pip install audio2vtt[dev]
```

## Quick Start

### Basic Usage

```python
from audio2vtt import Audio2VTT

converter = Audio2VTT(
    model_size_or_path="base",
    device="cpu",
    compute_type="int8"
)

vtt_content, word_count = converter.transcribe("audio.mp3", language="en")
print(vtt_content)
print(f"Transcribed {word_count} words")
```

### Transcribe from File Object

```python
with open("audio.mp3", "rb") as audio_file:
    vtt_content, word_count = converter.transcribe(audio_file)
```

### Save to File

```python
word_count = converter.transcribe_to_file(
    "audio.mp3",
    "output.vtt",
    language="en"
)
```

### Advanced Configuration

```python
converter = Audio2VTT(
    model_size_or_path="large-v3",
    device="cuda",
    compute_type="float16",
    cpu_threads=8,
    num_workers=4,
    download_root="./models"
)

vtt_content, word_count = converter.transcribe(
    "audio.mp3",
    language="en",
    beam_size=5,
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 500}
)
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

## Output Format

```
WEBVTT

00:00:00.000 --> 00:00:03.500
Hello, this is a test transcription.

00:00:03.500 --> 00:00:07.200
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
