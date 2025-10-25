from faster_whisper import WhisperModel
from typing import BinaryIO, Optional
from src.config import (
    MODELS_DIR,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_CPU_THREADS,
    WHISPER_NUM_WORKERS,
    WHISPER_MODEL,
    WHISPER_BEAM_SIZE,
    WHISPER_DEVICE_INDEX,
)


STOP_CHARS = set(
    ".!?,:;…‥"  # English & common
    "。！？，、；："  # Chinese/Japanese
    "।"  # Hindi
    "܀።፧"  # Semitic (Syriac, Ge'ez)
    "؟؛"  # Arabic/Persian
    "၊။"  # Burmese
    "⸮⁇⁈⁉"  # Rare multilingual
)

_whisper_model: WhisperModel = None


def detect_audio_format(audio_data: bytes) -> Optional[str]:
    """
    Detect audio format from the actual audio data using magic bytes.
    
    Returns the file extension (with dot) or None if format cannot be detected.
    """
    # Check first few bytes for magic numbers
    if len(audio_data) < 12:
        return None
    
    # MP3: FF FB or FF F3 or FF F2 or ID3
    if audio_data[:3] == b'ID3' or audio_data[:2] in (b'\xff\xfb', b'\xff\xf3', b'\xff\xf2'):
        return '.mp3'
    
    # WAV: RIFF....WAVE
    if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
        return '.wav'
    
    # OGG: OggS
    if audio_data[:4] == b'OggS':
        return '.ogg'
    
    # FLAC: fLaC
    if audio_data[:4] == b'fLaC':
        return '.flac'
    
    # M4A/MP4: ftyp
    if audio_data[4:8] == b'ftyp':
        # Check for M4A specific markers
        if b'M4A' in audio_data[8:20] or b'mp42' in audio_data[8:20]:
            return '.m4a'
        return '.mp4'
    
    # WebM: 0x1A 0x45 0xDF 0xA3
    if audio_data[:4] == b'\x1a\x45\xdf\xa3':
        return '.webm'
    
    return None


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            model_size_or_path=WHISPER_MODEL,
            device=WHISPER_DEVICE,
            device_index=WHISPER_DEVICE_INDEX,
            compute_type=WHISPER_COMPUTE_TYPE,
            cpu_threads=WHISPER_CPU_THREADS,
            num_workers=WHISPER_NUM_WORKERS,
            download_root=MODELS_DIR.joinpath(WHISPER_MODEL).as_posix(),
            local_files_only=True,
        )
    return _whisper_model


def whisper_transcribe(audio_data: BinaryIO, language: str = None) -> (str, str):
    transcribe_kwargs = {
        "audio": audio_data,
        "beam_size": WHISPER_BEAM_SIZE,
        "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": dict(min_silence_duration_ms=500),
    }

    # Add language parameter if provided
    if language:
        transcribe_kwargs["language"] = language

    segments, _ = get_whisper_model().transcribe(**transcribe_kwargs)

    subtitles, sub_text = convert_to_subtitles(segments)

    items = []
    for subtitle in subtitles:
        text = subtitle.get("msg").strip()
        if text:
            items.append(
                text_to_vtt(text, subtitle.get("start_time"), subtitle.get("end_time"))
            )

    vtt_content = "WEBVTT\n\n" + "\n".join(items)
    return vtt_content, sub_text


def convert_to_subtitles(segments) -> (list, str):
    subtitles = []
    subtitle_content = ""
    for segment in segments:
        subtitle_content += segment.text
        words_idx = 0
        words_len = len(segment.words)

        seg_start = 0
        seg_end = 0
        seg_text = ""

        if segment.words:
            is_segmented = False
            for word in segment.words:
                if not is_segmented:
                    seg_start = word.start
                    is_segmented = True

                seg_end = word.end
                # If it contains punctuation, then break the sentence.
                seg_text += word.word

                if end_with_stop_char(word.word):
                    # remove last char
                    seg_text = seg_text[:-1]
                    if not seg_text:
                        continue

                    # Ensure start_time is less than end_time
                    if seg_start < seg_end and seg_text.strip():
                        subtitles.append(
                            {
                                "msg": seg_text,
                                "start_time": seg_start,
                                "end_time": seg_end,
                            }
                        )

                    is_segmented = False
                    seg_text = ""

                if words_idx == 0 and segment.start < word.start:
                    seg_start = word.start
                if words_idx == (words_len - 1) and segment.end > word.end:
                    seg_end = word.end
                words_idx += 1

        if not seg_text:
            continue

        # Ensure start_time is less than end_time
        if seg_start < seg_end and seg_text.strip():
            subtitles.append(
                {"msg": seg_text, "start_time": seg_start, "end_time": seg_end}
            )

    return subtitles, subtitle_content


def time_convert_seconds_to_hmsm(seconds) -> str:
    hours = int(seconds // 3600)
    seconds = seconds % 3600
    minutes = int(seconds // 60)
    milliseconds = int(seconds * 1000) % 1000
    seconds = int(seconds % 60)
    return "{:02d}:{:02d}:{:02d}.{:03d}".format(hours, minutes, seconds, milliseconds)


def capitalize_first_letter(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


def text_to_vtt(msg: str, start_time: float, end_time: float) -> str:
    start_time_str = time_convert_seconds_to_hmsm(start_time)
    end_time_str = time_convert_seconds_to_hmsm(end_time)

    return (
        f"{start_time_str} --> {end_time_str}\n{capitalize_first_letter(msg.strip())}\n"
    )


def end_with_stop_char(text: str) -> bool:
    if not text:
        return False

    for c in STOP_CHARS:
        if text.endswith(c):
            return True
    return False
