from pathlib import Path
from faster_whisper import WhisperModel
from src.config import (
    MODELS_DIR,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_CPU_THREADS,
    WHISPER_NUM_WORKERS,
    WHISPER_LOCAL_FILES_ONLY,
    WHISPER_MODEL,
    WHISPER_BEAM_SIZE,
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


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        download_root = MODELS_DIR.joinpath(WHISPER_MODEL)
        download_root.mkdir(parents=True, exist_ok=True)
        _whisper_model = WhisperModel(
            model_size_or_path=WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            cpu_threads=WHISPER_CPU_THREADS,
            num_workers=WHISPER_NUM_WORKERS,
            download_root=download_root.as_posix(),
            local_files_only=WHISPER_LOCAL_FILES_ONLY,
        )
    return _whisper_model


def whisper_transcribe(audio_path: Path, language: str = None):
    """Transcribe audio file to VTT format"""
    transcribe_kwargs = {
        "audio": audio_path.as_posix(),
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
