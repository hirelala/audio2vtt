from .audio2vtt import Audio2VTT
from .audio_utils import detect_audio_format
from .text_utils import (
    convert_to_subtitles,
    text_to_vtt,
    time_convert_seconds_to_hmsm,
    capitalize_first_letter,
    end_with_stop_char,
)

__version__ = "1.0.0"
__author__ = "Gary Lab"
__all__ = [
    "Audio2VTT",
    "detect_audio_format",
    "convert_to_subtitles",
    "text_to_vtt",
    "time_convert_seconds_to_hmsm",
    "capitalize_first_letter",
    "end_with_stop_char",
]
