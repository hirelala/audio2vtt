from faster_whisper import WhisperModel
from typing import BinaryIO, Union, Tuple, List, Dict, Any
import numpy as np

STOP_CHARS = set(
    ".!?,:;…‥"
    "。！？，、；："
    "।"
    "܀።፧"
    "؟؛"
    "၊။"
    "⸮⁇⁈⁉"
)


class AudioToVTT:
    def __init__(self, **kwargs):
        self.model = WhisperModel(**kwargs)
    
    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        kwargs.setdefault("word_timestamps", True)
        kwargs.setdefault("vad_filter", True)
        kwargs.setdefault("vad_parameters", {"min_silence_duration_ms": 500})
        
        segments, _ = self.model.transcribe(audio=audio, **kwargs)
        
        subtitles, word_count = self.segments_to_subtitle(segments)
        
        items = []
        for subtitle in subtitles:
            text = subtitle.get("msg")
            if text:
                items.append(
                    self.segment_text_to_vtt(
                        text=text,
                        start_time=subtitle.get("start_time"),
                        end_time=subtitle.get("end_time")
                    )
                )
        
        vtt_content = "WEBVTT\n\n" + "\n".join(items)
        return {
            "vtt": vtt_content,
            "word_count": word_count
        }

    def segments_to_subtitle(self, segments) -> Tuple[List[Dict], int]:
        subtitles = []
        word_count = 0
        
        for segment in segments:
            word_count += len(segment.words)
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
                    seg_text += word.word

                    if self.end_with_stop_char(word.word):
                        seg_text = seg_text[:-1]
                        if not seg_text:
                            continue

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

            if seg_start < seg_end and seg_text.strip():
                subtitles.append(
                    {"msg": seg_text, "start_time": seg_start, "end_time": seg_end}
                )

        return subtitles, word_count


    def seconds_to_hmsm(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        seconds = seconds % 3600
        minutes = int(seconds // 60)
        milliseconds = int(seconds * 1000) % 1000
        seconds = int(seconds % 60)
        return "{:02d}:{:02d}:{:02d}.{:03d}".format(hours, minutes, seconds, milliseconds)

    def segment_text_to_vtt(self, text: str, start_time: float, end_time: float) -> str:
        start_time_str = self.seconds_to_hmsm(start_time)
        end_time_str = self.seconds_to_hmsm(end_time)
        text = text.strip()
        text = text[0].upper() + text[1:] if text else text

        return (
            f"{start_time_str} --> {end_time_str}\n{text}\n"
        )


    def end_with_stop_char(self, text: str) -> bool:
        if not text:
            return False

        for c in STOP_CHARS:
            if text.endswith(c):
                return True
        return False
