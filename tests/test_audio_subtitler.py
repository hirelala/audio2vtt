import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from src.audio_subtitler import AudioSubtitler, STOP_CHARS


class MockWord:
    """Mock object for Whisper word timestamps"""
    def __init__(self, word: str, start: float, end: float):
        self.word = word
        self.start = start
        self.end = end


class MockSegment:
    """Mock object for Whisper segment"""
    def __init__(self, text: str, start: float, end: float, words: list):
        self.text = text
        self.start = start
        self.end = end
        self.words = words


class TestAudioSubtitler:
    """Test suite for AudioSubtitler class"""

    @patch('src.audio_subtitler.WhisperModel')
    def test_init_default_params(self, mock_whisper_model):
        """Test initialization with default parameters"""
        converter = AudioSubtitler(model_size_or_path="base")
        mock_whisper_model.assert_called_once_with(model_size_or_path="base")
        assert converter.model is not None

    @patch('src.audio_subtitler.WhisperModel')
    def test_init_custom_params(self, mock_whisper_model):
        """Test initialization with custom parameters"""
        converter = AudioSubtitler(
            model_size_or_path="base",
            device="cpu",
            compute_type="int8"
        )
        mock_whisper_model.assert_called_once_with(
            model_size_or_path="base",
            device="cpu",
            compute_type="int8"
        )

    @patch('src.audio_subtitler.WhisperModel')
    def test_seconds_to_vtt_time_basic(self, mock_whisper_model):
        """Test time conversion with basic values"""
        converter = AudioSubtitler(model_size_or_path="base")
        
        assert converter.seconds_to_vtt_time(0) == "00:00:00.000"
        assert converter.seconds_to_vtt_time(1.5) == "00:00:01.500"
        assert converter.seconds_to_vtt_time(60) == "00:01:00.000"
        assert converter.seconds_to_vtt_time(3661.123) == "01:01:01.123"

    @patch('src.audio_subtitler.WhisperModel')
    def test_seconds_to_vtt_time_edge_cases(self, mock_whisper_model):
        """Test time conversion with edge cases"""
        converter = AudioSubtitler(model_size_or_path="base")
        
        assert converter.seconds_to_vtt_time(0.001) == "00:00:00.001"
        assert converter.seconds_to_vtt_time(0.999) == "00:00:00.999"
        assert converter.seconds_to_vtt_time(7200) == "02:00:00.000"
        assert converter.seconds_to_vtt_time(3599.999) == "00:59:59.999"

    @patch('src.audio_subtitler.WhisperModel')
    def test_format_vtt_segment(self, mock_whisper_model):
        """Test VTT segment formatting"""
        converter = AudioSubtitler(model_size_or_path="base")
        
        result = converter._format_vtt_segment("hello world", 0.0, 2.5)
        assert result == "00:00:00.000 --> 00:00:02.500\nHello world\n"
        
        result = converter._format_vtt_segment("  test  ", 1.0, 3.0)
        assert result == "00:00:01.000 --> 00:00:03.000\nTest\n"

    @patch('src.audio_subtitler.WhisperModel')
    def test_format_vtt_segment_capitalization(self, mock_whisper_model):
        """Test that first letter is capitalized"""
        converter = AudioSubtitler(model_size_or_path="base")
        
        result = converter._format_vtt_segment("hello", 0.0, 1.0)
        assert "Hello" in result
        
        result = converter._format_vtt_segment("", 0.0, 1.0)
        assert result == "00:00:00.000 --> 00:00:01.000\n\n"

    @patch('src.audio_subtitler.WhisperModel')
    def test_end_with_stop_char_english(self, mock_whisper_model):
        """Test stop character detection for English punctuation"""
        converter = AudioSubtitler(model_size_or_path="base")
        
        assert converter.end_with_stop_char("hello.") is True
        assert converter.end_with_stop_char("hello!") is True
        assert converter.end_with_stop_char("hello?") is True
        assert converter.end_with_stop_char("hello,") is True
        assert converter.end_with_stop_char("hello:") is True
        assert converter.end_with_stop_char("hello;") is True

    @patch('src.audio_subtitler.WhisperModel')
    def test_end_with_stop_char_multilingual(self, mock_whisper_model):
        """Test stop character detection for various languages"""
        converter = AudioSubtitler(model_size_or_path="base")
        
        # Chinese
        assert converter.end_with_stop_char("你好。") is True
        assert converter.end_with_stop_char("你好！") is True
        assert converter.end_with_stop_char("你好？") is True
        
        # Arabic
        assert converter.end_with_stop_char("مرحبا؟") is True
        assert converter.end_with_stop_char("مرحبا؛") is True

    @patch('src.audio_subtitler.WhisperModel')
    def test_end_with_stop_char_negative(self, mock_whisper_model):
        """Test stop character detection returns False for non-stop chars"""
        converter = AudioSubtitler(model_size_or_path="base")
        
        assert converter.end_with_stop_char("hello") is False
        assert converter.end_with_stop_char("hello world") is False
        assert converter.end_with_stop_char("") is False

    @patch('src.audio_subtitler.WhisperModel')
    def test_end_with_stop_char_empty(self, mock_whisper_model):
        """Test stop character detection with empty string"""
        converter = AudioSubtitler(model_size_or_path="base")
        assert converter.end_with_stop_char("") is False

    @patch('src.audio_subtitler.WhisperModel')
    def test_segments_to_subtitle_basic(self, mock_whisper_model):
        """Test conversion of segments to subtitles"""
        converter = AudioSubtitler(model_size_or_path="base")
        
        # Create mock segments with words
        words = [
            MockWord("Hello", 0.0, 0.5),
            MockWord(" world.", 0.5, 1.0)
        ]
        segment = MockSegment("Hello world.", 0.0, 1.0, words)
        
        subtitles, word_count = converter.segments_to_subtitle([segment])
        
        assert word_count == 2
        assert len(subtitles) == 1
        assert subtitles[0]["msg"] == "Hello world"
        assert subtitles[0]["start_time"] == 0.0
        assert subtitles[0]["end_time"] == 1.0

    @patch('src.audio_subtitler.WhisperModel')
    def test_segments_to_subtitle_multiple_sentences(self, mock_whisper_model):
        """Test conversion with multiple sentences"""
        converter = AudioSubtitler(model_size_or_path="base")
        
        words = [
            MockWord("Hello.", 0.0, 0.5),
            MockWord(" How", 0.6, 0.8),
            MockWord(" are", 0.8, 1.0),
            MockWord(" you?", 1.0, 1.5)
        ]
        segment = MockSegment("Hello. How are you?", 0.0, 1.5, words)
        
        subtitles, word_count = converter.segments_to_subtitle([segment])
        
        assert word_count == 4
        assert len(subtitles) == 2
        assert subtitles[0]["msg"] == "Hello"
        assert subtitles[1]["msg"] == " How are you"

    @patch('src.audio_subtitler.WhisperModel')
    def test_segments_to_subtitle_empty_segments(self, mock_whisper_model):
        """Test with empty segments"""
        converter = AudioSubtitler(model_size_or_path="base")
        
        segment = MockSegment("", 0.0, 0.0, [])
        subtitles, word_count = converter.segments_to_subtitle([segment])
        
        assert word_count == 0
        assert len(subtitles) == 0

    @patch('src.audio_subtitler.WhisperModel')
    def test_segments_to_subtitle_strips_whitespace(self, mock_whisper_model):
        """Test that whitespace is properly stripped"""
        converter = AudioSubtitler(model_size_or_path="base")
        
        words = [
            MockWord("  Hello  ", 0.0, 0.5),
            MockWord(" world.  ", 0.5, 1.0)
        ]
        segment = MockSegment("  Hello   world.  ", 0.0, 1.0, words)
        
        subtitles, word_count = converter.segments_to_subtitle([segment])
        
        assert len(subtitles) == 1
        # The subtitle should have content stripped when used

    @patch('src.audio_subtitler.WhisperModel')
    def test_transcribe_default_params(self, mock_whisper_model):
        """Test transcribe with default parameters"""
        # Setup mock
        mock_model_instance = MagicMock()
        mock_whisper_model.return_value = mock_model_instance
        
        words = [
            MockWord("Hello", 0.0, 0.5),
            MockWord(" world.", 0.5, 1.0)
        ]
        segment = MockSegment("Hello world.", 0.0, 1.0, words)
        
        mock_model_instance.transcribe.return_value = ([segment], None)
        
        converter = AudioSubtitler(model_size_or_path="base")
        result = converter.transcribe("test.mp3")
        
        # Verify default parameters were set
        call_kwargs = mock_model_instance.transcribe.call_args[1]
        assert call_kwargs["word_timestamps"] is True
        assert call_kwargs["vad_filter"] is True
        assert call_kwargs["vad_parameters"] == {"min_silence_duration_ms": 500}
        
        # Verify result structure
        assert "vtt" in result
        assert "word_count" in result
        assert result["word_count"] == 2
        assert result["vtt"].startswith("WEBVTT\n\n")

    @patch('src.audio_subtitler.WhisperModel')
    def test_transcribe_custom_params(self, mock_whisper_model):
        """Test transcribe with custom parameters"""
        mock_model_instance = MagicMock()
        mock_whisper_model.return_value = mock_model_instance
        
        words = [MockWord("Test", 0.0, 0.5)]
        segment = MockSegment("Test", 0.0, 0.5, words)
        mock_model_instance.transcribe.return_value = ([segment], None)
        
        converter = AudioSubtitler(model_size_or_path="base")
        result = converter.transcribe(
            "test.mp3",
            language="en",
            beam_size=10,
            temperature=0.5
        )
        
        # Verify custom parameters were passed
        call_kwargs = mock_model_instance.transcribe.call_args[1]
        assert call_kwargs["language"] == "en"
        assert call_kwargs["beam_size"] == 10
        assert call_kwargs["temperature"] == 0.5

    @patch('src.audio_subtitler.WhisperModel')
    def test_transcribe_override_defaults(self, mock_whisper_model):
        """Test that defaults can be overridden"""
        mock_model_instance = MagicMock()
        mock_whisper_model.return_value = mock_model_instance
        
        words = [MockWord("Test", 0.0, 0.5)]
        segment = MockSegment("Test", 0.0, 0.5, words)
        mock_model_instance.transcribe.return_value = ([segment], None)
        
        converter = AudioSubtitler(model_size_or_path="base")
        result = converter.transcribe(
            "test.mp3",
            word_timestamps=False,
            vad_filter=False,
            vad_parameters={"min_silence_duration_ms": 1000}
        )
        
        call_kwargs = mock_model_instance.transcribe.call_args[1]
        assert call_kwargs["word_timestamps"] is False
        assert call_kwargs["vad_filter"] is False
        assert call_kwargs["vad_parameters"] == {"min_silence_duration_ms": 1000}

    @patch('src.audio_subtitler.WhisperModel')
    def test_transcribe_vtt_format(self, mock_whisper_model):
        """Test that VTT output is correctly formatted"""
        mock_model_instance = MagicMock()
        mock_whisper_model.return_value = mock_model_instance
        
        words = [
            MockWord("Hello", 0.0, 0.5),
            MockWord(" world.", 0.5, 1.0)
        ]
        segment = MockSegment("Hello world.", 0.0, 1.0, words)
        mock_model_instance.transcribe.return_value = ([segment], None)
        
        converter = AudioSubtitler(model_size_or_path="base")
        result = converter.transcribe("test.mp3")
        
        vtt = result["vtt"]
        
        # Check VTT header
        assert vtt.startswith("WEBVTT\n\n")
        
        # Check timestamp format
        assert "00:00:00.000 --> 00:00:01.000" in vtt
        
        # Check text is capitalized
        assert "Hello world" in vtt

    @patch('src.audio_subtitler.WhisperModel')
    def test_transcribe_with_numpy_array(self, mock_whisper_model):
        """Test transcribe with numpy array input"""
        mock_model_instance = MagicMock()
        mock_whisper_model.return_value = mock_model_instance
        
        words = [MockWord("Test", 0.0, 0.5)]
        segment = MockSegment("Test", 0.0, 0.5, words)
        mock_model_instance.transcribe.return_value = ([segment], None)
        
        converter = AudioSubtitler(model_size_or_path="base")
        audio_array = np.zeros(16000, dtype=np.float32)
        result = converter.transcribe(audio_array)
        
        assert "vtt" in result
        assert "word_count" in result

    @patch('src.audio_subtitler.WhisperModel')
    def test_transcribe_multiple_segments(self, mock_whisper_model):
        """Test transcribe with multiple segments"""
        mock_model_instance = MagicMock()
        mock_whisper_model.return_value = mock_model_instance
        
        segment1_words = [MockWord("First", 0.0, 0.5), MockWord(" sentence.", 0.5, 1.0)]
        segment2_words = [MockWord("Second", 1.5, 2.0), MockWord(" sentence.", 2.0, 2.5)]
        
        segment1 = MockSegment("First sentence.", 0.0, 1.0, segment1_words)
        segment2 = MockSegment("Second sentence.", 1.5, 2.5, segment2_words)
        
        mock_model_instance.transcribe.return_value = ([segment1, segment2], None)
        
        converter = AudioSubtitler(model_size_or_path="base")
        result = converter.transcribe("test.mp3")
        
        assert result["word_count"] == 4
        vtt = result["vtt"]
        assert "First sentence" in vtt
        assert "Second sentence" in vtt

    def test_stop_chars_defined(self):
        """Test that STOP_CHARS constant is properly defined"""
        assert isinstance(STOP_CHARS, set)
        assert len(STOP_CHARS) > 0
        assert "." in STOP_CHARS
        assert "!" in STOP_CHARS
        assert "?" in STOP_CHARS

    @patch('src.audio_subtitler.WhisperModel')
    def test_seconds_to_srt_time(self, mock_whisper_model):
        """Test SRT time format conversion"""
        converter = AudioSubtitler(model_size_or_path="base")
        
        # Test SRT uses comma instead of period
        assert converter.seconds_to_srt_time(0) == "00:00:00,000"
        assert converter.seconds_to_srt_time(1.5) == "00:00:01,500"
        assert converter.seconds_to_srt_time(60) == "00:01:00,000"
        assert converter.seconds_to_srt_time(3661.123) == "01:01:01,123"

    @patch('src.audio_subtitler.WhisperModel')
    def test_format_srt_segment(self, mock_whisper_model):
        """Test SRT segment formatting"""
        converter = AudioSubtitler(model_size_or_path="base")
        
        result = converter._format_srt_segment(1, "hello world", 0.0, 2.5)
        assert "1\n" in result
        assert "00:00:00,000 --> 00:00:02,500" in result
        assert "Hello world" in result
        
        result = converter._format_srt_segment(2, "  test  ", 1.0, 3.0)
        assert "2\n" in result
        assert "00:00:01,000 --> 00:00:03,000" in result
        assert "Test" in result

    @patch('src.audio_subtitler.WhisperModel')
    def test_transcribe_srt_format(self, mock_whisper_model):
        """Test transcription with SRT format"""
        mock_model_instance = MagicMock()
        mock_whisper_model.return_value = mock_model_instance
        
        words = [
            MockWord("Hello", 0.0, 0.5),
            MockWord(" world.", 0.5, 1.0)
        ]
        segment = MockSegment("Hello world.", 0.0, 1.0, words)
        mock_model_instance.transcribe.return_value = ([segment], None)
        
        converter = AudioSubtitler(model_size_or_path="base")
        result = converter.transcribe("test.mp3", format="srt")
        
        # Verify result structure
        assert "srt" in result
        assert "word_count" in result
        assert result["word_count"] == 2
        
        srt = result["srt"]
        
        # Check SRT format (uses comma, has index)
        assert "1\n" in srt
        assert "00:00:00,000 --> 00:00:01,000" in srt
        assert "Hello world" in srt

    @patch('src.audio_subtitler.WhisperModel')
    def test_transcribe_vtt_vs_srt_format(self, mock_whisper_model):
        """Test difference between VTT and SRT formats"""
        mock_model_instance = MagicMock()
        mock_whisper_model.return_value = mock_model_instance
        
        words = [MockWord("Test", 0.0, 0.5)]
        segment = MockSegment("Test", 0.0, 0.5, words)
        mock_model_instance.transcribe.return_value = ([segment], None)
        
        converter = AudioSubtitler(model_size_or_path="base")
        
        # Get VTT format
        vtt_result = converter.transcribe("test.mp3", format="vtt")
        assert "vtt" in vtt_result
        assert "WEBVTT" in vtt_result["vtt"]
        assert "00:00:00.000" in vtt_result["vtt"]  # Period separator
        
        # Get SRT format
        mock_model_instance.transcribe.return_value = ([segment], None)
        srt_result = converter.transcribe("test.mp3", format="srt")
        assert "srt" in srt_result
        assert "1\n" in srt_result["srt"]  # Index
        assert "00:00:00,000" in srt_result["srt"]  # Comma separator
        assert "WEBVTT" not in srt_result["srt"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

