import pytest
import sys
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import io

from src.cli import main


class TestCLI:
    """Test suite for CLI functionality"""

    @patch('src.cli.AudioToVTT')
    @patch('sys.argv', ['audiotovtt', '--help'])
    def test_help_message(self, mock_audio_to_vtt):
        """Test that help message is displayed"""
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    @patch('src.cli.AudioToVTT')
    @patch('sys.argv', ['audiotovtt', '--version'])
    def test_version(self, mock_audio_to_vtt):
        """Test version flag"""
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    @patch('src.cli.AudioToVTT')
    @patch('sys.argv', ['audiotovtt', 'nonexistent.mp3'])
    def test_nonexistent_file(self, mock_audio_to_vtt, capsys):
        """Test error handling for nonexistent file"""
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower()

    @patch('src.cli.AudioToVTT')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    @patch('sys.argv', ['audiotovtt', 'test.mp3'])
    def test_basic_transcription_stdout(self, mock_is_file, mock_exists, mock_audio_to_vtt_class, capsys):
        """Test basic transcription to stdout"""
        # Setup mocks
        mock_exists.return_value = True
        mock_is_file.return_value = True
        
        mock_converter = MagicMock()
        mock_audio_to_vtt_class.return_value = mock_converter
        
        mock_converter.transcribe.return_value = {
            "vtt": "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nTest\n",
            "word_count": 1
        }
        
        main()
        
        # Verify converter was initialized with correct defaults
        mock_audio_to_vtt_class.assert_called_once_with(
            model_size_or_path="base",
            device="cpu",
            device_index=0,
            compute_type="int8",
            cpu_threads=4,
            download_root=None,
            local_files_only=False,
        )
        
        # Verify transcribe was called
        assert mock_converter.transcribe.called
        
        # Check stdout contains VTT content
        captured = capsys.readouterr()
        assert "WEBVTT" in captured.out

    @patch('src.cli.AudioToVTT')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    @patch('pathlib.Path.write_text')
    @patch('sys.argv', ['audiotovtt', 'test.mp3', '-o', 'output.vtt'])
    def test_transcription_to_file(self, mock_write_text, mock_is_file, mock_exists, mock_audio_to_vtt_class, capsys):
        """Test transcription to file"""
        # Setup mocks
        mock_exists.return_value = True
        mock_is_file.return_value = True
        
        mock_converter = MagicMock()
        mock_audio_to_vtt_class.return_value = mock_converter
        
        mock_converter.transcribe.return_value = {
            "vtt": "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nTest\n",
            "word_count": 1
        }
        
        main()
        
        # Verify file was written
        mock_write_text.assert_called_once_with(
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nTest\n",
            encoding="utf-8"
        )

    @patch('src.cli.AudioToVTT')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    @patch('sys.argv', ['audiotovtt', 'test.mp3', '--model', 'large-v2', '--language', 'en'])
    def test_custom_parameters(self, mock_is_file, mock_exists, mock_audio_to_vtt_class, capsys):
        """Test CLI with custom parameters"""
        # Setup mocks
        mock_exists.return_value = True
        mock_is_file.return_value = True
        
        mock_converter = MagicMock()
        mock_audio_to_vtt_class.return_value = mock_converter
        
        mock_converter.transcribe.return_value = {
            "vtt": "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nTest\n",
            "word_count": 1
        }
        
        main()
        
        # Verify converter was initialized with custom model
        mock_audio_to_vtt_class.assert_called_once_with(
            model_size_or_path="large-v2",
            device="cpu",
            device_index=0,
            compute_type="int8",
            cpu_threads=4,
            download_root=None,
            local_files_only=False,
        )
        
        # Verify language was passed to transcribe
        call_kwargs = mock_converter.transcribe.call_args[1]
        assert call_kwargs["language"] == "en"

    @patch('src.cli.AudioToVTT')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    @patch('sys.argv', ['audiotovtt', 'test.mp3', '--device', 'cuda', '--compute-type', 'float16'])
    def test_gpu_parameters(self, mock_is_file, mock_exists, mock_audio_to_vtt_class, capsys):
        """Test CLI with GPU parameters"""
        # Setup mocks
        mock_exists.return_value = True
        mock_is_file.return_value = True
        
        mock_converter = MagicMock()
        mock_audio_to_vtt_class.return_value = mock_converter
        
        mock_converter.transcribe.return_value = {
            "vtt": "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nTest\n",
            "word_count": 1
        }
        
        main()
        
        # Verify GPU settings
        mock_audio_to_vtt_class.assert_called_once_with(
            model_size_or_path="base",
            device="cuda",
            device_index=0,
            compute_type="float16",
            cpu_threads=4,
            download_root=None,
            local_files_only=False,
        )

    @patch('src.cli.AudioToVTT')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    @patch('sys.argv', ['audiotovtt', 'test.mp3', '--no-vad', '--beam-size', '10'])
    def test_transcription_parameters(self, mock_is_file, mock_exists, mock_audio_to_vtt_class, capsys):
        """Test CLI with transcription parameters"""
        # Setup mocks
        mock_exists.return_value = True
        mock_is_file.return_value = True
        
        mock_converter = MagicMock()
        mock_audio_to_vtt_class.return_value = mock_converter
        
        mock_converter.transcribe.return_value = {
            "vtt": "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nTest\n",
            "word_count": 1
        }
        
        main()
        
        # Verify transcription parameters
        call_kwargs = mock_converter.transcribe.call_args[1]
        assert call_kwargs["vad_filter"] is False
        assert call_kwargs["beam_size"] == 10

    @patch('src.cli.AudioToVTT')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    @patch('sys.argv', ['audiotovtt', 'test.mp3', '--quiet'])
    def test_quiet_mode(self, mock_is_file, mock_exists, mock_audio_to_vtt_class, capsys):
        """Test quiet mode suppresses stderr messages"""
        # Setup mocks
        mock_exists.return_value = True
        mock_is_file.return_value = True
        
        mock_converter = MagicMock()
        mock_audio_to_vtt_class.return_value = mock_converter
        
        mock_converter.transcribe.return_value = {
            "vtt": "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nTest\n",
            "word_count": 1
        }
        
        main()
        
        # In quiet mode, stderr should be minimal (no info messages)
        captured = capsys.readouterr()
        assert "Loading model" not in captured.err
        assert "Transcribing" not in captured.err

    @patch('src.cli.AudioToVTT')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    @patch('sys.argv', ['audiotovtt', 'test.mp3'])
    def test_error_handling(self, mock_is_file, mock_exists, mock_audio_to_vtt_class, capsys):
        """Test error handling during transcription"""
        # Setup mocks
        mock_exists.return_value = True
        mock_is_file.return_value = True
        
        mock_converter = MagicMock()
        mock_audio_to_vtt_class.return_value = mock_converter
        
        # Make transcribe raise an exception
        mock_converter.transcribe.side_effect = Exception("Test error")
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

