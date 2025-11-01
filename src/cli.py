#!/usr/bin/env python3
"""Command-line interface for audio-subtitler."""

import sys
import argparse
from pathlib import Path

from src.audio_subtitler import AudioSubtitler

__version__ = "0.1.0"


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Convert audio files to subtitle format (VTT, SRT) using Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - VTT output to stdout
  audiosubtitler input.mp3 > output.vtt
  
  # Generate SRT format
  audiosubtitler input.mp3 --format srt > output.srt
  
  # Specify output file
  audiosubtitler input.mp3 -o output.vtt
  
  # Use a different model
  audiosubtitler input.mp3 --model large-v2 > output.vtt
  
  # Specify language
  audiosubtitler input.mp3 --language en > output.vtt
  
  # Use GPU
  audiosubtitler input.mp3 --device cuda > output.vtt
        """,
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Input audio file path",
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output subtitle file path (if not specified, outputs to stdout)",
    )
    
    parser.add_argument(
        "-f", "--format",
        type=str,
        default="vtt",
        choices=["vtt", "srt"],
        help="Subtitle format (default: vtt)",
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="base",
        help="Whisper model size (tiny, base, small, medium, large, large-v2, large-v3) or path to model (default: base)",
    )
    
    parser.add_argument(
        "-l", "--language",
        type=str,
        default=None,
        help="Language code (e.g., en, es, fr, de, zh, ja) (default: auto-detect)",
    )
    
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device to use for inference (default: cpu)",
    )
    
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Device index for CUDA (default: 0)",
    )
    
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        help="Compute type (int8, int8_float16, int16, float16, float32) (default: int8)",
    )
    
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (default: 5)",
    )
    
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable voice activity detection",
    )
    
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="Number of CPU threads to use (default: 4)",
    )
    
    parser.add_argument(
        "--download-root",
        type=str,
        default=None,
        help="Directory to download models to (default: ~/.cache/huggingface)",
    )
    
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only local model files, don't download",
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress informational messages",
    )
    
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    if not input_path.is_file():
        print(f"Error: Input path is not a file: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Print info if not quiet
    if not args.quiet:
        print(f"Loading model: {args.model}", file=sys.stderr)
    
    try:
        # Initialize the converter
        converter = AudioSubtitler(
            model_size_or_path=args.model,
            device=args.device,
            device_index=args.device_index,
            compute_type=args.compute_type,
            cpu_threads=args.cpu_threads,
            download_root=args.download_root,
            local_files_only=args.local_files_only,
        )
        
        if not args.quiet:
            print(f"Transcribing: {args.input}", file=sys.stderr)
        
        # Prepare transcription kwargs
        transcribe_kwargs = {
            "format": args.format,
            "beam_size": args.beam_size,
            "vad_filter": not args.no_vad,
        }
        
        if args.language:
            transcribe_kwargs["language"] = args.language
        
        # Transcribe
        result = converter.transcribe(args.input, **transcribe_kwargs)
        word_count = result["word_count"]
        
        # Get the content based on format
        if args.format == "vtt":
            content = result["vtt"]
        else:  # srt
            content = result["srt"]
        
        if not args.quiet:
            print(f"Transcription complete: {word_count} words", file=sys.stderr)
        
        # Output to file or stdout
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(content, encoding="utf-8")
            if not args.quiet:
                print(f"Output written to: {args.output}", file=sys.stderr)
        else:
            # Output to stdout for piping
            print(content, end="")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

