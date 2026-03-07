from faster_whisper import WhisperModel
import os
import sys
import threading
from datetime import datetime


def load_my_model(model_size="small", cores=2):
    """
    Initializes and returns the Faster-Whisper model.

    Args:
        model_size: one of "tiny", "small", "medium", "large"
        cores:      number of CPU threads to use
    """
    print(f"Loading model '{model_size}' with {cores} cores...")
    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=cores,
        num_workers=1,
    )


def transcribe_file(
    file_path,
    model_instance=None,
    language="ur",
    cores=2,
    segment_callback=None,
    stop_event: threading.Event = None,
):
    """
    Transcribes a single audio/video file and writes output to a .txt file
    in the same directory as the source file.

    Args:
        file_path:         path to the audio/video file
        model_instance:    a loaded WhisperModel (will auto-load if None)
        language:          Whisper language code e.g. "ur", "en", or None for auto-detect
        cores:             CPU threads (only used when auto-loading the model here)
        segment_callback:  optional callable(text: str) called for each segment
                           — used by the GUI to stream text live to the screen
        stop_event:        optional threading.Event; transcription stops cleanly
                           between segments when this is set

    Returns:
        output filename (not full path) on success,
        None if the file was not found or processing was stopped before finishing
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    if model_instance is None:
        print("No model provided — loading default...")
        model_instance = load_my_model(cores=cores)

    print(f"Processing: {os.path.basename(file_path)}...")

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name   = os.path.splitext(os.path.basename(file_path))[0]
    output_name = f"{base_name}_{timestamp}.txt"
    output_path = os.path.join(os.path.dirname(file_path), output_name)

    # language=None tells Whisper to auto-detect
    segments, info = model_instance.transcribe(
        file_path,
        language=language,
        beam_size=5,
    )

    print(f"Detected language: {info.language} (confidence {info.language_probability:.0%})")

    completed = False
    with open(output_path, "a", encoding="utf-8", buffering=1) as f:
        for segment in segments:
            # Check stop flag between every segment
            if stop_event is not None and stop_event.is_set():
                print("Transcription stopped by user.")
                break

            f.write(segment.text + " ")
            f.flush()
            print(f"  [{segment.start:.1f}s] {segment.text}")

            # Push text live to the GUI if a callback was provided
            if segment_callback is not None:
                segment_callback(segment.text)
        else:
            # for/else: only runs if the loop wasn't broken
            completed = True

    if completed:
        print(f"Saved: {output_path}")
        return output_name
    else:
        # Return partial file name so the GUI can still show/open what was saved
        print(f"Partial save: {output_path}")
        return output_name


# ── CLI usage ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python whisper_engine.py <file> [model_size] [language] [cores]")
        print("  model_size : tiny | small | medium | large  (default: small)")
        print("  language   : ur | en | ar | ...             (default: ur)")
        print("  cores      : integer                        (default: 2)")
        sys.exit(1)

    file_arg  = sys.argv[1]
    size_arg  = sys.argv[2] if len(sys.argv) > 2 else "small"
    lang_arg  = sys.argv[3] if len(sys.argv) > 3 else "ur"
    cores_arg = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    if lang_arg.lower() == "none":
        lang_arg = None

    model = load_my_model(model_size=size_arg, cores=cores_arg)
    transcribe_file(
        file_arg, model,
        language=lang_arg,
        cores=cores_arg,
        segment_callback=lambda t: None,   # no GUI in CLI mode
    )
