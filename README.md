# Urdu Transcriber

A desktop application for transcribing audio and video files using [Faster-Whisper](https://github.com/guillaumekynast/faster-whisper), with a PyQt6 GUI. Supports live transcription output, multiple languages, and adjustable performance settings.

---

## Requirements

- Python 3.10 or higher
- `pip` (comes with Python)
- Internet connection (first run only, to download the Whisper model)

---

## Installation

### macOS

**1. Install Python 3**

Check if Python 3 is already installed:
```bash
python3 --version
```

If not installed, download it from [python.org](https://www.python.org/downloads/) or install via Homebrew:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python
```

**2. (Recommended) Create a virtual environment**
```bash
cd /path/to/urdu-transcriber
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install PyQt6
pip install faster-whisper
```

**4. Run the app**
```bash
python3 transcribe.py
```

---

### Ubuntu / Debian

**1. Update packages and install Python**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
```

**2. Install system dependencies**

PyQt6 on Linux requires a few system libraries:
```bash
sudo apt install libgl1 libglib2.0-0 libegl1 -y
```

**3. (Recommended) Create a virtual environment**
```bash
cd /path/to/urdu-transcriber
python3 -m venv venv
source venv/bin/activate
```

**4. Install dependencies**
```bash
pip install PyQt6
pip install faster-whisper
```

**5. Run the app**
```bash
python3 transcribe.py
```

---

## First Run

On first run, Whisper will automatically download the selected model from the internet and cache it locally. This only happens once per model size.

| Model  | Download Size | Speed    | Accuracy |
|--------|--------------|----------|----------|
| tiny   | ~75 MB       | Fastest  | Basic    |
| small  | ~244 MB      | Fast     | Good     |
| medium | ~769 MB      | Moderate | Better   |
| large  | ~1.5 GB      | Slow     | Best     |

Model files are cached at:
- **macOS:** `~/.cache/huggingface/hub/`
- **Ubuntu:** `~/.cache/huggingface/hub/`

---

## Command Line Usage

You can also run the engine directly without the GUI:

```bash
python3 whisper_engine.py <audio_file> [model_size] [language_code] [cores]
```

**Examples:**
```bash
# Transcribe with defaults (small model, Urdu, 2 cores)
python3 whisper_engine.py recording.mp3

# Transcribe with specific settings
python3 whisper_engine.py recording.mp3 medium ur 4

# Auto-detect language
python3 whisper_engine.py recording.mp3 small None 2
```

Output is saved as a `.txt` file in the same folder as the input file.

---

## Supported File Formats

`mp3`, `wav`, `mp4`, `m4a`, `ogg`, `flac`

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'PyQt6'`**
Run `pip install PyQt6` and make sure your virtual environment is activated if you are using one.

**`ModuleNotFoundError: No module named 'faster_whisper'`**
Run `pip install faster-whisper`.

**App is slow on first file**
This is normal — the model is loading into memory. Subsequent files in the same session process faster.

**Ubuntu: app won't launch / display errors**
Make sure the system libraries are installed:
```bash
sudo apt install libgl1 libglib2.0-0 libegl1 -y
```

**macOS: `command not found: python`**
Use `python3` instead of `python` on macOS.
