#!/usr/bin/env bash
# One-time setup on macOS / Linux. Run from the repo folder:  ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

command -v git >/dev/null    || { echo "git is required"; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg is required (brew install ffmpeg / sudo apt install ffmpeg)"; exit 1; }

# ClipsAI and WhisperX are most reliable on Python 3.10-3.11
PY=$(command -v python3.11 || command -v python3.10 || true)
[ -n "$PY" ] || { echo "Python 3.11 is required (brew install python@3.11 / sudo apt install python3.11 python3.11-venv)"; exit 1; }

git submodule update --init --depth 1
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install python-magic   # needs libmagic: brew install libmagic / sudo apt install libmagic1

[ -f .env ] || cp .env.example .env
python -c "import yt_dlp, clipsai, whisperx, requests; print('imports ok')"
echo
echo "Setup done. Next:  . .venv/bin/activate && python pipeline.py --help"
