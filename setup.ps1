# One-time setup on Windows (PowerShell). Run from the repo folder:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Need($cmd, $hint) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Host "Missing: $cmd. Install it with:  $hint" -ForegroundColor Yellow
        Write-Host "Then close and reopen PowerShell and run setup.ps1 again."
        exit 1
    }
}
Need git    "winget install Git.Git"
Need ffmpeg "winget install Gyan.FFmpeg"

# ClipsAI and WhisperX are most reliable on Python 3.11
$py = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Installing Python 3.11 (next to any other Python you have)..."
    winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
}
if (-not (Test-Path $py)) { Write-Host "Python 3.11 not found at $py - reopen PowerShell and run setup.ps1 again."; exit 1 }

# the three projects live in submodules
git submodule update --init --depth 1

if (-not (Test-Path .venv)) { & $py -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Windows needs the bundled libmagic for ClipsAI's file-type checks
& .\.venv\Scripts\python.exe -m pip install python-magic-bin

if (-not (Test-Path .env)) { Copy-Item .env.example .env }

& .\.venv\Scripts\python.exe -c "import yt_dlp, clipsai, whisperx, requests; print('imports ok')"
Write-Host ""
Write-Host "Setup done. Next:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python pipeline.py --help"
