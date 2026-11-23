# clips — YouTube video → viral vertical shorts → scheduled posts

Turns a long YouTube video into short vertical clips (TikTok / YouTube Shorts / Instagram Reels) and schedules them through [Postiz](https://github.com/Fogged1/Postiz-app).

```
YouTube URL ──► yt-dlp ──► transcript (WhisperX) ──► hot-moment finder ──► edit + render ──► Postiz
                                                    (replays, loudness,    (9:16, captions,    (drafts or
                                                     hype words)            jump cuts, hook)    scheduled)
```

Made by [@ksimpaa](https://github.com/ksimpaa).

## What it does

**Picks the moments worth watching.** Each second of the video is scored by YouTube's *Most replayed* data (when the video has it), audio intensity (shouting, explosions, sudden jumps), and speech energy (fast talking, words like "no", "bro", "what"). The best non-overlapping 15–30 s windows win, with cuts snapped to word boundaries.

**Edits each clip like a short.**
- 1080×1920 vertical: the full frame over a blurred background (`blur`), a centre crop (`crop`), or face tracking (`speaker`)
- Word-by-word captions with the spoken word highlighted
- A hook title at the top, plus a cover image with the hook in big type
- Jump cuts that remove quiet pauses (loud game action is never cut)
- A cold open that flashes the peak moment first
- Punch-in zooms on the loudest beats

**Writes the copy and schedules it.** It writes a hook, title, caption and hashtags per clip, either from built-in templates or by Claude if you add an API key. It then uploads to Postiz with platform-specific text: YouTube gets a title with `#shorts`, the cover as thumbnail and tags; TikTok and Instagram get the caption and hashtags. Clips are scheduled at daily time slots, hottest first.

## What's in the repo

| Path | What it is |
|---|---|
| `pipeline.py` | The whole pipeline: one file, one command-line tool |
| `setup.ps1` / `setup.sh` | One-time setup for Windows / macOS & Linux |
| `repos/yt-dlp` | [yt-dlp](https://github.com/yt-dlp/yt-dlp) (submodule): downloads the video |
| `repos/video-editor` | [ClipsAI video-editor](https://github.com/ClipsAI/video-editor) (submodule): optional review UI |
| `repos/postiz` | [Postiz](https://github.com/Fogged1/Postiz-app) (submodule): the scheduler you run yourself |
| `runs/<video id>/` | Created per video: source, transcript, `clips/`, `clips.json` (not committed) |

## Setup

You need **git**, **ffmpeg** and **Python 3.11**. On Windows, `setup.ps1` installs Python 3.11 for you if it's missing. About 5 GB of disk space is needed for PyTorch and the models, and at least 8 GB of RAM, ideally free while it runs.

### 1. Clone, including the three submodules

```bash
git clone --recursive https://github.com/ksimpaa/clips.git
cd clips
```

If you already cloned without `--recursive`, run `git submodule update --init --depth 1`.

### 2. Install

**Windows (PowerShell):**
```powershell
winget install Git.Git
winget install Gyan.FFmpeg
# close and reopen PowerShell, then in the clips folder:
powershell -ExecutionPolicy Bypass -File setup.ps1
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
# macOS: brew install ffmpeg python@3.11 libmagic
# Ubuntu: sudo apt install ffmpeg python3.11 python3.11-venv libmagic1
./setup.sh
. .venv/bin/activate
```

The install step downloads PyTorch, so it takes a while. Setup ends with `imports ok`.

### 3. Configure (optional for testing)

Setup creates a `.env` file. Fill it in only for what you use:

| Key | Needed for |
|---|---|
| `POSTIZ_API_URL`, `POSTIZ_API_KEY` | Posting (Postiz → Settings → Public API) |
| `ANTHROPIC_API_KEY` | AI-written hooks, titles and captions (otherwise built-in templates) |
| `HUGGINGFACE_TOKEN` | Only `--layout speaker` |

## Test it

**1. Make clips from a short video.** Pick a 5–15 minute video with talking in it. `--skip-post` stops before Postiz, so no Postiz is needed:

```bash
python pipeline.py run "https://www.youtube.com/watch?v=VIDEO_ID" --max-clips 3 --max 30 --whisper-model base --language en --skip-post
```

It downloads the video, transcribes it (the slow part), picks the 3 hottest moments and renders them. You should see `[1/3] clip_01.mp4 … hook: …` for each clip.

**2. Look at the result** in `runs/VIDEO_ID/clips/`, which holds `clip_01.mp4` … plus `clip_01_cover.jpg` …
- Windows: `explorer runs\VIDEO_ID\clips`
- macOS: `open runs/VIDEO_ID/clips`

**3. Change a hook and re-render.** Edit `"hook"` in `runs/VIDEO_ID/clips.json`, then:
```bash
python pipeline.py render runs/VIDEO_ID
```

**4. Post as drafts.** This needs Postiz running and `.env` filled in:
```bash
python pipeline.py channels                                   # lists your channel ids
python pipeline.py post runs/VIDEO_ID --integrations ID1,ID2 --mode draft
```
The drafts then appear in Postiz's calendar. Use `--mode schedule` when you're happy with them.

## Command reference

```
python pipeline.py run URL [clip options] [post options] [--skip-post]
python pipeline.py download URL
python pipeline.py clip runs/ID [clip options]      # reuses the saved transcript
python pipeline.py render runs/ID [edit options]    # after editing clips.json
python pipeline.py post runs/ID [post options]
python pipeline.py channels
```

| Clip option | Default | |
|---|---|---|
| `--pick hot\|topics` | hot | hot moments, or ClipsAI's topic-based clips |
| `--min` / `--max` | 15 / 20 s | clip length window |
| `--max-clips` | 5 | 0 = all |
| `--layout blur\|crop\|speaker\|original` | blur | how 16:9 becomes 9:16 |
| `--skip-start` | 30 | ignore the intro (seconds) |
| `--no-jumpcut`, `--no-cold-open`, `--no-zoom`, `--no-hook`, `--no-captions` | | turn edits off |
| `--hook-seconds N` | whole clip | show the hook only for the first N seconds |
| `--caption-words`, `--caption-position lower\|middle` | 3, lower | |
| `--whisper-model tiny\|base\|small\|…`, `--language en` | tiny on CPU | |
| `--batch-size` | 4 | lower it if you run out of memory |

| Post option | Default | |
|---|---|---|
| `--integrations` | `POSTIZ_INTEGRATIONS` | channel ids or names, comma-separated |
| `--mode draft\|schedule\|now` | draft | |
| `--slots 12:30,18:30,21:30` | those | daily posting times (local); `--slots ''` uses `--every` |
| `--start`, `--every` | +30 min, 24 h | |
| `--hashtags a,b` | | extra hashtags on every post |
| `--youtube-privacy`, `--tiktok-privacy` | public | |
| `--credit` | off | add the source link to YouTube descriptions |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `mkl_malloc: failed to allocate memory`, `Unable to allocate … MiB`, `OpenBLAS … Memory allocation` | Out of RAM. Close heavy apps (Docker Desktop's `com.docker.backend` can grow to 15+ GB; quit it while clipping), then use `--whisper-model tiny --batch-size 2`. |
| `OSError: [WinError 127]` importing torchaudio | torch and torchaudio versions differ: `pip install "torch==2.8.0" "torchaudio==2.8.0"` |
| `[Errno 22] Invalid argument` during pip install (Windows) | Antivirus locked a file mid-install. Run the same pip command again. |
| Long `torchcodec is not installed correctly` warning | Harmless unless you use `--layout speaker`. |
| `No supported JavaScript runtime` (yt-dlp) | Harmless for downloading. `winget install DenoLand.Deno` may bring back YouTube's replay data. |
| `finding hot moments (no replay data…)` | The video has no "Most replayed" graph; clips are picked from audio and speech. |
| `.ps1 cannot be loaded because running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

## Credits & licences

This repo links to these projects as submodules; each keeps its own licence:
- [yt-dlp](https://github.com/yt-dlp/yt-dlp): Unlicense (public domain)
- [ClipsAI](https://github.com/ClipsAI/clipsai) and its [video-editor](https://github.com/ClipsAI/video-editor): MIT
- [Postiz](https://github.com/Fogged1/Postiz-app): AGPL-3.0

Only download and repost videos you have the rights to. Reposting other creators' content can get your accounts struck.
