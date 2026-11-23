#!/usr/bin/env python3
"""
clip-pipeline: yt-dlp -> ClipsAI -> Postiz

    python pipeline.py run "https://youtube.com/watch?v=..." --integrations <id>[,<id>...]

Stages (each can also be run on its own, and each is resumable):

  download   yt-dlp fetches the video into runs/<video_id>/source.mp4
  clip       ClipsAI transcribes, finds clips, cuts them (and crops to 9:16)
             into runs/<video_id>/clips/*.mp4 and writes runs/<video_id>/clips.json.
             With --ui it also loads the result into repos/video-editor so you can
             review the clips in the ClipsAI editor UI (npm run dev).
  post       uploads every clip in clips.json with "post": true to Postiz and
             creates one Postiz post per clip (draft, scheduled, or now).

Between "clip" and "post" you can edit runs/<video_id>/clips.json by hand:
set "post": false on clips you don't want, change "title" or "caption".
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
UI_DIR = ROOT / "repos" / "video-editor"

# Use the vendored yt-dlp in repos/ if it hasn't been pip-installed.
sys.path.insert(0, str(ROOT / "repos" / "yt-dlp"))


def load_env(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


def read_manifest(workdir: Path) -> dict:
    return json.loads((workdir / "clips.json").read_text(encoding="utf-8"))


def write_manifest(workdir: Path, manifest: dict) -> None:
    (workdir / "clips.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Stage 1: yt-dlp
# --------------------------------------------------------------------------- #
def download(url: str, cookies: str | None = None) -> Path:
    import yt_dlp

    RUNS.mkdir(exist_ok=True)
    opts = {
        # best mp4 video + m4a audio, merged into one mp4 (needs ffmpeg)
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(RUNS / "%(id)s" / "source.%(ext)s"),
        "noplaylist": True,
        "writeinfojson": False,
    }
    if cookies:
        opts["cookiefile"] = cookies

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    workdir = RUNS / info["id"]
    src = Path(info["requested_downloads"][0]["filepath"])
    if src.suffix != ".mp4":
        target = workdir / "source.mp4"
        src.rename(target)
        src = target

    meta = {
        "id": info["id"],
        "title": info.get("title") or info["id"],
        "url": info.get("webpage_url") or url,
        "uploader": info.get("uploader"),
        "duration": info.get("duration"),
        "source": src.name,
        "heatmap": info.get("heatmap"),
    }
    (workdir / "video.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"downloaded '{meta['title']}' -> {src}")
    return workdir


# --------------------------------------------------------------------------- #
# Stage 2: ClipsAI
# --------------------------------------------------------------------------- #
LAYOUTS = ("blur", "crop", "speaker", "original")


def _title_from_text(text: str, limit: int = 90) -> str:
    text = " ".join(text.split())
    first = text.split(". ")[0].strip()
    if len(first) > limit:
        first = first[:limit].rsplit(" ", 1)[0] + "…"
    return first or "Clip"


def _select_clips(found, *, min_len: float, max_len: float, skip_start: float, max_clips: int) -> list:
    """ClipsAI returns overlapping candidates in timeline order with no score.
    Keep the ones in the length window, drop overlaps, and spread the picks
    across the whole video instead of taking the earliest ones."""
    cands = sorted(
        (c for c in found if min_len <= c.end_time - c.start_time <= max_len and c.start_time >= skip_start),
        key=lambda c: (c.start_time, -(c.end_time - c.start_time)),
    )
    picked, last_end = [], -1.0
    for c in cands:
        if c.start_time >= last_end:
            picked.append(c)
            last_end = c.end_time
    if max_clips and len(picked) > max_clips:
        step = len(picked) / max_clips
        picked = [picked[int(i * step + step / 2)] for i in range(max_clips)]
    return picked


def fetch_heatmap(url: str) -> list | None:
    """YouTube's 'Most replayed' graph: ~100 segments with a 0-1 intensity."""
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True, "noplaylist": True}) as ydl:
        return ydl.extract_info(url, download=False).get("heatmap")


def _rank01(x):
    """Percentile rank in 0..1, robust to outliers and to the units of each signal."""
    import numpy as np

    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.allclose(x, x[0]):
        return np.zeros_like(x)
    return x.argsort().argsort() / (x.size - 1)


def loudness_per_second(source: Path):
    """RMS loudness of every second of audio (decoded at 8 kHz mono to keep memory small)."""
    import subprocess

    import numpy as np

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(source), "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout
    a = np.frombuffer(raw, np.int16)
    n = a.size // 8000
    a = a[: n * 8000].reshape(n, 8000).astype(np.float32) / 32768.0
    return 20 * np.log10(np.sqrt((a ** 2).mean(axis=1)) + 1e-6)


HYPE_WORDS = {
    "no", "what", "wow", "oh", "omg", "god", "bro", "yo", "wait", "dude", "crazy", "insane",
    "run", "help", "dead", "die", "died", "kill", "killed", "yes", "lets", "let's", "go", "how",
    "why", "holy", "aah", "ahh", "huh", "stop", "look", "bruh", "nah", "wtf",
}


def find_hot_moments(words: list[dict], source: Path, heatmap: list | None, *, min_len: float,
                     max_len: float, skip_start: float, max_clips: int, loud=None):
    """Score every second by replays (YouTube heatmap), audio intensity and speech
    energy, then return the best non-overlapping windows, snapped to word boundaries."""
    import numpy as np

    loud = loudness_per_second(source) if loud is None else loud
    dur = loud.size
    # loudness: absolute level plus sudden jumps (shouting, explosions)
    jump = np.maximum(0, np.diff(loud, prepend=loud[:1]))
    loud_s = 0.7 * _rank01(loud) + 0.3 * _rank01(jump)

    rate = np.zeros(dur)
    hype = np.zeros(dur)
    for w in words:
        st = w.get("start_time")
        if st is None or not (0 <= st < dur):
            continue
        rate[int(st)] += 1
        txt = (w.get("word") or "").lower()
        if txt.strip(".,!?\"'") in HYPE_WORDS or "!" in txt or "?" in txt:
            hype[int(st)] += 1
    speech_s = 0.6 * _rank01(np.convolve(rate, np.ones(3) / 3, "same")) + 0.4 * _rank01(np.convolve(hype, np.ones(5), "same"))

    heat = np.zeros(dur)
    if heatmap:
        for seg in heatmap:
            a, b = int(seg["start_time"]), int(np.ceil(seg["end_time"]))
            heat[max(0, a): min(dur, b)] = seg.get("value") or 0
        score = 0.55 * heat + 0.30 * loud_s + 0.15 * speech_s
    else:
        score = 0.65 * loud_s + 0.35 * speech_s

    L = int(round(max_len))
    if dur <= L:
        return []
    starts = np.arange(int(skip_start), dur - L)
    if starts.size == 0:
        starts = np.arange(0, dur - L)
    csum = np.concatenate([[0], np.cumsum(score)])
    win_mean = (csum[starts + L] - csum[starts]) / L
    win_peak = np.array([score[s: s + L].max() for s in starts])
    win_score = 0.7 * win_mean + 0.3 * win_peak

    timed = [w for w in words if w.get("start_time") is not None and w.get("end_time") is not None]
    picked, taken = [], np.zeros(dur, bool)
    for i in np.argsort(-win_score):
        s0 = int(starts[i])
        if taken[max(0, s0 - 5): s0 + L + 5].any():
            continue
        start, end = _snap(timed, s0, s0 + L, min_len, max_len)
        picked.append({
            "start_time": start, "end_time": end, "score": round(float(win_score[i]), 3),
            "replays": round(float(heat[s0: s0 + L].mean()), 2) if heatmap else None,
            "loudness": round(float(loud_s[s0: s0 + L].mean()), 2),
            "speech": round(float(speech_s[s0: s0 + L].mean()), 2),
        })
        taken[s0: s0 + L] = True
        if max_clips and len(picked) >= max_clips:
            break
    return picked


def _snap(words: list[dict], start: float, end: float, min_len: float, max_len: float) -> tuple[float, float]:
    """Move cuts to sentence/word boundaries so nobody is cut off mid-word."""
    if not words:
        return float(start), float(end)
    # start: a word beginning within 2 s before / 1 s after, preferring one after a pause
    best, best_gap = None, -1.0
    for j, w in enumerate(words):
        if start - 2 <= w["start_time"] <= start + 1:
            gap = w["start_time"] - words[j - 1]["end_time"] if j else 9.0
            if gap > best_gap:
                best, best_gap = w["start_time"], gap
    s = max(0.0, (best - 0.15) if best is not None else float(start))
    # end: a word ending inside [s+min, s+max], preferring one followed by a pause
    best, best_gap = None, -1.0
    for j, w in enumerate(words):
        if s + min_len <= w["end_time"] <= s + max_len:
            gap = words[j + 1]["start_time"] - w["end_time"] if j + 1 < len(words) else 9.0
            if gap > best_gap:
                best, best_gap = w["end_time"], gap
    e = (best + 0.25) if best is not None else s + max_len
    return round(s, 3), round(min(e, s + max_len + 0.25), 3)


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def _ass_escape(t: str) -> str:
    return t.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def _no_emoji(t: str) -> str:
    """On-screen text only: emoji fonts are unreliable in burned-in subtitles."""
    import re

    return re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]", "", t).strip()


# --------------------------------------------------------------------------- #
# Editing: jump cuts, cold open, zoom punch-ins
# --------------------------------------------------------------------------- #
def _timed(words: list[dict]) -> list[dict]:
    return [w for w in words if w.get("start_time") is not None and w.get("end_time") is not None]


def plan_edit(start: float, end: float, words: list[dict], loud_rank, *, jumpcut: bool = True,
              cold_open: bool = True, zoom: bool = True, max_gap: float = 0.45, pad: float = 0.12) -> list[dict]:
    """Turn one clip window into a list of source segments to play in order:
    quiet pauses removed, the peak second teased up front, punch-ins on the loudest beats."""
    import numpy as np

    n = len(loud_rank)

    def quiet(a: float, b: float) -> bool:
        idx = list(range(max(0, int(a)), min(n, int(np.ceil(b)))))
        return not idx or float(np.mean([loud_rank[i] for i in idx])) < 0.6

    ws = [w for w in _timed(words) if start <= w["start_time"] < end]

    # 1. jump cuts: drop pauses between words that are long *and* quiet (keep loud game action)
    keep = [[start, end]]
    if jumpcut and ws:
        marks = [(start, start)] + [(w["start_time"], w["end_time"]) for w in ws] + [(end, end)]
        cuts = []
        for (_, prev_end), (nxt_start, _) in zip(marks, marks[1:]):
            a, b = prev_end + pad, nxt_start - pad
            if b - a > max_gap and quiet(a, b):
                cuts.append((a, b))
        keep, cur = [], start
        for a, b in cuts:
            if a - cur >= 0.25:
                keep.append([cur, a])
            cur = b
        if end - cur >= 0.25:
            keep.append([cur, end])
        if not keep:
            keep = [[start, end]]

    # 2. punch-ins: the 3 loudest seconds of the clip (if they are loud for this video)
    zooms = []
    if zoom:
        secs = [i for i in range(int(start), min(n, int(np.ceil(end))))]
        secs = sorted(secs, key=lambda i: -loud_rank[i])[:3]
        zooms = sorted((i, i + 1.0) for i in secs if loud_rank[i] >= 0.85)

    segs = []
    for a, b in keep:
        cuts_at = sorted({a, b, *[z for zz in zooms for z in zz if a < z < b]})
        for x, y in zip(cuts_at, cuts_at[1:]):
            if y - x >= 0.1:
                mid = (x + y) / 2
                segs.append({"start": round(x, 3), "end": round(y, 3),
                             "zoom": any(z0 <= mid < z1 for z0, z1 in zooms)})

    # 3. cold open: flash the loudest moment first if it isn't already at the start
    if cold_open and n:
        secs = [i for i in range(int(start), min(n, int(np.ceil(end))))]
        if secs:
            peak = max(secs, key=lambda i: loud_rank[i])
            if peak - start > 4 and loud_rank[peak] >= 0.8:
                ta, tb = max(start, peak - 0.4), min(end, peak + 1.4)
                segs.insert(0, {"start": round(ta, 3), "end": round(tb, 3), "zoom": True, "teaser": True})
    return segs


def map_words(words: list[dict], segs: list[dict]) -> tuple[list[dict], float]:
    """Word timings on the edited timeline (words inside removed pauses disappear)."""
    out, t0 = [], 0.0
    timed = _timed(words)
    for sg in segs:
        for w in timed:
            if sg["start"] <= w["start_time"] < sg["end"]:
                out.append({"word": w.get("word", ""), "start_time": t0 + w["start_time"] - sg["start"],
                            "end_time": t0 + min(w["end_time"], sg["end"]) - sg["start"]})
        t0 += sg["end"] - sg["start"]
    return out, t0


def write_ass(words: list[dict], total: float, path: Path, *, width: int, height: int,
              hook: str | None = None, hook_seconds: float = 0, per_line: int = 3,
              position: str = "lower", captions: bool = True) -> None:
    """Subtitles file for the edited clip: hook title at the top + word-by-word captions
    (a few words at a time, white with black outline, current word yellow and bigger)."""
    vertical = height > width
    size = round(height * (0.044 if vertical else 0.075))
    outline = max(3, size // 12)
    align, margin_v = (2, round(height * 0.29)) if position == "lower" else (5, 0)
    if not vertical and position == "lower":
        margin_v = round(height * 0.08)
    hook_size = round(height * (0.038 if vertical else 0.06))
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Caption,Arial Black,{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,"
        f"100,100,0,0,1,{outline},{outline // 2},{align},60,60,{margin_v},1",
        # black text on a white box, top centre (sits in the blurred band above the gameplay)
        f"Style: Hook,Arial Black,{hook_size},&H00101010,&H00101010,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,"
        f"100,100,0,0,3,{max(10, hook_size // 3)},0,8,90,90,{round(height * (0.13 if vertical else 0.05))},1",
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    if hook:
        end = min(total, hook_seconds) if hook_seconds else total
        lines.append(f"Dialogue: 1,{_ass_time(0)},{_ass_time(end)},Hook,,0,0,0,,"
                     "{\\fad(120,200)}" + _ass_escape(_no_emoji(hook).upper()))
    if captions:
        ws = [[w["start_time"], max(w["end_time"], w["start_time"] + 0.08),
               _ass_escape((w.get("word") or "").strip().upper())] for w in words if (w.get("word") or "").strip()]
        chunks, cur = [], []
        for w in ws:
            if cur and (len(cur) >= per_line or w[0] - cur[-1][1] > 0.7 or cur[-1][2][-1:] in ".?!,"):
                chunks.append(cur)
                cur = []
            cur.append(w)
        if cur:
            chunks.append(cur)
        for ci, ch in enumerate(chunks):
            nxt = chunks[ci + 1][0][0] if ci + 1 < len(chunks) else ch[-1][1] + 0.4
            for wi, w in enumerate(ch):
                st = ch[0][0] if wi == 0 else w[0]
                en = ch[wi + 1][0] if wi + 1 < len(ch) else min(nxt, ch[-1][1] + 0.4, total)
                if en <= st:
                    en = st + 0.05
                text = " ".join(("{\\1c&H00FFFF&\\fscx112\\fscy112}" + x[2] + "{\\r}") if j == wi else x[2]
                                for j, x in enumerate(ch))
                lines.append(f"Dialogue: 0,{_ass_time(st)},{_ass_time(en)},Caption,,0,0,0,,{text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_captions(words: list[dict], clip_start: float, clip_end: float, path: Path, *,
                   width: int, height: int, per_line: int = 3, position: str = "lower") -> int:
    """Captions for an unedited window of the source (kept for --layout speaker)."""
    mapped, total = map_words(words, [{"start": clip_start, "end": clip_end}])
    write_ass(mapped, total, path, width=width, height=height, per_line=per_line, position=position)
    return len(mapped)


def _probe_size(path: Path) -> tuple[int, int]:
    import subprocess

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    w, h = out.split("x")[:2]
    return int(w), int(h)


def _layout_graph(layout: str, zoom: bool, size: tuple[int, int]) -> str:
    W, H = size
    graphs = {
        "blur": "[0:v]split=2[a][b];"
                "[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
                "gblur=sigma=30,eq=brightness=-0.08[bg];"
                "[b]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v0]",
        "crop": "[0:v]crop='min(iw,ih*9/16)':ih,scale=1080:1920,setsar=1[v0]",
        "speaker": "[0:v]scale=1080:1920,setsar=1[v0]",
        "original": "[0:v]null[v0]",
    }
    z = f";[v0]crop=iw/1.18:ih/1.18,scale={W}:{H},setsar=1[v1]" if zoom else ";[v0]null[v1]"
    return graphs[layout] + z


def _run_ffmpeg(cmd: list[str], cwd: Path, what: str) -> None:
    import subprocess

    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"ffmpeg failed on {what}:\n{r.stderr[-1500:]}")


ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
       "-c:a", "aac", "-b:a", "160k", "-ar", "48000"]


def _ffmpeg_render(src: Path, out: Path, *, start: float, duration: float, layout: str,
                   ass_name: str | None, zoom: bool = False, size: tuple[int, int] = (1080, 1920),
                   fade: bool = False) -> None:
    """Cut [start, start+duration] of src, lay it out (and punch in), burn subtitles."""
    graph = _layout_graph(layout, zoom, size) + (f";[v1]ass={ass_name}[v]" if ass_name else ";[v1]null[v]")
    af = ["-af", f"afade=t=in:d=0.03,afade=t=out:st={max(0, duration - 0.04):.3f}:d=0.04"] if fade else []
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(src.resolve()),
           "-filter_complex", graph, "-map", "[v]", "-map", "0:a?", *af, *ENC,
           "-movflags", "+faststart", out.name]
    # run inside the clips folder so the .ass path needs no escaping (Windows drive letters)
    _run_ffmpeg(cmd, out.parent, out.name)


def render_edit(src: Path, segs: list[dict], out: Path, *, layout: str, ass_name: str | None,
                size: tuple[int, int]) -> None:
    """Render each segment (layout + punch-in), join them, burn hook + captions on top."""
    tmp = []
    for k, sg in enumerate(segs):
        f = out.parent / f"{out.stem}_part{k:02d}.mp4"
        _ffmpeg_render(src, f, start=sg["start"], duration=sg["end"] - sg["start"], layout=layout,
                       ass_name=None, zoom=sg.get("zoom", False), size=size, fade=True)
        tmp.append(f)
    lst = out.parent / f"{out.stem}_parts.txt"
    lst.write_text("".join(f"file '{f.name}'\n" for f in tmp), encoding="utf-8")
    vf = ["-vf", f"ass={ass_name}"] if ass_name else []
    _run_ffmpeg(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                 "-i", lst.name, *vf, *ENC, "-movflags", "+faststart", out.name], out.parent, out.name)
    for f in [*tmp, lst]:
        f.unlink(missing_ok=True)


def make_cover(src: Path, t: float, out: Path, *, layout: str, hook: str | None, size: tuple[int, int]) -> None:
    """Cover / thumbnail: the clip's peak frame in the same layout, with the hook in big type."""
    W, H = size
    ass = out.with_suffix(".ass")
    big = round(H * (0.06 if H > W else 0.09))
    ass.write_text("\n".join([
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 0",
        "ScaledBorderAndShadow: yes", "", "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Cover,Arial Black,{big},&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,0,-4,1,"
        f"{max(5, big // 9)},{max(3, big // 14)},8,70,70,{round(H * 0.1)},1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        *([f"Dialogue: 0,0:00:00.00,0:00:10.00,Cover,,0,0,0,,{_ass_escape(_no_emoji(hook).upper())}"] if hook else []),
    ]) + "\n", encoding="utf-8")
    graph = _layout_graph(layout, True, size) + f";[v1]eq=saturation=1.25:contrast=1.08,ass={ass.name}[v]"
    _run_ffmpeg(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", str(src.resolve()),
                 "-filter_complex", graph, "-map", "[v]", "-frames:v", "1", "-q:v", "2", out.name],
                out.parent, out.name)
    ass.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Copy: hook, title, caption, hashtags
# --------------------------------------------------------------------------- #
STOP = {"the", "a", "an", "in", "on", "of", "to", "and", "or", "i", "my", "me", "is", "it", "got", "get", "for",
        "with", "this", "that", "free", "roam", "part", "ep", "episode", "video", "official", "new", "vs",
        "all", "from", "every", "best", "most", "how", "what", "when", "why", "who", "we", "you", "your",
        "our", "they", "their", "at", "by", "as", "be", "are", "was", "were", "has", "have", "had", "into",
        "out", "up", "down", "over", "after", "before", "ever", "only", "just", "very", "so", "too", "but",
        "stealing", "trying", "playing", "making", "using", "going", "getting", "first", "last", "time"}
HOOKS = ["WAIT FOR IT 😳", "THIS GOT OUT OF HAND 💀", "DID NOT SEE THAT COMING 😭", "HOW DID THIS HAPPEN 😂",
         "THE ENDING THO 💀", "NOBODY EXPECTED THIS 😳"]
QUESTIONS = ["Would you have survived this? 👇", "What would you have done? 👇", "Rate this 1-10 👇",
             "Has this ever happened to you? 👇", "Tag someone who would do this 😂"]


def _title_tags(title: str) -> list[str]:
    import re

    title = re.sub(r"[\(\[].*?[\)\]]", " ", title)
    toks = re.findall(r"[A-Za-z0-9]+", title)
    tags, i = [], 0
    while i < len(toks):
        t = toks[i]
        if i + 1 < len(toks) and toks[i + 1].isdigit() and not t.isdigit():  # "GTA 6" -> gta6
            t, i = t + toks[i + 1], i + 1
        if t.lower() not in STOP and len(t) > 2 and not t.isdigit():
            tags.append(t.lower())
        i += 1
    tags = sorted(dict.fromkeys(tags), key=lambda t: not any(ch.isdigit() for ch in t))  # game tags (gta6) first
    return tags[:4]


def _quote_from(text: str) -> str | None:
    """Shortest exciting exclamation/question in the transcript, as a quote hook."""
    import re

    best = None
    for sent in re.split(r"(?<=[.!?])\s+", " ".join(text.split())):
        words = sent.split()
        hype = sum(w.lower().strip(".,!?\"'") in HYPE_WORDS for w in words)
        if 2 <= len(words) <= 7 and (hype >= 2 or (hype >= 1 and sent[-1:] in "!?")):
            score = hype + sent.count("!") + 0.5 * sent.count("?")
            if best is None or score > best[0]:
                best = (score, sent)
    return f"“{best[1].rstrip('.')}”" if best else None


def heuristic_copy(entries: list[dict], meta: dict) -> None:
    import re

    vt = re.sub(r"[\(\[].*?[\)\]]", "", meta.get("title") or "").strip()
    tags = _title_tags(meta.get("title") or "")
    for i, e in enumerate(entries):
        if e.get("hook"):
            continue
        quote = _quote_from(e.get("transcript", ""))
        if i == 0 and vt and len(vt.split()) <= 8:
            hook = vt
        elif quote and i % 2 == 1:
            hook = quote
        else:
            hook = HOOKS[i % len(HOOKS)]
        e["hook"] = hook
        e["title"] = (f"{vt} 😳" if i == 0 and vt else f"{hook.strip(chr(8220) + chr(8221))} | {vt}")[:90]
        h = hook.strip(chr(8220) + chr(8221))
        h = h if any(ch.islower() for ch in h) else h.capitalize()  # keep "GTA 6", tame ALL CAPS
        e["caption"] = f"{h[:1].upper() + h[1:]} {QUESTIONS[i % len(QUESTIONS)]}"
        e["hashtags"] = list(dict.fromkeys(tags + ["gaming", "fyp", "viral", "shorts"]))[:8]
        e["copy_by"] = "built-in"


def llm_copy(entries: list[dict], meta: dict) -> bool:
    """Ask Claude for scroll-stopping hooks, titles, captions and hashtags (needs ANTHROPIC_API_KEY)."""
    import requests

    key = os.environ.get("ANTHROPIC_API_KEY")
    todo = [e for e in entries if not e.get("hook")]
    if not key or not todo:
        return False
    clips = [{"id": e["id"], "seconds": round(e["end_time"] - e["start_time"]),
              "transcript": e.get("transcript", "")[:1500], "why_picked": e.get("hotness")} for e in todo]
    prompt = (
        "You write copy for short vertical gaming clips (TikTok, YouTube Shorts, Instagram Reels).\n"
        f"Source video: {meta.get('title')!r} by {meta.get('uploader')!r}.\n"
        "For each clip return: hook (on-screen text, max 7 words, creates curiosity about what happens, "
        "may end with one emoji, must not lie about the content), title (YouTube Shorts title, max 80 chars), "
        "caption (1-2 short sentences for TikTok/Instagram ending with a question that invites comments), "
        "hashtags (5-8 lowercase tags without '#', mixing broad gaming tags and tags specific to the game/moment).\n"
        "Reply with only a JSON array of objects with keys id, hook, title, caption, hashtags.\n\n"
        + json.dumps(clips, ensure_ascii=False)
    )
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5"), "max_tokens": 4000,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=120,
        )
        r.raise_for_status()
        text = "".join(b.get("text", "") for b in r.json().get("content", []))
        data = json.loads(text[text.index("["): text.rindex("]") + 1])
    except Exception as e:  # never let copywriting stop the pipeline
        log(f"AI copywriting failed ({e.__class__.__name__}: {str(e)[:120]}); using built-in copy")
        return False
    by_id = {d.get("id"): d for d in data if isinstance(d, dict)}
    for e in todo:
        d = by_id.get(e["id"])
        if d and d.get("hook"):
            e["hook"], e["title"] = str(d["hook"]), str(d.get("title") or d["hook"])[:95]
            e["caption"] = str(d.get("caption") or d["hook"])
            e["hashtags"] = [str(h).lstrip("#").replace(" ", "").lower() for h in d.get("hashtags", [])][:8]
            e["copy_by"] = "claude"
    return True


# --------------------------------------------------------------------------- #
# Stage 2: pick moments -> write copy -> render
# --------------------------------------------------------------------------- #
def _load_transcription(workdir: Path, source: Path, whisper_model, language, batch_size):
    from clipsai import Transcription, Transcriber
    from clipsai.filesys.json_file import JSONFile

    tpath = workdir / "transcription.json"
    if tpath.exists():
        log("reusing transcription.json")
        return Transcription(JSONFile(str(tpath)))
    log("transcribing (WhisperX)…")
    transcriber = Transcriber(model_size=whisper_model) if whisper_model else Transcriber()
    t = transcriber.transcribe(audio_file_path=str(source), iso6391_lang_code=language, batch_size=batch_size)
    t.store_as_json_file(str(tpath))
    return t


def render_clips(workdir: Path, manifest: dict, words: list[dict], loud_rank, opts: dict) -> None:
    meta = manifest["video"]
    source = workdir / meta["source"]
    clips_dir = workdir / "clips"
    clips_dir.mkdir(exist_ok=True)
    layout = opts["layout"]
    size = _probe_size(source) if layout == "original" else (1080, 1920)
    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("PYANNOTE_AUTH_TOKEN")
    if layout == "speaker" and not hf_token:
        raise SystemExit("--layout speaker needs HUGGINGFACE_TOKEN in .env (pyannote); use --layout blur or crop.")

    for i, e in enumerate(manifest["clips"], 1):
        stem = e["id"]
        final, ass_name = clips_dir / f"{stem}.mp4", f"{stem}.ass"
        hook = e.get("hook") if opts["hook"] else None
        if layout == "speaker":
            from clipsai import AudioVideoFile, MediaEditor, resize

            dur = e["end_time"] - e["start_time"]
            write_captions(words, e["start_time"], e["end_time"], clips_dir / ass_name, width=1080, height=1920,
                           per_line=opts["caption_words"], position=opts["caption_position"])
            horizontal, vertical = clips_dir / f"{stem}_full.mp4", clips_dir / f"{stem}_v.mp4"
            _ffmpeg_render(source, horizontal, start=e["start_time"], duration=dur, layout="original",
                           ass_name=None, size=_probe_size(source))
            crops = resize(video_file_path=str(horizontal), pyannote_auth_token=hf_token, aspect_ratio=opts["aspect"])
            MediaEditor().resize_video(original_video_file=AudioVideoFile(str(horizontal)),
                                       resized_video_file_path=str(vertical), width=crops.crop_width,
                                       height=crops.crop_height, segments=[s.to_dict() for s in crops.segments])
            _ffmpeg_render(vertical, final, start=0, duration=dur, layout="speaker",
                           ass_name=ass_name if opts["captions"] else None)
            horizontal.unlink(missing_ok=True)
            vertical.unlink(missing_ok=True)
            segs, total = [{"start": e["start_time"], "end": e["end_time"]}], dur
        else:
            segs = plan_edit(e["start_time"], e["end_time"], words, loud_rank, jumpcut=opts["jumpcut"],
                             cold_open=opts["cold_open"], zoom=opts["zoom"])
            mapped, total = map_words(words, segs)
            write_ass(mapped, total, clips_dir / ass_name, width=size[0], height=size[1], hook=hook,
                      hook_seconds=opts["hook_seconds"], per_line=opts["caption_words"],
                      position=opts["caption_position"], captions=opts["captions"])
            render_edit(source, segs, final, layout=layout, ass_name=ass_name, size=size)
        (clips_dir / ass_name).unlink(missing_ok=True)

        # cover image at the clip's loudest second
        secs = range(int(e["start_time"]), min(len(loud_rank), int(e["end_time"])))
        peak = max(secs, key=lambda k: loud_rank[k]) if len(secs) else e["start_time"]
        cover = clips_dir / f"{stem}_cover.jpg"
        cover_layout = "crop" if layout == "speaker" else layout
        make_cover(source, float(peak) + 0.5, cover, layout=cover_layout, hook=e.get("hook"), size=size)

        cut = (e["end_time"] - e["start_time"]) - sum(s["end"] - s["start"] for s in segs if not s.get("teaser"))
        e.update({"file": f"clips/{final.name}", "cover": f"clips/{cover.name}", "duration": round(total, 1),
                  "edit": {"segments": len(segs), "silence_cut_s": round(cut, 1),
                           "cold_open": any(s.get("teaser") for s in segs),
                           "punch_ins": sum(1 for s in segs if s.get("zoom") and not s.get("teaser"))}})
        log(f"[{i}/{len(manifest['clips'])}] {final.name}  {total:.1f}s  hook: {e.get('hook')}")
        write_manifest(workdir, manifest)


def clip(
    workdir: Path,
    *,
    min_len: int,
    max_len: int,
    max_clips: int,
    layout: str,
    aspect: tuple[int, int],
    whisper_model: str | None,
    language: str | None,
    ui: bool,
    batch_size: int = 4,
    skip_start: float = 30,
    captions: bool = True,
    caption_words: int = 3,
    caption_position: str = "lower",
    pick: str = "hot",
    jumpcut: bool = True,
    cold_open: bool = True,
    zoom: bool = True,
    hook: bool = True,
    hook_seconds: float = 0,
) -> dict:
    meta = json.loads((workdir / "video.json").read_text(encoding="utf-8"))
    source = workdir / meta["source"]
    transcription = _load_transcription(workdir, source, whisper_model, language, batch_size)
    words = transcription.get_word_info()
    full_text = transcription.text
    log("measuring audio…")
    loud = loudness_per_second(source)
    loud_rank = _rank01(loud)

    # 1. pick moments
    if pick == "hot":
        if "heatmap" not in meta and meta.get("url"):
            log("fetching YouTube 'Most replayed' data…")
            try:
                meta["heatmap"] = fetch_heatmap(meta["url"])
            except Exception as e:  # network/extractor problems shouldn't stop clipping
                log(f"couldn't fetch it ({e.__class__.__name__}); using audio and speech only")
                meta["heatmap"] = None
            (workdir / "video.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        hm = meta.get("heatmap")
        log("finding hot moments (" + ("replays + " if hm else "no replay data; ") + "audio + speech)…")
        moments = find_hot_moments(words, source, hm, min_len=min_len, max_len=max_len,
                                   skip_start=skip_start, max_clips=max_clips, loud=loud)
        picked = [SimpleNamespace(start_time=m["start_time"], end_time=m["end_time"],
                                  stats={k: m[k] for k in ("score", "replays", "loudness", "speech")})
                  for m in moments]
    else:
        from clipsai import ClipFinder

        log("finding clips by topic (ClipsAI)…")
        found = ClipFinder(min_clip_duration=min_len, max_clip_duration=max_len).find_clips(transcription=transcription)
        picked = _select_clips(found, min_len=min_len, max_len=max_len, skip_start=skip_start, max_clips=max_clips)
        log(f"ClipsAI found {len(found)} candidates; using {len(picked)} non-overlapping clips")
    if not picked:
        raise SystemExit("No clips fit the length window; try --min/--max or --skip-start 0.")

    # keep your edits and posting status from an earlier run (so re-clipping never re-posts)
    old = {}
    try:
        for e in read_manifest(workdir).get("clips", []):
            old[(e.get("start_time"), e.get("end_time"))] = e
    except (OSError, ValueError):  # no clips.json yet, or a half-written one from a crash
        pass

    entries = []
    for i, c in enumerate(picked, 1):
        inside = [w for w in _timed(words) if c.start_time <= w["start_time"] < c.end_time]
        text = full_text[inside[0]["start_char"]: inside[-1]["end_char"]] if inside else ""
        e = {"id": f"clip_{i:02d}", "start_time": round(c.start_time, 3), "end_time": round(c.end_time, 3),
             "transcript": text, **({"hotness": c.stats} if getattr(c, "stats", None) else {}),
             "post": True, "postiz": None}
        prev = old.get((e["start_time"], e["end_time"]))
        if prev:
            for k in ("hook", "title", "caption", "hashtags", "copy_by", "post", "postiz"):
                if k in prev:
                    e[k] = prev[k]
        entries.append(e)
    log(f"{len(entries)} clips: " + ", ".join(f"{e['start_time']:.0f}s" for e in entries))

    # 2. hooks, titles, captions, hashtags
    if not llm_copy(entries, meta):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            log("writing hooks/captions with built-in rules (set ANTHROPIC_API_KEY in .env for AI-written copy)")
    heuristic_copy(entries, meta)

    manifest = {"video": meta, "clips": entries,
                "render": {"layout": layout, "jumpcut": jumpcut, "cold_open": cold_open, "zoom": zoom,
                           "hook": hook, "hook_seconds": hook_seconds, "captions": captions,
                           "caption_words": caption_words, "caption_position": caption_position,
                           "aspect": list(aspect)}}
    write_manifest(workdir, manifest)

    # 3. render
    render_clips(workdir, manifest, words, loud_rank, manifest["render"] | {"aspect": aspect})
    log(f"wrote {workdir / 'clips.json'}")
    if ui:
        export_to_ui(workdir, manifest, transcription, None, [])
    return manifest


def rerender(workdir: Path, overrides: dict) -> None:
    """Re-render existing clips after you edit hooks/captions in clips.json."""
    from clipsai import Transcription
    from clipsai.filesys.json_file import JSONFile

    manifest = read_manifest(workdir)
    opts = manifest.get("render", {}) | {k: v for k, v in overrides.items() if v is not None}
    opts.setdefault("aspect", [9, 16])
    opts["aspect"] = tuple(opts["aspect"])
    for k, v in (("layout", "blur"), ("jumpcut", True), ("cold_open", True), ("zoom", True), ("hook", True),
                 ("hook_seconds", 0), ("captions", True), ("caption_words", 3), ("caption_position", "lower")):
        opts.setdefault(k, v)
    manifest["render"] = {**opts, "aspect": list(opts["aspect"])}
    words = Transcription(JSONFile(str(workdir / "transcription.json"))).get_word_info()
    loud_rank = _rank01(loudness_per_second(workdir / manifest["video"]["source"]))
    render_clips(workdir, manifest, words, loud_rank, opts)


def export_to_ui(workdir: Path, manifest: dict, transcription, crop_dims, segments) -> None:
    """Load the run into repos/video-editor in the format its src/data/*.ts files use."""
    import subprocess
    import uuid

    meta = manifest["video"]
    source = workdir / meta["source"]
    public = UI_DIR / "public"
    public.mkdir(exist_ok=True)
    shutil.copy2(source, public / "source.mp4")

    now = int(dt.datetime.now().timestamp())
    vid = str(uuid.uuid5(uuid.NAMESPACE_URL, meta["url"]))

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:format=duration", "-of", "json", str(source)],
        capture_output=True, text=True, check=True,
    )
    p = json.loads(probe.stdout)
    width, height = p["streams"][0]["width"], p["streams"][0]["height"]
    duration = float(p["format"]["duration"])

    video = {
        "id": vid,
        "object": "video",
        "clips": [
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, meta["url"] + c["id"])),
                "object": "clip",
                "created": now,
                "start_time": c["start_time"],
                "end_time": c["end_time"],
                "start_char": c["start_char"],
                "end_char": c["end_char"],
                "video_id": vid,
                "favorited": False,
                "deleted": False,
                "scores": {"embedding_norm": 0},
                "title": c["title"],
            }
            for c in manifest["clips"]
        ],
        "created": now,
        "metadata": {"duration": duration, "file_size": source.stat().st_size, "mime_type": "video/mp4"},
        "source": "/source.mp4",
        "status": "complete",
        "title": meta["title"],
    }

    transcript = {
        "id": vid,
        "object": "transcript",
        "created": now,
        "words": [
            {k: w[k] for k in ("start_char", "end_char", "start_time", "end_time")}
            for w in transcription.get_word_info()
        ],
        "transcription": transcription.text,
    }

    # The editor needs crop segments covering the whole timeline: use the real
    # ClipsAI crops inside each clip and a centred crop everywhere else.
    if crop_dims:
        cw, ch = crop_dims["crop_width"], crop_dims["crop_height"]
    else:
        ch = height
        cw = int(round(height * 9 / 16)) if width > height else width
    centre = {"speakers": [], "x": max(0, (width - cw) // 2), "y": 0}
    filled, t = [], 0.0
    for s in sorted(segments, key=lambda s: s["start_time"]):
        if s["start_time"] > t:
            filled.append({**centre, "start_time": t, "end_time": s["start_time"]})
        filled.append(s)
        t = max(t, s["end_time"])
    if t < duration:
        filled.append({**centre, "start_time": t, "end_time": duration})
    crops = {"original_width": width, "original_height": height,
             "crop_width": cw, "crop_height": ch, "segments": filled}

    data = UI_DIR / "src" / "data"
    for name, var, typ, obj in (
        ("video", "video", "Video", video),
        ("transcript", "transcript", "Transcript", transcript),
        ("crops", "crops", "Crops", crops),
    ):
        backup = data / f"{name}.demo.ts"
        if not backup.exists() and (data / f"{name}.ts").exists():
            shutil.copy2(data / f"{name}.ts", backup)
        (data / f"{name}.ts").write_text(
            f"export const {var}: {typ} = {json.dumps(obj, indent=2, ensure_ascii=False)}\n", encoding="utf-8"
        )
    log(f"loaded into the editor UI: cd {UI_DIR} && npm i && npm run dev")


# --------------------------------------------------------------------------- #
# Stage 3: Postiz
# --------------------------------------------------------------------------- #
class Postiz:
    def __init__(self, base_url: str, api_key: str):
        import requests

        self.base = base_url.rstrip("/")
        self.s = requests.Session()
        self.s.headers["Authorization"] = api_key

    def _check(self, r):
        if r.status_code >= 400:
            raise SystemExit(f"Postiz {r.request.method} {r.url} -> {r.status_code}: {r.text[:500]}")
        return r.json()

    def integrations(self) -> list[dict]:
        return self._check(self.s.get(f"{self.base}/integrations", timeout=60))

    def upload(self, path: Path) -> dict:
        import mimetypes

        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        with path.open("rb") as f:
            r = self.s.post(f"{self.base}/upload", files={"file": (path.name, f, ctype)}, timeout=900)
        return self._check(r)

    def create_post(self, body: dict) -> dict:
        return self._check(self.s.post(f"{self.base}/posts", json=body, timeout=120))


def provider_settings(identifier: str, clip: dict, opts: argparse.Namespace, thumb: dict | None = None) -> dict:
    """Per-platform settings Postiz validates (see postiz/libraries/.../providers-settings)."""
    if identifier == "youtube":
        title = clip.get("title") or clip.get("hook") or "Clip"
        if "#shorts" not in title.lower() and len(title) <= 91:
            title += " #shorts"
        st = {"__type": "youtube", "title": title[:100], "type": opts.youtube_privacy,
              "tags": [{"value": h, "label": h} for h in clip.get("hashtags", [])[:10]]}
        if thumb:
            st["thumbnail"] = {"id": thumb["id"], "path": thumb["path"]}
        return st
    if identifier == "tiktok":
        return {
            "__type": "tiktok",
            "privacy_level": opts.tiktok_privacy,
            "duet": False,
            "stitch": False,
            "comment": True,
            "autoAddMusic": "no",
            "brand_content_toggle": False,
            "brand_organic_toggle": False,
            "content_posting_method": "DIRECT_POST",
        }
    if identifier.startswith("instagram"):
        return {"__type": identifier, "post_type": "post"}
    return {"__type": identifier}


def post_text(identifier: str, clip: dict, extra_tags: list[str], opts: argparse.Namespace, meta: dict) -> str:
    """The text each platform gets: caption + hashtags, sized for that platform."""
    tags = list(dict.fromkeys([*extra_tags, *clip.get("hashtags", [])]))
    if opts.caption_template:
        text = opts.caption_template.format(
            title=clip.get("title", ""), caption=clip.get("caption", ""), hook=clip.get("hook", ""),
            hashtags=" ".join(f"#{h}" for h in tags), source=meta.get("url", ""), channel=meta.get("uploader") or "")
    elif identifier == "x":
        text = clip.get("caption", "")
        for h in tags[:3]:
            if len(text) + len(h) + 2 <= 270:
                text += f" #{h}"
    else:
        n = 5 if identifier.startswith("instagram") or identifier == "tiktok" else 8
        text = f"{clip.get('caption', '')}\n\n" + " ".join(f"#{h}" for h in tags[:n])
        if identifier == "youtube" and opts.credit and meta.get("uploader"):
            text += f"\n\nOriginal video: {meta.get('url')} ({meta['uploader']})"
    text = text.strip()
    return text if len(text) >= 6 else (text + " " + (meta.get("title") or "")).strip()


def schedule_times(n: int, opts: argparse.Namespace) -> list[dt.datetime]:
    """--slots 12:30,18:30,21:30 -> the next n of those times (local); else --start + i * --every."""
    now = dt.datetime.now().astimezone()
    start = dt.datetime.fromisoformat(opts.start) if opts.start else now + dt.timedelta(minutes=30)
    if start.tzinfo is None:
        start = start.astimezone()
    if not opts.slots:
        return [start + i * dt.timedelta(hours=opts.every) for i in range(n)]
    slots = sorted(dt.time.fromisoformat(x.strip()) for x in opts.slots.split(",") if x.strip())
    out, day = [], start.date()
    while len(out) < n:
        for t in slots:
            when = dt.datetime.combine(day, t).astimezone()
            if when >= start and len(out) < n:
                out.append(when)
        day += dt.timedelta(days=1)
    return out


def post(workdir: Path, opts: argparse.Namespace) -> None:
    base = opts.postiz_url or os.environ.get("POSTIZ_API_URL")
    key = os.environ.get("POSTIZ_API_KEY")
    if not base or not key:
        raise SystemExit("Set POSTIZ_API_URL and POSTIZ_API_KEY in .env (see .env.example).")
    api = Postiz(base, key)

    available = {i["id"]: i for i in api.integrations() if not i.get("disabled")}
    wanted = [x.strip() for x in (opts.integrations or os.environ.get("POSTIZ_INTEGRATIONS", "")).split(",") if x.strip()]
    if not wanted:
        raise SystemExit("No channels chosen. Run `python pipeline.py channels` and pass --integrations <id,...>.")
    targets = []
    for w in wanted:
        match = available.get(w) or next((i for i in available.values() if i["name"] == w), None)
        if not match:
            raise SystemExit(f"Postiz channel '{w}' not found or disabled. Run `python pipeline.py channels`.")
        targets.append(match)

    manifest = read_manifest(workdir)
    meta = manifest["video"]
    # hottest first (clip_01 is the best-scoring moment)
    todo = [c for c in manifest["clips"] if c.get("post") and not c.get("postiz")]
    if not todo:
        log("nothing to post (every clip is posted or marked \"post\": false)")
        return
    times = schedule_times(len(todo), opts)
    extra = [h.strip().lstrip("#").lower() for h in (opts.hashtags or "").split(",") if h.strip()]

    for c, when in zip(todo, times):
        log(f"uploading {c['file']} to Postiz…")
        media = api.upload(workdir / c["file"])
        thumb = None
        if c.get("cover") and (workdir / c["cover"]).exists() and any(t["identifier"] == "youtube" for t in targets):
            thumb = api.upload(workdir / c["cover"])
        body = {
            "type": opts.mode,
            "shortLink": False,
            "date": when.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "tags": [],
            "posts": [
                {
                    "integration": {"id": t["id"]},
                    "value": [{"content": post_text(t["identifier"], c, extra, opts, meta),
                               "image": [{"id": media["id"], "path": media["path"]}]}],
                    "settings": provider_settings(t["identifier"], c, opts, thumb),
                }
                for t in targets
            ],
        }
        result = api.create_post(body)
        c["postiz"] = {"mode": opts.mode, "date": body["date"], "channels": [t["name"] for t in targets],
                       "media": media, "thumbnail": thumb, "response": result}
        write_manifest(workdir, manifest)  # save after each clip so a failure can resume
        log(f"{opts.mode}: {c['id']} \"{c.get('hook')}\" -> {', '.join(t['name'] for t in targets)}"
            f" @ {when:%a %d %b %H:%M}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _aspect(v: str) -> tuple[int, int]:
    a, b = v.split(":")
    return int(a), int(b)


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description="yt-dlp -> ClipsAI -> Postiz")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def clip_args(p):
        p.add_argument("--pick", choices=["hot", "topics"], default="hot",
                       help="hot: most replayed / loudest / most hyped moments (default); "
                            "topics: ClipsAI's topic-based clips")
        p.add_argument("--min", dest="min_len", type=int, default=None,
                       help="min clip length, seconds (hot: 15, topics: 15)")
        p.add_argument("--max", dest="max_len", type=int, default=None,
                       help="max clip length, seconds (hot: 20, topics: 60)")
        p.add_argument("--max-clips", type=int, default=5, help="how many clips to cut, 0 = all (5)")
        p.add_argument("--layout", choices=LAYOUTS, default="blur",
                       help="blur: whole frame on a blurred 9:16 background (default, good for gaming); "
                            "crop: centre crop to 9:16; speaker: ClipsAI face tracking (needs HF token); "
                            "original: keep the source format")
        p.add_argument("--no-resize", action="store_true", help="same as --layout original")
        p.add_argument("--aspect", type=_aspect, default=(9, 16), help="ratio for --layout speaker (9:16)")
        p.add_argument("--skip-start", type=float, default=30, help="ignore the first N seconds (intro) (30)")
        p.add_argument("--no-captions", action="store_true", help="don't burn in subtitles")
        p.add_argument("--caption-words", type=int, default=3, help="words on screen at once (3)")
        p.add_argument("--caption-position", choices=["lower", "middle"], default="lower")
        p.add_argument("--whisper-model", default=None, help="tiny/base/small/medium/large-v2 (ClipsAI default)")
        p.add_argument("--language", default=None, help="ISO 639-1 code, e.g. en, fr (auto-detect)")
        p.add_argument("--batch-size", type=int, default=4,
                       help="audio chunks transcribed at once; lower it if you run out of memory (4)")
        p.add_argument("--ui", action="store_true", help="also load the result into repos/video-editor")
        edit_args(p)

    def edit_args(p, render=False):
        d = None if render else True  # render: only override what you pass
        p.add_argument("--no-jumpcut", dest="jumpcut", action="store_false", default=d,
                       help="keep the quiet pauses (default: cut them)")
        p.add_argument("--no-cold-open", dest="cold_open", action="store_false", default=d,
                       help="don't flash the peak moment at the start")
        p.add_argument("--no-zoom", dest="zoom", action="store_false", default=d, help="no punch-in zooms")
        p.add_argument("--no-hook", dest="hook", action="store_false", default=d, help="no hook title on screen")
        p.add_argument("--hook-seconds", type=float, default=None if render else 0,
                       help="show the hook only for the first N seconds (default: whole clip)")

    def post_args(p):
        p.add_argument("--integrations", help="Postiz channel ids or names, comma-separated")
        p.add_argument("--mode", choices=["draft", "schedule", "now"], default="draft",
                       help="draft (default), schedule, or now")
        p.add_argument("--slots", default="12:30,18:30,21:30",
                       help="daily posting times, local (12:30,18:30,21:30); pass '' to use --every")
        p.add_argument("--start", help="don't post before this time, ISO 8601 (default: 30 min from now)")
        p.add_argument("--every", type=float, default=24, help="hours between clips when --slots ''")
        p.add_argument("--hashtags", default="", help="extra hashtags for every post, comma-separated")
        p.add_argument("--caption-template", default=None,
                       help="override the text, fields: {title} {caption} {hook} {hashtags} {source} {channel}")
        p.add_argument("--credit", action="store_true", help="add the source video link to YouTube descriptions")
        p.add_argument("--youtube-privacy", choices=["public", "private", "unlisted"], default="public")
        p.add_argument("--tiktok-privacy", default="PUBLIC_TO_EVERYONE",
                       choices=["PUBLIC_TO_EVERYONE", "MUTUAL_FOLLOW_FRIENDS", "FOLLOWER_OF_CREATOR", "SELF_ONLY"])
        p.add_argument("--postiz-url", help="overrides POSTIZ_API_URL")

    p = sub.add_parser("run", help="download, clip and post in one go")
    p.add_argument("url")
    p.add_argument("--cookies", help="cookies.txt for yt-dlp (age-gated/private videos)")
    p.add_argument("--skip-post", action="store_true", help="stop after clipping so you can review")
    clip_args(p); post_args(p)

    p = sub.add_parser("download", help="yt-dlp only"); p.add_argument("url"); p.add_argument("--cookies")
    p = sub.add_parser("clip", help="ClipsAI only"); p.add_argument("workdir", type=Path); clip_args(p)
    p = sub.add_parser("post", help="Postiz only"); p.add_argument("workdir", type=Path); post_args(p)
    p = sub.add_parser("render", help="re-render clips after editing hooks/captions in clips.json")
    p.add_argument("workdir", type=Path)
    p.add_argument("--layout", choices=LAYOUTS, default=None)
    p.add_argument("--no-captions", dest="captions", action="store_false", default=None)
    edit_args(p, render=True)
    p = sub.add_parser("channels", help="list your Postiz channels and their ids")
    p.add_argument("--postiz-url")

    a = ap.parse_args()
    clip_kw = lambda: dict(pick=a.pick, min_len=a.min_len or 15,
                           max_len=a.max_len or (20 if a.pick == "hot" else 60), max_clips=a.max_clips,
                           layout="original" if a.no_resize else a.layout, aspect=a.aspect,
                           whisper_model=a.whisper_model, language=a.language, ui=a.ui,
                           batch_size=a.batch_size, skip_start=a.skip_start, captions=not a.no_captions,
                           caption_words=a.caption_words, caption_position=a.caption_position,
                           jumpcut=a.jumpcut, cold_open=a.cold_open, zoom=a.zoom, hook=a.hook,
                           hook_seconds=a.hook_seconds)

    if a.cmd == "download":
        print(download(a.url, a.cookies))
    elif a.cmd == "clip":
        clip(a.workdir, **clip_kw())
    elif a.cmd == "post":
        post(a.workdir, a)
    elif a.cmd == "render":
        rerender(a.workdir, {k: getattr(a, k) for k in ("layout", "captions", "jumpcut", "cold_open", "zoom",
                                                       "hook", "hook_seconds")})
    elif a.cmd == "channels":
        api = Postiz(a.postiz_url or os.environ["POSTIZ_API_URL"], os.environ["POSTIZ_API_KEY"])
        for i in api.integrations():
            print(f"{i['id']}\t{i['identifier']:<12}\t{i['name']}{'  (disabled)' if i.get('disabled') else ''}")
    elif a.cmd == "run":
        workdir = download(a.url, a.cookies)
        clip(workdir, **clip_kw())
        if a.skip_post:
            log(f"review {workdir / 'clips.json'}, then: python pipeline.py post {workdir} --integrations …")
        else:
            post(workdir, a)


if __name__ == "__main__":
    main()
