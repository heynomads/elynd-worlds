"""A world's shared clip: 12 seconds that loop without a seam, for the share card to play under its words.

The world is drawn by the phone's own scene code (`flutter/elynd_worlds_scene`, test/render_preview_test.dart) from
a moment no bird crosses, at the share card's 9:16; every object's motion repeats every 12 s (motion.dart), and the
water and the sky blend their last 1.5 s into their first. It is encoded as H.264 at a bitrate a phone can send
anywhere, with a bed of music that loops the same way if one is given, and marked as made by AI like every picture
of a world (the XMP packet in a 'uuid' box appended to the MP4: appending moves no chunk, so the file stays valid).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import wave
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .meta import AI_MADE_XMP

LOOP_SECONDS = 12.0
BLEND_SECONDS = 1.5
FPS = 30
WAKE_SECONDS = 4.0           # the scene's first seconds, before its life is settled
FISH_RING_SECONDS = 6.0      # how long a fish's rings spread on the water
XMP_UUID = bytes.fromhex("BE7ACFCB97A942E89C71999491E3AFAC")  # XMP in ISO media files (Adobe)
AUDIO_RATE = 48000
SCENE_PACKAGE = Path(__file__).resolve().parent.parent / "flutter" / "elynd_worlds_scene"


# ---- when a world's life happens: the shader's own timing (living.frag) ----
def _fract(x: float) -> float:
    return x - math.floor(x)


def _hash(x: float, y: float) -> float:
    """The shader's hash (Dave Hoskins' hash without sine), so the clip is aimed where the phone has its life."""
    p = [_fract(x * 0.1031), _fract(y * 0.1031), _fract(x * 0.1031)]
    d = p[0] * (p[1] + 33.33) + p[1] * (p[2] + 33.33) + p[2] * (p[0] + 33.33)
    p = [v + d for v in p]
    return _fract((p[0] + p[1]) * p[2])


def bird_passes(kind: str, seed: float, until: float) -> list[tuple[float, float]]:
    """(start, end) of every flock or gull pass across the sky until `until` seconds of the world's clock."""
    if kind not in ("flock", "gulls"):
        return []
    gulls = 1.0 if kind == "gulls" else 0.0
    period = 70.0 + 25.0 * gulls
    passes = []
    for slot in range(int(until // period) + 1):
        start = slot * period + 6.0 + 22.0 * _hash(slot + 1.0, seed * 97.0 + 3.0)
        crossing = (14.0 + 20.0 * gulls) * (0.85 + 0.3 * _hash(float(slot), 7.0 + seed))
        passes.append((start, start + crossing))
    return passes


def shooting_stars(seed: float, until: float) -> list[tuple[float, float]]:
    """(start, end) of the faint shooting stars of a night sky."""
    stars = []
    for slot in range(int(until // 47) + 1):
        if _hash(float(slot), 83.0 + seed) > 0.6:
            continue
        start = slot * 47.0 + 5.0 + 30.0 * _hash(float(slot), 89.0)
        stars.append((start, start + 0.8))
    return stars


def fish_rises(until: float) -> list[tuple[float, float]]:
    """(start, end) of a fish's rings on calm water."""
    rises = []
    for slot in range(int(until // 11) + 1):
        if _hash(float(slot), 61.0) > 0.75:
            continue
        start = slot * 11.0 + 9.0 * _hash(float(slot), 67.0)
        rises.append((start, start + FISH_RING_SECONDS))
    return rises


def quiet_window(living: dict[str, Any] | None, length: float = LOOP_SECONDS, blend: float = BLEND_SECONDS,
                 earliest: float = WAKE_SECONDS, until: float = 1200.0) -> int:
    """The first whole second of the world's clock from which a clip of `length` + `blend` seconds has no bird in
    the sky at all (a pass takes 12-39 s, so it can only be whole or absent: half of one would jump at the seam)
    and no shooting star or fish ring where the seam blends two moments (either end)."""
    living = living or {}
    span = length + blend
    seed = float(living.get("seed") or 0.0)
    birds = bird_passes(str(living.get("birds") or "none"), seed, until + span)
    brief = (shooting_stars(seed, until + span) if living.get("night") else []) + \
            (fish_rises(until + span) if living.get("fish") else [])
    for clock in range(math.ceil(earliest), int(until) + 1):
        if any(a < clock + span and b > clock for a, b in birds):
            continue
        ends = ((clock, clock + blend), (clock + length, clock + span))
        if any(a < z1 and b > z0 for a, b in brief for z0, z1 in ends):
            continue
        return clock
    raise ValueError("no quiet window in the world's first 20 minutes")


# ---- the frames: the phone's scene code, then the seam ----
def render(world_dir: Path, frames_dir: Path, clock: int, seconds: float, size: tuple[int, int] = (360, 640),
           ratio: float = 3.0, flutter: str | None = None, package: Path | None = None) -> list[Path]:
    """The world drawn by the scene package's renderer, `seconds` from `clock`, 30 frames a second, at
    `size` points x `ratio` pixels each (360 x 640 x 3 = the share card's 1080 x 1920)."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    command = [
        flutter or os.environ.get("FLUTTER", "flutter"), "test", "test/render_preview_test.dart", "--timeout", "none",
        f"--dart-define=WORLD_DIR={Path(world_dir).resolve()}", f"--dart-define=FRAMES_DIR={frames_dir.resolve()}",
        f"--dart-define=SECONDS={math.ceil(seconds)}", f"--dart-define=CLOCK={int(clock)}",
        f"--dart-define=RATIO={ratio}", "--dart-define=SILENT=true", f"--dart-define=SIZE={size[0]}x{size[1]}",
        "--dart-define=LOOP=true",
    ]
    subprocess.run(command, cwd=package or SCENE_PACKAGE, check=True, capture_output=True, stdin=subprocess.DEVNULL)
    return sorted(frames_dir.glob("*.png"))


def loop_frames(paths: list[Path], length: float = LOOP_SECONDS, blend: float = BLEND_SECONDS,
                fps: int = FPS) -> Iterator[np.ndarray]:
    """The clip's frames: the first `blend` seconds are blended with the moments just after its end, so its last
    frame runs straight into its first."""
    n, b = round(length * fps), round(blend * fps)
    if len(paths) < n + b:
        raise ValueError(f"{len(paths)} frames, a loop needs {n + b}")
    for k in range(n):
        frame = np.asarray(Image.open(paths[k]).convert("RGB"), dtype=np.float32)
        if k < b:
            w = k / b
            w = w * w * (3 - 2 * w)
            later = np.asarray(Image.open(paths[k + n]).convert("RGB"), dtype=np.float32)
            frame = frame * w + later * (1 - w)
        yield np.clip(np.rint(frame), 0, 255).astype(np.uint8)


# ---- the sound: a bed of music that loops the same way ----
def looped_music(music: Path, start: float, out_wav: Path, length: float = LOOP_SECONDS,
                 blend: float = BLEND_SECONDS) -> Path:
    """`length` seconds of `music` from `start`, its first `blend` seconds blended with the music just after its
    end (equal power), as a WAV: played in a loop it never stops or jumps."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(length + blend), "-i", str(music),
                          "-f", "s16le", "-ac", "2", "-ar", str(AUDIO_RATE), "-"],
                         check=True, capture_output=True).stdout
    pcm = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2).astype(np.float32)
    n, b = round(length * AUDIO_RATE), round(blend * AUDIO_RATE)
    if len(pcm) < n + b:
        raise ValueError(f"{len(pcm) / AUDIO_RATE:.1f} s of music from {start} s; a loop needs {length + blend}")
    body, tail = pcm[:n].copy(), pcm[n:n + b]
    x = np.linspace(0.0, 1.0, b, endpoint=False)[:, None]
    body[:b] = body[:b] * np.sin(0.5 * np.pi * x) + tail * np.cos(0.5 * np.pi * x)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(AUDIO_RATE)
        w.writeframes(np.clip(np.rint(body), -32768, 32767).astype(np.int16).tobytes())
    return out_wav


def mark_mp4(path: Path) -> None:
    """Mark the MP4 as made by AI: the same XMP the pictures carry, in a 'uuid' box appended at its end."""
    box = XMP_UUID + AI_MADE_XMP
    with open(path, "ab") as f:
        f.write((8 + len(box)).to_bytes(4, "big") + b"uuid" + box)


def encode(frames: Iterable[np.ndarray], size: tuple[int, int], out: Path, fps: int = FPS, bitrate: str = "3M",
           sound: Path | None = None, length: float = LOOP_SECONDS) -> Path:
    """H.264 High (what phones and every social network take), a keyframe every second (no quality jump at the
    seam), and the made-by-AI mark."""
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-"]
    if sound:
        command += ["-i", str(sound), "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "128k"]
    rate = float(bitrate.rstrip("Mm"))
    command += ["-c:v", "libx264", "-profile:v", "high", "-preset", "medium", "-b:v", bitrate,
                "-maxrate", f"{rate * 1.5:g}M", "-bufsize", f"{rate * 2:g}M", "-g", str(fps), "-keyint_min", str(fps),
                "-sc_threshold", "0", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
                "-color_trc", "bt709", "-movflags", "+faststart", "-t", str(length), str(out)]
    ff = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert ff.stdin is not None
    for frame in frames:
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError(f"ffmpeg could not encode {out}")
    mark_mp4(out)
    return out


def make_loop(world_dir: Path, out: Path, size: tuple[int, int] = (360, 640), ratio: float = 3.0,
              bitrate: str = "3M", music: Path | None = None, music_from: float = 60.0, flutter: str | None = None,
              package: Path | None = None, renderer=None) -> dict[str, Any]:
    """`world_dir`'s shared clip at `out`; what was made, for the log."""
    manifest = json.loads((Path(world_dir) / "world.json").read_text())
    clock = quiet_window(manifest.get("living"))
    with tempfile.TemporaryDirectory(prefix="elynd-loop-") as tmp:
        frames = (renderer or render)(Path(world_dir), Path(tmp) / "frames", clock, LOOP_SECONDS + BLEND_SECONDS + 0.5,
                          size=size, ratio=ratio, flutter=flutter, package=package)
        first = Image.open(frames[0])
        sound = looped_music(music, music_from, Path(tmp) / "bed.wav") if music else None
        encode(loop_frames(frames), first.size, out, bitrate=bitrate, sound=sound)
    return {"world": Path(world_dir).name, "title": manifest.get("title"), "clock": clock, "size": list(first.size),
            "bytes": out.stat().st_size, "sound": bool(music), "out": str(out)}
