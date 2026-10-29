"""The shared clip (loop.py): a moment without birds, a seam that cannot be seen or heard, the made-by-AI mark."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from elynd_worlds import cli, loop
from elynd_worlds.meta import is_marked

FFMPEG = shutil.which("ffmpeg") is not None


def boxes(data: bytes) -> list[tuple[str, bytes]]:
    out, i = [], 0
    while i < len(data):
        size, kind = int.from_bytes(data[i:i + 4], "big"), data[i + 4:i + 8].decode("latin-1")
        assert size >= 8 and i + size <= len(data), f"a broken box at {i}"
        out.append((kind, data[i + 8:i + size]))
        i += size
    return out


@pytest.mark.parametrize("kind", ["gulls", "flock"])
@pytest.mark.parametrize("seed", [0.0, 0.1234, 0.5, 0.7192, 0.9448])
def test_the_clip_starts_where_no_bird_crosses_it(kind, seed):
    clock = loop.quiet_window({"birds": kind, "seed": seed})
    span = loop.LOOP_SECONDS + loop.BLEND_SECONDS
    assert clock >= loop.WAKE_SECONDS
    assert not any(a < clock + span and b > clock for a, b in loop.bird_passes(kind, seed, clock + span + 100))


def test_the_dog_worlds_gulls_first_pass_at_23_6_s_leaves_the_first_clip_free():
    assert [round(a, 1) for a, _ in loop.bird_passes("gulls", 0.7192, 200)][:2] == [23.6, 122.7]
    assert loop.quiet_window({"birds": "gulls", "seed": 0.7192}) == 4


def test_no_shooting_star_or_fish_ring_where_the_seam_blends():
    for seed in (0.0, 0.3, 0.6, 0.9):
        clock = loop.quiet_window({"night": True, "fish": True, "seed": seed})
        ends = ((clock, clock + loop.BLEND_SECONDS), (clock + loop.LOOP_SECONDS, clock + loop.LOOP_SECONDS + loop.BLEND_SECONDS))
        events = loop.shooting_stars(seed, clock + 60) + loop.fish_rises(clock + 60)
        assert not any(a < z1 and b > z0 for a, b in events for z0, z1 in ends)


def test_a_world_without_life_starts_after_the_wake_up():
    assert loop.quiet_window(None) == 4 and loop.quiet_window({}) == 4


def write_frames(folder: Path, n: int, size=(36, 64)) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", size, (i % 256, (2 * i) % 256, 100)).save(folder / f"{i + 1:04d}.png")
    return sorted(folder.glob("*.png"))


def test_the_last_frame_runs_straight_into_the_first(tmp_path):
    paths = write_frames(tmp_path, 420)
    frames = list(loop.loop_frames(paths))
    n = round(loop.LOOP_SECONDS * loop.FPS)
    assert len(frames) == n
    red = [int(f[0, 0, 0]) for f in frames]
    assert red[0] == n % 256                    # the clip opens on the moment just after its own end
    assert red[-1] == (n - 1) % 256             # and ends on the frame before it
    assert red[45:] == [k % 256 for k in range(45, n)]   # after the blend, the frames as rendered


def test_too_few_frames_is_an_error_not_a_jumping_clip(tmp_path):
    with pytest.raises(ValueError):
        list(loop.loop_frames(write_frames(tmp_path, 400)))


def test_the_mp4_carries_the_made_by_ai_mark_at_its_end(tmp_path):
    mp4 = tmp_path / "x.mp4"
    mp4.write_bytes((16).to_bytes(4, "big") + b"ftypisom" + b"\0" * 4 + (12).to_bytes(4, "big") + b"mdat" + b"abcd")
    loop.mark_mp4(mp4)
    found = boxes(mp4.read_bytes())
    assert [k for k, _ in found] == ["ftyp", "mdat", "uuid"]
    assert found[-1][1][:16] == loop.XMP_UUID and is_marked(found[-1][1])


def sine(path: Path, seconds: float) -> Path:
    t = np.arange(int(seconds * loop.AUDIO_RATE)) / loop.AUDIO_RATE
    pcm = (8000 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(loop.AUDIO_RATE)
        w.writeframes(pcm.tobytes())
    return path


@pytest.mark.skipif(not FFMPEG, reason="needs ffmpeg")
def test_the_music_bed_loops_without_a_gap(tmp_path):
    bed = loop.looped_music(sine(tmp_path / "m.wav", 20), 2.0, tmp_path / "bed.wav")
    with wave.open(str(bed)) as w:
        assert w.getnframes() == round(loop.LOOP_SECONDS * loop.AUDIO_RATE) and w.getnchannels() == 2
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, 2).astype(np.float32)
    # never quiet at the seam: the end and the start both carry the music
    assert np.abs(pcm[-480:]).mean() > 2000 and np.abs(pcm[:480]).mean() > 2000


@pytest.mark.skipif(not FFMPEG, reason="needs ffmpeg")
def test_a_world_becomes_a_marked_12_s_clip_through_the_cli(tmp_path, monkeypatch):
    world = tmp_path / "quiet_lake"
    world.mkdir()
    (world / "world.json").write_text(json.dumps({"title": "Quiet Lake", "living": {"birds": "gulls", "seed": 0.7192}}))
    seen = {}

    def fake_render(world_dir, frames_dir, clock, seconds, size, ratio, flutter, package):
        seen.update(clock=clock, seconds=seconds, size=size, ratio=ratio, package=package)
        return write_frames(frames_dir, math.ceil(seconds) * loop.FPS, (72, 128))

    monkeypatch.setattr(loop, "render", fake_render)
    music = sine(tmp_path / "music.wav", 80)
    assert cli.main(["loop", str(world), "--out-dir", str(tmp_path / "loops"), "--music", str(music),
                     "--package", str(tmp_path / "pkg")]) == 0
    clip = tmp_path / "loops" / "quiet_lake.mp4"
    assert seen == {"clock": 4, "seconds": 14.0, "size": (360, 640), "ratio": 3.0, "package": tmp_path / "pkg"}
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_name,width,height",
                                       "-of", "json", str(clip)], capture_output=True, text=True, check=True).stdout)
    assert abs(float(probe["format"]["duration"]) - loop.LOOP_SECONDS) < 0.05
    assert {s["codec_name"] for s in probe["streams"]} == {"h264", "aac"}
    assert [k for k, _ in boxes(clip.read_bytes())][-1] == "uuid" and is_marked(clip.read_bytes())


def test_the_clip_renders_with_the_clip_periods_and_only_the_clip(tmp_path, monkeypatch):
    # A meditation's world never repeats (the live periods, the scene's default); only the shared clip's renderer
    # asks for the 12-s periods (a product decision, 29 Oct 2026: the world during a session must not loop).
    seen = []
    monkeypatch.setattr(loop.subprocess, "run", lambda command, **kw: seen.append(command))
    loop.render(tmp_path / "world", tmp_path / "frames", clock=4, seconds=13.5, flutter="flutter")
    assert "--dart-define=LOOP=true" in seen[0]
    motion = (Path(loop.SCENE_PACKAGE) / "lib/src/motion.dart").read_text()
    assert "static WorldPeriods periods = WorldPeriods.live;" in motion
