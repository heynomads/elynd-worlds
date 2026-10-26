"""When the world breathes, and how much of it is visible.

People meditate with their eyes closed, so a world that stays bright the whole
session would be wrong. The timeline follows the session second by second:

* visibility (0-1): the whole world while the listener arrives (45 s), fading
  to a faint glow while their eyes are closed, rising again in long silences
  (30 s or more) and over the last 30 s. A sleep world never rises above a
  glow and ends dark.
* breath (0 exhaled - 1 inhaled): a slow wave, about 5.5 breaths a minute in
  silences and gentler while the voice speaks. A spoken cue ("breathe in",
  "breathe out") overrides the wave: the world inhales over 4 s and exhales
  over 6 s exactly when the voice says so.

The app builds the same timeline on the phone from what it already knows (the
length, the voice's silences, the cues); `flutter/elynd_worlds_scene` mirrors
this file and a parity test keeps the two identical.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

RESONANT_PERIOD = 60 / 5.5
SPEECH_PERIOD = 10.0
INHALE_SECONDS = 4.0
EXHALE_SECONDS = 6.0
ARRIVE_SECONDS = 45.0
FADE_SECONDS = 20.0
RETURN_SECONDS = 30.0
LONG_SILENCE = 30.0
GLOW = 0.25
SLEEP_GLOW = 0.15


@dataclass(frozen=True)
class Silence:
    start: float
    end: float


@dataclass(frozen=True)
class Cue:
    at: float
    kind: str  # "inhale" | "exhale"


def _silence_at(silences: list[Silence], t: float) -> Silence | None:
    for s in silences:
        if s.start <= t < s.end:
            return s
    return None


def breath_at(t: float, silences: list[Silence], cues: list[Cue]) -> float:
    for cue in cues:
        if cue.kind == "inhale" and cue.at <= t < cue.at + INHALE_SECONDS:
            return (t - cue.at) / INHALE_SECONDS
        if cue.kind == "exhale" and cue.at <= t < cue.at + EXHALE_SECONDS:
            return 1.0 - (t - cue.at) / EXHALE_SECONDS
    quiet = _silence_at(silences, t) is not None
    period = RESONANT_PERIOD if quiet else SPEECH_PERIOD
    depth = 0.8 if quiet else 0.3
    return 0.5 + 0.5 * depth * math.sin(2 * math.pi * t / period - math.pi / 2)


def visibility_at(t: float, duration: float, silences: list[Silence], sleep: bool) -> float:
    if t < ARRIVE_SECONDS:
        v = 1.0
    elif t < ARRIVE_SECONDS + FADE_SECONDS:
        v = 1.0 - (1.0 - GLOW) * (t - ARRIVE_SECONDS) / FADE_SECONDS
    else:
        v = GLOW
    silence = _silence_at(silences, t)
    if silence is not None and silence.end - silence.start >= LONG_SILENCE:
        ramp = min(t - silence.start, silence.end - t, 8.0) / 8.0
        v = max(v, GLOW + 0.5 * ramp)
    left = duration - t
    if left < RETURN_SECONDS:
        v = max(v, 1.0 - (1.0 - GLOW) * left / RETURN_SECONDS)
    if sleep and t >= ARRIVE_SECONDS:
        v = min(v, SLEEP_GLOW)
        if left < RETURN_SECONDS:
            v = SLEEP_GLOW * max(left, 0.0) / RETURN_SECONDS
    return v


def build(duration: float, silences: list[Silence] = (), cues: list[Cue] = (), sleep: bool = False) -> dict:
    """One frame per second: {"t", "breath", "visibility"}."""
    silences, cues = list(silences), sorted(cues, key=lambda c: c.at)
    frames = []
    for second in range(int(math.ceil(duration)) + 1):
        t = float(second)
        frames.append({
            "t": second,
            "breath": round(min(max(breath_at(t, silences, cues), 0.0), 1.0), 3),
            "visibility": round(min(max(visibility_at(t, duration, silences, sleep), 0.0), 1.0), 3),
        })
    return {"version": 1, "duration": duration, "sleep": sleep, "frames": frames}
