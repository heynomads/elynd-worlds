"""climb_days.py STAGES OUT: the climb's media for an app, day by day, from climb_weeks.py's paintings and motion.

OUT/day01.webp .. day30.webp   the view standing at each day (720x1280, no text: the app draws its own labels)
OUT/step02.mp4 .. step30.mp4   the walk into each day from the day before (a step; at a new week, the cloud)
OUT/finale.mp4                 day 30 made: the last view glows into the sunrise, turns to the way back, 30 lights
OUT/keepsake.webp              the view back with the 30 lights
OUT/manifest.json              file names, sizes, durations
Each clip starts on the day before's still and ends on its day's still, so the app can rest on the still after it.
Every still and clip is marked as made by AI, like every picture of a world (meta.encode, loop.mark_mp4). The climb
is made once and is the same for everyone: nothing of a listener's journey goes into it.

    python scripts/climb/climb_days.py climb out/climb-days"""
from __future__ import annotations
import json, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from elynd_worlds.loop import mark_mp4
from elynd_worlds.meta import encode

STAGES_DIR, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
sys.argv = [sys.argv[0], str(STAGES_DIR), str(OUT / "_unused")]
sys.path.insert(0, str(Path(__file__).parent))
import climb_weeks as cw                                # noqa: E402  (loads the paintings; renders nothing yet)

W, H, FPS = cw.W, cw.H, cw.FPS


def clean(img: Image.Image, day: int, key: str, factor: float | None = None) -> Image.Image:
    """The day's light, no label and no shade (the app draws those)."""
    a = np.asarray(img, dtype=np.float32) * (cw.grade(day, key) if factor is None else factor)
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


class Clip:
    def __init__(self, name: str):
        self.name, self.dir, self.n = name, Path(tempfile.mkdtemp(dir=OUT)), 0

    def put(self, img: Image.Image):
        img.save(self.dir / f"{self.n:05d}.jpg", quality=92); self.n += 1
        cw.n += 1                                       # climb_weeks.mist drifts with its frame counter

    def close(self) -> dict:
        path = OUT / self.name
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(self.dir / "%05d.jpg"),
                        # Standard video colour, tagged: untagged full-range BT.601 from the JPEG frames shows
                        # wrongly on some Android decoders, and the clip's last frame then jumps against its WebP still.
                        "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-color_range", "tv",
                        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                        "-c:v", "libx264", "-profile:v", "main", "-crf", "20", "-movflags",
                        "+faststart", "-an", str(path)], check=True)
        mark_mp4(path)
        shutil.rmtree(self.dir)
        return {"file": self.name, "bytes": path.stat().st_size, "seconds": round(self.n / FPS, 2)}


def still(img: Image.Image, day: int) -> dict:
    name = f"day{day:02d}.webp"
    (OUT / name).write_bytes(encode(img, "WEBP", quality=82, method=6))
    return {"file": name, "bytes": (OUT / name).stat().st_size}


manifest = {"mountain": "climb-1", "size": [W, H], "fps": FPS, "days": {}, "finale": None, "keepsake": None,
            "marked": "every still and clip carries the made-by-AI XMP (IPTC trainedAlgorithmicMedia)"}
days = manifest["days"]
ease, frames = cw.ease, cw.frames


def step_clip(w, key, day, tz):
    """The ordinary step from `day`-1 to `day` inside one painting; returns the new tz."""
    c = Clip(f"step{day:02d}.mp4")
    m = frames(0.85)
    for j in range(0, m + 1):
        t = ease(j / m)
        c.put(clean(w.render(tz + cw.STEP * t), day, key, cw.grade(day - 1, key) * (1 - t) + cw.grade(day, key) * t))
    days[day] = {"step": c.close()}
    return tz + cw.STEP


def cloud_clip(w, key, w2, key2, day, tz):
    """Into the cloud in `key` (the last day of its week, `day`-1) and out higher up in `key2` on `day`."""
    colour = cw.fog_colour(key, key2)
    c = Clip(f"step{day:02d}.mp4")
    m = frames(1.1)
    for j in range(0, m + 1):
        t = j / m
        c.put(clean(cw.mist(cw.fogged(w, tz + 0.16 * ease(t), cw.KMAX * t ** 3, colour), t ** 2), day - 1, key))
    a, b = cw.fogged(w, tz + 0.18, cw.KMAX, colour), cw.fogged(w2, 0.0, cw.KMAX, colour)
    m = frames(0.35)
    for j in range(1, m + 1):
        t = ease(j / m)
        c.put(clean(cw.mist(Image.blend(a, b, t), 1.0), day, key2,
                    cw.grade(day - 1, key) * (1 - t) + cw.grade(day, key2) * t))
    m = frames(1.1)
    for j in range(1, m + 1):
        t = j / m
        c.put(clean(cw.mist(cw.fogged(w2, cw.STEP * ease(t), cw.KMAX * (1 - t) ** 3, colour), (1 - t) ** 2), day, key2))
    days[day] = {"step": c.close()}
    return cw.STEP


def main():
    w = cw.Walker(cw.PICS["A"], pp=cw.STAGES[0][1])
    tz = 0.0
    days[1] = {"step": None, "still": still(clean(w.render(0.0), 1, "A"), 1)}
    for s, (key, _) in enumerate(cw.STAGES):
        first, last = cw.WEEKS[key]
        if key == "F":
            break
        for day in range(first + 1, last + 1):          # e.g. days 2-7 inside the forest
            tz = step_clip(w, key, day, tz)
            days[day]["still"] = still(clean(w.render(tz), day, key), day)
        nxt, pp = cw.STAGES[s + 1]
        w2 = cw.Walker(cw.PICS[nxt], pp=pp)
        tz = cloud_clip(w, key, w2, nxt, last + 1, tz)
        days[last + 1]["still"] = still(clean(w2.render(tz), last + 1, nxt), last + 1)
        del w
        w = w2
    # Day 30: the last steps up to the cairn (an ordinary step), then the finale once day 30 is made.
    c = Clip("step30.mp4")
    m = frames(0.85)
    for j in range(0, m + 1):
        t = ease(j / m)
        c.put(clean(w.render(tz + 0.22 * t), 30, "F", cw.grade(29, "F") * (1 - t) + cw.grade(30, "F") * t))
    days[30] = {"step": c.close()}
    tz += 0.22
    top = clean(w.render(tz), 30, "F")
    days[30]["still"] = still(top, 30)
    warm = np.array([255, 230, 186], dtype=np.float32)
    view = cw.load("G2").resize((W, H), Image.LANCZOS)
    lights = cw.lights_on(view)
    c = Clip("finale.mp4")
    for _ in range(frames(0.4)):
        c.put(top)
    m = frames(1.4)
    for j in range(1, m + 1):                           # into the sunrise glow
        t = j / m
        f = np.asarray(clean(w.render(tz + 0.08 * ease(t)), 30, "F"), dtype=np.float32)
        g = 0.88 * t ** 1.6
        c.put(Image.fromarray((f * (1 - g) + warm * g).clip(0, 255).astype(np.uint8)))
    total = frames(1.4) + frames(2.6) + frames(1.5)
    k = 0
    def back(levels, g):
        nonlocal k
        sc = 1.0 + 0.03 * k / (total - 1)              # the last frame is the keepsake exactly (zoom 1.03)
        f = np.asarray(cw.glow(cw.zoomed(view, sc), lights, levels, sc), dtype=np.float32)
        k += 1
        c.put(Image.fromarray((f * (1 - g) + warm * g).clip(0, 255).astype(np.uint8)))
    m = frames(1.4)
    for j in range(1, m + 1):                           # the glow settles: the way back down
        back([0] * 30, 0.88 * (1 - ease(j / m)))
    m = frames(2.6)
    for j in range(1, m + 1):                           # day 1's light first (far below), day 30's last
        t = j / m
        back([float(np.clip((t * 33 - (29 - i)) / 3, 0, 1)) for i in range(30)], 0.0)
    for _ in range(frames(1.5)):
        back([1.0] * 30, 0.0)
    manifest["finale"] = c.close()
    keep = cw.glow(cw.zoomed(view, 1.03), lights, [1.0] * 30, 1.03)
    (OUT / "keepsake.webp").write_bytes(encode(keep, "WEBP", quality=85, method=6))
    manifest["keepsake"] = {"file": "keepsake.webp", "bytes": (OUT / "keepsake.webp").stat().st_size}
    manifest["days"] = {str(d): days[d] for d in sorted(days)}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total_bytes = sum((v.get("still") or {}).get("bytes", 0) + ((v.get("step") or {}).get("bytes", 0))
                      for v in manifest["days"].values()) + manifest["finale"]["bytes"] + manifest["keepsake"]["bytes"]
    print(len(manifest["days"]), "days;", round(total_bytes / 1e6, 1), "MB in all;", OUT)


if __name__ == "__main__":
    main()
