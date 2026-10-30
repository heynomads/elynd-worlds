"""climb_weeks.py STAGES OUT: a journey's thirty days as one climb with ONE PAINTING PER WEEK, as one video with each
day's label (OUT/climb30_weeks.mp4) and the view back with its 30 lights (OUT/still_day30_lights.jpg).

The paintings are STAGES/{A,B,C,E,F,G2}.png (or .jpg), from climb_stage.py: week 1 the forest (A), week 2 the meadow
(B), week 3 the lower slope (C), week 4 the ridge (E), days 29-30 the top (F), then the view back down (G2). Inside a
painting each day is one small step into it (walker.py); a longer step tears the picture, so the distance comes from
the paintings. The cloud comes only when a new week starts, so each jump means a new week, higher up: depth fog (far
things fade first, near things last), the next painting under the same fog, then the fog clears near-first. The light
rises a little each day. Day 30 is the top: the sunrise glow, then the view back down (G2), where the 30 lights come
on along the path, day 1's far below and day 30's at the viewer's feet.

    python scripts/climb/climb_weeks.py climb out/climb
    python scripts/climb/climb_weeks.py climb out/climb --lights      # only the day-30 still, to check the lights

Needs ffmpeg for the video; climb_days.py makes the same climb as an app's per-day stills and clips."""
from __future__ import annotations
import math, subprocess, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).parent))
from walker import Walker, W, H, DEV, depth

ST, OUT = Path(sys.argv[1]), Path(sys.argv[2])
FR = OUT / "frames"; FR.mkdir(parents=True, exist_ok=True)
FPS, STEP, KMAX = 30, 0.085, 2.6                        # seven days in one painting: a slightly shorter step
TITLE = "Speaking up at work"                           # an example journey's title, on the video's labels
# Where each painting's path leads (the point the walk moves toward), as fractions of width and height.
STAGES = [("A", (0.50, 0.57)), ("B", (0.53, 0.52)), ("C", (0.66, 0.52)), ("E", (0.54, 0.40)), ("F", (0.51, 0.40))]
WEEKS = {"A": (1, 7), "B": (8, 14), "C": (15, 21), "E": (22, 28), "F": (29, 30)}
FONTS = Path(__file__).resolve().parents[2] / "elynd_worlds" / "fonts"        # the bundled Poppins (SIL OFL 1.1)
font = ImageFont.truetype(str(FONTS / "Poppins-Medium.ttf"), 24)
big = ImageFont.truetype(str(FONTS / "Poppins-SemiBold.ttf"), 40)


def load(key):
    return Image.open(next(p for p in (ST / f"{key}.png", ST / f"{key}.jpg") if p.exists())).convert("RGB")


def luma(img):
    a = np.asarray(img.resize((90, 160)), dtype=np.float32)
    return float((a @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)).mean())


PICS = {k: load(k) for k, _ in STAGES}
LUMA = {k: luma(v) for k, v in PICS.items()}
print("luma", {k: round(v, 1) for k, v in LUMA.items()})


def grade(day: int, key: str) -> float:
    """A little more light each day: the brightness rises evenly from day 1's painting to day 30's, within reach of
    each painting's own light (never more than 10 % away from it)."""
    lo, hi = LUMA["A"], LUMA["F"]
    target = lo + (hi - lo) * (day - 1) / 29
    return float(np.clip(target / LUMA[key], 0.90, 1.10))


def fog_colour(a: str, b: str) -> torch.Tensor:
    """A pale version of the two paintings' skies."""
    sky = np.mean([np.asarray(PICS[k].resize((90, 160)), dtype=np.float32)[:56].reshape(-1, 3).mean(0) for k in (a, b)], 0)
    return torch.tensor((0.45 * sky + 0.55 * np.array([236, 238, 242])) / 255, device=DEV, dtype=torch.float32)


def fogged(w: Walker, tz: float, k: float, colour: torch.Tensor | None) -> Image.Image:
    if colour is None or k <= 1e-4:
        return w.render(tz)
    a = torch.exp(-k * (w.z - tz).clamp(min=0.12))[:, None]
    kept = w.c
    w.c = kept * a + colour[None] * (1 - a)
    try:
        return w.render(tz)
    finally:
        w.c = kept


NOISE: np.ndarray | None = None


def mist(img: Image.Image, amount: float) -> Image.Image:
    """Soft brighter and darker patches drifting sideways through the fog, so a cloud never looks like a flat grey
    screen (run 2, 30 Oct). `amount`: how much fog there is (0-1)."""
    global NOISE
    if NOISE is None:
        rng = np.random.default_rng(7)
        a = np.zeros((H, W + 600), np.float32)
        for gh, gw, amp in ((12, 6, 1.0), (24, 12, 0.5), (48, 24, 0.25)):    # wider than tall, as clouds are
            g = Image.fromarray(rng.standard_normal((gh, gw)).astype(np.float32))
            a += amp * np.asarray(g.resize((W + 600, H), Image.BICUBIC), dtype=np.float32)
        NOISE = a / np.abs(a).max()
    if amount <= 0:
        return img
    s = int(n * 2.5) % 600
    f = np.asarray(img, dtype=np.float32) * (1 + 0.10 * amount * NOISE[:, s:s + W, None])
    return Image.fromarray(f.clip(0, 255).astype(np.uint8))


def look(img: Image.Image, day: int, factor: float) -> Image.Image:
    a = np.asarray(img, dtype=np.float32) * factor
    f = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    shade = Image.new("RGB", (W, 260), (8, 24, 26))
    f.paste(shade, (0, 0), Image.fromarray(np.linspace(150, 0, 260).astype(np.uint8)[:, None].repeat(W, 1)))
    d = ImageDraw.Draw(f)
    d.text((36, 70), f"DAY {day} OF 30" + (f"  ·  WEEK {(day - 1) // 7 + 1}" if day <= 28 else ""), font=font,
           fill=(255, 255, 255))
    d.text((36, 104), TITLE, font=big, fill=(255, 255, 255))
    return f


n = 0
def put(img):
    global n
    img.save(FR / f"{n:05d}.jpg", quality=90); n += 1


def ease(t): return t * t * (3 - 2 * t)


def frames(seconds): return max(1, int(round(seconds * FPS)))


def walk_days(w, key, first_day, days, tz0):
    """`days` daily steps inside one painting, starting on `first_day`'s view at tz0; returns the last day's tz."""
    tz = tz0
    for k in range(days):
        day = first_day + k
        for _ in range(frames(0.45)):
            put(look(w.render(tz), day, grade(day, key)))
        m = frames(0.85)
        for j in range(1, m + 1):
            t = ease(j / m)
            d = day if t < .5 else day + 1
            put(look(w.render(tz + STEP * t), d, grade(day, key) * (1 - t) + grade(day + 1, key) * t))
        tz += STEP
    return tz


def through_the_cloud(w, key, w2, key2, day, tz):
    """Day `day` (the last in painting `key`) to day+1 (the first in `key2`): into the cloud, out higher up."""
    colour = fog_colour(key, key2)
    for _ in range(frames(0.45)):
        put(look(w.render(tz), day, grade(day, key)))
    m = frames(1.1)
    for j in range(1, m + 1):                       # into the cloud: far things fade first, near ones last
        t = j / m
        put(look(mist(fogged(w, tz + 0.16 * ease(t), KMAX * t ** 3, colour), t ** 2), day, grade(day, key)))
    a = fogged(w, tz + 0.18, KMAX, colour)
    b = fogged(w2, 0.0, KMAX, colour)
    m = frames(0.35)
    for j in range(1, m + 1):                       # inside the cloud: the next painting under the same fog
        t = ease(j / m)
        put(look(mist(Image.blend(a, b, t), 1.0), day if t < .5 else day + 1,
                 grade(day, key) * (1 - t) + grade(day + 1, key2) * t))
    m = frames(1.1)
    for j in range(1, m + 1):                       # out of the cloud: near things first
        t = j / m
        put(look(mist(fogged(w2, STEP * ease(t), KMAX * (1 - t) ** 3, colour), (1 - t) ** 2), day + 1,
                 grade(day + 1, key2)))
    return STEP


# ─── The 30 lights on the view back (G2) ───
# The path from the viewer's feet (day 30) down the ridge, the zigzags, through the trees and across the valley
# (day 1), traced on a 720x1280 copy of our G2 (another painting needs its own trace) and snapped to the path's
# pale stones below.
PATH = [(385, 1215), (392, 1150), (395, 1090), (385, 1030), (362, 975), (340, 925), (325, 880), (330, 850), (352, 822),
        (385, 795), (418, 765), (442, 735), (456, 705), (445, 684), (420, 672), (390, 660), (360, 648), (330, 637),
        (300, 625), (282, 614), (300, 598), (330, 585), (360, 572), (385, 560), (399, 547), (390, 540), (365, 532),
        (340, 520), (315, 508), (303, 502), (325, 492), (350, 485), (375, 475), (400, 462), (415, 455), (405, 447),
        (390, 440), (378, 430), (371, 415), (372, 400), (377, 385), (380, 370), (370, 357), (355, 345), (342, 337),
        (339, 320), (349, 310), (360, 300), (362, 290), (350, 280), (340, 270), (342, 260), (350, 250)]


def lights_on(view: Image.Image) -> list[tuple[float, float, float]]:
    """30 points (x, y, radius) along PATH, day 30 first: spaced so the far ones sit closer on screen (as steps do),
    each nudged onto the palest, least saturated spot within 8 px (the path's stones; blurred, so a white flower in
    the grass never wins over the path)."""
    a = np.asarray(view, dtype=np.float32) / 255
    pale = a.mean(2) - (a.max(2) - a.min(2)) * 1.5
    pale = torch.nn.functional.avg_pool2d(torch.from_numpy(pale)[None, None], 7, stride=1, padding=3)[0, 0].numpy()
    z = depth(view)
    pts = np.array(PATH, dtype=np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    dense = np.concatenate([np.linspace(pts[i], pts[i + 1], max(2, int(seg[i])), endpoint=False) for i in range(len(seg))]
                           + [pts[-1:]])
    zs = np.array([z[int(min(H - 1, y)), int(min(W - 1, x))] for x, y in dense])
    step = np.linalg.norm(np.diff(dense, axis=0), axis=1) * np.sqrt(zs[1:])
    s = np.concatenate([[0], np.cumsum(step)])
    out = []
    for target in np.linspace(0, s[-1], 30):
        x, y = dense[int(np.searchsorted(s, target).clip(0, len(dense) - 1))]
        r = 8
        win = pale[int(y) - r:int(y) + r + 1, int(x) - r:int(x) + r + 1]
        dy, dx = np.unravel_index(int(np.argmax(win)), win.shape)
        zz = z[int(y), int(x)]
        out.append((x + dx - r, y + dy - r, float(np.clip(10 / zz ** 0.75, 1.8, 9.0))))
    return out


def glow(img: Image.Image, lights, levels, scale=1.0, centre=(W / 2, H / 2)) -> Image.Image:
    """Warm lights added onto the picture (a soft halo and a bright core each)."""
    a = np.asarray(img, dtype=np.float32) / 255
    colour = np.array([1.0, 0.84, 0.56], dtype=np.float32)
    for (x, y, r), level in zip(lights, levels):
        if level <= 0:
            continue
        x, y, r = centre[0] + (x - centre[0]) * scale, centre[1] + (y - centre[1]) * scale, r * scale
        R = int(r * 5) + 2
        x0, x1, y0, y1 = max(0, int(x) - R), min(W, int(x) + R + 1), max(0, int(y) - R), min(H, int(y) + R + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        g = 0.7 * np.exp(-d2 / (2 * (2.0 * r) ** 2)) + 1.2 * np.exp(-d2 / (2 * (0.55 * r) ** 2))
        a[y0:y1, x0:x1] += (level * g)[..., None] * colour
    return Image.fromarray((a.clip(0, 1) * 255).astype(np.uint8))


def zoomed(img: Image.Image, scale: float) -> Image.Image:
    w, h = int(round(W * scale)), int(round(H * scale))
    big_img = img.resize((w, h), Image.LANCZOS)
    left, top = (w - W) // 2, (h - H) // 2
    return big_img.crop((left, top, left + W, top + H))


def the_top(w, tz):
    """Day 29 to 30 on the last painting: the last steps into the sunrise glow, then the view back down with the 30
    lights coming on from day 1 (far below) to day 30 (at the viewer's feet)."""
    warm = np.array([255, 230, 186], dtype=np.float32)
    for _ in range(frames(0.45)):
        put(look(w.render(tz), 29, grade(29, "F")))
    m = frames(1.4)
    for j in range(1, m + 1):
        t = j / m
        f = np.asarray(w.render(tz + 0.22 * ease(t)), dtype=np.float32)
        g = 0.88 * t ** 1.6
        put(look(Image.fromarray((f * (1 - g) + warm * g).clip(0, 255).astype(np.uint8)), 29 if t < .8 else 30,
                 grade(29, "F")))
    view = load("G2").resize((W, H), Image.LANCZOS)
    lights = lights_on(view)
    total = frames(1.4) + frames(2.6) + frames(3.0)
    k = 0
    def frame(levels, g):
        nonlocal k
        s = 1.0 + 0.03 * k / total
        f = np.asarray(glow(zoomed(view, s), lights, levels, s), dtype=np.float32)
        k += 1
        put(look(Image.fromarray((f * (1 - g) + warm * g).clip(0, 255).astype(np.uint8)), 30, 1.0))
    m = frames(1.4)
    for j in range(1, m + 1):                       # the glow settles: the view back down
        frame([0] * 30, 0.88 * (1 - ease(j / m)))
    m = frames(2.6)
    for j in range(1, m + 1):                       # day 1's light first (far below), day 30's last
        t = j / m
        levels = [float(np.clip((t * 33 - (29 - i)) / 3, 0, 1)) for i in range(30)]
        frame(levels, 0.0)
    m = frames(3.0)
    for j in range(m):                              # all thirty, breathing a little
        frame([0.9 + 0.1 * math.sin(2 * math.pi * (j / FPS * 0.6 + i * 0.37)) for i in range(30)], 0.0)
    return view, lights


def main():
    tz = 0.0
    w = Walker(PICS["A"], pp=STAGES[0][1])
    for _ in range(frames(0.8)):
        put(look(w.render(0.0), 1, grade(1, "A")))
    for s, (key, _) in enumerate(STAGES):
        first, last = WEEKS[key]
        if key == "F":                              # day 29 at the foot of the top, then day 30
            view, lights = the_top(w, tz)
            break
        tz = walk_days(w, key, first, last - first, tz)   # e.g. days 1-7: six steps
        nxt, pp = STAGES[s + 1]
        w2 = Walker(PICS[nxt], pp=pp)
        tz = through_the_cloud(w, key, w2, nxt, last, tz)
        del w
        if DEV == "mps":
            torch.mps.empty_cache()
        w = w2
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(FR / "%05d.jpg"), "-c:v",
                    "libx264", "-pix_fmt", "yuv420p", "-crf", "19", str(OUT / "climb30_weeks.mp4")], check=True)
    glow(view, lights, [1.0] * 30).save(OUT / "still_day30_lights.jpg", quality=92)
    print(n, "frames", round(n / FPS, 1), "s", OUT / "climb30_weeks.mp4")


if __name__ == "__main__":
    if sys.argv[3:] == ["--lights"]:                # only the day-30 still, to check the lights sit on the path
        view = load("G2").resize((W, H), Image.LANCZOS)
        glow(view, lights_on(view), [1.0] * 30).save(OUT / "still_day30_lights.jpg", quality=92)
    else:
        main()
