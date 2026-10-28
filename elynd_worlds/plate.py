"""The plate: the painting as it would look without its moving subjects.

The layers move a few pixels as such things move. Wherever a subject moves
away, the plate shows through, so it must hold what lies behind the subject
there. A still subject (a stone, a post, a lantern) never moves away: it stays
in the plate exactly as painted, and nothing is repainted around it. For the
moving ones the plate is built in four steps:

1. Muse Image repaints the painting without them (painter.py).
2. That repaint is kept only where the subject's layer is opaque: its outline
   grown by two pixels, where the layer carries the subject's soft edge
   (`confine`, cutter.EDGE_PX).
   A full repaint shifts colours and loses detail elsewhere (on the harbour test
   it removed the rope next to the post as well), and any difference outside
   the subjects would show as a jump when the still painting gives way to the
   moving world. Confined, plate + layers at rest equals the painting.
3. Under each subject, the repaint's colours are matched to the painting's
   colours around it (`tone_match`), so a moving subject does not uncover a
   paler band.
4. SAM checks the result for the very wordings it matched on the painting. If a
   subject is still there, the plate is filled from its surroundings instead
   (`fill`), which is enough for the few pixels a layer moves. The fill, too,
   stays where the layer is opaque, so no blurred ring shows around a subject
   at rest.
"""
from __future__ import annotations

import asyncio
import io

import numpy as np
from PIL import Image, ImageFilter

from .cutter import Layer, grown, near, sam_image_url, segment, to_canvas
from .meta import Meter, encode
from .settings import DEFAULT, Settings


def fit(png: bytes, size: tuple[int, int]) -> Image.Image:
    """Scale and centre-crop a picture to exactly `size` (an edit may come back a little larger)."""
    image = Image.open(io.BytesIO(png)).convert("RGB")
    if image.size == size:
        return image
    scale = max(size[0] / image.width, size[1] / image.height)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    left, top = (image.width - size[0]) // 2, (image.height - size[1]) // 2
    return image.crop((left, top, left + size[0], top + size[1]))


def tone_match(edit: Image.Image, painting: Image.Image, mask01: np.ndarray, ring_px: int = 24) -> Image.Image:
    """Shift the edit's colours under one subject by how far the edit drifted from the painting around it.

    The shift is the median, per channel, of painting minus edit over a ring around the subject:
    where the repaint rightly differs there (it also took away the subject's reflection or shadow),
    those pixels are outliers and do not count. (Matching the ring's mean and contrast instead once
    turned a clean repaint of water into a pale boat-shaped patch, which SAM then took for the boat.)
    """
    ring = grown(mask01, ring_px).astype(bool) & (mask01 == 0)
    if ring.sum() < 50:
        return edit
    e = np.asarray(edit.convert("RGB"), dtype=np.float32)
    p = np.asarray(painting.convert("RGB"), dtype=np.float32)
    shift = np.median(p[ring] - e[ring], axis=0)
    out = e.copy()
    inside = mask01.astype(bool)
    out[inside] = e[inside] + shift
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))


def confine(painting: Image.Image, edit: Image.Image, mask01: np.ndarray, soft_px: float = 1.0) -> Image.Image:
    """The edit inside the mask (its edge one pixel soft, on the inside), the painting everywhere else.

    The soft edge lies inside the mask, where the layer is opaque: at rest no trace of the edit
    shows, not even a seam."""
    weight = _inside(mask01, soft_px)
    out = (np.asarray(edit.convert("RGB"), dtype=np.float32) * weight
           + np.asarray(painting.convert("RGB"), dtype=np.float32) * (1.0 - weight))
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _inside(mask01: np.ndarray, soft_px: float = 1.0) -> np.ndarray:
    """A weight for blending into the mask: 1 well inside, soft at its edge, 0 everywhere outside."""
    region = Image.fromarray(mask01.astype(np.uint8) * 255)
    weight = np.asarray(region.filter(ImageFilter.GaussianBlur(soft_px)), dtype=np.float32) / 255.0
    return (weight * mask01.astype(np.float32))[..., None]


def _box(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    """Mean over a window of 2r+1 along one axis (edges repeat)."""
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def _smooth(a: np.ndarray, sigma: int) -> np.ndarray:
    """A near-Gaussian blur in float (three box passes per axis): no 8-bit rounding."""
    r = max(1, int(sigma))
    for _ in range(3):
        a = _box(_box(a, r, 0), r, 1)
    return a


def fill(painting: Image.Image, mask01: np.ndarray, grow_px: int = 12, keep_px: int = 0,
         radii: tuple[int, ...] = (8, 24, 72, 216, 648)) -> Image.Image:
    """Fill the subjects' area from its surroundings, at growing scales until nothing is left empty.

    Each pass blurs the known pixels and their weight (normalised convolution)
    and fills every hole pixel the blur reaches with enough weight. Small holes
    fill from close by in the first pass; the middle of a large subject fills in
    a later, wider pass instead of staying dark. The fill is computed from
    pixels at least `grow_px` away (the subject's soft edge in the painting
    never bleeds into it) but kept only inside the mask grown by `keep_px`
    (one pixel soft, on the inside): beyond that the painting stays as painted.
    """
    hole = grown(mask01, grow_px).astype(bool)
    known = (~hole).astype(np.float32)
    rgb = np.asarray(painting.convert("RGB"), dtype=np.float32) * known[..., None]
    for r in radii:
        if known.min() >= 1:
            break
        weight = _smooth(known, r)
        values = _smooth(rgb, r) / np.maximum(weight[..., None], 1e-6)
        reach = (known < 1) & (weight > 0.1)
        rgb[reach] = values[reach]
        known[reach] = 1.0
    keep = mask01
    if keep_px > 0:
        keep = grown(mask01, keep_px)
    weight = _inside(keep)
    out = rgb * weight + np.asarray(painting.convert("RGB"), dtype=np.float32) * (1.0 - weight)
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))


def to_png(image: Image.Image) -> bytes:
    # zlib's default level: optimize=True took 2.2 s per plate for a 5% smaller file (28 Oct 2026),
    # and no phone gets this PNG (the server sends WebP).
    return encode(image, "PNG")


async def is_clean(client, plate_png: bytes, layers: list[Layer], meter: Meter,
                   settings: Settings = DEFAULT) -> bool:
    """True if SAM finds none of the subjects (by their matched wording) on the plate."""
    plate = Image.open(io.BytesIO(plate_png))
    width, height = plate.size
    url = sam_image_url(plate_png)
    answers = await asyncio.gather(*(segment(client, url, layer.wording, meter, settings) for layer in layers),
                                   return_exceptions=True)
    for layer, records in zip(layers, answers):
        if isinstance(records, BaseException):
            return False  # cannot prove it is clean
        sure = [r for r in records if r.confidence is None or r.confidence >= settings.min_confidence]
        if not sure:
            continue
        found = np.zeros((height, width), dtype=np.uint8)
        for record in near(sure, settings.near_share):
            found |= to_canvas(record, width, height)
        # Only what is still where the subject was counts: the same kind of
        # object further away in the scene is fine.
        overlap = float((found.astype(bool) & (layer.mask > 0)).mean()) if layer.mask is not None else 0.0
        if overlap > 0.25 * layer.coverage:
            return False
    return True


def moves(layer: Layer) -> bool:
    """Whether the phone moves this layer (a still one only ever sits where it is painted)."""
    return layer.motion != "still"


async def make_plate(client, painting_png: bytes, edit_png: bytes | None, layers: list[Layer], meter: Meter,
                     settings: Settings = DEFAULT) -> tuple[bytes | None, str]:
    """(plate, source); source is 'muse_image', 'filled', 'painting' (no subject moves: the
    plate is the painting) or 'none' (no layers, no plate needed)."""
    if not layers:
        return None, "none"
    painting = Image.open(io.BytesIO(painting_png)).convert("RGB")
    moving = [layer for layer in layers if moves(layer)]
    if not moving:
        return await asyncio.to_thread(to_png, painting), "painting"
    # Where the moving layers are opaque (cutter.cutout): their outlines grown by EDGE_PX.
    covered = [grown(layer.mask) for layer in moving if layer.mask is not None]
    mask = np.zeros((painting.height, painting.width), dtype=np.uint8)
    for region in covered:
        mask |= region
    if edit_png:
        # The pixel work runs in a worker thread (like the cutter's), never holding up other worlds' answers.
        candidate = await asyncio.to_thread(_from_edit, painting, edit_png, covered, mask)
        try:
            if await is_clean(client, candidate, moving, meter, settings):
                return candidate, "muse_image"
        except Exception:  # noqa: BLE001 - fall back to the local fill
            pass
    return await asyncio.to_thread(lambda: to_png(fill(painting, mask))), "filled"


def _from_edit(painting: Image.Image, edit_png: bytes, covered: list[np.ndarray], mask: np.ndarray) -> bytes:
    """The repaint, tone-matched around each moving subject and kept only where the moving layers are opaque."""
    edit = fit(edit_png, painting.size)
    for region in covered:
        edit = tone_match(edit, painting, region)
    return to_png(confine(painting, edit, mask))
