"""SAM 3.1 cuts the designed subjects out of the painting, one layer each.

SAM takes a picture and a short noun phrase and returns the mask of every
match. Three things make that reliable on paintings:

* Wording. SAM is literal: a subject can match as "smooth stones" but not as
  "smooth stone". Each subject is asked by its name and its plural at once; if
  both miss, by its head noun alone ("stone"). Branches are the exception that
  no wording rescues (on the lake test, "pine branch", "pine branches", "tree
  branch" and "branch" all found nothing), so the art director avoids them.
* Distance. A harbour painted with one rowboat in front often holds three more
  far away. Only the near instances move: those whose area is at least
  `Settings.near_share` of the largest one's.
* Solidity. A teacup's dark tea can come back as a hole in the cup's mask.
  Holes enclosed by the mask are filled, so the layer moves as one object.

Each layer also records its outline box, so the app can anchor the motion where
the object is attached: its base if it rests on the ground, the edge it grows
from if it is a branch, its centre if it floats.
"""
from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from meta_sam_parser import SegmentationMaskRecord, decode_mask_to_raster, image_segmentation_format, \
    parse_responses_stream
from openai import APIConnectionError, InternalServerError, NotFoundError, RateLimitError
from PIL import Image, ImageDraw, ImageFilter

from .meta import Meter, data_url, encode, privacy_fields
from .settings import DEFAULT, Settings

# Worth asking again (APITimeoutError is an APIConnectionError); anything else is a real answer.
TRANSIENT = (APIConnectionError, InternalServerError, NotFoundError, RateLimitError, asyncio.TimeoutError)

IRREGULAR_PLURALS = {
    "leaf": "leaves", "knife": "knives", "shelf": "shelves", "loaf": "loaves", "wolf": "wolves",
    "goose": "geese", "mouse": "mice", "foot": "feet", "tooth": "teeth", "child": "children",
}
IRREGULAR_SINGULARS = {plural: single for single, plural in IRREGULAR_PLURALS.items()}


@dataclass
class Layer:
    """One cut-out subject: the painting's own pixels inside its outline, transparent outside."""

    name: str              # as designed
    depth: str             # "near" | "mid"
    attached: str          # "ground" | "edge" | "air"
    motion: str            # "sway" | "drift" | "float" | "still"
    wording: str           # the phrase SAM matched
    png: bytes             # RGBA, the painting's size
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 as fractions of the painting
    coverage: float        # share of the painting the subject covers
    confidence: float | None
    mask: np.ndarray | None = field(default=None, repr=False)  # 0/1, the painting's size

    def anchor(self) -> tuple[float, float]:
        """The point (fractions of the painting) the layer's motion pivots on."""
        x0, y0, x1, y1 = self.box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if self.attached == "ground":
            return (round(cx, 4), round(y1, 4))
        if self.attached == "edge":
            # The side of the box closest to a picture edge is where it grows from.
            sides = {"left": x0, "right": 1 - x1, "top": y0, "bottom": 1 - y1}
            side = min(sides, key=sides.get)
            point = {"left": (x0, cy), "right": (x1, cy), "top": (cx, y0), "bottom": (cx, y1)}[side]
            return (round(point[0], 4), round(point[1], 4))
        return (round(cx, 4), round(cy, 4))


def plural(word: str) -> str:
    if word in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[word]
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def singular(word: str) -> str:
    if word in IRREGULAR_SINGULARS:
        return IRREGULAR_SINGULARS[word]
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def wordings(name: str) -> tuple[list[str], str | None]:
    """(first round, fallback): the name and its plural, then the head noun alone."""
    words = name.split()
    *rest, head = words
    other = plural(head) if singular(head) == head else singular(head)
    first = [name, " ".join([*rest, other])]
    first = list(dict.fromkeys(first))
    fallback = head if rest and head not in first else None
    return first, fallback


def to_canvas(record: SegmentationMaskRecord, width: int, height: int) -> np.ndarray:
    """One box-local SAM mask placed on a 0/1 canvas of the painting's size."""
    raster = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8)
    raster = raster.reshape(record.mask.height, record.mask.width)
    b = record.bounds
    left, top = int(round(b.left)), int(round(b.top))
    box_w = max(int(round(b.right)) - left, 1)
    box_h = max(int(round(b.bottom)) - top, 1)
    if raster.shape != (box_h, box_w):
        raster = (np.asarray(Image.fromarray(raster * 255).resize((box_w, box_h), Image.NEAREST)) > 127)
        raster = raster.astype(np.uint8)
    canvas = np.zeros((height, width), dtype=np.uint8)
    y0, x0 = max(top, 0), max(left, 0)
    y1, x1 = min(top + box_h, height), min(left + box_w, width)
    if y1 > y0 and x1 > x0:
        canvas[y0:y1, x0:x1] = raster[y0 - top:y1 - top, x0 - left:x1 - left]
    return canvas


def near(records: list[SegmentationMaskRecord], share: float) -> list[SegmentationMaskRecord]:
    """The instances in front: area at least `share` of the largest one's."""
    def area(r: SegmentationMaskRecord) -> float:
        return max(r.bounds.right - r.bounds.left, 0) * max(r.bounds.bottom - r.bounds.top, 0)
    biggest = max((area(r) for r in records), default=0)
    return [r for r in records if biggest and area(r) >= share * biggest]


# A boat's oars, which SAM leaves out of the boat, are gaps up to about this wide where they cross
# its side: closed before the holes are filled, so they ride with the boat (and the repaint behind
# it never shows through them).
CLOSE_PX = 12


def _count(mask01: np.ndarray, r: int) -> np.ndarray:
    """How many set pixels lie in the (2r+1)^2 square around each pixel (outside the picture: none)."""
    h, w = mask01.shape
    k = 2 * r + 1
    s = np.pad(mask01.astype(np.int32), ((k, 0), (k, 0))).cumsum(0).cumsum(1)
    s = np.pad(s, ((0, r), (0, r)), mode="edge")
    return s[k + r:k + r + h, k + r:k + r + w] - s[r:r + h, k + r:k + r + w] - s[k + r:k + r + h, r:r + w] + s[r:r + h, r:r + w]


def closed(mask01: np.ndarray, px: int = CLOSE_PX) -> np.ndarray:
    """The mask with gaps narrower than about 2*px closed (grown, then shrunk back); never smaller."""
    if px <= 0:
        return mask01.astype(np.uint8)
    grown_ = (_count(mask01, px) > 0).astype(np.uint8)
    shrunk = _count(grown_, px) == (2 * px + 1) ** 2
    return (mask01.astype(bool) | shrunk).astype(np.uint8)


def rides_the_water(subject: dict[str, Any]) -> bool:
    """A boat and the like: floating on the water. What lies in it moves with it."""
    return subject.get("motion") == "float" and subject.get("attached", "ground") == "ground"


def solid(mask01: np.ndarray) -> np.ndarray:
    """The mask with every hole it encloses filled."""
    h, w = mask01.shape
    # .copy(): an image made from an array can be read-only, and floodfill would silently do nothing.
    padded = Image.fromarray(np.pad(mask01, 1).astype(np.uint8) * 255).copy()
    ImageDraw.floodfill(padded, (0, 0), 128)  # everything reachable from outside
    outside = np.asarray(padded)[1:h + 1, 1:w + 1] == 128
    return (~outside).astype(np.uint8)


def box_of(mask01: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask01)
    if len(xs) == 0:
        return (0.0, 0.0, 1.0, 1.0)
    h, w = mask01.shape
    return (round(xs.min() / w, 4), round(ys.min() / h, 4), round((xs.max() + 1) / w, 4), round((ys.max() + 1) / h, 4))


# A layer is opaque this far past the subject's outline: it carries the
# subject's own soft edge (a dark rim fading into the water) with it when it
# moves, and the plate is repainted exactly where the layer is opaque.
EDGE_PX = 2


def grown(mask01: np.ndarray, px: int = EDGE_PX) -> np.ndarray:
    """The mask grown by `px` pixels on every side.

    Exactly what PIL's MaxFilter(2*px+1) gives, counted from a summed-area table: MaxFilter
    compares every pixel with all (2*px+1)^2 around it, 5.7 s per subject for the plate's
    24-pixel ring on a 1152x2048 painting (28 Oct 2026); this takes milliseconds at any width."""
    if px <= 0:
        return mask01.astype(np.uint8)
    return (_count(mask01, px) > 0).astype(np.uint8)


def cutout(painting: Image.Image, mask01: np.ndarray, feather_px: float = 2.0) -> bytes:
    """RGBA: the painting inside the mask grown by EDGE_PX, a soft edge just outside, transparent beyond.

    Alpha is 1 everywhere inside the grown mask, so plate + layers at rest is
    exactly the painting (the plate changes only there, see plate.py).
    """
    hard = Image.fromarray(grown(mask01) * 255)
    soft = np.asarray(hard.filter(ImageFilter.GaussianBlur(feather_px)))
    alpha = np.maximum(np.asarray(hard), soft).astype(np.uint8)
    rgba = painting.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha))
    # zlib's default level: optimize=True took 3 s per layer for a 7% smaller file (28 Oct 2026),
    # and no phone gets this PNG (the server sends WebP).
    return encode(rgba, "PNG")


async def segment(client, image_url: str, phrase: str, meter: Meter,
                  settings: Settings = DEFAULT) -> list[SegmentationMaskRecord]:
    """One SAM 3.1 request (one concept per request); the mask records it returns.

    A request that fails on the way (a timeout, a dropped line, a busy server, the model briefly missing from
    the account's catalog) is asked again, `settings.sam_attempts` times in all: on 27 Oct a slow hour turned
    the water and the sky of 36 library worlds into "none", and those worlds stood still.
    """
    attempts = max(1, settings.sam_attempts)
    for attempt in range(1, attempts + 1):
        try:
            stream = await client.responses.create(
                model=settings.cutter_model,
                input=[{"role": "user", "content": [
                    {"type": "input_image", "image_url": image_url},
                    {"type": "input_text", "text": phrase},
                ]}],
                metadata={"include_confidence": "true", "mask_encoding": settings.sam_mask_encoding},
                stream=True,
                timeout=settings.cut_timeout,
                **privacy_fields(),
            )
            meter.sam("segment")
            parsed = parse_responses_stream(stream, image_segmentation_format())
            result = await parsed.final_result()
            return [r for r in result.records if isinstance(r, SegmentationMaskRecord)]
        except TRANSIENT:
            if attempt == attempts:
                raise
            await asyncio.sleep(settings.sam_retry_seconds * attempt)


def _confident(records: list[SegmentationMaskRecord], settings: Settings) -> list[SegmentationMaskRecord]:
    return [r for r in records if r.confidence is None or r.confidence >= settings.min_confidence]


async def find(client, image_url: str, name: str, meter: Meter,
               settings: Settings = DEFAULT) -> tuple[str, list[SegmentationMaskRecord]] | None:
    """(wording, records) for the first wording of `name` that SAM matches, else None."""
    first, fallback = wordings(name)
    if not settings.sam_plurals:
        first = first[:1]
    answers = await asyncio.gather(*(segment(client, image_url, w, meter, settings) for w in first),
                                   return_exceptions=True)
    for wording, records in zip(first, answers):
        if not isinstance(records, BaseException) and _confident(records, settings):
            return wording, _confident(records, settings)
    if fallback:
        try:
            records = _confident(await segment(client, image_url, fallback, meter, settings), settings)
        except Exception:  # noqa: BLE001 - a missing layer only means a stiller world
            records = []
        if records:
            return fallback, records
    return None


def sam_image_url(painting: bytes) -> str:
    """The painting as a high-quality JPEG data URL: a fraction of the PNG's upload, same masks."""
    image = Image.open(io.BytesIO(painting)).convert("RGB")
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=92)
    return data_url(out.getvalue(), "image/jpeg")


async def cut(client, painting: bytes, scene: dict[str, Any], meter: Meter,
              settings: Settings = DEFAULT) -> list[Layer]:
    """One layer per designed subject that SAM finds; nearest first.

    The pixel work runs in a worker thread: the critic's answer, which arrives meanwhile, is read on time.
    """
    url = await asyncio.to_thread(sam_image_url, painting)
    subjects = scene["subjects"]
    found = await asyncio.gather(*(find(client, url, s["name"], meter, settings) for s in subjects),
                                 return_exceptions=True)
    return await asyncio.to_thread(_layers, painting, subjects, found, settings)


def _layers(painting: bytes, subjects: list[dict[str, Any]], found: list[Any], settings: Settings) -> list[Layer]:
    image = Image.open(io.BytesIO(painting)).convert("RGB")
    width, height = image.size
    layers: list[Layer] = []
    taken = np.zeros((height, width), dtype=bool)
    for subject, result in zip(subjects, found):
        if isinstance(result, BaseException) or result is None:
            continue
        wording, records = result
        records = near(records, settings.near_share)
        mask = np.zeros((height, width), dtype=np.uint8)
        for record in records:
            mask |= to_canvas(record, width, height)
        if rides_the_water(subject):
            mask = closed(mask)
        mask = solid(mask)
        mask[taken] = 0  # two subjects never claim the same pixel
        coverage = float(mask.mean())
        if not settings.min_layer_share <= coverage <= settings.max_layer_share:
            continue
        taken |= mask.astype(bool)
        confidences = [r.confidence for r in records if r.confidence is not None]
        layers.append(Layer(
            name=subject["name"], depth=subject["depth"], attached=subject["attached"],
            motion=subject["motion"], wording=wording, png=cutout(image, mask), box=box_of(mask),
            coverage=round(coverage, 5), confidence=max(confidences) if confidences else None, mask=mask,
        ))
    layers.sort(key=lambda layer: 0 if layer.depth == "near" else 1)
    return layers
