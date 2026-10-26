"""SAM 3.1 finds the parts of a painting that move in nature, so the phone can move them.

The art director lists them in `scene["nature"]` (water, sky, grass, stream,
trees). For each of the first four, one SAM 3.1 request on the painting finds it;
every confident mask of that concept is merged into one soft mask. The four masks
become one picture, the region map (red = water, green = sky, blue = grass,
alpha = stream), which the phone's shader reads to ripple the water, drift the
clouds, wave the grass and let the stream flow. "trees" needs no mask: the phone
sways the upper part of the painting outside the sky.

A glowing subject (a lantern, a candle) becomes a light that flickers, at the
place SAM cut it out (`lights_from`).
"""
from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .cutter import Layer, sam_image_url, segment, solid, to_canvas
from .meta import Meter
from .settings import DEFAULT, Settings

# Channel order of the region map, and what SAM 3.1 is asked for each.
CHANNELS = ("water", "sky", "grass", "stream")
PHRASES = {"water": "water", "sky": "sky", "grass": "grass", "stream": "stream"}


@dataclass
class Regions:
    """What moves in a painting, for the phone."""

    png: bytes | None = None                     # the region map, RGBA, half the painting's size
    present: dict[str, bool] = field(default_factory=dict)
    coverage: dict[str, float] = field(default_factory=dict)
    horizon: float | None = None                 # where the water starts (fraction of the height)
    trees: bool = False

    @property
    def any(self) -> bool:
        return self.trees or any(self.present.values())

    def manifest(self) -> dict[str, Any]:
        return {**{name: bool(self.present.get(name)) for name in CHANNELS}, "trees": self.trees,
                "horizon": None if self.horizon is None else round(self.horizon, 4),
                "coverage": {k: round(v, 4) for k, v in self.coverage.items()}}


def lights_from(layers: list[Layer], scene: dict[str, Any]) -> list[dict[str, float]]:
    """A light for each glowing subject that was cut out: its flame a little above the box's middle."""
    glowing = {(s.get("name") or "").lower() for s in scene.get("subjects") or [] if s.get("glows")}
    lights = []
    for layer in layers:
        if layer.name in glowing:
            x0, y0, x1, y1 = layer.box
            lights.append({"x": round((x0 + x1) / 2, 4), "y": round(y0 + (y1 - y0) * 0.45, 4),
                           "size": round(max(x1 - x0, 0.01), 4)})
    return lights


def _horizon(water: np.ndarray) -> float | None:
    rows = np.where(water.mean(axis=1) > 0.5)[0]
    return float(rows.min() / water.shape[0]) if len(rows) else None


async def find_regions(client, painting: bytes, scene: dict[str, Any], meter: Meter,
                       settings: Settings = DEFAULT) -> Regions:
    """The moving parts the design names, as SAM 3.1 finds them in the painting."""
    wanted = [name for name in CHANNELS if name in (scene.get("nature") or [])]
    regions = Regions(trees="trees" in (scene.get("nature") or []))
    if not wanted:
        return regions
    image = Image.open(io.BytesIO(painting)).convert("RGB")
    width, height = image.size
    url = sam_image_url(painting)
    answers = await asyncio.gather(*(segment(client, url, PHRASES[name], meter, settings) for name in wanted),
                                   return_exceptions=True)
    w2, h2 = max(width // 2, 1), max(height // 2, 1)
    channels = {name: np.zeros((h2, w2), dtype=np.uint8) for name in CHANNELS}
    for name, records in zip(wanted, answers):
        if isinstance(records, BaseException):
            continue  # a missing region only means that part stays still
        mask = np.zeros((height, width), dtype=np.uint8)
        for record in records:
            if record.confidence is None or record.confidence >= settings.min_confidence:
                mask |= to_canvas(record, width, height)
        share = float(mask.mean())
        if share < settings.min_region_share:
            continue
        mask = solid(mask)
        regions.present[name] = True
        regions.coverage[name] = share
        if name == "water":
            regions.horizon = _horizon(mask)
        small = Image.fromarray(mask * 255).resize((w2, h2), Image.BILINEAR)
        # Soft edges, so a rippling lake meets a still shore without a seam.
        channels[name] = np.asarray(small.filter(ImageFilter.GaussianBlur(max(w2 / 180, 1.0))))
    if any(regions.present.values()):
        rgba = np.dstack([channels[name] for name in CHANNELS])
        out = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(out, format="PNG", optimize=True)
        regions.png = out.getvalue()
    return regions
