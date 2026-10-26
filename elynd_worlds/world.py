"""A finished world, and the manifest the app reads.

Manifest schema "elynd.world/1" (world.json next to the pictures):

    {
      "schema": "elynd.world/1",
      "title": "Still Morning Lake", "sleep": false, "palette": ["#..."],
      "size": [1152, 2048],
      "focus": [0.42, 0.55],               # where the subjects are: phones crop around it
      "painting": "painting.png",          # the full painting (shown first)
      "plate": "plate.png" | null,         # the painting without its subjects
      "layers": [                          # nearest first
        {"file": "layer_0.png", "name": "smooth stone", "depth": "near",
         "motion": "still", "attached": "ground",
         "anchor": [0.31, 0.88],           # pivot of its motion, fractions of the painting
         "box": [0.18, 0.79, 0.44, 0.88]}
      ],
      "living": {                          # what moves in nature (regions.py)
        "regions": "regions.png" | null,   # RGBA: water, sky, grass, stream (half size, soft)
        "water": true, "sky": true, "grass": false, "stream": false, "trees": false,
        "horizon": 0.52,                   # where the water starts
        "lights": [{"x": 0.5, "y": 0.6, "size": 0.08}],   # glowing subjects flicker here
        "particles": "none"                # leaves | snow | pollen | fireflies | none
      },
      "cost_usd": 0.0432, "seconds": 41.2, "trace": [...]
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cutter import Layer
from .meta import Meter, png_size
from .regions import Regions

SCHEMA = "elynd.world/1"


WHERE_X = {"left": 0.33, "centre": 0.5, "right": 0.67}


def focus_of_scene(scene: dict[str, Any]) -> tuple[float, float]:
    """The focus before any layer is cut: the mean of the subjects' designed places."""
    xs = [WHERE_X.get(s.get("where", ""), 0.5) for s in scene.get("subjects") or []]
    return (round(sum(xs) / len(xs), 4) if xs else 0.5, 0.5)


@dataclass
class Step:
    """One step of the agent's run, for the trace."""

    name: str
    seconds: float
    note: str = ""


@dataclass
class World:
    scene: dict[str, Any]
    painting: bytes
    verdict: dict[str, Any]
    layers: list[Layer] = field(default_factory=list)
    plate: bytes | None = None
    plate_source: str = "none"     # "muse_image" | "filled" | "none"
    paintings: int = 1             # how many paintings the agent made
    trace: list[Step] = field(default_factory=list)
    meter: Meter = field(default_factory=Meter)
    seconds: float = 0.0
    regions: Regions = field(default_factory=Regions)
    lights: list[dict[str, float]] = field(default_factory=list)

    def focus(self) -> tuple[float, float]:
        """Where the subjects are (fractions of the painting), so a screen narrower than the
        painting crops around them instead of around the middle. From the cut layers when there
        are any (weighted by size), else from where the art director placed the subjects."""
        if self.layers:
            weights = [max(layer.coverage, 1e-4) for layer in self.layers]
            total = sum(weights)
            x = sum(w * (layer.box[0] + layer.box[2]) / 2 for w, layer in zip(weights, self.layers)) / total
            y = sum(w * (layer.box[1] + layer.box[3]) / 2 for w, layer in zip(weights, self.layers)) / total
            return (round(x, 4), round(y, 4))
        return focus_of_scene(self.scene)

    @property
    def moving(self) -> bool:
        """True if the world has a clean plate and at least one layer to move."""
        return bool(self.plate and self.layers)

    def manifest(self) -> dict[str, Any]:
        width, height = png_size(self.painting)
        layers = self.layers if self.moving else []
        return {
            "schema": SCHEMA,
            "title": self.scene["title"],
            "sleep": bool(self.scene.get("sleep")),
            "palette": list(self.scene.get("palette") or []),
            "size": [width, height],
            "focus": list(self.focus()),
            "painting": "painting.png",
            "plate": "plate.png" if self.moving else None,
            "layers": [
                {"file": f"layer_{i}.png", "name": layer.name, "depth": layer.depth, "motion": layer.motion,
                 "attached": layer.attached, "anchor": list(layer.anchor()), "box": list(layer.box)}
                for i, layer in enumerate(layers)
            ],
            "living": {
                "regions": "regions.png" if self.regions.png else None,
                **self.regions.manifest(),
                "lights": list(self.lights) if self.moving else [],
                "particles": self.scene.get("particles") or "none",
            },
            "cost_usd": self.meter.usd,
            "seconds": round(self.seconds, 2),
            "trace": [{"step": s.name, "seconds": round(s.seconds, 2), "note": s.note} for s in self.trace],
        }

    def save(self, folder: str | Path, *, extra: dict[str, Any] | None = None) -> Path:
        """Write the pictures and world.json (plus scene.json and verdict.json) into `folder`."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "painting.png").write_bytes(self.painting)
        if self.moving:
            (folder / "plate.png").write_bytes(self.plate)
            for i, layer in enumerate(self.layers):
                (folder / f"layer_{i}.png").write_bytes(layer.png)
        if self.regions.png:
            (folder / "regions.png").write_bytes(self.regions.png)
        manifest = {**self.manifest(), **(extra or {})}
        (folder / "world.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        (folder / "scene.json").write_text(json.dumps(self.scene, indent=2, ensure_ascii=False))
        (folder / "verdict.json").write_text(json.dumps(self.verdict, indent=2, ensure_ascii=False))
        (folder / "cost.json").write_text(json.dumps(self.meter.as_dict(), indent=2))
        return folder
