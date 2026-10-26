"""A finished world, and the manifest the app reads.

Manifest schema "elynd.world/1" (world.json next to the pictures):

    {
      "schema": "elynd.world/1",
      "title": "Still Morning Lake", "sleep": false, "palette": ["#..."],
      "size": [1152, 2048],
      "painting": "painting.png",          # the full painting (shown first)
      "plate": "plate.png" | null,         # the painting without its subjects
      "layers": [                          # nearest first
        {"file": "layer_0.png", "name": "smooth stone", "depth": "near",
         "motion": "still", "attached": "ground",
         "anchor": [0.31, 0.88],           # pivot of its motion, fractions of the painting
         "box": [0.18, 0.79, 0.44, 0.88]}
      ],
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

SCHEMA = "elynd.world/1"


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
            "painting": "painting.png",
            "plate": "plate.png" if self.moving else None,
            "layers": [
                {"file": f"layer_{i}.png", "name": layer.name, "depth": layer.depth, "motion": layer.motion,
                 "attached": layer.attached, "anchor": list(layer.anchor()), "box": list(layer.box)}
                for i, layer in enumerate(layers)
            ],
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
        manifest = {**self.manifest(), **(extra or {})}
        (folder / "world.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        (folder / "scene.json").write_text(json.dumps(self.scene, indent=2, ensure_ascii=False))
        (folder / "verdict.json").write_text(json.dumps(self.verdict, indent=2, ensure_ascii=False))
        (folder / "cost.json").write_text(json.dumps(self.meter.as_dict(), indent=2))
        return folder
