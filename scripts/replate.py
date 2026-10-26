"""Rebuilds the layers, the plate and the region map of saved worlds with the current rule, offline
(no API call).

    python scripts/replate.py out/<time>/<name> [out/<time>/<name> ...]

Since 27 Oct 2026:
- a layer is opaque two pixels past its subject's outline (cutter.EDGE_PX);
- a still subject stays in the plate as painted, and the plate changes only where a moving layer is
  opaque (plate.py);
- still subjects are taken out of the region map (regions.without);
- a boat's outline is closed over its oars (cutter.closed).

Worlds saved before that are rebuilt from what they saved: the painting, the layers (a subject's
outline is where its old layer is fully opaque), the old plate and region map. A plate Muse Image
repainted keeps that repaint inside the moving subjects' outlines and is filled locally in the two
pixels around them; a filled one is filled again. A world whose subjects are all still, none
glowing, becomes a still painting (its water, sky and grass still move).

world.json records the step as "replate" and gets "edge_px". A world that has either (made by the
current engine, or rebuilt already) is left as it is. Everything is checked and computed before
anything is written: each file is written under a temporary name and moved into place, world.json
last, and only then is anything deleted. A run that stops halfway leaves a ".replating" file, and
the folder is then left alone until it is restored from a copy.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from elynd_worlds import cutter, plate
from elynd_worlds.regions import Regions, without


def replate(folder: Path) -> str:
    lock = folder / ".replating"
    if lock.exists():
        return "skipped: an earlier run stopped halfway here; restore this folder from a copy, then run again"
    manifest = json.loads((folder / "world.json").read_text())
    if not manifest.get("plate"):
        return "still painting, nothing to do"
    if "edge_px" in manifest or any(step["step"] == "replate" for step in manifest.get("trace", [])):
        return "made or rebuilt with the current rule already, left as it is"
    layers = manifest["layers"]
    living = manifest.get("living") or {}
    needed = ["painting.png", manifest["plate"], *(layer["file"] for layer in layers)]
    if living.get("regions"):
        needed.append(living["regions"])
    missing = [name for name in needed if not (folder / name).exists()]
    if missing:
        return f"skipped: missing {', '.join(missing)}"

    painting = Image.open(folder / "painting.png").convert("RGB")
    raw = [(np.asarray(Image.open(folder / layer["file"]).getchannel("A")) == 255).astype(np.uint8)
           for layer in layers]
    # A boat's outline is closed over its oars, which then ride with it (cutter.closed).
    outlines = [cutter.solid(cutter.closed(outline)) if cutter.rides_the_water(layer) else outline
                for layer, outline in zip(layers, raw)]
    moving = [outline for layer, outline in zip(layers, outlines) if layer["motion"] != "still"]
    still = [outline for layer, outline in zip(layers, outlines) if layer["motion"] == "still"]
    writes: dict[str, bytes] = {}
    deletes: list[str] = []
    if living.get("regions") and still:
        regions = Regions(png=(folder / living["regions"]).read_bytes())
        writes[living["regions"]] = without(regions, still).png
    if not moving and not any(layer.get("glows") for layer in layers):
        deletes = [manifest["plate"], *(layer["file"] for layer in layers)]
        manifest.update(plate=None, layers=[])
        if "living" in manifest:
            manifest["living"]["lights"] = []
        note = "none: only still subjects, the world is the still painting"
    else:
        for layer, outline in zip(layers, outlines):
            writes[layer["file"]] = cutter.cutout(painting, outline)
        if not moving:
            new, note = painting, "painting (only still subjects; one glows)"
        else:
            covered = np.zeros_like(moving[0])
            inside = np.zeros_like(moving[0])
            for layer, outline, old_outline in zip(layers, outlines, raw):
                if layer["motion"] != "still":
                    covered |= cutter.grown(outline)
                    # The old repaint was kept only inside the old outline; elsewhere (a boat's
                    # oars, now part of it) the old plate is the painting: fill those parts afresh.
                    inside |= old_outline
            new = plate.fill(painting, covered)
            note = "filled where the moving layers are opaque"
            source = next((s["note"] for s in manifest.get("trace", []) if s["step"] == "plate"), "")
            if source == "muse_image":
                old = np.asarray(Image.open(folder / manifest["plate"]).convert("RGB"))
                new = Image.fromarray(np.where(inside[..., None].astype(bool), old, np.asarray(new)))
                note = "muse_image inside the moving subjects, filled in the two pixels around them"
        writes[manifest["plate"]] = plate.to_png(new)
    manifest["edge_px"] = cutter.EDGE_PX
    manifest.setdefault("trace", []).append({"step": "replate", "seconds": 0.0, "note": note})
    for name, data in writes.items():
        Image.open(io.BytesIO(data)).verify()  # every picture decodes before anything changes on disk
    lock.write_text("replate.py was writing here\n")
    staged = {}
    for name, data in writes.items():
        staged[name] = folder / f"{name}.replate-tmp"
        staged[name].write_bytes(data)
    staged["world.json"] = folder / "world.json.replate-tmp"
    staged["world.json"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    for name, path in staged.items():  # world.json last (dicts keep their order)
        os.replace(path, folder / name)
    for name in deletes:
        (folder / name).unlink(missing_ok=True)
    lock.unlink()
    return note


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    for arg in argv:
        print(f"{arg}: {replate(Path(arg))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
