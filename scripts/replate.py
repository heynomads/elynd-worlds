"""Rebuilds the layers and the plate of saved worlds with the current rule, offline (no API call).

    python scripts/replate.py out/<time>/<name> [out/<time>/<name> ...]

Since 27 Oct 2026 a layer is opaque two pixels past its subject's outline
(cutter.EDGE_PX), a still subject stays in the plate as painted, and the plate
changes only where a moving layer is opaque (plate.py). Worlds saved before
that are rebuilt from what they saved: the painting, the layers (a subject's
outline is where its old layer is fully opaque) and the old plate. A plate Muse
Image repainted keeps that repaint inside the moving subjects' outlines and is
filled locally in the two pixels around them; a filled one is filled again. A
world whose subjects are all still, none glowing, becomes a still painting (its
water, sky and grass still move). world.json records the step as "replate"; a
world that has one is left as it is.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from elynd_worlds import cutter, plate


def replate(folder: Path) -> str:
    manifest = json.loads((folder / "world.json").read_text())
    if not manifest.get("plate"):
        return "still painting, nothing to do"
    if any(step["step"] == "replate" for step in manifest["trace"]):
        return "already rebuilt"
    painting = Image.open(folder / "painting.png").convert("RGB")
    layers = manifest["layers"]
    outlines = [(np.asarray(Image.open(folder / layer["file"]).getchannel("A")) == 255).astype(np.uint8)
                for layer in layers]
    moving = [outline for layer, outline in zip(layers, outlines) if layer["motion"] != "still"]
    if not moving and not any(layer.get("glows") for layer in layers):
        (folder / manifest["plate"]).unlink()
        for layer in layers:
            (folder / layer["file"]).unlink()
        manifest.update(plate=None, layers=[])
        manifest["living"]["lights"] = []
        note = "none: only still subjects, the world is the still painting"
    else:
        for layer, outline in zip(layers, outlines):
            (folder / layer["file"]).write_bytes(cutter.cutout(painting, outline))
        if not moving:
            new, note = painting, "painting (only still subjects; one glows)"
        else:
            covered = np.zeros_like(moving[0])
            inside = np.zeros_like(moving[0])
            for outline in moving:
                covered |= cutter.grown(outline)
                inside |= outline
            new = plate.fill(painting, covered)
            note = "filled where the moving layers are opaque"
            source = next((s["note"] for s in manifest["trace"] if s["step"] == "plate"), "")
            if source == "muse_image":
                old = np.asarray(Image.open(folder / manifest["plate"]).convert("RGB"))
                new = Image.fromarray(np.where(inside[..., None].astype(bool), old, np.asarray(new)))
                note = "muse_image inside the moving subjects, filled in the two pixels around them"
        (folder / manifest["plate"]).write_bytes(plate.to_png(new))
    manifest["trace"].append({"step": "replate", "seconds": 0.0, "note": note})
    (folder / "world.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
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
