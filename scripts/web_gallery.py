"""Builds the web gallery's worlds (web/worlds/) from saved runs.

    python scripts/web_gallery.py out/<time>/<name> [out/<time>/<name> ...]

For each world folder (as `elynd-worlds make` saves it): the plate, the layers
and the painting as WebP at phone size (each keeps the "made by AI" mark), and
one entry in web/worlds/index.json with what the page shows: the manifest, the
example meditation it was made from (its first lines), the art director's design,
the critic's verdict, the agent's steps, the cost and the time. Only worlds made
from the example meditations in examples/: never a listener's.
"""
from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

from elynd_worlds.meta import encode
from elynd_worlds.world import focus_of_scene

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web" / "worlds"
WIDTH = 720


def webp(png: Path, out: Path, *, quality: int) -> None:
    image = Image.open(png)
    image = image.convert("RGBA" if image.mode in ("RGBA", "LA", "P") else "RGB")
    image = image.resize((WIDTH, round(image.height * WIDTH / image.width)), Image.LANCZOS)
    options = {"quality": quality, "method": 6}
    if image.mode == "RGBA":
        options["alpha_quality"] = 90
    out.write_bytes(encode(image, "WEBP", **options))


def focus_of(manifest: dict, scene: dict) -> list[float]:
    """The saved focus; early runs (before it was saved) get it from their layers' boxes."""
    if manifest.get("focus"):
        return manifest["focus"]
    boxes = [layer["box"] for layer in manifest.get("layers") or [] if layer.get("box")]
    if not boxes:
        return list(focus_of_scene(scene))
    areas = [max((b[2] - b[0]) * (b[3] - b[1]), 1e-4) for b in boxes]
    total = sum(areas)
    return [round(sum(a * (b[0] + b[2]) / 2 for a, b in zip(areas, boxes)) / total, 4),
            round(sum(a * (b[1] + b[3]) / 2 for a, b in zip(areas, boxes)) / total, 4)]


def example_of(name: str) -> tuple[dict, str]:
    """The header and the opening of the example meditation the world was made from."""
    lines = (ROOT / "examples" / f"{name}.txt").read_text().splitlines()
    header = {}
    if lines and lines[0].startswith("#"):
        header = dict(part.split("=", 1) for part in lines.pop(0).lstrip("# ").split())
    opening = " ".join(line.strip() for line in lines if line.strip())
    return header, (opening[:240].rsplit(" ", 1)[0] + "…") if len(opening) > 240 else opening


def build(folder: Path) -> dict:
    manifest = json.loads((folder / "world.json").read_text())
    scene = json.loads((folder / "scene.json").read_text())
    verdict = json.loads((folder / "verdict.json").read_text())
    name = folder.name
    out = WEB / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    webp(folder / "painting.png", out / "painting.webp", quality=82)
    plate = manifest.get("plate")
    if plate:
        webp(folder / plate, out / "plate.webp", quality=82)
    layers = []
    for i, layer in enumerate(manifest.get("layers") or []):
        webp(folder / layer["file"], out / f"layer_{i}.webp", quality=85)
        layers.append({k: layer[k] for k in ("name", "depth", "motion", "attached", "anchor", "box")}
                      | {"file": f"layer_{i}.webp"})
    header, opening = example_of(name)
    width, height = manifest["size"]
    return {
        "id": name,
        "title": manifest["title"],
        "sleep": manifest["sleep"],
        "aspect": round(width / height, 5),
        "focus": focus_of(manifest, scene),
        "palette": manifest["palette"],
        "painting": "painting.webp",
        "plate": "plate.webp" if plate else None,
        "layers": layers,
        "meditation": {"technique": header.get("technique", ""), "language": header.get("language", ""),
                       "minutes": int(header.get("minutes", 10)), "opening": opening},
        "design": {k: scene[k] for k in ("place", "time_of_day", "light", "air", "subjects")},
        "verdict": {k: verdict[k] for k in ("score", "reason", "subjects_seen")},
        "steps": manifest["trace"],
        "cost_usd": manifest["cost_usd"],
        "seconds": manifest["seconds"],
    }


def main(argv: list[str]) -> int:
    folders = [Path(a) for a in argv]
    if not folders:
        print(__doc__, file=sys.stderr)
        return 2
    WEB.mkdir(parents=True, exist_ok=True)
    worlds = [build(folder) for folder in folders]
    (WEB / "index.json").write_text(json.dumps({"schema": "elynd.gallery/1", "worlds": worlds},
                                               ensure_ascii=False, indent=1) + "\n")
    size = sum(p.stat().st_size for p in WEB.rglob("*") if p.is_file())
    print(f"{len(worlds)} worlds, {size // 1024} KB in {WEB.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
