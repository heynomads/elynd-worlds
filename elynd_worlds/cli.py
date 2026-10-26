"""Command line: make worlds from meditation texts, look at them, keep the spend in view.

    elynd-worlds make examples/lake.txt --technique visualization --mood stressed
    elynd-worlds make examples/*.txt --parallel 4            # a quality batch
    elynd-worlds sheet out/batch-*/                          # one contact sheet
    elynd-worlds timeline --duration 600 --silence 120:180   # the breathing timeline

`make` refuses to start when the spend ledger (a CSV of every run's cost) is
already at the budget. The key comes from $MODEL_API_KEY or --key-file.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import timeline
from .art_director import Meditation
from .director import make_world
from .meta import Meter, connect, encode


def read_key(path: str | None) -> str | None:
    if path:
        return Path(path).expanduser().read_text().strip()
    return os.environ.get("MODEL_API_KEY")


def spent(ledger: Path) -> float:
    if not ledger.exists():
        return 0.0
    with ledger.open() as f:
        return round(sum(float(row["usd"]) for row in csv.DictReader(f)), 4)


def record(ledger: Path, what: str, usd: float, note: str) -> None:
    new = not ledger.exists()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["utc", "what", "usd", "note"])
        writer.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), what, f"{usd:.5f}", note])


def meditation_for(path: Path, args: argparse.Namespace) -> Meditation:
    """The meditation in `path`; an optional first line "# technique=... mood=... language=... minutes=..."."""
    text = path.read_text()
    meta = {"technique": args.technique, "mood": args.mood, "language": args.language, "minutes": args.minutes}
    first, _, rest = text.partition("\n")
    if first.startswith("#"):
        for pair in first[1:].split():
            key, _, value = pair.partition("=")
            if key in meta and value:
                meta[key] = int(value) if key == "minutes" else value.replace("_", " ")
        text = rest
    return Meditation(script=text, **meta)


async def _make(args: argparse.Namespace) -> int:
    ledger = Path(args.ledger).expanduser()
    already = spent(ledger)
    estimate = 0.07 * len(args.scripts)
    if already + estimate > args.budget:
        print(f"refusing: the ledger shows ${already:.2f} spent; this run may add ~${estimate:.2f}, "
              f"over the ${args.budget:.2f} budget", file=sys.stderr)
        return 2
    client = connect(read_key(args.key_file))
    out = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    gate = asyncio.Semaphore(args.parallel)

    async def one(path: Path) -> dict:
        async with gate:
            meter = Meter()
            started = time.monotonic()
            try:
                world = await make_world(client, meditation_for(path, args), meter=meter)
            except Exception as error:  # noqa: BLE001 - report and keep the batch going
                return {"name": path.stem, "ok": False, "error": f"{type(error).__name__}: {error}"[:300],
                        "usd": meter.usd}
            folder = world.save(out / path.stem)
            return {"name": path.stem, "ok": True, "title": world.scene["title"], "usd": meter.usd,
                    "seconds": round(time.monotonic() - started, 1), "paintings": world.paintings,
                    "score": world.verdict.get("score"), "plate": world.plate_source,
                    "layers": [layer.wording for layer in world.layers] if world.moving else [],
                    "wanted": [s["name"] for s in world.scene["subjects"]], "folder": str(folder)}

    results = await asyncio.gather(*(one(Path(p)) for p in args.scripts))
    total = round(sum(r["usd"] for r in results), 5)
    record(ledger, "make", total, f"{len(results)} worlds -> {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    print(f"{sum(r['ok'] for r in results)}/{len(results)} worlds, ${total:.4f} this run, "
          f"${already + total:.4f} in the ledger -> {out}")
    return 0


def _sheet(args: argparse.Namespace) -> int:
    from PIL import Image, ImageDraw
    tiles = []
    for folder in sorted({p.parent for d in args.folders for p in Path(d).rglob("world.json")}):
        manifest = json.loads((folder / "world.json").read_text())
        world = Image.open(folder / "painting.png").convert("RGB")
        world.thumbnail((360, 640))
        layers = Image.new("RGBA", world.size, (255, 0, 255, 255))
        if manifest.get("plate"):
            layers = Image.open(folder / "plate.png").convert("RGBA")
            layers.thumbnail((360, 640))
        for layer in manifest["layers"]:
            img = Image.open(folder / layer["file"]).convert("RGBA")
            img.thumbnail((360, 640))
            tint = Image.new("RGBA", img.size, (255, 0, 255, 90))
            layers.alpha_composite(Image.composite(tint, Image.new("RGBA", img.size), img.getchannel("A")))
        tile = Image.new("RGB", (world.width * 2 + 10, world.height + 54), (14, 16, 20))
        tile.paste(world, (0, 0))
        tile.paste(layers.convert("RGB"), (world.width + 10, 0))
        draw = ImageDraw.Draw(tile)
        draw.text((6, world.height + 6), f"{folder.name}: {manifest['title']}", fill=(255, 255, 255))
        draw.text((6, world.height + 28), f"${manifest['cost_usd']:.3f}  {manifest['seconds']}s  "
                  f"layers {[l['name'] for l in manifest['layers']]}"[:80], fill=(190, 190, 190))
        tiles.append(tile)
    if not tiles:
        print("no worlds found", file=sys.stderr)
        return 1
    cols = min(3, len(tiles))
    w, h = tiles[0].size
    sheet = Image.new("RGB", (cols * (w + 10), ((len(tiles) + cols - 1) // cols) * (h + 10)), (6, 6, 8))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * (w + 10), (i // cols) * (h + 10)))
    output = Path(args.output)
    output.write_bytes(encode(sheet, "PNG" if output.suffix.lower() == ".png" else "JPEG", quality=88))
    print(args.output)
    return 0


def _timeline(args: argparse.Namespace) -> int:
    silences = [timeline.Silence(*map(float, s.split(":"))) for s in args.silence]
    cues = [timeline.Cue(float(c.split(":")[1]), c.split(":")[0]) for c in args.cue]
    print(json.dumps(timeline.build(args.duration, silences, cues, args.sleep)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="elynd-worlds", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("make", help="make worlds from meditation texts")
    make.add_argument("scripts", nargs="+")
    make.add_argument("--technique", default="")
    make.add_argument("--mood", default="")
    make.add_argument("--language", default="en")
    make.add_argument("--minutes", type=int, default=10)
    make.add_argument("--out", default="out")
    make.add_argument("--parallel", type=int, default=4)
    make.add_argument("--key-file", default=os.environ.get("MODEL_API_KEY_FILE"))
    make.add_argument("--ledger", default=os.environ.get("ELYND_WORLDS_LEDGER", "out/spend.csv"))
    make.add_argument("--budget", type=float, default=float(os.environ.get("ELYND_WORLDS_BUDGET", "25")))

    sheet = sub.add_parser("sheet", help="one contact sheet of saved worlds")
    sheet.add_argument("folders", nargs="+")
    sheet.add_argument("--output", default="out/sheet.jpg")

    tl = sub.add_parser("timeline", help="print the breathing timeline as JSON")
    tl.add_argument("--duration", type=float, required=True)
    tl.add_argument("--silence", action="append", default=[], help="start:end in seconds")
    tl.add_argument("--cue", action="append", default=[], help="inhale:12.5 or exhale:16")
    tl.add_argument("--sleep", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "make":
        return asyncio.run(_make(args))
    if args.command == "sheet":
        return _sheet(args)
    return _timeline(args)


if __name__ == "__main__":
    sys.exit(main())
