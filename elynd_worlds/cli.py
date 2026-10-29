"""Command line: make worlds from meditation texts, look at them, keep the spend in view.

    elynd-worlds make examples/lake.txt --technique visualization --mood stressed
    elynd-worlds make examples/*.txt --parallel 4            # a quality batch
    elynd-worlds sheet out/batch-*/                          # one contact sheet
    elynd-worlds pick examples/*.txt --library library/worlds  # which library world, or its own?
    elynd-worlds timeline --duration 600 --silence 120:180   # the breathing timeline

`make` refuses to start when the spend ledger (a CSV of every run's cost) is
already at the budget. The key comes from $MODEL_API_KEY or --key-file.

With an open-model key ($OPEN_MODEL_API_KEY or --open-key-file), Muse Spark Open
on Baseten is the critic, and its spend goes to a ledger of its own. Once that
ledger reaches --open-budget, Muse Spark on Meta Model API judges again.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import library, timeline
from .art_director import Meditation
from .director import make_world
from .meta import Meter, connect, connect_open, encode
from .settings import DEFAULT

# The most one world's critic can cost on the open model: two paintings, about 1.2 cents each at most.
OPEN_ESTIMATE_USD = 0.03


def read_key(path: str | None, env: str = "MODEL_API_KEY") -> str | None:
    if path:
        return Path(path).expanduser().read_text().strip()
    return os.environ.get(env)


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


def open_critic(args: argparse.Namespace):
    """(the open-model client for the critic, or None; and why): None without a key or at the budget."""
    key = read_key(args.open_key_file, "OPEN_MODEL_API_KEY")
    if not key:
        return None, "no open-model key"
    already = spent(Path(args.open_ledger).expanduser())
    if already + OPEN_ESTIMATE_USD * len(args.scripts) > args.open_budget:
        return None, f"the open-model ledger shows ${already:.2f} of its ${args.open_budget:.2f} budget"
    return connect_open(key), f"${already:.2f} of ${args.open_budget:.2f} spent"


async def _make(args: argparse.Namespace) -> int:
    ledger = Path(args.ledger).expanduser()
    already = spent(ledger)
    estimate = 0.07 * len(args.scripts)
    if already + estimate > args.budget:
        print(f"refusing: the ledger shows ${already:.2f} spent; this run may add ~${estimate:.2f}, "
              f"over the ${args.budget:.2f} budget", file=sys.stderr)
        return 2
    client = connect(read_key(args.key_file))
    # Several worlds share one process here, so steps wait on each other more than on a server: the open
    # critic gets longer before Muse Spark takes over.
    settings = dataclasses.replace(DEFAULT, open_critic_timeout=args.open_timeout, open_designer=args.open_designer)
    open_client, why = open_critic(args)
    print(f"critic: {'Muse Spark Open' if open_client else 'Muse Spark'} ({why})", file=sys.stderr)
    if args.open_designer and open_client is None:
        print("refusing: --open-designer needs the open model (its key, and its ledger under budget)", file=sys.stderr)
        return 2
    print(f"designer: {'Muse Spark Open' if args.open_designer else 'Muse Spark'}", file=sys.stderr)
    out = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    gate = asyncio.Semaphore(args.parallel)

    async def one(path: Path) -> dict:
        async with gate:
            meter = Meter()
            started = time.monotonic()
            try:
                world = await make_world(client, meditation_for(path, args), settings=settings, meter=meter,
                                         open_client=open_client)
            except Exception as error:  # noqa: BLE001 - report and keep the batch going
                return {"name": path.stem, "ok": False, "error": f"{type(error).__name__}: {error}"[:300],
                        "usd": meter.usd, "meta_usd": meter.meta_usd, "open_usd": meter.open_usd}
            folder = world.save(out / path.stem)
            return {"name": path.stem, "ok": True, "title": world.scene["title"], "usd": meter.usd,
                    "meta_usd": meter.meta_usd, "open_usd": meter.open_usd,
                    "seconds": round(time.monotonic() - started, 1), "paintings": world.paintings,
                    "score": world.verdict.get("score"), "judge": world.verdict.get("judge"),
                    "designer": settings.open_designer_model if args.open_designer else "muse-spark",
                    "design_seconds": next((s.seconds for s in world.trace if s.name == "design"), None),
                    "plate": world.plate_source,
                    "layers": [layer.wording for layer in world.layers] if world.moving else [],
                    "wanted": [s["name"] for s in world.scene["subjects"]], "folder": str(folder)}

    results = await asyncio.gather(*(one(Path(p)) for p in args.scripts))
    total = round(sum(r["meta_usd"] for r in results), 5)
    open_total = round(sum(r["open_usd"] for r in results), 5)
    record(ledger, "make", total, f"{len(results)} worlds -> {out}")
    if open_total:
        record(Path(args.open_ledger).expanduser(), "make", open_total,
               f"{'design and critic' if args.open_designer else 'critic'} of {len(results)} worlds -> {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    print(f"{sum(r['ok'] for r in results)}/{len(results)} worlds, ${total:.4f} Meta + ${open_total:.4f} open model "
          f"this run, ${already + total:.4f} in the Meta ledger -> {out}")
    return 0


async def _pick(args: argparse.Namespace) -> int:
    ledger = Path(args.ledger).expanduser()
    if spent(ledger) + 0.01 * len(args.scripts) > args.budget:
        print(f"refusing: the ledger is at its ${args.budget:.2f} budget", file=sys.stderr)
        return 2
    entries = library.entries_in(args.library)
    if not entries:
        print(f"no library worlds under {args.library}", file=sys.stderr)
        return 1
    client, meter = connect(read_key(args.key_file)), Meter()
    for path in args.scripts:
        started = time.monotonic()
        chosen = await library.pick(client, meditation_for(Path(path), args), entries, meter)
        print(json.dumps({"name": Path(path).stem, "library": chosen.entry.id if chosen else None,
                          "fit": chosen.fit if chosen else None, "own_world": chosen.own_world if chosen else None,
                          "place": chosen.place if chosen else None,
                          "paint_its_own": library.needs_own_world(chosen),
                          "seconds": round(time.monotonic() - started, 1)}))
    record(ledger, "pick", meter.usd, f"{len(args.scripts)} library choices from {len(entries)} worlds")
    print(f"{len(args.scripts)} choices among {len(entries)} library worlds, ${meter.usd:.4f}")
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


def _loop(args: argparse.Namespace) -> int:
    from . import loop
    w, h = (int(v) for v in args.size.split("x"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for folder in map(Path, args.worlds):
        made = loop.make_loop(folder, out_dir / f"{folder.name}.mp4", size=(w, h), ratio=args.ratio,
                              bitrate=args.bitrate, music=Path(args.music) if args.music else None,
                              music_from=args.music_from, flutter=args.flutter,
                              package=Path(args.package) if args.package else None)
        print(json.dumps(made))
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
    make.add_argument("--open-key-file", default=os.environ.get("OPEN_MODEL_API_KEY_FILE"),
                      help="Muse Spark Open (Baseten) as the critic; without a key, Muse Spark judges")
    make.add_argument("--open-ledger", default=os.environ.get("ELYND_WORLDS_OPEN_LEDGER", "out/open_spend.csv"))
    make.add_argument("--open-budget", type=float, default=DEFAULT.open_budget_usd)
    make.add_argument("--open-timeout", type=float, default=45.0,
                      help="seconds the open critic gets before Muse Spark judges (a batch shares one process)")
    make.add_argument("--open-designer", action="store_true",
                      help="Muse Spark Open designs each world too: everything but painting and cutting on open weights")

    pick = sub.add_parser("pick", help="the library world each meditation gets, or whether it needs its own")
    pick.add_argument("scripts", nargs="+")
    pick.add_argument("--library", nargs="+", required=True, help="folders of saved library worlds")
    pick.add_argument("--technique", default="")
    pick.add_argument("--mood", default="")
    pick.add_argument("--language", default="en")
    pick.add_argument("--minutes", type=int, default=10)
    pick.add_argument("--key-file", default=os.environ.get("MODEL_API_KEY_FILE"))
    pick.add_argument("--ledger", default=os.environ.get("ELYND_WORLDS_LEDGER", "out/spend.csv"))
    pick.add_argument("--budget", type=float, default=float(os.environ.get("ELYND_WORLDS_BUDGET", "25")))

    sheet = sub.add_parser("sheet", help="one contact sheet of saved worlds")
    sheet.add_argument("folders", nargs="+")
    sheet.add_argument("--output", default="out/sheet.jpg")

    lp = sub.add_parser("loop", help="each world's shared clip: 12 s that loop without a seam (MP4, marked as made by AI)")
    lp.add_argument("worlds", nargs="+", help="saved world folders (world.json and its pictures)")
    lp.add_argument("--out-dir", default="out/loops")
    lp.add_argument("--size", default="360x640", help="the screen in points; times --ratio in pixels (1080 x 1920)")
    lp.add_argument("--ratio", type=float, default=3.0)
    lp.add_argument("--bitrate", default="3M")
    lp.add_argument("--music", help="a bed of our own music, looped like the picture")
    lp.add_argument("--music-from", type=float, default=60.0, help="where in the music the bed starts (s)")
    lp.add_argument("--flutter", default=os.environ.get("FLUTTER", "flutter"))
    lp.add_argument("--package", help="a copy of flutter/elynd_worlds_scene to render in (one per parallel run)")

    tl = sub.add_parser("timeline", help="print the breathing timeline as JSON")
    tl.add_argument("--duration", type=float, required=True)
    tl.add_argument("--silence", action="append", default=[], help="start:end in seconds")
    tl.add_argument("--cue", action="append", default=[], help="inhale:12.5 or exhale:16")
    tl.add_argument("--sleep", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(format="%(asctime)s %(name)s %(message)s", level=logging.WARNING)
    logging.getLogger("elynd_worlds").setLevel(logging.INFO)  # e.g. why Muse Spark judged instead
    if args.command == "make":
        return asyncio.run(_make(args))
    if args.command == "pick":
        return asyncio.run(_pick(args))
    if args.command == "sheet":
        return _sheet(args)
    if args.command == "loop":
        return _loop(args)
    return _timeline(args)


if __name__ == "__main__":
    sys.exit(main())
