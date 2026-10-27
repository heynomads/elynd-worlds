"""library_summary.py [catalog folder]: one line per library world -> library/catalog.json (committed: what
the library holds, without its pictures). Reads the catalog written from the saved worlds (scene.json,
world.json, verdict.json per theme)."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(folder: str = "library/catalog") -> int:
    rows = []
    for theme in sorted(p for p in Path(folder).iterdir() if (p / "world.json").exists()):
        world = json.loads((theme / "world.json").read_text())
        scene = json.loads((theme / "scene.json").read_text())
        verdict = json.loads((theme / "verdict.json").read_text()) if (theme / "verdict.json").exists() else {}
        rows.append({
            "id": theme.name, "title": world["title"], "place": scene["place"], "time_of_day": scene["time_of_day"],
            "sleep": bool(world.get("sleep")), "moods": world.get("moods") or [],
            "subjects": [s["name"] for s in scene["subjects"]],
            "moving": [layer["name"] for layer in world.get("layers") or []],
            "nature": [k for k in ("water", "sky", "grass", "stream", "trees") if (world.get("living") or {}).get(k)],
            "particles": (world.get("living") or {}).get("particles", "none"),
            "critic": {"score": verdict.get("score"), "judge": verdict.get("judge")},
            "cost_usd": world.get("cost_usd"),
        })
    out = Path("library/catalog.json")
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    judges = {}
    for r in rows:
        judges[r["critic"]["judge"]] = judges.get(r["critic"]["judge"], 0) + 1
    print(f"{len(rows)} worlds, {sum(r['sleep'] for r in rows)} for sleep, {sum(bool(r['moving']) for r in rows)} with "
          f"moving layers; judged by {judges}; ${sum(r['cost_usd'] or 0 for r in rows):.2f} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
