"""Which library worlds would look dead on the phone (library.looks_dead), from the catalog's small JSON files.

    python scripts/library_audit.py            # counts, and each dead world with the reason
    python scripts/library_audit.py --ids      # only the ids of the dead worlds, one per line

The catalog (library/catalog/<id>/) holds each library world's world.json and scene.json as on S3
(library_catalog.py in the workspace fetches them); no pictures are needed.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from elynd_worlds.library import looks_dead

CATALOG = Path(__file__).resolve().parent.parent / "library" / "catalog"


def audit(catalog: Path = CATALOG) -> dict[str, str | None]:
    """Each library world's id → why it looks dead, or None."""
    found = {}
    for folder in sorted(p for p in catalog.iterdir() if (p / "world.json").exists()):
        manifest = json.loads((folder / "world.json").read_text())
        scene = json.loads((folder / "scene.json").read_text())
        found[folder.name] = looks_dead(manifest, scene)
    return found


def main(argv: list[str]) -> int:
    found = audit()
    dead = {id_: why for id_, why in found.items() if why}
    if "--ids" in argv:
        print("\n".join(dead))
        return 0
    print(f"{len(found)} library worlds: {len(found) - len(dead)} look alive, {len(dead)} look dead")
    kinds = Counter(why.split(" (")[0] for why in dead.values())
    for kind, n in kinds.most_common():
        print(f"\n{n} × {kind}:")
        print("  " + ", ".join(id_ for id_, why in dead.items() if why.split(" (")[0] == kind))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
