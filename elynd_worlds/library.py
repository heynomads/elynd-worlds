"""The library: worlds painted ahead of time, and Muse Spark choosing one for a meditation.

A world made for a meditation takes about a minute, and listeners close their
eyes early. So a session opens on a library world: one of the worlds the agent
painted in advance, from short themes across the places guided meditations
visit most (scripts/library_briefs.py), each checked by the critic like any
other. Muse Spark reads the meditation, the same text the art director reads
and nothing more, and names the three library worlds that belong to it best;
the first one the listener has not seen lately is shown. The world made for the
meditation takes over when it is ready.

Sleep is decided before Muse Spark is asked: a meditation for falling asleep
chooses only among the dark, quiet worlds, any other only among the rest.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .art_director import Meditation
from .meta import Meter, privacy_fields, reasoning
from .settings import DEFAULT, Settings

SLEEP_TECHNIQUES = {"sleep_journey"}


@dataclass(frozen=True)
class Entry:
    """One library world, as the chooser sees it."""

    id: str
    title: str
    place: str
    time_of_day: str
    sleep: bool = False

    def line(self) -> str:
        return f"{self.id} | {self.title} | {self.place} | {self.time_of_day}"

    @classmethod
    def of_scene(cls, id: str, scene: dict[str, Any]) -> Entry:
        return cls(id=id, title=scene["title"], place=scene["place"], time_of_day=scene["time_of_day"],
                   sleep=bool(scene.get("sleep")))


def entries_in(folders: Iterable[str | Path]) -> list[Entry]:
    """The library worlds saved under `folders` (each world's folder name is its id)."""
    found = {}
    for folder in folders:
        for scene in sorted(Path(folder).rglob("scene.json")):
            if (scene.parent / "world.json").exists():
                found[scene.parent.name] = Entry.of_scene(scene.parent.name, json.loads(scene.read_text()))
    return list(found.values())


@dataclass(frozen=True)
class Pick:
    entry: Entry
    fit: int     # 0-10: how well it belongs to the meditation (9-10: the same place and light)


PICK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "choices": {
            "type": "array",
            "description": "The three worlds that belong to the meditation best, best first",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "description": "Exactly as listed"},
                    "fit": {"type": "integer", "description": "0-10"},
                },
                "required": ["id", "fit"],
            },
        },
    },
    "required": ["choices"],
}

PICK_INSTRUCTIONS = """You choose the painted world that fills a listener's phone screen while their
guided meditation plays in ELYND, a meditation app. Choose from the library below the three worlds
that belong to the meditation best, best first:
1. The place the meditation describes or travels to (a beach, a forest, a garden, a lake...), then its
   time of day, season and weather.
2. If it describes no place, worlds whose light and mood suit its technique and feeling: soft and
   gentle for sadness, bright and fresh for energy or focus, warm for gratitude.
fit (0-10): 9-10 the same place and light; 6-8 the same kind of place; 3-5 only the mood fits.
Use the ids exactly as listed. Return only the JSON object.

The library (id | title | place | time of day):
"""


def is_sleep(meditation: Meditation) -> bool:
    return meditation.technique in SLEEP_TECHNIQUES


async def pick(client, meditation: Meditation, entries: list[Entry], meter: Meter, settings: Settings = DEFAULT,
               recent: Iterable[str] = ()) -> Pick | None:
    """The library world for one meditation; None if the library has none of its kind or the answer is unusable.

    `recent`: ids the listener saw lately; the next best is shown instead when it fits nearly as well.
    """
    sleep = is_sleep(meditation)
    candidates = [e for e in entries if e.sleep == sleep]
    if not candidates:
        return None
    # The library first and the meditation last: the long, unchanging start of the request can be cached.
    response = await client.chat.completions.create(
        model=settings.library_model,
        messages=[
            {"role": "system", "content": PICK_INSTRUCTIONS + "\n".join(e.line() for e in candidates)},
            {"role": "user", "content": _shortened(meditation, settings.pick_words).as_message()},
        ],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "pick", "schema": PICK_SCHEMA, "strict": True}},
        **reasoning(settings.pick_reasoning),
        extra_body=privacy_fields(),
        timeout=settings.pick_timeout,
    )
    meter.spark(getattr(response, "usage", None), "pick")
    choices = json.loads(response.choices[0].message.content).get("choices") or []
    return choose(choices, candidates, recent)


def choose(choices: list[dict[str, Any]], candidates: list[Entry], recent: Iterable[str] = (),
           near: int = 2) -> Pick | None:
    """The first choice the listener has not seen lately and that fits within `near` of the best."""
    by_id = {e.id: e for e in candidates}
    valid = []
    for choice in choices:
        entry = by_id.get(str(choice.get("id", "")).strip())
        if entry is not None and entry.id not in {p.entry.id for p in valid}:
            valid.append(Pick(entry, max(0, min(10, int(choice.get("fit") or 0)))))
    if not valid:
        return None
    seen = set(recent)
    best = valid[0]
    return next((p for p in valid if p.entry.id not in seen and p.fit >= best.fit - near), best)


def _shortened(meditation: Meditation, words: int) -> Meditation:
    """The meditation's first `words` words: the place is set early, and the rest only costs tokens."""
    script = meditation.script.split()
    if len(script) <= words:
        return meditation
    return Meditation(script=" ".join(script[:words]) + " ...", technique=meditation.technique,
                      mood=meditation.mood, language=meditation.language, minutes=meditation.minutes)
