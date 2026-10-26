"""Muse Spark as the art director: it reads a meditation and designs its world.

The design ("scene") is strict JSON. Two parts of it drive everything after:

* `prompt`: what Muse Image paints.
* `subjects`: the 1-3 foreground objects that SAM 3.1 will cut out of the
  painting so the app can move them with the listener's breath. The director is
  told to make them solid, clearly outlined objects, because only those cut out
  cleanly. It also says how each one is attached (resting on the ground, growing
  in from an edge, or floating), which tells the app where to anchor its motion.

The listener's own words never reach the image model: the director turns the
meditation into scenery, and only scenery goes on. When the critic rejects a
painting, the director revises its own design from the critic's note (`revise`).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .meta import Meter
from .settings import DEFAULT, Settings

SUBJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "description": "Singular noun phrase, 1-3 words, e.g. 'smooth stone'"},
        "depth": {"type": "string", "enum": ["near", "mid"]},
        "attached": {"type": "string", "enum": ["ground", "edge", "air"],
                     "description": "ground: rests on ground or water; edge: grows in from a picture edge; air: floats"},
        "motion": {"type": "string", "enum": ["sway", "drift", "float", "still"]},
        "where": {"type": "string", "enum": ["lower left", "lower centre", "lower right",
                                             "left edge", "right edge", "upper left", "upper right"]},
    },
    "required": ["name", "depth", "attached", "motion", "where"],
}

SCENE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string", "description": "2-4 plain words, English, e.g. 'Still Morning Lake'"},
        "place": {"type": "string", "description": "The place, one sentence"},
        "time_of_day": {"type": "string", "enum": ["dawn", "morning", "midday", "golden hour", "dusk", "night"]},
        "palette": {"type": "array", "items": {"type": "string"}, "description": "3-5 colours as #RRGGBB"},
        "light": {"type": "string", "description": "Quality and direction of the light"},
        "air": {"type": "string", "description": "Atmosphere: haze, mist, still water, warm dust..."},
        "subjects": {"type": "array", "items": SUBJECT_SCHEMA, "description": "1-3 foreground objects"},
        "prompt": {"type": "string", "description": "The painting, for the image model: English, 60-120 words"},
        "sleep": {"type": "boolean", "description": "A dark, quiet world for falling asleep"},
    },
    "required": ["title", "place", "time_of_day", "palette", "light", "air", "subjects", "prompt", "sleep"],
}

DIRECTOR_INSTRUCTIONS = """You are the art director of ELYND, a meditation app.
A listener is about to hear the guided meditation below, written for them alone. Design ONE calm,
painted world that belongs to it. The world fills their phone screen while they listen: it greets
them as they arrive, fades while their eyes are closed, and returns at the end.

Rules:
1. Belong to this meditation. Use the place, the images and the journey it describes. If it describes
   none, choose a natural place that fits its technique and mood.
2. Calm, spacious and beautiful; painterly but believable; soft light. Portrait (9:16) composition
   with open, quiet space in the upper half.
3. Never: people, faces, hands, bodies, animals looking at the viewer, text, letters, numbers, logos,
   symbols, flags, religious icons, weapons.
4. Nothing unsettling: no storms, threatening darkness, falling, deep water seen from below,
   enclosed spaces, ruins, decay.
5. subjects: 1-3 foreground objects that will be cut out of the painting and move gently with the
   listener's breath. Each must be ONE solid, free-standing object with a clear outline ("smooth
   stone", "wooden rowboat", "paper lantern", "seashell", "ceramic teacup", "white butterfly"), never
   a texture, mass or substance (grass, waves, water, mist, clouds, sand, a field of flowers, a
   forest). Avoid branches and plants: they cannot be cut out cleanly. Name each with a singular
   noun phrase of 1-3 words. Keep them apart from each other, fully in view, and clearly separated
   from what is behind them.
   attached: "ground" if it rests on the ground or on water, "edge" if it grows into the picture from
   an edge (a branch), "air" if it floats or hangs free.
6. prompt: one paragraph of 60-120 words for the image model, in English: the place, the light, the
   palette, the air, and every subject by exactly its name and with its position.
7. Never copy the listener's words, names, people or personal details into any field. Scenery only.
8. A meditation for falling asleep gets a dark, warm, quiet world (night or late dusk, low contrast)
   and sleep = true.
Return only the JSON object."""

REVISE_INSTRUCTIONS = """You are the art director of ELYND, revising your own design.
The painting made from the design below was rejected by the critic. Change what the critic's note
asks for and nothing else: keep the place, the palette and the subjects unless the note names them as
the problem. All rules of the original brief still apply. Return the complete design as JSON."""

HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


@dataclass(frozen=True)
class Meditation:
    """What the art director needs to know about one meditation."""

    script: str
    technique: str = ""
    mood: str = ""
    language: str = "en"
    minutes: int = 10

    def as_message(self) -> str:
        return (
            f"Technique: {self.technique or 'not given'}\n"
            f"Mood before the session: {self.mood or 'not given'}\n"
            f"Length: {self.minutes} minutes\n"
            f"Language of the meditation: {self.language}\n\n"
            f"Meditation:\n{self.script.strip()}"
        )


class SceneError(ValueError):
    """A design the rest of the pipeline cannot work with."""


def check(scene: dict[str, Any]) -> dict[str, Any]:
    """Checks the schema cannot express; returns the scene with a clean palette."""
    palette = [c for c in scene.get("palette") or [] if isinstance(c, str) and HEX.match(c)]
    if len(palette) < 2:
        raise SceneError("the palette needs at least two #RRGGBB colours")
    subjects = scene.get("subjects") or []
    if not 1 <= len(subjects) <= 3:
        raise SceneError("a scene needs 1-3 subjects")
    names = set()
    for subject in subjects:
        name = " ".join((subject.get("name") or "").split()).lower()
        if not name or len(name.split()) > 4:
            raise SceneError(f"subject is not a short noun phrase: {subject.get('name')!r}")
        if name in names:
            raise SceneError(f"subject named twice: {name!r}")
        names.add(name)
        subject["name"] = name
    if len((scene.get("prompt") or "").split()) < 20:
        raise SceneError("the painting prompt is too short")
    if not (scene.get("title") or "").strip():
        raise SceneError("the scene has no title")
    return {**scene, "palette": palette[:5]}


def _response_format() -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": "scene", "schema": SCENE_SCHEMA, "strict": True}}


async def design(client, meditation: Meditation, meter: Meter, settings: Settings = DEFAULT) -> dict[str, Any]:
    """Muse Spark designs the world of one meditation."""
    response = await client.chat.completions.create(
        model=settings.director_model,
        messages=[
            {"role": "system", "content": DIRECTOR_INSTRUCTIONS},
            {"role": "user", "content": meditation.as_message()},
        ],
        response_format=_response_format(),
        timeout=settings.design_timeout,
    )
    meter.spark(getattr(response, "usage", None), "design")
    return check(json.loads(response.choices[0].message.content))


async def revise(client, scene: dict[str, Any], critique: dict[str, Any], meter: Meter,
                 settings: Settings = DEFAULT) -> dict[str, Any]:
    """The director rewrites its design from the critic's note on the painting."""
    note = critique.get("fix") or critique.get("reason") or "The painting did not pass."
    response = await client.chat.completions.create(
        model=settings.director_model,
        messages=[
            {"role": "system", "content": DIRECTOR_INSTRUCTIONS + "\n\n" + REVISE_INSTRUCTIONS},
            {"role": "user", "content": (
                f"Your design:\n{json.dumps(scene, ensure_ascii=False)}\n\n"
                f"The critic's verdict on the painting:\n{json.dumps(critique, ensure_ascii=False)}\n\n"
                f"What to change: {note}"
            )},
        ],
        response_format=_response_format(),
        timeout=settings.design_timeout,
    )
    meter.spark(getattr(response, "usage", None), "revise")
    return check(json.loads(response.choices[0].message.content))
