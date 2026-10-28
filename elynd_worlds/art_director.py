"""Muse Spark as the art director: it reads a meditation and designs its world.

The design ("scene") is strict JSON. Two parts of it drive everything after:

* `prompt`: what Muse Image paints.
* `subjects`: the 1-3 foreground objects that SAM 3.1 will cut out of the
  painting so the app can move them as such things move. The director is told to
  make them solid, clearly outlined objects, because only those cut out cleanly.
  It also says how each one is attached (resting on the ground, growing in from
  an edge, or floating), how it moves, and whether it glows (a lantern flickers).
* `nature` and `particles`: the parts of the place that move in nature (water,
  sky, grass, a stream, tree crowns), which SAM 3.1 then finds in the painting
  (regions.py), and what floats in the air (leaves, snow, pollen, fireflies).

The listener's own words never reach the image model: the director turns the
meditation into scenery, and only scenery goes on. When the critic rejects a
painting, the director revises its own design from the critic's note (`revise`).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from .meta import Meter, cache_fields, json_object, privacy_fields, reasoning, spark_chat
from .settings import DEFAULT, Settings

log = logging.getLogger("elynd_worlds")

# The parts of a place that move in nature, as the app animates them: open water
# ripples, the sky drifts, grass waves in the wind, a stream flows, tree crowns sway.
NATURE = ["water", "sky", "grass", "stream", "trees"]
PARTICLES = ["none", "leaves", "snow", "pollen", "fireflies"]

SUBJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "description": "Singular noun phrase, 1-3 words, e.g. 'smooth stone'"},
        "depth": {"type": "string", "enum": ["near", "mid"]},
        "attached": {"type": "string", "enum": ["ground", "edge", "air"],
                     "description": "ground: rests on ground or water; edge: grows in from a picture edge; air: floats"},
        "motion": {"type": "string", "enum": ["sway", "drift", "float", "flutter", "still"],
                   "description": "sway: a branch or flower in the wind; drift: mist, a cloud; float: a boat on water, "
                                  "a leaf in the air; flutter: a butterfly or moth (folds its wings now and then); "
                                  "still: stone, post, shell, cabin, lantern, a resting animal or person"},
        "glows": {"type": "boolean", "description": "A light that glows by itself: lantern, candle, campfire"},
        "where": {"type": "string", "enum": ["left", "centre", "right"],
                  "description": "Across the picture; every subject sits in its middle band (35-60% down)"},
    },
    "required": ["name", "depth", "attached", "motion", "glows", "where"],
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
        "nature": {"type": "array", "items": {"type": "string", "enum": NATURE},
                   "description": "The parts of the place that move in nature and are clearly visible"},
        "particles": {"type": "string", "enum": PARTICLES, "description": "What floats through the air, if anything"},
        "prompt": {"type": "string", "description": "The painting, for the image model: English, 60-120 words"},
        "sleep": {"type": "boolean", "description": "A dark, quiet world for falling asleep"},
    },
    "required": ["title", "place", "time_of_day", "palette", "light", "air", "subjects", "nature", "particles",
                 "prompt", "sleep"],
}

DIRECTOR_INSTRUCTIONS = """You are the art director of ELYND, a meditation app.
A listener is about to hear the guided meditation below, written for them alone. Design ONE calm,
painted world that belongs to it. The world fills their phone screen while they listen: it greets
them as they arrive, fades while their eyes are closed, and returns at the end.

Rules:
1. Belong to this meditation. Use the place, the images and the journey it describes. If it describes
   none, choose a natural place that fits its technique and mood.
2. Calm, spacious and beautiful; painterly but believable; soft light. Portrait (9:16) composition.
   The phone shows its title over the top 15% of the picture and its words and controls over the
   bottom 35%, under a soft shade: keep both calm and simple (open sky; still water, soft ground or
   sand), and put everything that matters in the middle band, 35-60% down the picture.
3. Animals and people belong in a world when the meditation brings them (a dog on its beach, a monkey
   asleep on its blanket, swans on the lake, someone sitting on a bench): always at rest (lying,
   sitting, standing, looking out, asleep, floating on water), never running, jumping or walking,
   since they cannot move like that on the phone; an animal never looks at the viewer; a person is
   small, at a distance or seen from behind, never a face up close. Paint what the meditation says of
   them (a golden retriever, a small monkey, a grey-haired woman), never a name.
   Never: text, letters, numbers, logos, symbols, flags, religious icons, weapons.
4. Nothing unsettling: no storms, threatening darkness, falling, deep water seen from below,
   enclosed spaces, ruins, decay.
5. subjects: 1-3 foreground objects that will be cut out of the painting and move gently, the way
   such things move (a boat rides the ripples, a butterfly flutters, a lantern flickers, a stone
   stays still). Each must be ONE solid, free-standing object with a clear outline ("smooth
   stone", "wooden rowboat", "paper lantern", "seashell", "ceramic teacup", "white butterfly"), never
   a texture, mass or substance (grass, waves, water, mist, clouds, sand, a field of flowers, a
   forest). An animal or a person is one subject too ("sleeping dog", "small monkey", "white swan"). Avoid branches and plants: they cannot be cut out cleanly. Name each with a singular
   noun phrase of 1-3 words. Make each large enough to notice on a phone (at least a tenth of the
   picture's width). Keep them apart from each other, fully in view, clearly separated from
   what is behind them, inside the middle band (35-60% down the picture), never at the bottom, and
   within the middle 70% of the width (phones crop the sides).
   attached: "ground" if it rests on the ground or on water, "edge" if it grows into the picture from
   an edge (a branch), "air" if it floats or hangs free. motion: how it moves in nature (a boat on
   water floats, a butterfly flutters, a stone, a lantern, a resting animal or person is still: it
   stays as painted while the world moves around it; a swan or a duck on the water floats). glows: true only for a light
   that glows by itself (a lantern, a candle, a campfire).
6. prompt: one paragraph of 60-120 words for the image model, in English: the place, the light, the
   palette, the air, and every subject by exactly its name and with its position in the middle of the
   picture. Say that the bottom of the picture is calm and simple.
7. Never copy the listener's words or anyone's name into any field: describe the place, its things,
   animals and people in your own words.
8. A meditation for falling asleep gets a dark, warm, quiet world (night or late dusk, low contrast)
   and sleep = true.
9. nature: the parts of the place that move in nature and are clearly visible in the painting:
   "water" (a lake or the sea), "sky", "grass" (a meadow), "stream" (a brook or a river),
   "trees" (leafy crowns). Choose places with some of them: a world comes alive through them.
   particles: what drifts through the air, if it suits the place and the season: "leaves" (autumn),
   "snow" (winter), "pollen" (a summer meadow), "fireflies" (a warm summer night), else "none".
Return only the JSON object."""

REVISE_INSTRUCTIONS = """You are the art director of ELYND, revising your own design.
The painting made from the design below was rejected by the critic. Change what the critic's note
asks for and nothing else: keep the place, the palette and the subjects unless the note names them as
the problem. All rules of the original brief still apply. Return the complete design as JSON."""

HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")
DEFAULT_PALETTE = ("#2E4A52", "#9DB7B0", "#E8D9C0")


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
        # Only tints in the app come from it: a calm default, rather than losing a good design (27 Oct: 2 of
        # 55 library designs had one).
        palette = list(DEFAULT_PALETTE)
    subjects = scene.get("subjects") or []
    if not 1 <= len(subjects) <= 3:
        raise SceneError("a scene needs 1-3 subjects")
    names, kept = set(), []
    for subject in subjects:
        name = " ".join((subject.get("name") or "").split()).lower()
        if not name or len(name.split()) > 4:
            raise SceneError(f"subject is not a short noun phrase: {subject.get('name')!r}")
        if name in names:
            continue  # "two wooden chairs" designed as two subjects: one layer holds both (cutter.find)
        names.add(name)
        subject["name"] = name
        kept.append(subject)
    subjects = kept
    if len((scene.get("prompt") or "").split()) < 20:
        raise SceneError("the painting prompt is too short")
    if not (scene.get("title") or "").strip():
        raise SceneError("the scene has no title")
    for subject in subjects:
        subject["glows"] = subject.get("glows") is True
    nature = [n for n in dict.fromkeys(scene.get("nature") or []) if n in NATURE]
    particles = scene.get("particles") if scene.get("particles") in PARTICLES else "none"
    return {**scene, "palette": palette[:5], "subjects": subjects, "nature": nature, "particles": particles}


def _response_format() -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": "scene", "schema": SCENE_SCHEMA, "strict": True}}


async def design(client, meditation: Meditation, meter: Meter, settings: Settings = DEFAULT,
                 open_client=None) -> dict[str, Any]:
    """Muse Spark designs the world of one meditation (Muse Spark Open, with `Settings.open_designer`)."""
    messages = [
        {"role": "system", "content": DIRECTOR_INSTRUCTIONS},
        {"role": "user", "content": meditation.as_message()},
    ]
    return await _ask(client, messages, "design", meter, settings, open_client)


async def revise(client, scene: dict[str, Any], critique: dict[str, Any], meter: Meter,
                 settings: Settings = DEFAULT, open_client=None) -> dict[str, Any]:
    """The director rewrites its design from the critic's note on the painting."""
    note = critique.get("fix") or critique.get("reason") or "The painting did not pass."
    messages = [
        {"role": "system", "content": DIRECTOR_INSTRUCTIONS + "\n\n" + REVISE_INSTRUCTIONS},
        {"role": "user", "content": (
            f"Your design:\n{json.dumps(scene, ensure_ascii=False)}\n\n"
            f"The critic's verdict on the painting:\n"
            f"{json.dumps({k: v for k, v in critique.items() if k != 'judge'}, ensure_ascii=False)}\n\n"
            f"What to change: {note}"
        )},
    ]
    return await _ask(client, messages, "revise", meter, settings, open_client)


async def _ask(client, messages: list[dict[str, Any]], step: str, meter: Meter, settings: Settings,
               open_client) -> dict[str, Any]:
    """A design (or its revision) from Muse Spark, or from Muse Spark Open with `Settings.open_designer`."""
    if settings.open_designer and open_client is not None:
        # The open model's host enforces no answer shape, and its filter may refuse a calm meditation (a desert
        # under the stars, a hand on the belly: 2 of 20 on 28 Oct 2026). So: once more when the answer is out of
        # shape, and Muse Spark designs whenever the open model cannot.
        for attempt in (1, 2):
            try:
                # No privacy fields: they are Meta's (critic.py does the same).
                response = await open_client.chat.completions.create(
                    model=settings.open_designer_model, messages=messages, response_format=_response_format(),
                    timeout=settings.open_design_timeout, **reasoning(settings.open_designer_reasoning))
                meter.spark_open(getattr(response, "usage", None), step)
                return check(json_object(response.choices[0].message.content))
            except ValueError as error:  # out of shape (a SceneError or a JSONDecodeError)
                log.info("the open designer answered out of shape (attempt %d): %s", attempt, error)
            except Exception as error:  # noqa: BLE001 - refused, failed or too slow: Muse Spark designs
                log.info("the open designer failed, Muse Spark designs: %s: %s", type(error).__name__, error)
                break
    response, _ = await spark_chat(
        client, (settings.director_model, settings.spark_fallback_model),
        messages=messages,
        response_format=_response_format(),
        extra_body={**privacy_fields(), **cache_fields("design")},
        timeout=settings.design_timeout,
    )
    meter.spark(getattr(response, "usage", None), step)
    return check(json.loads(response.choices[0].message.content))
