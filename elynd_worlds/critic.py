"""The critic: it looks at each painting before a listener does.

Image models sometimes slip: a dog with five legs, a face in the clouds, letters on a
boat, a sky that turned stormy, a figure nobody designed. A world is the place alone unless its
meditation brings an animal or a person (art_director.figures_in), and then holds only those, asleep or
lying at rest and painted naturally. The critic is told which ones belong (`figures_rule`) and turns
away any other (29 Oct 2026), one drawn wrong, a face up close, a figure looking at the viewer, and one
awake and upright: a cut-out cannot turn its head or walk, and an awake figure that never moves reads
as a still picture. The critic sees the painting next to the design it was made from
and answers in strict JSON. A painting is shown only if it is safe, calm,
belongs to the meditation and scores at least `Settings.pass_score`. When it
fails, the critic says in one sentence what to change, and the art director
revises its design from that note.

The critic is Muse Spark Open, Meta's open-weights model, when the deployment
passes an open-model client (Baseten serves it). It sees only the painting and
the scenery design, never the listener's words or id. When it fails, is slow or
answers out of shape, Muse Spark on Meta Model API judges instead, so a world
never waits on one provider, unless `Settings.open_critic_only` says no other
model may judge: then the painting is never shown (CriticUnavailable).
`verdict["judge"]` names the model that judged.

A journey's later stages (journeys/stages.py) are judged next to the first stage's painting (`reference`): the
critic then also says whether the stage shows the same place from the same viewpoint (`same_place`), and a stage
that does not never passes.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .art_director import figures_in, the_names
from .meta import Meter, cache_fields, data_url, json_object, preview_jpeg, privacy_fields, reasoning, spark_chat
from .settings import DEFAULT, Settings

log = logging.getLogger("elynd_worlds")

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "belongs": {"type": "boolean", "description": "The painting shows the designed place and mood"},
        "calm": {"type": "boolean", "description": "Calm and inviting; nothing threatening or busy"},
        "figures_wrong": {"type": "boolean",
                          "description": "A person or animal that does not belong, drawn wrong, awake and upright, a "
                                         "face up close, or a figure looking at the viewer"},
        "text_or_symbols": {"type": "boolean", "description": "Any letters, numbers, logos or symbols"},
        "subjects_seen": {"type": "array", "items": {"type": "string"},
                          "description": "The designed subjects that are clearly visible and separate"},
        "score": {"type": "integer", "description": "0-10: how well it would serve a meditation"},
        "reason": {"type": "string", "description": "One sentence"},
        "fix": {"type": "string", "description": "If score < 8: one sentence on what to change in the design"},
    },
    "required": ["belongs", "calm", "figures_wrong", "text_or_symbols", "subjects_seen", "score", "reason", "fix"],
}

CRITIC_INSTRUCTIONS = """You are the critic of a meditation app. Before a listener sees a painting,
you check it against the design it was made from. Be strict about safety and calm, fair about art.
- figures_wrong: true if a person or an animal is drawn wrong (an extra or missing limb, a twisted or
  melted body, odd eyes, a figure that is half there), if a face is seen up close, if a hand fills the
  foreground, if a figure looks at the viewer, or if a person or an animal is awake and upright (standing,
  sitting up, walking, looking out). Animals asleep and people lying at rest, painted naturally, are fine
  where they belong in this world (below); so are birds floating on water there.
- text_or_symbols: true if you see any letters, numbers, logos, signs or symbols.
- calm: false if anything feels threatening, chaotic, sad or busy.
- belongs: false if the painting does not show the designed place, time of day and mood.
- subjects_seen: list the designed subjects (by their design names) that are clearly visible,
  whole and separate from what is behind them.
- reason: one sentence on what you see and why it passes or fails; never empty.
- fix: when the score is below 8, say in one sentence what the art director should change in the
  design; otherwise an empty string.
Return only the JSON object."""

FLAGS = ("belongs", "calm", "figures_wrong", "text_or_symbols")

# A journey's later stage, judged next to its first stage: the same checks, and whether it is the same place.
STAGE_VERDICT_SCHEMA: dict[str, Any] = {
    **VERDICT_SCHEMA,
    "properties": {**VERDICT_SCHEMA["properties"], "same_place": {
        "type": "boolean",
        "description": "The painting shows the first picture's place from the same viewpoint, only at another stage"}},
    "required": [*VERDICT_SCHEMA["required"], "same_place"],
}
STAGE_FLAGS = (*FLAGS, "same_place")
# A world of the place alone: its design brings no animal and no person, or `Settings.figures` is off (a journey's
# world).
PLACE_ALONE = ("This world is the place alone: any person or animal in the painting, at rest or not, makes "
               "figures_wrong true.")
# A world whose meditation brings animals or people: those, and no others.
ONLY_THESE = ("The only animals or people that belong in this world: {}. Any other person or animal, at rest or not, "
              "makes figures_wrong true.")
STAGE_INSTRUCTIONS = """This painting is a later stage of a world that grows: the place of the first picture you are
shown, at another time. Judge the painting named as the one you judge; the first picture is only for comparing.
- same_place: true if the painting shows the same place as the first picture, from the same viewpoint: the same
  landforms, path, water and landmarks and the same designed subjects, in about the same places and at about the
  same sizes. The light, the sky, the season, snow or blossom, and plants that have grown may all differ. False if
  the view moved or turned, if it is another place, or if something was added that was not there (a building, an
  object, a figure, text)."""

# The answer's shape, first in the open model's instructions: its host ignores `response_format`, and Meta's guide
# says Muse Spark Open "pays more attention to the beginning of the system prompt" (dev.meta.ai, Muse Spark Open,
# prompting).
OPEN_SHAPE = ('Answer with one JSON object and nothing else, with exactly these keys: {"belongs": true or false, '
              '"calm": true or false, "figures_wrong": true or false, "text_or_symbols": true or false, '
              '"subjects_seen": [design names], "score": an integer 0-10, "reason": "one sentence", '
              '"fix": "one sentence, or empty"}.')
OPEN_STAGE_SHAPE = OPEN_SHAPE.replace('"subjects_seen"', '"same_place": true or false, "subjects_seen"')


def figures_rule(scene: dict[str, Any], settings: Settings = DEFAULT) -> str:
    """What the critic is told of figures: the animals and people the design brings (`Settings.figures` on), the
    only ones that belong, or else that the world is the place alone."""
    brought = figures_in(scene) if settings.figures else []
    return ONLY_THESE.format(the_names(brought)) if brought else PLACE_ALONE


def passes(verdict: dict[str, Any], settings: Settings = DEFAULT) -> bool:
    return (bool(verdict.get("belongs")) and bool(verdict.get("calm"))
            and not verdict.get("figures_wrong") and not verdict.get("text_or_symbols")
            and verdict.get("same_place") is not False  # a journey's later stage only (`reference`)
            and int(verdict.get("score") or 0) >= settings.pass_score)


class CriticUnavailable(RuntimeError):
    """Muse Spark Open could not judge a painting, and no other model may (`Settings.open_critic_only`)."""


async def critique(client, painting: bytes, scene: dict[str, Any], meter: Meter,
                   settings: Settings = DEFAULT, open_client=None, *, reference: bytes | None = None) -> dict[str, Any]:
    """The verdict on one painting: Muse Spark Open's when `open_client` is given and answers, else Muse Spark's
    (or, with `settings.open_critic_only`, CriticUnavailable). With `reference`, the first stage of a journey's world,
    the painting is a later stage of it, and the verdict also says whether it is the same place (`same_place`)."""
    if open_client is None and settings.open_critic_only:
        raise CriticUnavailable("no open model to judge the painting")
    design = {k: scene[k] for k in ("title", "place", "time_of_day", "light", "air", "subjects", "sleep")}

    async def picture(png: bytes) -> dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": data_url(
            await asyncio.to_thread(preview_jpeg, png, settings.critic_image_width), "image/jpeg")}}

    instructions = CRITIC_INSTRUCTIONS + "\n" + figures_rule(scene, settings)
    schema, shape, flags = VERDICT_SCHEMA, OPEN_SHAPE, FLAGS
    pictures = [await picture(painting)]
    if reference is not None:
        instructions, schema = instructions + "\n\n" + STAGE_INSTRUCTIONS, STAGE_VERDICT_SCHEMA
        shape, flags = OPEN_STAGE_SHAPE, STAGE_FLAGS
        pictures = [{"type": "text", "text": "The first picture: the place as first painted."},
                    await picture(reference),
                    {"type": "text", "text": "The painting you judge: a later stage of the same place."},
                    *pictures]
    request = {
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": [
                {"type": "text", "text": "The design:\n" + json.dumps(design, ensure_ascii=False)},
                *pictures,
            ]},
        ],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "verdict", "schema": schema, "strict": True}},
    }
    if open_client is not None:
        open_request = {**request, "messages": [
            {"role": "system", "content": shape + "\n\n" + instructions}, *request["messages"][1:]]}
        failure: Exception | None = None
        for attempt in (1, 2):  # once more when an answer is out of shape; never after an error or a timeout
            try:
                # No privacy fields: they are Meta's, and the listener's id never goes to the open model's host.
                response = await asyncio.wait_for(open_client.chat.completions.create(
                    model=settings.open_critic_model, timeout=settings.open_critic_timeout,
                    **reasoning(settings.open_critic_reasoning), **open_request),
                    settings.open_critic_timeout)
                meter.spark_open(getattr(response, "usage", None), "critique")
                return _verdict(response, settings.open_critic_model, flags)
            except ValueError as error:  # out of shape (a JSONDecodeError too)
                failure = error
                log.info("the open critic answered out of shape (attempt %d): %s", attempt, error)
            except Exception as error:  # noqa: BLE001 - an error or a timeout: Muse Spark judges instead
                failure = error
                break
        if settings.open_critic_only:
            raise CriticUnavailable(f"the open critic failed: {type(failure).__name__}: {str(failure)[:120]}") \
                from failure
        log.info("the open critic failed, Muse Spark judges: %s: %s", type(failure).__name__, failure)
    response, model = await spark_chat(client, (settings.critic_model, settings.spark_fallback_model),
                                       extra_body={**privacy_fields(), **cache_fields("critic")}, timeout=settings.critic_timeout,
                                       **request)
    meter.spark(getattr(response, "usage", None), "critique")
    return _verdict(response, model, flags)


def _verdict(response: Any, judge: str, flags: tuple[str, ...] = FLAGS) -> dict[str, Any]:
    """The verdict in a critic's answer, checked: a flag that is not true or false never passes a painting."""
    verdict = json_object(response.choices[0].message.content)
    if "same_place" not in flags:
        verdict.pop("same_place", None)  # only a journey's later stage is judged on it
    for flag in flags:
        if not isinstance(verdict.get(flag), bool):
            raise ValueError(f"the verdict's {flag!r} is not true or false")
    verdict["score"] = max(0, min(10, int(verdict.get("score") or 0)))
    if not str(verdict.get("reason") or "").strip():
        verdict["reason"] = reason_from_checks(verdict)
    verdict["judge"] = judge
    return verdict


def reason_from_checks(verdict: dict[str, Any]) -> str:
    """A reason built from the verdict's own checks, for a critic that left its sentence out."""
    faults = [text for flag, fault, text in (
        ("figures_wrong", True, "shows a person or an animal that does not belong, drawn wrong, awake and upright, "
                                "or a face up close"),
        ("text_or_symbols", True, "shows letters or symbols"),
        ("calm", False, "is not calm"),
        ("belongs", False, "does not show the designed place and mood"),
        ("same_place", False, "is not the same place as the first stage"),
    ) if flag in verdict and bool(verdict.get(flag)) is fault]
    if faults:
        return f"The painting {', '.join(faults)}."
    return f"No fault found; score {verdict.get('score')} of 10."
