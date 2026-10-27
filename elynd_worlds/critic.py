"""The critic: it looks at each painting before a listener does.

Image models sometimes slip: a face in the clouds, letters on a boat, a sky that
turned stormy. The critic sees the painting next to the design it was made from
and answers in strict JSON. A painting is shown only if it is safe, calm,
belongs to the meditation and scores at least `Settings.pass_score`. When it
fails, the critic says in one sentence what to change, and the art director
revises its design from that note.

The critic is Muse Spark Open, Meta's open-weights model, when the deployment
passes an open-model client (Baseten serves it). It sees only the painting and
the scenery design, never the listener's words or id. When it fails, is slow or
answers out of shape, Muse Spark on Meta Model API judges instead, so a world
never waits on one provider. `verdict["judge"]` names the model that judged.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .meta import Meter, cache_fields, data_url, preview_jpeg, privacy_fields, reasoning, spark_chat
from .settings import DEFAULT, Settings

log = logging.getLogger("elynd_worlds")

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "belongs": {"type": "boolean", "description": "The painting shows the designed place and mood"},
        "calm": {"type": "boolean", "description": "Calm and inviting; nothing threatening or busy"},
        "people_or_faces": {"type": "boolean", "description": "Any person, face, hand or figure, even small"},
        "text_or_symbols": {"type": "boolean", "description": "Any letters, numbers, logos or symbols"},
        "subjects_seen": {"type": "array", "items": {"type": "string"},
                          "description": "The designed subjects that are clearly visible and separate"},
        "score": {"type": "integer", "description": "0-10: how well it would serve a meditation"},
        "reason": {"type": "string", "description": "One sentence"},
        "fix": {"type": "string", "description": "If score < 8: one sentence on what to change in the design"},
    },
    "required": ["belongs", "calm", "people_or_faces", "text_or_symbols", "subjects_seen", "score", "reason", "fix"],
}

CRITIC_INSTRUCTIONS = """You are the critic of a meditation app. Before a listener sees a painting,
you check it against the design it was made from. Be strict about safety and calm, fair about art.
- people_or_faces: true if you see any person, face, hand or human figure, however small or vague.
- text_or_symbols: true if you see any letters, numbers, logos, signs or symbols.
- calm: false if anything feels threatening, chaotic, sad or busy.
- belongs: false if the painting does not show the designed place, time of day and mood.
- subjects_seen: list the designed subjects (by their design names) that are clearly visible,
  whole and separate from what is behind them.
- reason: one sentence on what you see and why it passes or fails; never empty.
- fix: when the score is below 8, say in one sentence what the art director should change in the
  design; otherwise an empty string.
Return only the JSON object."""

FLAGS = ("belongs", "calm", "people_or_faces", "text_or_symbols")


def passes(verdict: dict[str, Any], settings: Settings = DEFAULT) -> bool:
    return (bool(verdict.get("belongs")) and bool(verdict.get("calm"))
            and not verdict.get("people_or_faces") and not verdict.get("text_or_symbols")
            and int(verdict.get("score") or 0) >= settings.pass_score)


async def critique(client, painting: bytes, scene: dict[str, Any], meter: Meter,
                   settings: Settings = DEFAULT, open_client=None) -> dict[str, Any]:
    """The verdict on one painting: Muse Spark Open's when `open_client` is given and answers, else Muse Spark's."""
    design = {k: scene[k] for k in ("title", "place", "time_of_day", "light", "air", "subjects", "sleep")}
    request = {
        "messages": [
            {"role": "system", "content": CRITIC_INSTRUCTIONS},
            {"role": "user", "content": [
                {"type": "text", "text": "The design:\n" + json.dumps(design, ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": data_url(
                    await asyncio.to_thread(preview_jpeg, painting, settings.critic_image_width), "image/jpeg")}},
            ]},
        ],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "verdict", "schema": VERDICT_SCHEMA, "strict": True}},
    }
    if open_client is not None:
        try:
            # No privacy fields: they are Meta's, and the listener's id never goes to the open model's host.
            response = await asyncio.wait_for(open_client.chat.completions.create(
                model=settings.open_critic_model, timeout=settings.open_critic_timeout,
                **reasoning(settings.open_critic_reasoning), **request),
                settings.open_critic_timeout)
            meter.spark_open(getattr(response, "usage", None), "critique")
            return _verdict(response, settings.open_critic_model)
        except Exception as error:  # noqa: BLE001 - Muse Spark judges instead
            log.info("the open critic failed, Muse Spark judges: %s: %s", type(error).__name__, error)
    response, model = await spark_chat(client, (settings.critic_model, settings.spark_fallback_model),
                                       extra_body={**privacy_fields(), **cache_fields("critic")}, timeout=settings.critic_timeout,
                                       **request)
    meter.spark(getattr(response, "usage", None), "critique")
    return _verdict(response, model)


def _verdict(response: Any, judge: str) -> dict[str, Any]:
    """The verdict in a critic's answer, checked: a flag that is not true or false never passes a painting."""
    verdict = json.loads(response.choices[0].message.content)
    for flag in FLAGS:
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
        ("people_or_faces", True, "shows a person or a face"),
        ("text_or_symbols", True, "shows letters or symbols"),
        ("calm", False, "is not calm"),
        ("belongs", False, "does not show the designed place and mood"),
    ) if bool(verdict.get(flag)) is fault]
    if faults:
        return f"The painting {', '.join(faults)}."
    return f"No fault found; score {verdict.get('score')} of 10."
