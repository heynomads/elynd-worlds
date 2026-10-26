"""Muse Spark as the critic: it looks at each painting before a listener does.

Image models sometimes slip: a face in the clouds, letters on a boat, a sky that
turned stormy. The critic sees the painting next to the design it was made from
and answers in strict JSON. A painting is shown only if it is safe, calm,
belongs to the meditation and scores at least `Settings.pass_score`. When it
fails, the critic says in one sentence what to change, and the art director
revises its design from that note.
"""
from __future__ import annotations

import json
from typing import Any

from .meta import Meter, data_url, preview_jpeg
from .settings import DEFAULT, Settings

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
- fix: when the score is below 8, say in one sentence what the art director should change in the
  design; otherwise an empty string.
Return only the JSON object."""


def passes(verdict: dict[str, Any], settings: Settings = DEFAULT) -> bool:
    return (bool(verdict.get("belongs")) and bool(verdict.get("calm"))
            and not verdict.get("people_or_faces") and not verdict.get("text_or_symbols")
            and int(verdict.get("score") or 0) >= settings.pass_score)


async def critique(client, painting: bytes, scene: dict[str, Any], meter: Meter,
                   settings: Settings = DEFAULT) -> dict[str, Any]:
    """Muse Spark's verdict on one painting."""
    design = {k: scene[k] for k in ("title", "place", "time_of_day", "light", "air", "subjects", "sleep")}
    response = await client.chat.completions.create(
        model=settings.critic_model,
        messages=[
            {"role": "system", "content": CRITIC_INSTRUCTIONS},
            {"role": "user", "content": [
                {"type": "text", "text": "The design:\n" + json.dumps(design, ensure_ascii=False)},
                {"type": "image_url", "image_url": {
                    "url": data_url(preview_jpeg(painting, settings.critic_image_width), "image/jpeg")}},
            ]},
        ],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "verdict", "schema": VERDICT_SCHEMA, "strict": True}},
        timeout=settings.critic_timeout,
    )
    meter.spark(getattr(response, "usage", None), "critique")
    verdict = json.loads(response.choices[0].message.content)
    verdict["score"] = max(0, min(10, int(verdict.get("score") or 0)))
    return verdict
