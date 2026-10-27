"""Muse Image paints the world, and paints it once more without its subjects.

The second picture (the "plate") is what shows behind the moving layers: when
a subject breathes, the few pixels it uncovers must show what lies behind it,
not a copy of the subject. Muse Image's edit endpoint repaints the scene with
the subjects removed; `plate.py` then keeps that repaint only where the subjects
were, so the rest of the picture stays exactly as painted.
"""
from __future__ import annotations

import io
from typing import Any

from PIL import Image

from .meta import NO_IMAGE_TOOLS, Meter, encode, first_image, is_marked, listener
from .settings import DEFAULT, Settings

STYLE = ("Painterly, soft, calm and believable; gentle light; a quiet, spacious composition; "
         "any animal or person at rest and painted naturally, a person small or seen from behind, "
         "no face up close; no text, no letters, no logos, no symbols.")


def painting_prompt(scene: dict[str, Any]) -> str:
    palette = ", ".join(scene.get("palette") or [])
    return (f"{scene['prompt'].strip()} Time of day: {scene['time_of_day']}. Light: {scene['light']}. "
            f"Air: {scene['air']}. Palette: {palette}. Portrait 9:16 for a phone screen; the main subjects "
            f"in the middle of the picture, the bottom third calm and simple. {STYLE}")


def moving_subjects(scene: dict[str, Any]) -> list[dict[str, Any]]:
    """The designed subjects that move (a still one stays in the plate as painted)."""
    return [s for s in scene["subjects"] if s.get("motion", "still") != "still"]


def plate_prompt(scene: dict[str, Any]) -> str:
    subjects = moving_subjects(scene) or scene["subjects"]
    names = ", ".join(s["name"] for s in subjects)
    return (f"Remove the {names} from this painting completely and paint what would be behind "
            f"{'it' if len(subjects) == 1 else 'them'} instead, continuing the surroundings "
            "naturally. Change nothing else: same place, light, palette, brushwork and framing. "
            "No new objects, no people or animals, no text.")


def _for_the_listener(user: str | None = None) -> dict[str, Any]:
    who = user or listener.get()
    return {"user": who} if who else {}


def _marked(picture: bytes) -> bytes:
    """Muse Image marks its pictures as made by AI; one that arrives without the mark gets it."""
    return picture if is_marked(picture) else encode(Image.open(io.BytesIO(picture)), "PNG")


async def paint(client, scene: dict[str, Any], meter: Meter, settings: Settings = DEFAULT,
                user: str | None = None) -> bytes:
    """The painting of one scene, as PNG bytes."""
    response = await client.images.generate(
        model=settings.painter_model,
        prompt=painting_prompt(scene),
        n=1,
        size=settings.image_size,
        response_format="b64_json",
        extra_body={**NO_IMAGE_TOOLS, "output_format": "png"},
        timeout=settings.paint_timeout,
        **_for_the_listener(user),
    )
    meter.image("paint")
    return _marked(first_image(response))


async def paint_without_subjects(client, scene: dict[str, Any], painting: bytes, meter: Meter,
                                 settings: Settings = DEFAULT) -> bytes:
    """The same painting with every subject removed (Muse Image edit), as PNG bytes."""
    response = await client.images.edit(
        model=settings.painter_model,
        image=("painting.png", painting, "image/png"),
        prompt=plate_prompt(scene),
        n=1,
        size=settings.image_size,
        response_format="b64_json",
        extra_body={**NO_IMAGE_TOOLS, "output_format": "png"},
        timeout=settings.paint_timeout,
        **_for_the_listener(),
    )
    meter.image("plate_edit")
    return first_image(response)
