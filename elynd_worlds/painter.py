"""Muse Image paints the world, and paints it once more without its subjects.

The second picture (the "plate") is what shows behind the moving layers: when
a subject breathes, the few pixels it uncovers must show what lies behind it,
not a copy of the subject. Muse Image's edit endpoint repaints the scene with
the subjects removed; `plate.py` then keeps that repaint only where the subjects
were, so the rest of the picture stays exactly as painted.
"""
from __future__ import annotations

from typing import Any

from .meta import NO_IMAGE_TOOLS, Meter, first_image, listener
from .settings import DEFAULT, Settings

STYLE = ("Painterly, soft, calm and believable; gentle light; a quiet, spacious composition; "
         "no people, no faces, no hands, no text, no letters, no logos, no symbols.")


def painting_prompt(scene: dict[str, Any]) -> str:
    palette = ", ".join(scene.get("palette") or [])
    return (f"{scene['prompt'].strip()} Time of day: {scene['time_of_day']}. Light: {scene['light']}. "
            f"Air: {scene['air']}. Palette: {palette}. Portrait 9:16 for a phone screen; the main subjects "
            f"in the middle of the picture, the bottom third calm and simple. {STYLE}")


def plate_prompt(scene: dict[str, Any]) -> str:
    names = ", ".join(s["name"] for s in scene["subjects"])
    return (f"Remove the {names} from this painting completely and paint what would be behind "
            f"{'it' if len(scene['subjects']) == 1 else 'them'} instead, continuing the surroundings "
            "naturally. Change nothing else: same place, light, palette, brushwork and framing. "
            "No new objects, no people, no text.")


async def paint(client, scene: dict[str, Any], meter: Meter, settings: Settings = DEFAULT,
                user: str | None = None) -> bytes:
    """The painting of one scene, as PNG bytes."""
    who = user or listener.get()
    extra: dict[str, Any] = {"user": who} if who else {}
    response = await client.images.generate(
        model=settings.painter_model,
        prompt=painting_prompt(scene),
        n=1,
        size=settings.image_size,
        response_format="b64_json",
        extra_body={**NO_IMAGE_TOOLS, "output_format": "png"},
        timeout=settings.paint_timeout,
        **extra,
    )
    meter.image("paint")
    return first_image(response)


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
    )
    meter.image("plate_edit")
    return first_image(response)
