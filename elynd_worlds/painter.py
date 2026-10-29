"""Muse Image paints the world, and paints it once more without its subjects.

The second picture (the "plate") is what shows behind the moving layers: when
a subject breathes, the few pixels it uncovers must show what lies behind it,
not a copy of the subject. Muse Image's edit endpoint repaints the scene with
the subjects removed; `plate.py` then keeps that repaint only where the subjects
were, so the rest of the picture stays exactly as painted.

A journey's world grows (journeys/stages.py): each later stage is the same edit
endpoint on the first stage's painting, told to keep the viewpoint and the
composition and to change only the light, the season and what has grown.

The painting's style line ends its prompt (`style`). It names the animals and
people the design brings, asleep or lying at rest, and forbids any other; every
other world is painted as the place alone. A line on figures in general makes
Muse Image paint one nobody designed (29 Oct 2026, below).
"""
from __future__ import annotations

import io
from typing import Any

from PIL import Image

from .art_director import figures_in, the_names
from .meta import NO_IMAGE_TOOLS, Meter, encode, first_image, is_marked, listener, png_size
from .plate import fit
from .settings import DEFAULT, Settings

# 29 Oct 2026, designs with no animal and no person in them: told "any animal or person at rest", Muse Image painted a
# sitting person into 3 of 3; told "any animal asleep or person lying at rest", a lying figure into 5 of 5, and it
# refused 5 of 10 requests (a person lying in a landscape trips its filter); told "no people and no animals", no
# figure and no refusal in 4 of 4. So no line on figures in general ever reaches it: only the ones a design brings.
STYLE_START = "Painterly, soft, calm and believable; gentle light; a quiet, spacious composition; "
STYLE_END = "no text, no letters, no logos, no symbols."
PLACE_ALONE_STYLE = STYLE_START + "no people and no animals; " + STYLE_END
AT_REST = ("asleep or lying at rest, eyes closed or turned away, painted naturally (never standing, sitting up or "
           "walking)")
PERSON = ("small, at a distance or seen from behind, asleep where people rest (in a hammock, on a blanket or a beach "
          "towel), never on bare ground or in water, no face up close")


def style(scene: dict[str, Any], figures: bool = True) -> str:
    """The painting's style line. With `figures` on and a design that brings animals or people
    (art_director.figures_in), it names exactly those, at rest, and no others; otherwise the place alone."""
    brought = figures_in(scene) if figures else []
    if not brought:
        return PLACE_ALONE_STYLE
    people = figures_in(scene, ("person",))
    animals = len(people) < len(brought)
    return (f"{STYLE_START}{the_names(brought)} {AT_REST}; "
            + (f"{the_names(people)} {PERSON}; " if people else "")
            + f"no {'other ' if people else ''}people and no {'other ' if animals else ''}animals; {STYLE_END}")


def painting_prompt(scene: dict[str, Any], figures: bool = True) -> str:
    palette = ", ".join(scene.get("palette") or [])
    return (f"{scene['prompt'].strip()} Time of day: {scene['time_of_day']}. Light: {scene['light']}. "
            f"Air: {scene['air']}. Palette: {palette}. Portrait 9:16 for a phone screen; the main subjects "
            f"in the middle of the picture, the bottom third calm and simple. {style(scene, figures)}")


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


STAGE_STYLE = "Painterly, soft, calm and believable, in the same brushwork; no text, no letters, no logos, no symbols."


def clause(text: str) -> str:
    """A stage's words to follow "now:" ("Full light at the summit." -> "full light at the summit")."""
    text = " ".join(text.split()).rstrip(".")
    return text[0].lower() + text[1:] if text[1:2].islower() else text


def stage_prompt(scene: dict[str, Any], stage: str, fix: str = "") -> str:
    """The edit that turns a journey's first painting into a later stage: the same view, another light and season.

    `scene` is the stage's design (art_director.stage): its light, air and palette, and `change`, the art director's
    sentence on what changes in the picture. `fix`, after a rejected edit: the critic's note."""
    palette = ", ".join(scene.get("palette") or [])
    change = " ".join(str(scene.get("change") or "").split())
    return (f"The same place, painted again from exactly the same viewpoint, with the same framing and composition; "
            f"now: {clause(stage)}. {change + ' ' if change else ''}"
            f"Keep the land, the water, every landmark and {the_names([s['name'] for s in scene['subjects']])} where "
            "they are and as large as they are; "
            "change only the light, the sky, the air, the season and how far plants have grown. "
            f"Time of day: {scene['time_of_day']}. Light: {scene['light']}. Air: {scene['air']}. "
            f"Palette: {palette}. Add nothing that is not already there: no new objects, buildings, people, "
            f"animals or text. {'Mind this: ' + fix.strip() + ' ' if fix.strip() else ''}{STAGE_STYLE}")


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
        prompt=painting_prompt(scene, settings.figures),
        n=1,
        size=settings.image_size,
        response_format="b64_json",
        extra_body={**NO_IMAGE_TOOLS, "output_format": "png"},
        timeout=settings.paint_timeout,
        **_for_the_listener(user),
    )
    meter.image("paint")
    return _marked(first_image(response))


async def _edit(client, painting: bytes, prompt: str, meter: Meter, step: str, settings: Settings) -> bytes:
    """One Muse Image edit of a painting, as PNG bytes."""
    response = await client.images.edit(
        model=settings.painter_model,
        image=("painting.png", painting, "image/png"),
        prompt=prompt,
        n=1,
        size=settings.image_size,
        response_format="b64_json",
        extra_body={**NO_IMAGE_TOOLS, "output_format": "png"},
        timeout=settings.paint_timeout,
        **_for_the_listener(),
    )
    meter.image(step)
    return first_image(response)


async def paint_without_subjects(client, scene: dict[str, Any], painting: bytes, meter: Meter,
                                 settings: Settings = DEFAULT) -> bytes:
    """The same painting with every subject removed (Muse Image edit), as PNG bytes."""
    return await _edit(client, painting, plate_prompt(scene), meter, "plate_edit", settings)


async def paint_stage(client, scene: dict[str, Any], first: bytes, stage: str, meter: Meter,
                      settings: Settings = DEFAULT, fix: str = "") -> bytes:
    """A later stage of a journey's world: the first stage's painting edited by Muse Image (`stage_prompt`), as PNG
    bytes at the first painting's exact size (so the stages lie on one another) and marked as made by AI."""
    picture = await _edit(client, first, stage_prompt(scene, stage, fix), meter, "stage_edit", settings)
    size = png_size(first)
    if png_size(picture) != size:
        return encode(fit(picture, size), "PNG")
    return _marked(picture)
