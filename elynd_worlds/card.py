"""The share card: the listener's world with one line from their meditation.

1080x1920 (stories format). The painting fills the card; a soft shade at the
bottom carries the line, the world's name and the ELYND wordmark, set in
Poppins (SIL Open Font License, bundled in fonts/). The line is the session's
own share quote, which the listener sees before anything is shared. If the line
is in a script Poppins cannot draw, the card carries the world's name only.
"""
from __future__ import annotations

import io
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .meta import encode

SIZE = (1080, 1920)
FONTS = Path(__file__).parent / "fonts"
LATIN_EXTRA = set("“”‘’…–—·•")


def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONTS / f"Poppins-{weight}.ttf"
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default(size=size)


def drawable(text: str) -> bool:
    """True when Poppins (Latin) can draw every letter of `text`."""
    return all(ord(ch) < 0x250 or ch in LATIN_EXTRA for ch in text)


def cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    scale = max(size[0] / image.width, size[1] / image.height)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    left, top = (image.width - size[0]) // 2, (image.height - size[1]) // 2
    return image.crop((left, top, left + size[0], top + size[1]))


def spaced(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt, fill, tracking: float) -> None:
    """Letter-spaced text (Pillow has no tracking of its own)."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=fill)
        x += fnt.getlength(ch) + tracking


def make_card(painting_png: bytes, line: str = "", *, title: str = "",
              tagline: str = "Personalized Meditations", max_chars: int = 140) -> bytes:
    card = cover(Image.open(io.BytesIO(painting_png)).convert("RGB"), SIZE).convert("RGBA")

    shade = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shade)
    start = int(SIZE[1] * 0.48)
    for y in range(start, SIZE[1]):
        k = (y - start) / (SIZE[1] - start)
        draw.line([(0, y), (SIZE[0], y)], fill=(10, 18, 22, int(215 * k ** 1.4)))
    card = Image.alpha_composite(card, shade)
    draw = ImageDraw.Draw(card)

    line = " ".join(line.split())
    if not drawable(line):
        line = ""
    if len(line) > max_chars:
        line = line[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    quote_font = font("Medium", 56)
    lines = textwrap.wrap(f"“{line}”", width=26) if line else []
    y = SIZE[1] - 330 - 74 * len(lines)
    for text in lines:
        draw.text((96, y), text, font=quote_font, fill=(255, 255, 255, 248))
        y += 74
    if title and drawable(title):
        spaced(draw, (98, y + 26), title.upper(), font("Regular", 26), (255, 255, 255, 190), tracking=5)
    spaced(draw, (98, SIZE[1] - 150), "ELYND", font("SemiBold", 34), (255, 255, 255, 235), tracking=9)
    if tagline and drawable(tagline):
        draw.text((98, SIZE[1] - 104), tagline, font=font("Regular", 24), fill=(255, 255, 255, 170))

    return encode(card.convert("RGB"), "JPEG", quality=90, optimize=True)
