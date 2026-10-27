"""What lives in a world besides its water, sky, grass and trees: chosen from its design, never painted.

Life is what makes a painting feel like a world, as long as it stays realistic.
The phone (and the web gallery) draws them on the living painting, so a
world gets them whether it was painted today or for the library weeks ago; this module only decides which
belong to it, from the design's own words, time of day and what moves in nature:

* birds   "gulls" over the sea, a coast, a harbour or a lagoon (one to three, gliding), a small "flock"
          elsewhere; never at night, in a sleep world, in rain or falling snow, or without open sky
* night   the painted stars twinkle, and now and then a faint shooting star crosses (night skies only)
* fish    on calm fresh water by day (not in winter), now and then the rings of a fish rising
* shadows cloud shadows sliding over the grass on a clear morning or at midday
* particles "rain" when it rains (fine streaks, and drop rings on the water), "petals" in blossom worlds,
          else what the design chose (leaves, snow, pollen, fireflies) or nothing
"""
from __future__ import annotations

import re
import zlib
from typing import Any

# Only words of the sea itself: a lake also has a shore, an island and waves.
SEA = re.compile(r"\b(sea|seas|ocean|coast\w*|beach|harbou?r|bay|lagoon|lighthouse|fjord|surf|tide\w*)\b", re.I)
FRESH = re.compile(r"\b(lake|pond|river|stream|brook|marsh|loch|lilies|lily|reeds?|canal|jetty)\b", re.I)
RAIN = re.compile(r"\b(rain|raining|rainy|drizzle|showers?)\b", re.I)  # never "rainforest"
FALLING_SNOW = re.compile(r"\b(snowfall|falling snow|snow (is )?falling|snowflakes?)\b", re.I)
BLOSSOM = re.compile(r"\b(blossoms?|cherry|petals?|sakura)\b", re.I)
WINTER = re.compile(r"\b(snow\w*|ice|icy|frozen|winter|frost\w*)\b", re.I)
DAYLIGHT = {"dawn", "morning", "midday", "golden hour", "dusk"}


def _words(scene: dict[str, Any]) -> str:
    return " ".join(str(scene.get(k) or "") for k in ("title", "place", "air", "light"))


def life(scene: dict[str, Any], present: dict[str, bool]) -> dict[str, Any]:
    """The extra life of a world: {"birds", "night", "fish", "shadows", "particles"}.

    `present`: what moves in nature was found in the painting (Regions.present: water, sky, grass, stream).
    """
    words = _words(scene)
    time = scene.get("time_of_day") or ""
    sleep = bool(scene.get("sleep"))
    raining = bool(RAIN.search(words))
    snowing = bool(FALLING_SNOW.search(words)) or scene.get("particles") == "snow"
    sea = bool(SEA.search(words))
    sky = bool(present.get("sky"))
    water = bool(present.get("water"))

    particles = scene.get("particles") or "none"
    if raining:
        particles = "rain"
    elif particles == "none" and BLOSSOM.search(words):
        particles = "petals"

    birds = "none"
    if sky and time in DAYLIGHT and not sleep and not raining and not snowing:
        birds = "gulls" if sea else "flock"
    return {
        "birds": birds,
        "night": time == "night" and sky,
        "fish": water and not sea and bool(FRESH.search(words)) and time in DAYLIGHT and not raining
        and not WINTER.search(words),
        "shadows": bool(present.get("grass")) and time in {"morning", "midday"} and not raining
        and "mist" not in words.lower() and not snowing,
        "particles": particles,
    }


def seed(scene: dict[str, Any]) -> float:
    """A number in [0, 1) of this world's own, so that its birds and shooting stars come at its own moments
    (the same on every phone, and in the web gallery)."""
    return round(zlib.crc32((str(scene.get("title")) + "|" + str(scene.get("place"))).encode()) % 997 / 997, 4)

