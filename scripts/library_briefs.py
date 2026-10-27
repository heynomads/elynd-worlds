"""The library's themes: one short meditation per world, written to library/briefs/<id>.txt.

ELYND shows a library world from the first second of a session, while the
world made for that meditation is still being painted. The library covers the
places guided meditations visit most, each at a few times of day, in a few
seasons and weathers, with dark, quiet versions for falling asleep. Each brief
is a meditation in miniature: the art director designs a world from it exactly
as it does from a listener's meditation, and the critic checks the painting.

    .venv/bin/python scripts/library_briefs.py            # writes library/briefs/
    .venv/bin/elynd-worlds make library/briefs/*.txt ...  # paints them
"""
from __future__ import annotations

import sys
from pathlib import Path

# id: (the place, what is there)
PLACES = {
    "beach": ("a quiet sandy beach", "Small waves fold onto the sand and the sea stretches to the horizon."),
    "coast": ("a grassy cliff above a calm sea", "Far below, the water is still, and a few rocks rest near the shore."),
    "lagoon": ("a warm tropical lagoon", "Clear turquoise water lies still between white sand and palm trees."),
    "harbour": ("a small sheltered harbour", "A few wooden boats rest on glassy water behind an old stone pier."),
    "mountainlake": ("a clear mountain lake", "Still water mirrors the peaks around it, and smooth stones line the shore."),
    "forestlake": ("a quiet lake in a pine forest", "Dark pines stand along the shore and the water is perfectly still."),
    "lilypond": ("a pond covered with water lilies", "Round green leaves and pale flowers float on the calm water."),
    "river": ("a slow river winding through a meadow", "The water glides past soft grass and a few willows."),
    "stream": ("a clear stream in a green forest", "The water runs over round stones, moss covers the banks."),
    "waterfall": ("a small waterfall in a green gorge", "Water falls softly into a clear pool surrounded by ferns and rocks."),
    "beechforest": ("a path through an old beech forest", "Tall smooth trunks rise on both sides and the ground is soft."),
    "pineforest": ("a calm pine forest", "Straight trunks stand far apart and the air smells of resin."),
    "bamboo": ("a quiet bamboo grove", "Tall green stems rise gently and a narrow path leads between them."),
    "wildflowers": ("a meadow full of wildflowers", "Poppies, daisies and cornflowers sway in the long grass."),
    "hills": ("soft rolling green hills", "The land rises and falls gently all the way to the horizon."),
    "lavender": ("a lavender field", "Long purple rows run towards a single tree on a low hill."),
    "wheat": ("a golden wheat field", "The ripe wheat moves in slow waves under a wide sky."),
    "alpinemeadow": ("an alpine meadow in a mountain valley", "Green grass and small flowers lie below high peaks."),
    "snowpeaks": ("a valley of snowy mountains", "White peaks rise above a wide, untouched field of snow."),
    "winterforest": ("a quiet winter forest", "Snow rests on the branches and covers the ground in soft white."),
    "dunes": ("wide desert dunes", "Smooth waves of sand curve away under a clear sky."),
    "zengarden": ("a Japanese garden with a still pond", "A small wooden bridge, round stones and a stone lantern sit by the water."),
    "cottagegarden": ("a cottage flower garden", "Roses, foxgloves and lavender grow along a little gravel path."),
    "cabin": ("a small wooden cabin by the edge of a forest", "Warm light glows in its window."),
    "nightfield": ("an open field on a gentle hill", "Soft grass covers the hill, and the land around it is wide and dark."),
    "autumnforest": ("an autumn forest", "Red, orange and golden leaves glow on the trees and cover the ground."),
    "blossom": ("an orchard of cherry trees in blossom", "Pale pink blossoms fill the branches above soft green grass."),
    "greenlake": ("a calm lake among soft green hills", "Reeds grow near the shore and the far bank is a gentle blur."),
    "aboveclouds": ("a mountain top above the clouds", "A sea of soft clouds lies below, and the sky is wide and open."),
    "lighthouse": ("a small island with a white lighthouse", "Calm sea surrounds the island and the grass on it is soft."),
}

# variant: (technique, when, the light and the air)
V = {
    "dawn": ("visualization", "at dawn", "Soft pink and golden light rises slowly; the air is cool and fresh."),
    "morning": ("breath_awareness", "on a clear morning", "Fresh morning light makes everything bright and clean."),
    "midday": ("body_scan", "on a warm, bright day", "The sun is high, the light is clear and the air is warm and still."),
    "golden": ("gratitude_reflection", "in the golden hour", "Warm, low sunlight lies across everything and the shadows are long and soft."),
    "dusk": ("reflection", "at dusk", "The last light fades into lilac and deep blue; the air is soft and quiet."),
    "mist": ("mindfulness", "on a misty morning", "A soft mist hangs in the air and everything is hushed."),
    "rain": ("emotional_release", "in gentle rain", "A soft, steady rain falls; the light is grey and kind."),
    "snow": ("focus", "in winter", "Fresh snow lies everywhere and the air is clear and still."),
    "snowfall": ("mindfulness", "while snow is falling", "Big soft snowflakes drift down slowly through the still air."),
    "autumn": ("reflection", "in autumn", "Golden and red leaves drift down slowly in the mild air."),
    "spring": ("affirmation", "in spring", "Everything is fresh and new; blossoms and young leaves glow in soft light."),
    "fireflies": ("visualization", "on a warm summer night", "Fireflies glow softly here and there in the dark."),
    "night": ("sleep_journey", "at night", "Silver moonlight lies over everything; the world is dark, warm and still."),
    "stars": ("sleep_journey", "under the stars", "The sky is full of stars and the night is warm, dark and quiet."),
    "aurora": ("sleep_journey", "under the northern lights", "Soft green light ripples slowly across the dark sky."),
}

LIBRARY = {
    "beach": ["dawn", "golden", "midday", "night"],
    "coast": ["morning", "golden", "mist", "night"],
    "lagoon": ["midday", "golden", "dawn", "night"],
    "harbour": ["dawn", "golden", "mist", "night"],
    "mountainlake": ["dawn", "midday", "golden", "stars"],
    "forestlake": ["mist", "morning", "dusk", "night"],
    "lilypond": ["morning", "midday", "golden", "dusk"],
    "river": ["morning", "golden", "mist", "dusk"],
    "stream": ["morning", "midday", "autumn", "snow"],
    "waterfall": ["morning", "midday", "golden", "mist"],
    "beechforest": ["morning", "golden", "autumn", "mist"],
    "pineforest": ["morning", "snow", "mist", "night"],
    "bamboo": ["morning", "midday", "golden", "rain"],
    "wildflowers": ["morning", "midday", "golden", "fireflies"],
    "hills": ["dawn", "midday", "golden", "stars"],
    "lavender": ["morning", "golden", "dusk", "midday"],
    "wheat": ["golden", "dusk", "morning", "midday"],
    "alpinemeadow": ["morning", "midday", "golden", "dusk"],
    "snowpeaks": ["dawn", "midday", "golden", "night"],
    "winterforest": ["snow", "snowfall", "dusk", "night"],
    "dunes": ["dawn", "golden", "dusk", "stars"],
    "zengarden": ["morning", "autumn", "spring", "dusk"],
    "cottagegarden": ["morning", "midday", "golden", "rain"],
    "cabin": ["dusk", "night", "snowfall", "morning"],
    "nightfield": ["stars", "aurora", "fireflies", "night"],
    "autumnforest": ["morning", "golden", "mist", "rain"],
    "blossom": ["morning", "midday", "golden", "dusk"],
    "greenlake": ["rain", "mist", "dusk", "night"],
    "aboveclouds": ["dawn", "midday", "golden", "night"],
    "lighthouse": ["morning", "golden", "mist", "night"],
}


def brief(place: str, variant: str) -> str:
    where, what = PLACES[place]
    technique, when, light = V[variant]
    if technique == "sleep_journey":
        return (f"# technique=sleep_journey\nAs your body grows heavy and warm, picture {where} {when}. {what} "
                f"{light} Nothing here needs anything from you. Your breath slows, and the quiet carries you "
                f"gently towards sleep.\n")
    return (f"# technique={technique}\nTake a slow breath and picture {where} {when}. {what} {light} "
            f"Let your breath settle into the rhythm of this place, and feel how calm and safe it is here.\n")


def main(out: str = "library/briefs") -> int:
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    count = 0
    for place, variants in LIBRARY.items():
        assert len(set(variants)) == len(variants), place
        for variant in variants:
            (folder / f"{place}-{variant}.txt").write_text(brief(place, variant))
            count += 1
    print(f"{count} briefs -> {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
