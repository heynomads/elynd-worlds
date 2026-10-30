"""The library's themes: one short meditation per world, written to library/briefs/<id>.txt.

ELYND shows a library world from the first second of a session, while the
world made for that meditation is still being painted. The library covers the
places guided meditations visit most (90), each at a few times of day, in a
few seasons and weathers, with dark, quiet versions for falling asleep: 500
worlds. Each brief
is a meditation in miniature: the art director designs a world from it exactly
as it does from a listener's meditation, and the critic checks the painting.

    .venv/bin/python scripts/library_briefs.py            # writes library/briefs/
    .venv/bin/elynd-worlds make library/briefs/*.txt ...  # paints them (the push briefs with --mood unmotivated)
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
    # Added for the 500-world library (27 Oct 2026).
    "jetty": ("a wooden jetty reaching into a calm lake", "Its old boards end above still, clear water."),
    "fjord": ("a quiet fjord between steep green mountains", "The water lies like glass, and a small red cabin stands by the shore."),
    "canoelake": ("a northern lake with a wooden canoe resting at the shore", "Pine-covered islands float on the calm water."),
    "blacksand": ("a black sand beach with soft white surf", "Dark basalt cliffs rise behind it and the sea is calm."),
    "hotspring": ("a steaming hot spring among smooth rocks", "Warm mist rises gently from the clear blue water."),
    "marsh": ("a quiet marsh with reeds and still pools", "Small islands of grass rest between mirrors of water."),
    "canal": ("a quiet old canal between stone houses", "Small wooden boats are moored along the still water."),
    "lakeporch": ("the wooden porch of a lake house looking over the water", "Two empty chairs face the calm lake."),
    "tidepools": ("rocky tide pools by a calm sea", "Clear pools hold shells and smooth pebbles."),
    "bay": ("a sheltered bay with a few small sailing boats", "Green hills come down to the calm water."),
    "glacierlake": ("a turquoise glacier lake below white ice fields", "Small pieces of ice drift slowly on the still water."),
    "oasis": ("a desert oasis with palm trees around a clear pool", "Golden dunes surround the green shade."),
    "opensea": ("a calm open sea under a wide sky", "Gentle swells roll slowly towards a quiet shore."),
    "riverbend": ("a wide river bend with sandbanks and willows", "The water moves slowly, wide and calm."),
    "pond": ("a small round pond in a meadow", "Tall grass and wildflowers grow around its still water."),
    "lakeisland": ("a tiny island with a single tree in a still lake", "Its reflection lies perfectly in the water."),
    "birch": ("a bright birch forest", "White trunks stand in soft green light."),
    "redwoods": ("a forest of giant redwood trees", "Soft light falls in long rays between the huge trunks."),
    "rainforest": ("a lush rainforest with ferns and a clear pool", "Big leaves drip softly and the air is warm."),
    "mossforest": ("an ancient forest covered in soft green moss", "Old twisted trees and mossy stones rest in quiet light."),
    "clearing": ("a sunny clearing in the middle of a forest", "Soft grass and small flowers grow in a circle of trees."),
    "oak": ("a single old oak tree on a green meadow", "Its wide crown spreads over the soft grass."),
    "appleorchard": ("an old apple orchard", "Rows of apple trees stand in long soft grass."),
    "olivegrove": ("an olive grove on a sunny hillside", "Silver-green leaves shimmer above the warm earth."),
    "treehouse": ("a wooden treehouse in a big old tree", "A little ladder leads up to its open window."),
    "hammock": ("a hammock between two trees by a calm lake", "It hangs still in the warm air."),
    "willow": ("a great weeping willow by a quiet pond", "Its long branches hang down to the water."),
    "bluebells": ("a beech wood carpeted with bluebells", "A soft blue haze of flowers covers the forest floor."),
    "sunflowers": ("a field of sunflowers", "Tall golden flowers turn towards the light."),
    "tulips": ("colourful tulip fields with an old windmill", "Long stripes of red, yellow and pink reach to the horizon."),
    "riceterraces": ("green rice terraces on a hillside", "Water shines in the curved steps of the fields."),
    "teahills": ("rows of tea bushes on gentle hills", "Soft green lines curve over the slopes."),
    "vineyard": ("a vineyard on rolling hills", "Long rows of vines lead to a small stone house."),
    "tuscany": ("Tuscan hills with cypress trees and an old farmhouse", "A winding road disappears over the gentle hills."),
    "highlands": ("Scottish highlands with heather and a quiet loch", "Purple heather covers the slopes around the still water."),
    "canyon": ("a wide red-rock canyon", "Layered cliffs glow in warm colours and the air is still."),
    "savanna": ("a wide savanna with a lone acacia tree", "Golden grass stretches to a far horizon."),
    "steppe": ("wide open grassland under a huge sky", "The grass moves in long, slow waves."),
    "ridge": ("a quiet mountain ridge with a view over endless peaks", "Range after range fades into soft blue."),
    "alpinevillage": ("a small alpine village in a green valley", "Wooden chalets sit among green slopes below high peaks."),
    "countryside": ("green countryside with old stone walls", "Fields roll gently between low walls and lonely trees."),
    "heath": ("a heath of purple heather under a wide sky", "Small paths wind between the soft hills."),
    "frozenlake": ("a frozen lake surrounded by snowy pines", "The ice is smooth and the world is silent."),
    "meadowpath": ("a narrow path through a summer meadow", "Tall grass and flowers brush both sides of the path."),
    "dunesea": ("dunes by the sea with tall grass", "Soft sandy paths lead between the grassy dunes to the shore."),
    "barn": ("a red barn at the edge of wide fields", "A wooden fence runs along a quiet country lane."),
    "saltflat": ("a vast white salt flat that mirrors the sky", "A thin layer of water turns the ground into a perfect mirror."),
    "cherryriver": ("a river lined with cherry trees", "Soft pink branches lean over the slow water."),
    "greenhouse": ("a glass greenhouse full of green plants", "Light falls softly through the glass roof onto leaves and flowers."),
    "courtyard": ("a sunny courtyard with a tiled fountain and orange trees", "Water trickles quietly in the fountain."),
    "whitehouses": ("white houses on a hillside above a calm blue sea", "Pink flowers climb the walls and the sea glitters below."),
    "parkbench": ("a bench in a quiet park under tall old trees", "A gravel path curves past the bench."),
    "stonebridge": ("an old stone bridge over a quiet stream", "Ivy grows on its arches and the water runs softly beneath."),
    "rooftop": ("a quiet rooftop garden high above the city", "Potted plants and a wooden bench look out over the rooftops."),
    "lanterngarden": ("a garden strung with paper lanterns", "Soft round lights glow among the trees."),
    "rosegarden": ("a rose garden with an arched trellis", "Climbing roses bloom over a little white bench."),
    "kitchengarden": ("a vegetable garden with a wooden shed", "Neat rows of herbs and greens grow in the sun."),
    "pavilion": ("a wooden pavilion by a lotus pond", "Pink lotus flowers float on the calm water."),
    "seacottage": ("a little white cottage by the sea", "A garden of tall grasses leads down to the shore."),
    "mountainhut": ("a wooden mountain hut on a high meadow", "Peaks rise behind it and the air is clear."),
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
    "moon": ("sleep_journey", "under a full moon", "A round moon hangs low and silver light rests on everything; the night is warm and still."),
    "summer": ("body_scan", "in high summer", "The air is warm and full of light, and everything is green and in bloom."),
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
    "jetty": ["dawn", "midday", "golden", "dusk", "mist", "night"],
    "fjord": ["morning", "mist", "golden", "dusk", "snow", "stars"],
    "canoelake": ["dawn", "morning", "golden", "mist", "autumn", "night"],
    "blacksand": ["morning", "mist", "golden", "dusk", "midday", "aurora"],
    "hotspring": ["snow", "snowfall", "dusk", "morning", "mist", "night"],
    "marsh": ["dawn", "mist", "golden", "dusk", "morning", "moon"],
    "canal": ["morning", "golden", "dusk", "mist", "snow", "night"],
    "lakeporch": ["morning", "golden", "rain", "dusk", "mist", "night"],
    "tidepools": ["morning", "midday", "golden", "dusk", "mist", "moon"],
    "bay": ["morning", "midday", "golden", "dusk", "mist", "night"],
    "glacierlake": ["morning", "midday", "golden", "mist", "dusk", "stars"],
    "oasis": ["morning", "midday", "golden", "dusk", "dawn", "stars"],
    "opensea": ["dawn", "golden", "dusk", "midday", "night", "stars"],
    "riverbend": ["morning", "golden", "mist", "autumn", "dusk", "moon"],
    "pond": ["morning", "midday", "golden", "fireflies", "mist", "autumn"],
    "lakeisland": ["dawn", "mist", "golden", "snow", "dusk", "night"],
    "birch": ["spring", "morning", "autumn", "snow", "mist", "golden"],
    "redwoods": ["morning", "mist", "golden", "rain", "dusk", "night"],
    "rainforest": ["morning", "mist", "rain", "midday", "dusk", "night"],
    "mossforest": ["mist", "morning", "rain", "golden", "dusk", "night"],
    "clearing": ["morning", "midday", "golden", "fireflies", "snow", "stars"],
    "oak": ["dawn", "midday", "golden", "autumn", "snow", "stars"],
    "appleorchard": ["spring", "midday", "autumn", "golden", "mist", "dusk"],
    "olivegrove": ["morning", "midday", "golden", "dusk", "mist", "moon"],
    "treehouse": ["morning", "golden", "dusk", "autumn", "snow", "night"],
    "hammock": ["midday", "golden", "dusk", "morning", "summer", "night"],
    "willow": ["morning", "golden", "mist", "dusk", "spring", "moon"],
    "bluebells": ["spring", "morning", "golden", "mist", "dusk", "midday"],
    "sunflowers": ["morning", "midday", "golden", "dusk", "summer", "dawn"],
    "tulips": ["morning", "midday", "golden", "dusk", "spring", "dawn"],
    "riceterraces": ["morning", "mist", "golden", "dusk", "midday", "moon"],
    "teahills": ["mist", "morning", "golden", "rain", "midday", "dusk"],
    "vineyard": ["morning", "golden", "autumn", "dusk", "midday", "moon"],
    "tuscany": ["morning", "golden", "dusk", "mist", "midday", "night"],
    "highlands": ["morning", "mist", "golden", "rain", "dusk", "stars"],
    "canyon": ["dawn", "golden", "dusk", "midday", "morning", "stars"],
    "savanna": ["dawn", "golden", "dusk", "midday", "morning", "stars"],
    "steppe": ["morning", "golden", "dusk", "midday", "dawn", "stars"],
    "ridge": ["dawn", "golden", "dusk", "mist", "midday", "stars"],
    "alpinevillage": ["morning", "golden", "snow", "dusk", "snowfall", "night"],
    "countryside": ["morning", "mist", "golden", "rain", "dusk", "snow"],
    "heath": ["morning", "golden", "mist", "dusk", "autumn", "moon"],
    "frozenlake": ["aurora", "stars", "night", "snow", "dusk", "snowfall"],
    "meadowpath": ["morning", "midday", "golden", "fireflies", "summer", "dusk"],
    "dunesea": ["morning", "midday", "golden", "dusk", "mist", "moon"],
    "barn": ["morning", "golden", "dusk", "snow", "mist", "night"],
    "saltflat": ["dawn", "golden", "dusk", "midday", "morning", "stars"],
    "cherryriver": ["spring", "morning", "golden", "dusk", "mist", "night"],
    "greenhouse": ["morning", "rain", "golden", "midday", "dusk", "snow"],
    "courtyard": ["morning", "midday", "golden", "dusk", "spring", "night"],
    "whitehouses": ["morning", "midday", "golden", "dusk", "dawn", "night"],
    "parkbench": ["morning", "autumn", "golden", "rain", "snow", "spring"],
    "stonebridge": ["morning", "autumn", "golden", "mist", "spring", "snow"],
    "rooftop": ["golden", "dusk", "night", "morning", "rain", "stars"],
    "lanterngarden": ["dusk", "night", "fireflies", "rain", "golden", "snow"],
    "rosegarden": ["morning", "midday", "golden", "rain", "dusk", "summer"],
    "kitchengarden": ["morning", "midday", "golden", "rain", "dusk", "summer"],
    "pavilion": ["morning", "golden", "mist", "dusk", "rain", "moon"],
    "seacottage": ["morning", "golden", "mist", "dusk", "midday", "night"],
    "mountainhut": ["morning", "golden", "snow", "dusk", "mist", "stars"],
}

# A fifth version for twenty of the first places, to make 500.
for place, extra in {
    "beach": "dusk", "coast": "dusk", "lagoon": "stars", "harbour": "morning", "mountainlake": "mist",
    "forestlake": "snow", "lilypond": "rain", "river": "night", "stream": "spring", "waterfall": "stars",
    "beechforest": "spring", "pineforest": "dusk", "wildflowers": "summer", "hills": "mist",
    "lavender": "night", "wheat": "stars", "alpinemeadow": "stars", "dunes": "midday", "zengarden": "rain",
    "blossom": "night",
}.items():
    assert extra not in LIBRARY[place], (place, extra)
    LIBRARY[place] = LIBRARY[place] + [extra]


def brief(place: str, variant: str) -> str:
    where, what = PLACES[place]
    technique, when, light = V[variant]
    if technique == "sleep_journey":
        return (f"# technique=sleep_journey\nAs your body grows heavy and warm, picture {where} {when}. {what} "
                f"{light} Nothing here needs anything from you. Your breath slows, and the quiet carries you "
                f"gently towards sleep.\n")
    return (f"# technique={technique}\nTake a slow breath and picture {where} {when}. {what} {light} "
            f"Let your breath settle into the rhythm of this place, and feel how calm and safe it is here.\n")


# Worlds that give a push (27 Oct 2026): a sleepy tropical beach motivates no one.
# Every brief above ends "feel how calm and safe it is here", so the library had almost no
# world with a way forward in it. These show one: a path leading on and up, a summit at sunrise, water in full
# flow; bright and fresh, never stormy (the critic fails anything threatening or busy). Painted with --mood
# unmotivated, so the art director knows what they are for.
PUSH_LIGHT = {
    "dawn": "The sun is rising and its first light is warm and bright; the air is cool and clear.",
    "morning": "Bright morning sunlight fills the air, fresh and clear.",
    "midday": "The sun is high and the light is bright and clear.",
}
PUSH = {
    "summittrail-dawn": "a mountain trail at sunrise. The path climbs ahead over grass and stones toward a bright "
                        "summit, and the first sun is just touching the peak. The way is open and easy to follow.",
    "ridgetrail-morning": "a trail running along a high grassy ridge. It leads on toward far blue peaks, and the "
                          "wind moves through the grass beside it.",
    "mountainpath-dawn": "a path winding up a green mountainside in easy turns. Above, the crest glows with the first "
                         "light, and every turn of the path brings it closer.",
    "foreststeps-morning": "old stone steps climbing through a green forest toward an opening full of light. Sunbeams "
                           "fall across the steps, and the top is bright and near.",
    "bigwaterfall-morning": "a tall waterfall in full flow. The water pours down the rock face into a clear pool, and "
                            "its spray shines in the sun.",
    "riverrapids-morning": "a clear mountain river rushing over rocks through a sunny valley. A path follows the bank "
                           "upstream toward the hills.",
    "surfcoast-dawn": "the sea at sunrise from a rocky shore. Waves roll in steadily and break in white lines on the "
                      "dark rocks, and the spray catches the golden light.",
    "lighthousepath-morning": "a white lighthouse on a green headland on a breezy day. A path leads out along the "
                              "cliff top to the lighthouse, and waves break white on the rocks below.",
    "forestlight-morning": "a forest path leading toward a bright clearing ahead. Sunbeams pour through the trees onto "
                           "the path, and the light grows with every step.",
    "footbridge-morning": "a wooden footbridge across a rushing stream. On the far side a sunlit path leads on into "
                          "green hills.",
    "mountainpass-dawn": "a high mountain pass. A path crests the pass between two peaks, and beyond it a wide valley "
                         "opens up in the first sunlight.",
    "meadowtrail-dawn": "a path through a wildflower meadow leading toward the rising sun. The grass shines with dew "
                        "and the sky brightens ahead.",
    "coastpath-morning": "a cliff-top path along the sea. It runs on along the green edge toward a far headland, and "
                         "the sea sparkles below.",
    "windyhills-midday": "a grassy hilltop on a breezy day. The wind streams through the long grass and white clouds "
                         "sail across a deep blue sky.",
    "riceclimb-dawn": "green rice terraces climbing a hillside step by step toward the light. Water shines on each "
                      "terrace and the hilltop glows above.",
    "sunrisesea-dawn": "the sun rising out of the sea. Its path of light runs across the water toward the shore, and "
                       "gulls cross the brightening sky.",
    "alpinetrail-morning": "a trail through an alpine meadow leading up toward snowy peaks. A stream runs beside the "
                           "path and flowers line the way.",
    "glacierpath-morning": "a trail along a mountainside toward a shining glacier. White peaks stand ahead and a "
                           "bright river winds through the valley.",
    "peakcairn-dawn": "a small stone cairn on a mountain top at sunrise. Range after range of peaks spreads out below, "
                      "lit by the rising sun.",
    "valleytrail-morning": "a trail leading down into a wide green valley. A river shines along its floor and the "
                           "far hills are bright with sun.",
}


def push_brief(id: str) -> str:
    return (f"# technique=visualization\nTake a deep breath and picture {PUSH[id]} {PUSH_LIGHT[id.split('-')[1]]} "
            f"Breathe in the fresh air and feel your energy rising, ready for the way ahead.\n")


# The push worlds once more (30 Oct 2026), for the place-alone painter: the same briefs under new ids, <id>-v2.
AGAIN = "-v2"


def main(out: str = "library/briefs") -> int:
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    count = 0
    for place, variants in LIBRARY.items():
        assert len(set(variants)) == len(variants), place
        for variant in variants:
            (folder / f"{place}-{variant}.txt").write_text(brief(place, variant))
            count += 1
    assert count == 500, count
    for id in PUSH:
        assert f"{id.split('-')[0]}" not in LIBRARY, id
        (folder / f"{id}.txt").write_text(push_brief(id))
        (folder / f"{id}{AGAIN}.txt").write_text(push_brief(id))
    print(f"{count} briefs and {len(PUSH)} push briefs (each also as <id>{AGAIN}) -> {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
