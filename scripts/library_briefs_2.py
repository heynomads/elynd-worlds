"""The second library's themes (2 Nov 2026), one short meditation per world, written to library/briefs2/<id>.txt.

2 Nov 2026: the library grows past a thousand worlds. The first library (scripts/library_briefs.py, 26-27 Oct)
holds 90 places at a few times each.
This bank adds about 720 NEW nature places, times, weathers and seasons: no place word here is a place of the first
library, and no theme repeats a first-library title. No people and no animals in any theme (the painter paints the place alone). Every theme is
written for a feeling, as the music is chosen (FEEL): calm, letting go, a push, focus, joy, gratitude, strength, or
sleep; the closing sentence and the mood line of each brief say so, so the art director knows what the world is for
(27 Oct: a library written only as "calm and safe" had almost no world with a way forward in it). About 15 % are
sleep worlds (technique sleep_journey: a dark, quiet world).

    .venv/bin/python scripts/library_briefs_2.py [library/briefs2]       # writes <id>.txt per theme + themes.json
    .venv/bin/elynd-worlds make library/briefs2/*.txt --out library/worlds --parallel 4
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ENGINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENGINE / "scripts"))
from library_briefs import LIBRARY as FIRST_LIBRARY, PLACES as FIRST_PLACES, PUSH as FIRST_PUSH, V as FIRST_V  # noqa: E402

# The first library as the engine knows it (library/catalog.json): no new id or title repeats one of these.
CATALOG = ENGINE / "library" / "catalog.json"

# ─── Places: id -> (the place, what is there). Nature alone; no person, no animal. ─────────────────────────────
PLACES = {
    # Coast and sea
    "seastacks": ("a rocky coast with tall sea stacks standing in calm water", "The dark stacks rise quiet from the sea, and small waves wash gently around their feet."),
    "chalkcliffs": ("white chalk cliffs above a calm green sea", "The cliffs glow pale against the water, and soft grass runs along their top."),
    "pebblecove": ("a small cove of smooth grey pebbles", "Clear water laps at the stones, and rocky arms shelter the cove on both sides."),
    "tidalflats": ("wide tidal flats at low tide", "Shallow channels of water shine between banks of smooth wet sand, all the way out to the sea."),
    "mangroves": ("a quiet channel of still water between mangrove trees", "The green water winds between arching roots, and light falls in patches through the leaves."),
    "coralreef": ("a shallow coral reef seen from a quiet white shore", "Turquoise water lies clear over pale coral, and the sea beyond is calm and deep blue."),
    "atoll": ("a ring of white sand around a turquoise lagoon", "A few palms lean over the shallow water, and the open sea is calm beyond the reef."),
    "sandbar": ("a long pale sandbar reaching out between two calm seas", "Gentle waves meet from both sides, and the sand runs far out into the water."),
    "icebergbay": ("a quiet bay with icebergs floating in still water", "White and blue ice drifts slowly, and snowy mountains stand behind the bay."),
    "driftwood": ("a wild beach with silver driftwood", "Bleached logs lie on the sand, and long low waves roll in slowly from a wide sea."),
    "shellbeach": ("a beach of small white shells", "The shells shine in the shallow water, and the sea is clear and calm."),
    "estuary": ("a wide river estuary opening to the sea", "Sandbanks and still channels of water lie between grassy banks under a wide sky."),
    "seameadow": ("a flower meadow on a cliff high above the sea", "Grass and wildflowers run to the edge, and the sea lies calm and blue far below."),
    "rockarch": ("a natural stone arch standing in a calm sea", "The water is clear beneath it, and the rock glows warm in the light."),
    "archipelago": ("a scattering of small rocky islands in a calm sea", "Pines grow on the smooth grey rocks, and the water between them is still."),
    "longbeach": ("a long empty beach at low tide", "The wet sand mirrors the sky, and the waves are small and far away."),
    "machair": ("a flowery coastal meadow above a white sand beach", "The grass is full of small flowers, and the sea is turquoise beyond the dunes."),
    "headland": ("a green headland reaching into a calm sea", "The grass runs to the cliff edge, and the water lies calm and bright on both sides."),
    "windyshore": ("a wide shore with tall grass in the wind", "The wind combs the grass, and the waves run in long white lines up the sand."),
    "pinecoast": ("a pine forest on a sandy coast", "Tall pines lean over pale sand, and the sea is calm beyond them."),
    "lakebeach": ("a small sandy beach on a calm lake", "The water is clear and still, and pines stand along the shore."),
    "stoneshore": ("a lake shore of smooth round stones", "The clear water laps softly at the stones, and the far hills are blue."),
    "boathouse": ("an old wooden boathouse on a calm lake", "A small boat rests beside it, and the water is still."),
    # Lakes, rivers, springs
    "reedlake": ("a wide lake edged with tall reeds", "The reeds sway gently, and the water stretches calm to the far shore."),
    "oxbow": ("a quiet oxbow lake in a river meadow", "The curved water lies still among tall grass and willows."),
    "cypressswamp": ("a still swamp with tall cypress trees standing in the water", "Soft moss hangs from the branches, and the calm water mirrors the trees in gentle light."),
    "craterlake": ("a deep blue crater lake inside an old volcano", "Steep green slopes ring the still water, and the sky is wide above it."),
    "tarn": ("a small high mountain tarn among grey rocks", "The water lies dark and still, and patches of snow rest on the slopes above."),
    "springpool": ("a crystal spring pool in a green wood", "The water is so clear that every pebble shows, and it wells up softly from below."),
    "travertine": ("white terraces of warm spring water on a hillside", "Pale pools of turquoise water step down the slope, and soft steam rises from them."),
    "geyserbasin": ("a steaming valley of hot springs", "Soft plumes of steam drift over blue pools and pale ground."),
    "braidedriver": ("a wide braided river of gravel bars and shining channels", "The water runs in many threads under a huge sky, with mountains far away."),
    "riverisland": ("a small wooded island in a wide slow river", "The water parts around it quietly and joins again downstream."),
    "cascades": ("a chain of small waterfalls stepping down a mossy hillside", "Each pool spills softly into the next."),
    "lakecliffs": ("a long lake below tall cliffs", "The water is calm and deep green, and the cliffs rise straight from the shore."),
    "meltwater": ("a cold clear stream running down from snowy peaks", "It runs fast over pale stones through a high green valley."),
    "glacierfoot": ("the foot of a glacier, where a milky river begins", "Blue ice ends in grey stones, and the water runs away down the valley."),
    "dunelake": ("a small clear lake between high sand dunes", "The water lies still at the foot of the dunes, with a few reeds at its edge."),
    "canyonriver": ("a green river winding through a red-rock canyon", "The water is calm, and the cliffs glow warm above it."),
    "pinklake": ("a shallow pink salt lake under a wide sky", "The water is rose-coloured and perfectly still, with white salt along the shore."),
    "waterfallvalley": ("a green valley with many thin waterfalls", "White threads of water fall down the steep green walls into a quiet river."),
    "meadowbrook": ("a brook winding through a flower meadow", "The clear water chatters over stones between banks of grass and flowers."),
    "paddyfields": ("flat rice paddies full of water", "The young green shoots stand in still water that mirrors the sky."),
    "karst": ("a river winding between tall green limestone hills", "Mist rests between the hills, and the water is calm and green."),
    "halffrozen": ("a half-frozen waterfall", "Water still runs between columns of blue ice, and snow rests on the rocks around it."),
    "rainforestriver": ("a wide calm river through a rainforest", "Green walls of trees lean over the still water, and the air is warm."),
    "springfalls": ("a river in spring flood pouring white over wide rocks", "The water rushes bright and loud, and the trees above it are coming into leaf."),
    "lakeinlet": ("a narrow inlet of a calm lake between wooded shores", "The still water leads away between the trees, and a few rocks rest near the shore."),
    # Deserts and dry land
    "whitedesert": ("a desert of white sand and pale rock", "Smooth white dunes roll away under a clear blue sky."),
    "reddunes": ("tall red sand dunes", "Sharp crests and soft shadows curve across the dunes under a clear sky."),
    "joshuatrees": ("a high desert with Joshua trees", "The strange gentle trees stand far apart on pale sand, and the mountains are far and blue."),
    "saguaro": ("a desert of tall saguaro cacti", "The cacti stand like quiet pillars on the warm ground, and the hills are soft in the distance."),
    "mesa": ("a wide plain with flat red mesas", "The rock towers glow warm in the light, and the plain is wide and still."),
    "hoodoos": ("a valley of tall orange rock spires", "The spires stand in quiet rows under a clear sky, and the ground is soft and pale."),
    "paintedhills": ("rolling hills in bands of red, gold and cream", "The soft colours flow over the slopes under a wide sky."),
    "lavafield": ("an old lava field covered in soft green moss", "Rounded dark rocks lie under thick moss, and mountains stand blue in the distance."),
    # Forests and woods
    "larchforest": ("a larch forest turning golden in autumn", "The soft needles glow yellow against the dark mountains, and the ground is covered in gold."),
    "aspengrove": ("a grove of aspen trees with white trunks and golden leaves", "The leaves tremble in the light, and the grass below is soft."),
    "maplewood": ("a wood of red maples", "The leaves glow crimson and orange, and a soft path winds between the trunks."),
    "ginkgoavenue": ("an avenue of ginkgo trees in yellow autumn leaf", "A golden carpet of leaves covers the path, and the air is still."),
    "cedarforest": ("a forest of tall old cedars", "The great trunks rise into soft light, and the ground is soft with needles."),
    "eucalyptus": ("a grove of pale eucalyptus trees", "Their smooth trunks shine in the light, and the leaves hang silver-green in the warm air."),
    "palmgrove": ("a grove of tall palms", "Long fronds move slowly in the warm air, and soft sand lies between the trunks."),
    "cloudforest": ("a misty cloud forest high on a mountain", "Moss and ferns cover every branch, and soft cloud drifts between the trees."),
    "ferngully": ("a green gully full of tall ferns", "A small stream runs through the ferns, and the light is soft and green."),
    "bogcotton": ("a wide bog white with cotton grass", "The soft white tufts sway in the wind over pools of still water."),
    "tundra": ("an autumn tundra in red and gold", "Low bushes glow red and gold across the open land, and far hills are soft and blue."),
    "corkoaks": ("a wide pasture of old cork oaks on golden grass", "The spreading trees stand far apart, and the grass moves gently between them."),
    "mossgarden": ("a shady garden of soft green moss and stepping stones", "Old maples spread over the moss, and the light falls in soft patches."),
    "sprucelake": ("a northern lake ringed by dark spruce", "The trees stand close together along the shore, and the still water mirrors them exactly."),
    "hazelbrook": ("a brook running through a bright hazel wood", "Light falls through young leaves onto the clear water and the mossy stones."),
    "forestmeadow": ("a wide meadow at the heart of a forest", "The tall trees stand all around the soft grass, and the light is gentle."),
    # Meadows, fields and flowers
    "poppyfield": ("a field of red poppies", "The flowers move gently in the breeze, and the hills behind are soft and green."),
    "rapeseed": ("a bright yellow field of rapeseed", "The yellow flowers stretch away to a far line of trees under a wide sky."),
    "flaxfield": ("a field of pale blue flax flowers", "The soft blue field moves like water in the breeze."),
    "haymeadow": ("a summer hay meadow with a few round bales", "The cut grass lies pale and sweet, and the hills behind are soft."),
    "daisymeadow": ("a meadow white with daisies", "Thousands of small white flowers sway in the long grass."),
    "buttercups": ("a meadow golden with buttercups", "The bright flowers shine in the soft grass, and a line of trees stands at the far edge."),
    "daffodils": ("a hillside of yellow daffodils", "The flowers nod in the spring breeze below a few trees just coming into leaf."),
    "crocuses": ("a meadow of purple and white crocuses", "The small flowers open in the first spring light, and the grass is still pale."),
    "gorsehills": ("green hills bright with yellow gorse", "The flowering bushes glow in the sun, and the sea is a line of blue far away."),
    "chalkdowns": ("soft rolling chalk downs", "Short green grass flows over the long hills, and the sky is wide and bright."),
    "pampas": ("a wide plain of pale pampas grass", "The tall plumes move slowly in the wind under a huge sky."),
    "fells": ("wide green fells with a low stone wall and a small lake", "The land rises in soft green waves, and the water lies still in the valley."),
    "highpasture": ("a high summer pasture with an old wooden fence", "Flowers grow in the soft grass, and the peaks stand close under a clear sky."),
    "plateau": ("a wide grassy plateau above the valleys", "The land lies open under a huge sky, and far ridges fade into blue."),
    "almondblossom": ("an orchard of almond trees in white and pink blossom", "The light falls softly through the blossom onto the pale grass."),
    "orangegrove": ("an orange grove in the sun", "Bright fruit hangs among glossy leaves, and the grass between the trees is soft."),
    "wisteria": ("a garden walk under hanging wisteria", "Long purple flowers hang from the old trellis, and the light is soft and mauve."),
    "magnolias": ("a park of old magnolia trees in full bloom", "Big pink and white flowers cover the branches above soft grass."),
    "hydrangeas": ("a garden path lined with blue hydrangeas", "The big round flowers glow in soft light, and the path curves away."),
    "irispond": ("a pond edged with purple irises", "The tall flowers stand in the shallow water, and the pond is calm."),
    "azaleas": ("a hillside of pink azaleas", "The bushes glow pink and white over soft green moss and a few stone steps."),
    "rhododendrons": ("a mountain valley full of red rhododendrons", "The flowers glow against the dark leaves, and snowy peaks rise behind."),
    "peonies": ("a garden full of peonies", "The heavy pink and white flowers nod in the soft light."),
    "lupins": ("a field of blue and purple lupins by a river", "The tall flowers stand in bright rows, and the water shines behind them."),
    "cornfield": ("a tall green cornfield under a summer sky", "The leaves move in the warm wind, and a dirt track runs between the rows."),
    "vineterrace": ("steep terraced vineyards above a wide river", "The green rows step down to the calm water, and the far bank glows in the sun."),
    "bigsky": ("a wide flat grassland under an enormous sky", "The clouds pile high and bright, and the grass moves below them."),
    "heathpond": ("a small dark pond on a wide heath", "Heather and soft grass surround the still water, and the land is open all around."),
    "thymehills": ("dry hills covered in wild thyme and pale stones", "The low purple flowers spread over the warm ground, and the sea shines far away."),
    "orchidmeadow": ("a wet meadow dotted with wild orchids", "Pink and purple flowers stand in the long grass, and a slow stream runs at its edge."),
    # Mountains, snow and sky
    "snowline": ("a high trail where the meadow meets the snow", "Green grass gives way to white, and the peaks shine close above."),
    "sunrisepeaks": ("a line of snowy peaks at sunrise", "The first light turns the summits pink and gold above the dark blue valleys."),
    "icefield": ("a wide field of snow and ice high in the mountains", "Blue shadows lie in the hollows, and the peaks shine in the light."),
    "rockgarden": ("a high alpine rock garden of small bright flowers among boulders", "Tiny blossoms grow between the warm stones, and the valley lies far below."),
    "mountainmist": ("a mountain valley filling with soft morning mist", "The peaks stand clear above the white, and the slopes glow green where the mist thins."),
    "highlake": ("a turquoise lake high in the mountains with a grassy shore", "The still water holds the peaks upside down, and small flowers grow at its edge."),
    "stonepeaks": ("a quiet basin below grey stone peaks", "Pale rock and small green patches lie under the towering walls, and a thin stream runs through."),
    # A way forward (painted with mood=unmotivated, as the push worlds of 27 Oct: a path leading on and up, water in
    # full flow, the summit in view; bright, never stormy).
    "switchbacks": ("a trail zigzagging up a bright green slope toward a high pass", "Each turn climbs a little higher, and the pass glows in the light above."),
    "cliffstairs": ("stone steps cut into a cliff, climbing toward the light", "The sea shines below, and the top of the steps opens onto bright sky."),
    "riverpath": ("a path following a fast clear river upstream toward the mountains", "The water rushes bright beside the path, and the peaks stand ahead."),
    "duneclimb": ("a sandy path climbing a tall dune toward the sky", "The crest is close and bright, and the sand is smooth and warm."),
    "forestclimb": ("a forest trail climbing toward sunlit rocks above the trees", "Light pours down through the last trees, and the open rock shines ahead."),
    "waterfalltrail": ("a path climbing beside a tall white waterfall", "Spray shines in the sun, and the path leads up to the bright top of the falls."),
    "morningsummit": ("a grassy summit with the whole world spread out below it at sunrise", "Range after range of hills lights up, and the sky is clear and wide."),
    "lakeshoretrail": ("a trail along the shore of a mountain lake leading toward the peaks", "The water sparkles beside the path, and the mountains stand bright ahead."),
    "openroad": ("a quiet country road running straight toward far hills", "Fields lie bright on both sides, and the road leads on into the light."),
    "springridge": ("a green ridge path in spring, leading on toward snowy peaks", "Fresh grass and small flowers line the path, and the peaks shine beyond it."),
}

# ─── Variants: id -> (technique, when, the light and the air). The first library's seventeen, and new ones. ──
V = {
    **FIRST_V,
    "sunrise": ("visualization", "at sunrise", "The sun lifts over the horizon, and its first light spreads bright and warm across everything."),
    "afternoon": ("mindfulness", "on a soft afternoon", "The afternoon light is warm and mild, and small white clouds drift slowly across the sky."),
    "overcast": ("mindfulness", "under a soft grey sky", "An even, gentle light lies over everything; the clouds are calm and the air is cool."),
    "afterrain": ("emotional_release", "just after the rain", "Everything is washed and fresh; drops still hang on the leaves, and the light is soft and clear."),
    "drizzle": ("emotional_release", "in a fine drizzle", "A fine soft rain mists the air; everything glistens, and the light is grey and gentle."),
    "frost": ("focus", "on a frosty morning", "Hoarfrost silvers every blade and branch, the air is sharp and clear, and the low sun glitters on it."),
    "bluehour": ("reflection", "in the blue hour after sunset", "The world is deep blue and hushed, with a last band of warm light low in the sky."),
    "sunset": ("gratitude_reflection", "at sunset", "The sun sinks low and red, and the whole sky glows in orange and rose."),
    "clouds": ("focus", "under a sky of big white clouds", "Tall white clouds sail slowly across a deep blue sky, and their shadows drift over the land."),
    "breeze": ("emotional_release", "on a breezy day", "A steady wind moves through everything that can move, and the air is fresh and alive."),
    "milkyway": ("sleep_journey", "under the Milky Way", "A river of stars arches across the dark sky, and the night is warm, deep and quiet."),
    "moonrise": ("sleep_journey", "as the moon rises", "A big pale moon lifts over the horizon, and its light spreads slowly over the quiet land."),
    "thaw": ("affirmation", "at the spring thaw", "The snow is melting into bright trickles, the first green shows through, and the air smells of wet earth."),
    "lateautumn": ("reflection", "in late autumn", "The last golden leaves hang in still air, the light is low and soft, and the ground is covered in colour."),
    "earlysummer": ("body_scan", "in early summer", "Everything is lush and green, the light is warm and bright, and the air is full of life."),
    "firstsnow": ("mindfulness", "after the first snow", "A thin fresh snow lies over everything, and the air is clear, still and bright."),
    "haze": ("reflection", "in a soft summer haze", "A gentle haze softens the distance, and the light is warm and golden."),
}
SLEEP = {v for v, (technique, _, _) in V.items() if technique == "sleep_journey"}
assert SLEEP == {"night", "stars", "aurora", "moon", "milkyway", "moonrise"}, SLEEP
# Weather that moves the world: these variants speak of letting go whatever the place is for.
MOVING = {"rain", "breeze", "afterrain", "drizzle"}

# ─── What each world is for: the music's categories (engine library.MOODS), and what the brief says of it. ──
FEEL = {
    "calm": (None, "Let your breath settle into the rhythm of this place, and feel how calm and safe it is here."),
    "release": ("tense", "With every breath out, let the moving water and the open air carry away what you no longer need."),
    "energy": ("unmotivated", "Breathe in the fresh air and feel your energy rising, ready for the way ahead."),
    "focus": ("scattered", "Let the clear air and the wide view gather your attention into one calm, steady point."),
    "joy": ("happy", "Let the brightness and colour of this place lift you, and feel a smile rise with your breath."),
    "gratitude": ("grateful", "Let the warmth of this place fill you, and notice how much there is to be thankful for."),
    "strength": ("unsure", "Stand tall in this wide place, and feel its quiet strength become your own."),
}
PLACE_FEEL = {
    "seastacks": "strength", "chalkcliffs": "strength", "pebblecove": "calm", "tidalflats": "release", "mangroves": "calm",
    "coralreef": "joy", "atoll": "joy", "sandbar": "calm", "icebergbay": "focus", "driftwood": "release",
    "shellbeach": "joy", "estuary": "release", "seameadow": "joy", "rockarch": "strength", "archipelago": "calm",
    "longbeach": "release", "machair": "joy", "headland": "strength", "windyshore": "release", "pinecoast": "calm",
    "lakebeach": "calm", "stoneshore": "calm", "boathouse": "calm",
    "reedlake": "calm", "oxbow": "calm", "cypressswamp": "calm", "craterlake": "focus", "tarn": "focus",
    "springpool": "calm", "travertine": "gratitude", "geyserbasin": "calm", "braidedriver": "release",
    "riverisland": "calm", "cascades": "release", "lakecliffs": "focus", "meltwater": "energy", "glacierfoot": "focus",
    "dunelake": "calm", "canyonriver": "release", "pinklake": "joy", "waterfallvalley": "release", "meadowbrook": "calm",
    "paddyfields": "calm", "karst": "calm", "halffrozen": "focus", "rainforestriver": "calm", "springfalls": "release",
    "lakeinlet": "calm",
    "whitedesert": "focus", "reddunes": "strength", "joshuatrees": "calm", "saguaro": "calm", "mesa": "strength",
    "hoodoos": "focus", "paintedhills": "joy", "lavafield": "calm",
    "larchforest": "gratitude", "aspengrove": "joy", "maplewood": "gratitude", "ginkgoavenue": "joy", "cedarforest": "calm",
    "eucalyptus": "calm", "palmgrove": "calm", "cloudforest": "calm", "ferngully": "calm", "bogcotton": "release",
    "tundra": "calm", "corkoaks": "gratitude", "mossgarden": "calm", "sprucelake": "calm", "hazelbrook": "calm",
    "forestmeadow": "calm",
    "poppyfield": "joy", "rapeseed": "joy", "flaxfield": "calm", "haymeadow": "gratitude", "daisymeadow": "joy",
    "buttercups": "joy", "daffodils": "joy", "crocuses": "gratitude", "gorsehills": "energy", "chalkdowns": "focus",
    "pampas": "release", "fells": "focus", "highpasture": "gratitude", "plateau": "strength", "almondblossom": "gratitude",
    "orangegrove": "joy", "wisteria": "gratitude", "magnolias": "gratitude", "hydrangeas": "calm", "irispond": "calm",
    "azaleas": "joy", "rhododendrons": "strength", "peonies": "gratitude", "lupins": "joy", "cornfield": "energy",
    "vineterrace": "gratitude", "bigsky": "strength", "heathpond": "calm", "thymehills": "calm", "orchidmeadow": "calm",
    "snowline": "energy", "sunrisepeaks": "strength", "icefield": "strength", "rockgarden": "focus", "mountainmist": "focus",
    "highlake": "focus", "stonepeaks": "strength",
    "switchbacks": "energy", "cliffstairs": "energy", "riverpath": "energy", "duneclimb": "energy", "forestclimb": "energy",
    "waterfalltrail": "energy", "morningsummit": "energy", "lakeshoretrail": "energy", "openroad": "energy",
    "springridge": "energy",
}
assert set(PLACE_FEEL) == set(PLACES), set(PLACE_FEEL) ^ set(PLACES)

# ─── Which times, weathers and seasons each place gets (5-8 each; about one sleep variant each). ────────────
LIBRARY = {
    "seastacks": ["dawn", "golden", "mist", "breeze", "overcast", "stars", "bluehour"],
    "chalkcliffs": ["morning", "midday", "golden", "clouds", "afternoon", "night"],
    "pebblecove": ["morning", "golden", "dusk", "afterrain", "overcast", "moon", "haze"],
    "tidalflats": ["dawn", "golden", "dusk", "mist", "overcast", "sunset", "moonrise"],
    "mangroves": ["morning", "midday", "golden", "mist", "afterrain", "night"],
    "coralreef": ["morning", "midday", "golden", "afternoon", "sunset", "stars"],
    "atoll": ["dawn", "midday", "golden", "afternoon", "bluehour", "moon"],
    "sandbar": ["dawn", "midday", "golden", "dusk", "haze", "night"],
    "icebergbay": ["morning", "midday", "golden", "mist", "aurora", "bluehour", "snowfall"],
    "driftwood": ["morning", "golden", "mist", "breeze", "overcast", "sunset", "moon"],
    "shellbeach": ["dawn", "morning", "midday", "golden", "afternoon", "night"],
    "estuary": ["dawn", "golden", "mist", "breeze", "overcast", "sunset", "moonrise"],
    "seameadow": ["morning", "midday", "golden", "breeze", "earlysummer", "clouds", "stars"],
    "rockarch": ["dawn", "midday", "golden", "sunset", "mist", "night"],
    "archipelago": ["morning", "golden", "mist", "dusk", "overcast", "night", "firstsnow"],
    "longbeach": ["dawn", "golden", "mist", "breeze", "overcast", "sunset", "moon"],
    "machair": ["morning", "midday", "golden", "breeze", "earlysummer", "bluehour"],
    "headland": ["dawn", "morning", "golden", "clouds", "breeze", "sunset", "stars"],
    "windyshore": ["morning", "midday", "golden", "breeze", "overcast", "dusk", "moon"],
    "pinecoast": ["morning", "golden", "mist", "afternoon", "haze", "night"],
    "lakebeach": ["dawn", "morning", "midday", "golden", "dusk", "autumn", "stars"],
    "stoneshore": ["morning", "golden", "mist", "afterrain", "dusk", "night"],
    "boathouse": ["morning", "golden", "mist", "autumn", "dusk", "snow", "night"],
    "reedlake": ["dawn", "golden", "mist", "breeze", "dusk", "autumn", "moonrise"],
    "oxbow": ["morning", "golden", "mist", "afternoon", "autumn", "dusk", "night"],
    "cypressswamp": ["dawn", "morning", "golden", "mist", "afterrain", "dusk", "fireflies", "night"],
    "craterlake": ["morning", "midday", "golden", "clouds", "dusk", "stars"],
    "tarn": ["dawn", "morning", "midday", "mist", "dusk", "snow", "stars"],
    "springpool": ["morning", "midday", "golden", "afterrain", "autumn", "dusk"],
    "travertine": ["dawn", "morning", "golden", "dusk", "bluehour", "night"],
    "geyserbasin": ["dawn", "morning", "golden", "dusk", "frost", "night"],
    "braidedriver": ["morning", "midday", "golden", "clouds", "dusk", "autumn", "stars"],
    "riverisland": ["morning", "golden", "mist", "afternoon", "autumn", "dusk", "moon"],
    "cascades": ["morning", "midday", "golden", "afterrain", "autumn", "dusk", "night"],
    "lakecliffs": ["dawn", "morning", "golden", "mist", "dusk", "night"],
    "meltwater": ["sunrise", "morning", "midday", "clouds", "thaw", "dusk"],
    "glacierfoot": ["morning", "midday", "golden", "mist", "overcast", "dusk", "stars"],
    "dunelake": ["dawn", "golden", "dusk", "afternoon", "stars", "moon"],
    "canyonriver": ["dawn", "morning", "golden", "afternoon", "dusk", "stars"],
    "pinklake": ["dawn", "midday", "golden", "sunset", "dusk", "night"],
    "waterfallvalley": ["morning", "midday", "golden", "mist", "afterrain", "dusk", "night"],
    "meadowbrook": ["dawn", "morning", "midday", "golden", "earlysummer", "afterrain", "dusk", "fireflies"],
    "paddyfields": ["dawn", "morning", "golden", "mist", "afterrain", "dusk", "night"],
    "karst": ["dawn", "morning", "golden", "mist", "afterrain", "dusk", "night"],
    "halffrozen": ["morning", "midday", "golden", "snowfall", "dusk", "night"],
    "rainforestriver": ["dawn", "morning", "golden", "mist", "afterrain", "night"],
    "springfalls": ["sunrise", "morning", "midday", "clouds", "afterrain", "golden"],
    "lakeinlet": ["dawn", "morning", "golden", "mist", "autumn", "dusk", "stars"],
    "whitedesert": ["dawn", "morning", "golden", "dusk", "bluehour", "stars"],
    "reddunes": ["dawn", "sunrise", "golden", "dusk", "stars", "moon"],
    "joshuatrees": ["dawn", "golden", "sunset", "dusk", "stars", "milkyway"],
    "saguaro": ["dawn", "morning", "golden", "sunset", "dusk", "stars"],
    "mesa": ["dawn", "morning", "golden", "clouds", "sunset", "stars"],
    "hoodoos": ["dawn", "morning", "golden", "snow", "dusk", "stars"],
    "paintedhills": ["morning", "midday", "golden", "afterrain", "dusk", "night"],
    "lavafield": ["morning", "golden", "mist", "afterrain", "overcast", "dusk", "aurora"],
    "larchforest": ["morning", "golden", "mist", "autumn", "lateautumn", "dusk", "night"],
    "aspengrove": ["morning", "midday", "golden", "autumn", "breeze", "dusk", "night"],
    "maplewood": ["morning", "golden", "mist", "autumn", "lateautumn", "afterrain", "dusk"],
    "ginkgoavenue": ["morning", "golden", "autumn", "lateautumn", "afterrain", "dusk"],
    "cedarforest": ["morning", "golden", "mist", "afterrain", "snow", "dusk", "night"],
    "eucalyptus": ["morning", "midday", "golden", "afternoon", "haze", "dusk"],
    "palmgrove": ["dawn", "morning", "midday", "golden", "breeze", "dusk", "night"],
    "cloudforest": ["dawn", "morning", "mist", "afterrain", "golden", "dusk", "night"],
    "ferngully": ["morning", "midday", "golden", "mist", "afterrain", "dusk"],
    "bogcotton": ["morning", "midday", "golden", "breeze", "overcast", "dusk", "moonrise"],
    "tundra": ["morning", "golden", "mist", "frost", "dusk", "aurora"],
    "corkoaks": ["dawn", "morning", "golden", "afternoon", "haze", "dusk", "night"],
    "mossgarden": ["morning", "golden", "mist", "afterrain", "autumn", "dusk"],
    "sprucelake": ["dawn", "morning", "mist", "golden", "dusk", "snow", "aurora"],
    "hazelbrook": ["morning", "midday", "golden", "spring", "afterrain", "autumn", "dusk"],
    "forestmeadow": ["dawn", "morning", "midday", "golden", "earlysummer", "mist", "dusk", "fireflies"],
    "poppyfield": ["morning", "midday", "golden", "breeze", "earlysummer", "dusk"],
    "rapeseed": ["morning", "midday", "golden", "clouds", "breeze", "spring", "dusk"],
    "flaxfield": ["morning", "midday", "golden", "breeze", "earlysummer", "dusk"],
    "haymeadow": ["morning", "golden", "afternoon", "clouds", "sunset", "dusk", "stars"],
    "daisymeadow": ["morning", "midday", "golden", "breeze", "earlysummer", "dusk"],
    "buttercups": ["morning", "midday", "golden", "spring", "afterrain", "dusk"],
    "daffodils": ["morning", "midday", "golden", "spring", "thaw", "dusk"],
    "crocuses": ["morning", "midday", "golden", "spring", "thaw", "dusk"],
    "gorsehills": ["sunrise", "morning", "midday", "clouds", "breeze", "golden"],
    "chalkdowns": ["morning", "midday", "golden", "clouds", "breeze", "mist", "dusk"],
    "pampas": ["morning", "golden", "breeze", "afternoon", "sunset", "dusk", "moon"],
    "fells": ["morning", "golden", "mist", "clouds", "overcast", "dusk", "snow"],
    "highpasture": ["morning", "midday", "golden", "earlysummer", "clouds", "dusk", "stars"],
    "plateau": ["dawn", "sunrise", "midday", "golden", "clouds", "sunset", "stars"],
    "almondblossom": ["dawn", "morning", "midday", "golden", "spring", "dusk"],
    "orangegrove": ["morning", "midday", "golden", "afternoon", "dusk", "night"],
    "wisteria": ["morning", "midday", "golden", "afterrain", "spring", "dusk"],
    "magnolias": ["morning", "midday", "golden", "spring", "afterrain", "dusk"],
    "hydrangeas": ["morning", "golden", "rain", "afterrain", "overcast", "dusk"],
    "irispond": ["dawn", "morning", "golden", "afterrain", "dusk", "night"],
    "azaleas": ["morning", "midday", "golden", "mist", "spring", "dusk"],
    "rhododendrons": ["dawn", "morning", "midday", "golden", "mist", "dusk"],
    "peonies": ["morning", "midday", "golden", "afterrain", "dusk", "night"],
    "lupins": ["morning", "midday", "golden", "earlysummer", "breeze", "dusk"],
    "cornfield": ["sunrise", "morning", "midday", "golden", "breeze", "dusk"],
    "vineterrace": ["morning", "golden", "afternoon", "autumn", "haze", "dusk", "night"],
    "bigsky": ["dawn", "morning", "clouds", "breeze", "golden", "sunset", "stars", "milkyway"],
    "heathpond": ["dawn", "golden", "mist", "overcast", "dusk", "autumn", "moon"],
    "thymehills": ["morning", "midday", "golden", "haze", "dusk", "night"],
    "orchidmeadow": ["morning", "midday", "golden", "afterrain", "earlysummer", "dusk"],
    "snowline": ["sunrise", "morning", "midday", "clouds", "golden", "thaw"],
    "sunrisepeaks": ["sunrise", "dawn", "morning", "golden", "clouds", "bluehour", "stars"],
    "icefield": ["dawn", "morning", "midday", "golden", "dusk", "stars", "aurora"],
    "rockgarden": ["morning", "midday", "golden", "clouds", "earlysummer", "dusk"],
    "mountainmist": ["dawn", "sunrise", "morning", "golden", "autumn", "dusk"],
    "highlake": ["dawn", "morning", "midday", "golden", "clouds", "dusk", "stars"],
    "stonepeaks": ["dawn", "morning", "golden", "clouds", "dusk", "snow", "night"],
    "switchbacks": ["sunrise", "morning", "midday", "clouds", "golden", "breeze"],
    "cliffstairs": ["sunrise", "morning", "midday", "golden", "clouds", "breeze"],
    "riverpath": ["sunrise", "morning", "midday", "golden", "clouds", "thaw"],
    "duneclimb": ["sunrise", "morning", "golden", "clouds", "breeze", "dusk"],
    "forestclimb": ["sunrise", "morning", "midday", "golden", "autumn", "afterrain"],
    "waterfalltrail": ["sunrise", "morning", "midday", "golden", "clouds", "afterrain"],
    "morningsummit": ["sunrise", "dawn", "morning", "clouds", "golden", "breeze"],
    "lakeshoretrail": ["sunrise", "morning", "midday", "golden", "clouds", "autumn"],
    "openroad": ["sunrise", "morning", "midday", "golden", "clouds", "breeze", "sunset"],
    "springridge": ["sunrise", "morning", "midday", "golden", "clouds", "thaw"],
}
assert set(LIBRARY) == set(PLACES), set(LIBRARY) ^ set(PLACES)

# The budget for this job is $35 on Meta's API (about 4.5 cents a world), so about 705 themes; and about 15 % must be
# sleep worlds (the first bank gave 10 %). So: every place that has no sleep variant gets a night, except the push
# places (a way forward is a daytime thing) and places where a night would show nothing; then the longest lists lose
# their least telling variants, the quiet afternoon and haze first, until 705 are left.
NO_NIGHT = {"switchbacks", "cliffstairs", "riverpath", "duneclimb", "forestclimb", "waterfalltrail", "morningsummit",
            "lakeshoretrail", "openroad", "springridge", "snowline", "springfalls", "meltwater", "gorsehills",
            "cornfield", "daffodils", "crocuses", "flaxfield", "daisymeadow", "poppyfield", "rapeseed", "lupins",
            "azaleas", "rhododendrons", "magnolias", "wisteria", "almondblossom", "buttercups", "rockgarden",
            "mountainmist", "ferngully", "springpool"}
WARM_NIGHT_STARS = {"whitedesert", "reddunes", "saguaro", "mesa", "joshuatrees", "pinklake", "bigsky", "plateau"}
for place, variants in LIBRARY.items():
    if not set(variants) & SLEEP and place not in NO_NIGHT:
        variants.append("night" if place not in WARM_NIGHT_STARS else "stars")
TARGET = 705
LEAST = ["haze", "afternoon", "overcast", "earlysummer", "lateautumn", "firstsnow", "dusk", "midday", "mist", "golden"]
def _trim() -> None:
    while sum(len(v) for v in LIBRARY.values()) > TARGET:
        for least in LEAST:
            longest = max((p for p, v in LIBRARY.items() if least in v), key=lambda p: len(LIBRARY[p]), default=None)
            if longest is not None and len(LIBRARY[longest]) >= 6:
                LIBRARY[longest].remove(least)
                break
        else:
            raise AssertionError("nothing left to trim")
_trim()
assert sum(len(v) for v in LIBRARY.values()) == TARGET
# Weather that cannot be: no snow by the warm sea or in a desert, no fireflies in the snow.
WARM = {"coralreef", "atoll", "mangroves", "palmgrove", "rainforestriver", "shellbeach", "sandbar", "eucalyptus",
        "whitedesert", "reddunes", "joshuatrees", "saguaro", "mesa", "pinklake", "orangegrove", "thymehills", "dunelake"}
for place, variants in LIBRARY.items():
    assert len(set(variants)) == len(variants), place
    assert all(v in V for v in variants), (place, variants)
    if place in WARM:
        assert not {"snow", "snowfall", "firstsnow", "frost", "thaw", "aurora"} & set(variants), place


def brief(place: str, variant: str) -> str:
    where, what = PLACES[place]
    technique, when, light = V[variant]
    if technique == "sleep_journey":
        return (f"# technique=sleep_journey\nAs your body grows heavy and warm, picture {where} {when}. {what} "
                f"{light} Nothing here needs anything from you. Your breath slows, and the quiet carries you "
                f"gently towards sleep.\n")
    feel = "release" if variant in MOVING else PLACE_FEEL[place]
    mood, closing = FEEL[feel]
    head = f"# technique={technique}" + (f" mood={mood}" if mood else "")
    opening = "Take a deep breath" if feel == "energy" else "Take a slow breath"
    return f"{head}\n{opening} and picture {where} {when}. {what} {light} {closing}\n"


def feeling_of(place: str, variant: str) -> str:
    if variant in SLEEP:
        return "sleep"
    return "release" if variant in MOVING else PLACE_FEEL[place]


def first_library() -> tuple[set[str], set[str]]:
    """(ids, titles) of the first library: its catalog, its briefs and its push worlds."""
    catalog = json.loads(CATALOG.read_text()) if CATALOG.exists() else []
    entries = catalog if isinstance(catalog, list) else list(catalog.values())
    ids = {e.get("id") for e in entries if isinstance(e, dict) and e.get("id")}
    ids |= {f"{p}-{v}" for p, vs in FIRST_LIBRARY.items() for v in vs} | set(FIRST_PUSH) | {f"{p}-v2" for p in FIRST_PUSH}
    titles = {str(e.get("title", "")).lower() for e in entries if isinstance(e, dict)}
    return ids, titles


def main(out: str = "library/briefs2") -> int:
    taken_ids, taken_titles = first_library()
    taken_places = {i.split("-")[0] for i in taken_ids} | set(FIRST_PLACES) | {i.split("-")[0] for i in FIRST_PUSH}
    clash = sorted(set(PLACES) & taken_places)
    assert not clash, f"place words already in the library: {clash}"
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.txt"):
        old.unlink()
    themes = []
    for place, variants in LIBRARY.items():
        for variant in variants:
            id_ = f"{place}-{variant}"
            assert id_ not in taken_ids, id_
            assert len(id_) <= 40 and id_.replace("-", "").isalnum(), id_
            (folder / f"{id_}.txt").write_text(brief(place, variant))
            themes.append({"id": id_, "place": PLACES[place][0], "variant": variant, "when": V[variant][1],
                           "feeling": feeling_of(place, variant), "technique": V[variant][0],
                           "sleep": variant in SLEEP})
    (folder / "themes.json").write_text(json.dumps(themes, indent=1, ensure_ascii=False))
    feelings = Counter(t["feeling"] for t in themes)
    sleep = sum(t["sleep"] for t in themes)
    print(f"{len(themes)} themes of {len(PLACES)} new places -> {folder}")
    print(f"sleep {sleep} ({sleep / len(themes):.0%}); feelings: " + ", ".join(f"{k} {n}" for k, n in feelings.most_common()))
    print(f"variants used: {len({t['variant'] for t in themes})}; existing titles known: {len(taken_titles)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
