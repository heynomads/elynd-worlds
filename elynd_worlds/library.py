"""The library: worlds painted ahead of time, and Muse Spark choosing one for a meditation.

A world made for a meditation takes about a minute and costs about six cents,
and most meditations visit places a library can hold. So a session first gets a
library world: one of the worlds the agent painted in advance, from short themes
across the places guided meditations visit most (scripts/library_briefs.py),
each checked by the critic like any other. Muse Spark reads the meditation, the
same text the art director reads and nothing more, names the three library
worlds that belong to it best, and says whether the meditation is built around
something no library world shows (a particular animal or object that matters to
the listener, a particular person's place, an unusual setting). The first choice
the listener has not seen lately is shown at once. When it fits, it is the
session's world; when the meditation needs its own (`needs_own_world`), a world
is painted for it and takes over when ready.

Sleep is decided before Muse Spark is asked: a meditation for falling asleep
chooses only among the dark, quiet worlds, any other only among the rest.

There is always a world along with the meditation: when Muse Spark must not be
asked (a note that must stay on the server, a spent budget) or cannot answer, `local` chooses here, by the
place and time the meditation names, and nothing leaves the server.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from .art_director import Meditation
from .life import life
from .meta import Meter, cache_fields, data_url, preview_jpeg, privacy_fields, reasoning, spark_chat
from .settings import DEFAULT, Settings

log = logging.getLogger("elynd_worlds")

SLEEP_TECHNIQUES = {"sleep_journey"}


@dataclass(frozen=True)
class Entry:
    """One library world, as the chooser sees it."""

    id: str
    title: str
    place: str
    time_of_day: str
    sleep: bool = False
    moods: tuple[str, ...] = ()   # the feelings it suits (MOODS), best first

    def line(self) -> str:
        # Short on purpose: the theme (place and time, e.g. harbour-golden) and the painting's title. With
        # 500 worlds every word here is paid for on every session.
        return f"{self.id}: {self.title}"

    @classmethod
    def of_scene(cls, id: str, scene: dict[str, Any]) -> Entry:
        return cls(id=id, title=scene["title"], place=scene["place"], time_of_day=scene["time_of_day"],
                   sleep=bool(scene.get("sleep")))


def entries_in(folders: Iterable[str | Path]) -> list[Entry]:
    """The library worlds saved under `folders` (each world's folder name is its id)."""
    found = {}
    for folder in folders:
        for scene in sorted(Path(folder).rglob("scene.json")):
            world = scene.parent / "world.json"
            if world.exists():
                entry = Entry.of_scene(scene.parent.name, json.loads(scene.read_text()))
                moods = tuple(m for m in json.loads(world.read_text()).get("moods") or [] if m in MOODS)
                found[scene.parent.name] = dataclasses.replace(entry, moods=moods)
    return list(found.values())


@dataclass(frozen=True)
class Pick:
    entry: Entry
    fit: int                 # 0-10: how well it belongs to the meditation (9-10: the same place and light)
    own_world: bool = False  # the meditation is built around something no library world shows
    place: bool = True       # the meditation describes or travels to a place (not only the breath or the body)


PICK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "own_world": {"type": "boolean",
                      "description": "The meditation is built around something no library world shows"},
        "place": {"type": "boolean", "description": "The meditation describes or travels to a place"},
        "choices": {
            "type": "array",
            "description": "The three worlds that belong to the meditation best, best first",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "description": "Exactly as listed"},
                    "fit": {"type": "integer", "description": "0-10"},
                },
                "required": ["id", "fit"],
            },
        },
    },
    "required": ["own_world", "place", "choices"],
}

PICK_INSTRUCTIONS = """You choose the painted world that fills a listener's phone screen while their
guided meditation plays in ELYND, a meditation app. Choose from the library below the three worlds
that belong to the meditation best, best first:
1. The place the meditation describes or travels to (a beach, a forest, a garden, a lake...), then its
   time of day, season and weather.
2. If it describes no place, worlds whose light and mood suit its technique and feeling: soft and
   gentle for sadness, bright and fresh for energy or focus, warm for gratitude.
fit (0-10): 9-10 the same place and light; 6-8 the same kind of place; 3-5 only the mood fits.
own_world: true only if the meditation is built around something specific that none of these worlds
shows: a particular animal or object that matters to the listener, a particular person's place known by
its own things (her red bench and the blue watering can by the door, their old street), or an unusual
setting (a train, a temple). A similar-looking library world is not that place.
false if it describes no place, or a common place that one of these worlds shows well (a beach, a
river walk, a forest, a garden, a cabin...), even described in loving detail and full of memories.
place: true if the meditation describes or travels to a place (outdoors or a room), false if it stays
with the breath, the body or a feeling.
Use the ids exactly as listed. Return only the JSON object.

The library (id: title; the id names the place, then the time, season or weather):
"""


def is_sleep(meditation: Meditation) -> bool:
    return meditation.technique in SLEEP_TECHNIQUES


async def pick(client, meditation: Meditation, entries: list[Entry], meter: Meter, settings: Settings = DEFAULT,
               recent: Iterable[str] = ()) -> Pick | None:
    """The library world for one meditation; None if the library has none of its kind or the answer is unusable.

    `recent`: ids the listener saw lately; the next best is shown instead when it fits nearly as well.
    """
    sleep = is_sleep(meditation)
    candidates = [e for e in entries if e.sleep == sleep]
    if not candidates:
        return None
    # The library first and the meditation last: the long, unchanging start of the request can be cached.
    response, _ = await spark_chat(
        client, (settings.library_model, settings.spark_fallback_model),
        messages=[
            {"role": "system", "content": PICK_INSTRUCTIONS + "\n".join(e.line() for e in candidates)},
            {"role": "user", "content": _shortened(meditation, settings.pick_words).as_message()},
        ],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "pick", "schema": PICK_SCHEMA, "strict": True}},
        **reasoning(settings.pick_reasoning),
        extra_body={**privacy_fields(), **cache_fields("pick")},
        timeout=settings.pick_timeout,
    )
    meter.spark(getattr(response, "usage", None), "pick")
    answer = json.loads(response.choices[0].message.content)
    return choose(answer.get("choices") or [], candidates, recent, own_world=answer.get("own_world") is True,
                  place=answer.get("place") is not False)


# What a meditation's place is made of, and its time and weather: the words that tell library worlds apart
# (`local`). A synonym counts as its word; a library id's place may be a compound ("forestlake"), read by its
# whole words only. The details of a place (a bench, a path) count less than the place itself: a river walk with
# a bench is a river. Every place of the library has its word here (tests), except a clearing: "clearing the
# mind" is no place, and a clearing is a forest too.
PLACES = set("""beach sea shore coast cliff bay lagoon harbour island lake pond lilies river stream brook
waterfall pool fjord loch marsh reeds forest pine beech birch oak redwoods bamboo grove orchard garden meadow
field hills valley mountain peak ridge summit alpine glacier dunes desert canyon cabin chalet cottage barn
village city rooftop temple courtyard greenhouse treehouse park wheat lavender tulips heather vineyard cherry
rice tea lighthouse canal jetty oasis pavilion savanna steppe sunflowers wildflowers bluebells willow heath
highlands countryside tuscany hotspring saltflat blacksand aboveclouds whitehouses""".split())
DETAILS = set("pier bench bridge path window room fire lantern fountain porch hammock".split())
TIMES = set("dawn morning midday golden dusk night stars moon mist rain snow autumn spring summer blossom aurora".split())
SYNONYMS = {"ocean": "sea", "seaside": "sea", "sand": "beach", "sandy": "beach", "waves": "sea", "woods": "forest",
            "harbor": "harbour", "mountains": "mountain", "peaks": "peak", "hut": "cabin", "lodge": "cabin",
            "fireplace": "fire", "fields": "field", "trees": "forest", "creek": "stream", "lakeside": "lake",
            "sunrise": "dawn", "daybreak": "dawn", "noon": "midday", "afternoon": "golden", "sunset": "golden",
            "evening": "dusk", "twilight": "dusk", "starry": "stars", "starlight": "stars", "moonlight": "moon",
            "moonlit": "moon", "misty": "mist", "fog": "mist", "foggy": "mist", "rainy": "rain", "raindrops": "rain",
            "snowy": "snow", "snowfall": "snow", "snowflakes": "snow", "winter": "snow", "flowers": "garden",
            "sunflower": "sunflowers", "wildflower": "wildflowers", "bluebell": "bluebells", "tuscan": "tuscany",
            "savannah": "savanna", "grassland": "steppe", "grasslands": "steppe", "prairie": "steppe",
            "highland": "highlands", "jetties": "jetty", "oases": "oasis", "onsen": "hotspring",
            "moorland": "heath", "heathland": "heath"}
NAMED = PLACES | DETAILS | TIMES

# The same names in the app's other languages, by what they name. Read only in a meditation of that language:
# "see" is a lake in German and a verb in English. Left out on purpose where a word means something else too:
# Kiefer (pine, and the jaw of a body scan), morgen (tomorrow), Weg (away), Fluss in Einfluss (influence),
# Regen in anregen (to stimulate), mañana (tomorrow), haya (a beech, and "there may be"), fuente (a source),
# été (summer, and "been"), serre (a greenhouse, and "squeezes"), Weide (a willow, and a pasture), claro (a
# clearing, and "of course").
FOREIGN_NAMES = {
    "de": {
        "beach": "strand strände strandes", "sea": "meer meeres mittelmeer ozean ostsee nordsee", "shore": "ufer",
        "coast": "küste küsten", "cliff": "klippe klippen", "bay": "bucht", "lagoon": "lagune",
        "harbour": "hafen", "island": "insel inseln", "lake": "see seen sees", "pond": "teich teiche weiher",
        "lilies": "seerose seerosen", "river": "fluss flusses flüsse flusslauf", "stream": "bach baches bäche",
        "waterfall": "wasserfall wasserfälle", "marsh": "moor sumpf", "reeds": "schilf",
        "forest": "wald waldes wälder", "pine": "kiefern pinie pinien tanne tannen", "beech": "buche",
        "birch": "birke birken", "oak": "eiche eichen", "bamboo": "bambus", "grove": "hain",
        "orchard": "obstgarten obstwiese", "garden": "garten gärten", "meadow": "wiese wiesen",
        "field": "feld felder", "hills": "hügel", "valley": "tal täler", "mountain": "berg berge gebirge",
        "peak": "gipfel", "summit": "bergspitze", "ridge": "grat bergrücken", "alpine": "alm alpen",
        "glacier": "gletscher", "dunes": "düne dünen", "desert": "wüste", "canyon": "schlucht",
        "cabin": "hütte hütten", "cottage": "häuschen", "barn": "scheune", "village": "dorf dörfer",
        "city": "stadt", "rooftop": "dachterrasse", "temple": "tempel", "courtyard": "innenhof",
        "greenhouse": "gewächshaus", "treehouse": "baumhaus", "wheat": "weizen weizenfeld kornfeld",
        "lavender": "lavendel lavendelfeld", "tulips": "tulpen", "heather": "heidekraut",
        "vineyard": "weinberg weinberge", "cherry": "kirschbaum kirschbäume kirschblüte kirschblüten",
        "rice": "reisfeld reisfelder reisterrassen", "pier": "seebrücke", "bench": "bank", "bridge": "brücke",
        "path": "pfad wanderweg waldweg feldweg", "window": "fenster", "room": "zimmer",
        "fire": "feuer kamin lagerfeuer", "lantern": "laterne laternen", "fountain": "brunnen",
        "porch": "veranda", "hammock": "hängematte", "dawn": "sonnenaufgang morgendämmerung morgenröte",
        "morning": "morgens morgenlicht morgensonne vormittag", "midday": "mittag mittags mittagssonne",
        "golden": "sonnenuntergang abendsonne", "dusk": "abend abends abendlicht dämmerung",
        "night": "nacht nachts", "stars": "sterne sternen sternenhimmel sternenlicht", "moon": "mond mondlicht vollmond",
        "mist": "nebel neblig", "rain": "regen regentropfen", "snow": "schnee verschneit winter",
        "autumn": "herbst", "spring": "frühling", "summer": "sommer", "blossom": "blüte blüten",
        "aurora": "polarlicht polarlichter nordlicht nordlichter",
        "lighthouse": "leuchtturm leuchttürme", "canal": "kanal kanäle gracht grachten",
        "jetty": "steg stege bootssteg holzsteg anlegesteg", "oasis": "oase oasen", "pavilion": "pavillon",
        "savanna": "savanne", "sunflowers": "sonnenblume sonnenblumen sonnenblumenfeld",
        "wildflowers": "wildblumen wildblumenwiese", "bluebells": "hasenglöckchen",
        "willow": "trauerweide weidenbaum weidenbäume", "heath": "heide heidelandschaft", "highlands": "hochland",
        "tuscany": "toskana", "hotspring": "thermalquelle thermalquellen", "saltflat": "salzwüste salzpfanne",
        "aboveclouds": "wolkenmeer",
    },
    "es": {
        "beach": "playa playas arena", "sea": "mar océano olas", "shore": "orilla orillas", "coast": "costa",
        "cliff": "acantilado acantilados", "bay": "bahía", "lagoon": "laguna", "harbour": "puerto",
        "island": "isla islas", "lake": "lago lagos", "pond": "estanque", "lilies": "nenúfares",
        "river": "río ríos", "stream": "arroyo riachuelo", "waterfall": "cascada cascadas catarata",
        "fjord": "fiordo", "marsh": "pantano marisma", "reeds": "juncos", "forest": "bosque bosques selva",
        "pine": "pino pinos pinar", "beech": "hayedo", "birch": "abedul abedules", "oak": "roble robles",
        "redwoods": "secuoyas", "bamboo": "bambú", "grove": "arboleda", "orchard": "huerto huerta",
        "garden": "jardín jardines flores", "meadow": "prado prados pradera", "field": "campo campos",
        "hills": "colina colinas", "valley": "valle valles", "mountain": "montaña montañas monte",
        "peak": "pico picos", "summit": "cima cumbre", "ridge": "cresta", "glacier": "glaciar",
        "dunes": "dunas", "desert": "desierto", "canyon": "cañón", "cabin": "cabaña", "cottage": "casita",
        "barn": "granero", "village": "pueblo aldea", "city": "ciudad", "rooftop": "azotea",
        "temple": "templo", "courtyard": "patio", "greenhouse": "invernadero", "park": "parque",
        "wheat": "trigo trigal", "lavender": "lavanda", "tulips": "tulipanes", "heather": "brezo",
        "vineyard": "viñedo viñedos", "cherry": "cerezo cerezos", "rice": "arrozal arrozales", "pier": "muelle",
        "bench": "banco", "bridge": "puente", "path": "sendero senda camino", "window": "ventana",
        "room": "habitación", "fire": "fuego hoguera chimenea", "lantern": "farol farolillo linterna",
        "porch": "porche", "hammock": "hamaca", "dawn": "amanecer alba aurora", "midday": "mediodía",
        "golden": "atardecer", "dusk": "anochecer crepúsculo", "night": "noche", "stars": "estrellas estrellado",
        "moon": "luna", "mist": "niebla bruma neblina", "rain": "lluvia", "snow": "nieve nevado invierno",
        "autumn": "otoño", "spring": "primavera", "summer": "verano", "aurora": "boreal",
        "lighthouse": "faro faros", "jetty": "embarcadero embarcaderos", "pavilion": "pabellón",
        "savanna": "sabana sabanas", "steppe": "estepa estepas", "sunflowers": "girasol girasoles",
        "willow": "sauce sauces", "heath": "brezal brezales", "countryside": "campiña", "tuscany": "toscana",
        "hotspring": "termas", "saltflat": "salar salares",
    },
    "fr": {
        "beach": "plage plages sable", "sea": "mer océan vagues", "shore": "rivage rive", "coast": "côte",
        "cliff": "falaise falaises", "bay": "baie", "lagoon": "lagune", "harbour": "port",
        "island": "île îles", "lake": "lac lacs", "pond": "étang mare", "lilies": "nénuphars",
        "river": "rivière fleuve", "stream": "ruisseau", "waterfall": "cascade", "marsh": "marais",
        "reeds": "roseaux", "forest": "forêt forêts bois", "pine": "pin pins", "beech": "hêtre hêtres hêtraie",
        "birch": "bouleau bouleaux", "oak": "chêne chênes", "redwoods": "séquoias", "bamboo": "bambou bambous",
        "grove": "bosquet", "orchard": "verger", "garden": "jardin jardins fleurs",
        "meadow": "prairie pré prés", "field": "champ champs", "hills": "colline collines", "valley": "vallée",
        "mountain": "montagne montagnes", "peak": "pic", "summit": "sommet", "ridge": "crête",
        "desert": "désert", "cabin": "cabane", "cottage": "chaumière", "barn": "grange", "city": "ville",
        "courtyard": "cour", "park": "parc", "wheat": "blé", "lavender": "lavande",
        "tulips": "tulipes", "heather": "bruyère", "vineyard": "vigne vignes vignoble",
        "cherry": "cerisier cerisiers", "rice": "rizière rizières", "pier": "jetée", "bench": "banc",
        "bridge": "pont", "path": "sentier chemin", "window": "fenêtre", "room": "chambre pièce",
        "fire": "feu cheminée", "lantern": "lanterne", "fountain": "fontaine", "porch": "véranda",
        "hammock": "hamac", "dawn": "aube aurore", "morning": "matin matinée", "midday": "midi",
        "dusk": "soir soirée crépuscule", "night": "nuit", "stars": "étoiles étoilé", "moon": "lune",
        "mist": "brume brouillard", "rain": "pluie", "snow": "neige enneigé hiver", "autumn": "automne",
        "spring": "printemps", "aurora": "boréale",
        "lighthouse": "phare phares", "canal": "canaux", "jetty": "ponton pontons", "pavilion": "pavillon",
        "savanna": "savane savanes", "sunflowers": "tournesol tournesols", "willow": "saule saules",
        "heath": "lande landes", "countryside": "campagne", "tuscany": "toscane", "bluebells": "jacinthes",
    },
}
FOREIGN = {language: {word: name for name, words in names.items() for word in words.split()}
           for language, names in FOREIGN_NAMES.items()}
# German joins words: a compound names what its last part names ("Buchenwald": forest, "Bergsee": lake). Only
# endings that end no other common word (not Meer: a Blumenmeer is flowers); longest first, so a "Weinberg"
# stays a vineyard.
GERMAN_ENDINGS = sorted(
    {"wasserfall": "waterfall", "weinberg": "vineyard", "obstgarten": "orchard", "wald": "forest", "wälder": "forest",
     "see": "lake", "seen": "lake", "strand": "beach", "küste": "coast", "ufer": "shore",
     "bucht": "bay", "insel": "island", "teich": "pond", "bach": "stream", "garten": "garden",
     "wiese": "meadow", "wiesen": "meadow", "hütte": "cabin", "dorf": "village", "berg": "mountain",
     "berge": "mountain", "gipfel": "peak", "gletscher": "glacier", "dünen": "dunes", "tempel": "temple",
     "brücke": "bridge", "pfad": "path", "bank": "bench", "zimmer": "room", "laterne": "lantern",
     "schnee": "snow", "nebel": "mist", "mond": "moon", "sterne": "stars", "nacht": "night"}.items(),
    key=lambda pair: -len(pair[0]))
assert all(name in NAMED for names in FOREIGN_NAMES.values() for name in names), "a foreign word names nothing"
assert all(name in NAMED for _, name in GERMAN_ENDINGS), "a German ending names nothing"
# Places named in more than one word, joined before the words are read: a hot spring is no spring (the season).
# English ones in every language, as English words are. "" takes apart what only sounds like a place: the ear
# canal of a body scan.
PHRASES = {
    "en": {"hot spring": "hotspring", "hot springs": "hotspring", "salt flat": "saltflat",
           "salt flats": "saltflat", "black sand": "blacksand", "sea of clouds": "aboveclouds",
           "above the clouds": "aboveclouds", "white houses": "whitehouses", "ear canal": "", "ear canals": ""},
    "de": {"heiße quelle": "hotspring", "heißen quelle": "hotspring", "heiße quellen": "hotspring",
           "heißen quellen": "hotspring", "über den wolken": "aboveclouds", "schwarzer sand": "blacksand",
           "schwarzen sand": "blacksand", "weiße häuser": "whitehouses", "weißen häuser": "whitehouses",
           "weißen häusern": "whitehouses"},
    "es": {"aguas termales": "hotspring", "mar de nubes": "aboveclouds", "sobre las nubes": "aboveclouds",
           "arena negra": "blacksand", "tierras altas": "highlands", "flores silvestres": "wildflowers",
           "casas blancas": "whitehouses", "canal auditivo": ""},
    "fr": {"source chaude": "hotspring", "sources chaudes": "hotspring", "mer de nuages": "aboveclouds",
           "au-dessus des nuages": "aboveclouds", "sable noir": "blacksand", "fleurs sauvages": "wildflowers",
           "désert de sel": "saltflat", "maisons blanches": "whitehouses"},
}
_PHRASES = {language: [(re.compile(rf"\b{re.escape(phrase)}\b"), joined or " ") for phrase, joined in
                       sorted({**PHRASES["en"], **phrases}.items(), key=lambda pair: -len(pair[0]))]
            for language, phrases in PHRASES.items()}
assert all(joined in NAMED or not joined for phrases in PHRASES.values() for joined in phrases.values())


def _plain(word: str) -> str:
    """The word as NAMED lists it: a plural without its s or es ("pines": pine, "beaches": beach), never
    another word cut short ("seat" is not the sea, "team" not tea)."""
    if word not in NAMED and word.endswith("s"):
        for one in (word[:-1], word[:-2] if word.endswith("es") else ""):
            if one in NAMED:
                return one
    return word


def _name(word: str, language: str) -> str:
    """What one word names (a NAMED word), in English or in the meditation's own language, else the word."""
    foreign = FOREIGN.get(language)
    if foreign:
        if word in foreign:
            return foreign[word]
        if language == "de" and len(word) > 5:
            for ending, name in GERMAN_ENDINGS:
                if word.endswith(ending) and len(word) - len(ending) >= 3:
                    return name
    return _plain(SYNONYMS.get(word, word))


def _named(text: str, language: str = "en") -> Counter:
    """How often a text names each place, detail, time and weather, in English or in its own language
    (`language`: the meditation's, en, de, es or fr)."""
    language = (language or "en")[:2].lower()
    text = text.lower()
    for phrase, joined in _PHRASES.get(language, _PHRASES["en"]):
        text = phrase.sub(joined, text)
    words = (_name(w, language) for w in re.findall(r"[^\W\d_]+", text))
    return Counter(w for w in words if w in NAMED)


def _id_places(place: str) -> set[str]:
    """The places a library id's place ("pineforest", "alpinetrail") names, by whole words only: an alpine trail
    is no pine, a footbridge no ridge."""
    found = {w for w in NAMED if w in place}
    return {w for w in found & PLACES if not any(w != other and w in other for other in found)}


def local(meditation: Meditation, entries: list[Entry], recent: Iterable[str] = (),
          moods: Iterable[str] = ()) -> Pick | None:
    """A library world chosen here, without Muse Spark: nothing leaves the server. When the meditation names
    a place, a world of that place (then its details, time and weather); when it names none (the breath,
    the body), a world of the session's feeling: `moods`, the categories the app's music is chosen by
    (from the mood the listener picked or wrote), best first. Of its kind
    (a sleep world for a sleep meditation, any other rather than none), and never one seen lately while one
    that fits as well is fresh; among equals, the same meditation always gets the same one, another
    meditation another (as for the music: fit first, variety within the best). Fit 0: Muse Spark
    never rated it."""
    sleep = is_sleep(meditation)
    candidates = [e for e in entries if e.sleep == sleep] or list(entries)
    if not candidates:
        return None
    said = _named(" ".join(meditation.script.split()[:DEFAULT.pick_words]), meditation.language)
    moods = [m for m in moods if m in MOODS]
    names_place = any(w in PLACES for w in said)

    weight = {**dict.fromkeys(PLACES, 3), **dict.fromkeys(DETAILS, 1), **dict.fromkeys(TIMES, 2)}
    mains = {e.id: _id_places(e.id.split("-")[0]) for e in candidates}
    how_many = Counter(w for main in mains.values() for w in main)

    def score(entry: Entry) -> int:
        # What the world is (its id's place: "cabin", "zengarden" a garden) counts twice; what it shows besides once.
        main = mains[entry.id]
        its = set(_named(f"{entry.id.replace('-', ' ')} {entry.title} {entry.place} {entry.time_of_day}")) | main
        return sum(weight[w] * min(said[w], 3) for w in its if w in said) + sum(3 for w in main if w in said)

    def rare(entry: Entry) -> float:
        """Among worlds that fit as well, the rarer place the meditation names: a bamboo grove is a forest too,
        so "a forest of bamboo" opens on the bamboo, not on any forest."""
        return sum(1 / how_many[w] for w in mains[entry.id] if w in said)

    def feeling(entry: Entry) -> int:
        """How well the world suits the session's feelings, 0 best. As the music's tiers: every world with the
        session's first feeling before any with only its second, and among them the world's own order (its
        first feeling before its third). Nearly every world is calm somewhere, so calm as a world's first
        must never tie with letting go as its second (an angry listener got generic calm worlds)."""
        return min((3 * i + entry.moods.index(m) for i, m in enumerate(moods) if m in entry.moods),
                   default=3 * len(moods))

    # The place first when the meditation names one, else the feeling first.
    key = ((lambda e: (score(e), rare(e), -feeling(e))) if names_place else (lambda e: (-feeling(e), score(e))))
    seen = set(recent)
    fresh = [e for e in candidates if e.id not in seen] or candidates
    best = max(key(e) for e in fresh)
    tied = [e for e in fresh if key(e) == best]
    return Pick(tied[zlib.crc32(meditation.script.encode()) % len(tied)], 0,
                place=names_place or any(w in DETAILS for w in said))


# ─── The feelings a library world suits ─────────────────────────────────

# The categories the app's music is chosen by, so a world fits how
# the listener feels the way its music does: what each suits, and what in a world speaks to it. Plus joy: a
# happy listener and an unmotivated one hear the same lifting music (energy_motivation) but need different
# places: a sleepy tropical beach motivates no one. Here energy_motivation is the push, joy the smile.
MOODS = {
    "calm_ground": "calm and grounding, for stress and anxiety: still, soft, safe places",
    "release_let_go": "letting go, for anger and tension: flowing water, wind, rain, open space",
    "connection_gratitude": "warmth and connection, for sadness and loneliness, and gratitude: warm light, "
                            "gardens, homes, blossom",
    "restore_sleep": "rest and sleep, for tiredness: night, late dusk, dim and quiet places",
    "energy_motivation": "momentum, for low motivation: a way forward, upward or a new beginning, seen in the "
                         "place itself: a path or trail leading on, steps or a ridge climbing, a summit or peaks "
                         "in view, sunrise over open heights, water in full flow (a waterfall, a rushing river, "
                         "surf on rocks). Bright morning light alone is not enough: a sunny garden, courtyard, "
                         "grove or field is joy or calm. Never a place to lie back and do nothing (a lazy beach, "
                         "a hammock, an oasis pool, still midday heat)",
    "joy": "joy and lightness, for a happy listener: bright, colourful, sunlit places that smile: sunflowers, "
           "meadows in bloom, blossom, a sunny garden or courtyard, sparkling water",
    "reflection_insight": "quiet reflection, for a calm mind: mist, dusk, still water, temples, rain on water",
    "focus_clarity": "clarity and focus, for a scattered mind: clear air, wide views, mountains, open horizons",
    "confidence_strength": "strength and courage, for fear: peaks, cliffs, sunrise, big skies",
}

MOODS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "worlds": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "description": "Exactly as listed"},
                    "moods": {"type": "array", "items": {"type": "string", "enum": list(MOODS)},
                              "description": "1-3 feelings it suits best, best first"},
                },
                "required": ["id", "moods"],
            },
        },
    },
    "required": ["worlds"],
}

MOODS_INSTRUCTIONS = ("You tag the painted worlds of ELYND, a meditation app, by the feelings they suit, so a "
                      "listener's world fits how they feel. The feelings (use the ids exactly):\n"
                      + "\n".join(f"- {name}: {what}" for name, what in MOODS.items())
                      + "\nFor each world below, name the 1-3 feelings it suits best, best first: the feelings "
                      "of a listener who would be helped by being there (a lazy beach calms; it does not "
                      "motivate). Judge by what the listener would see: the painting when it is attached, its "
                      "place, light, time of day and air. Return only the JSON object.")


async def tag_moods(client, worlds: list[tuple[str, dict[str, Any]]], meter: Meter,
                    settings: Settings = DEFAULT, batch: int = 10,
                    picture: Callable[[str], Awaitable[bytes]] | None = None) -> dict[str, tuple[str, ...]]:
    """The feelings each library world suits, by Muse Spark: {id: moods, best first}, from each world's
    design (`worlds`: (id, scene)), ten worlds a request. With `picture` (a world's painting by id), Muse
    Spark also sees each painting, one world a request: the designs all speak softly ("a quiet ridge", "a
    slow river"), so only the picture tells a summit at sunrise from a sunny garden. A world left out of an
    answer, or whose request failed three times, is left out here."""
    gate = asyncio.Semaphore(4)   # also the most paintings held at once

    async def ask(part: list[tuple[str, dict[str, Any]]]) -> dict[str, tuple[str, ...]]:
        lines = "\n".join(f"{id}: {scene.get('title')}. {scene.get('place')} Time: {scene.get('time_of_day')}. "
                          f"Light: {scene.get('light')}. Air: {scene.get('air')}." for id, scene in part)
        for attempt in range(3):
            try:
                async with gate:
                    content: Any = lines
                    if picture is not None:
                        content = [{"type": "text", "text": lines}, *[
                            {"type": "image_url", "image_url": {"url": data_url(await asyncio.to_thread(
                                preview_jpeg, await picture(id), settings.moods_image_width), "image/jpeg")}}
                            for id, _ in part]]
                    response, _ = await spark_chat(
                        client, (settings.library_model, settings.spark_fallback_model),
                        messages=[{"role": "system", "content": MOODS_INSTRUCTIONS},
                                  {"role": "user", "content": content}],
                        response_format={"type": "json_schema",
                                         "json_schema": {"name": "moods", "schema": MOODS_SCHEMA, "strict": True}},
                        **reasoning(settings.pick_reasoning),
                        extra_body={**privacy_fields(), **cache_fields("moods")},
                        timeout=30,
                    )
                break
            except Exception as error:  # noqa: BLE001 - asked again, then left out
                log.info("tagging %s failed (%s), attempt %d", part[0][0], type(error).__name__, attempt + 1)
                await asyncio.sleep(2 * (attempt + 1))
        else:
            return {}
        meter.spark(getattr(response, "usage", None), "moods")
        asked = {id for id, _ in part}
        tags = {}
        for world in json.loads(response.choices[0].message.content).get("worlds") or []:
            moods = tuple(dict.fromkeys(m for m in world.get("moods") or [] if m in MOODS))[:3]
            if world.get("id") in asked and moods:
                tags[world["id"]] = moods
        return tags

    size = 1 if picture is not None else batch
    answers = await asyncio.gather(*(ask(worlds[i:i + size]) for i in range(0, len(worlds), size)))
    return {id: moods for answer in answers for id, moods in answer.items()}


def needs_own_world(chosen: Pick | None, settings: Settings = DEFAULT) -> bool:
    """Whether the meditation should get a world painted for it: no answer from Muse Spark, a meditation
    built around something no library world shows, or a place no library world fits. A meditation without
    a place (the breath, the body) never does: its library world fits by mood, and a painting would add
    nothing of its own (27 Oct 2026: a body scan rated 4 for mood would have been painted for 6 cents)."""
    return chosen is None or chosen.own_world or (chosen.place and chosen.fit < settings.library_min_fit)


def choose(choices: list[dict[str, Any]], candidates: list[Entry], recent: Iterable[str] = (),
           near: int = 2, own_world: bool = False, place: bool = True) -> Pick | None:
    """The first choice the listener has not seen lately and that fits within `near` of the best."""
    by_id = {e.id: e for e in candidates}
    valid = []
    for choice in choices:
        entry = by_id.get(str(choice.get("id", "")).strip())
        if entry is not None and entry.id not in {p.entry.id for p in valid}:
            valid.append(Pick(entry, max(0, min(10, int(choice.get("fit") or 0))), own_world, place))
    if not valid:
        return None
    seen = set(recent)
    best = valid[0]
    return next((p for p in valid if p.entry.id not in seen and p.fit >= best.fit - near), best)


def _shortened(meditation: Meditation, words: int) -> Meditation:
    """The meditation's first `words` words: the place is set early, and the rest only costs tokens."""
    script = meditation.script.split()
    if len(script) <= words:
        return meditation
    return Meditation(script=" ".join(script[:words]) + " ...", technique=meditation.technique,
                      mood=meditation.mood, language=meditation.language, minutes=meditation.minutes)


# ─── Only worlds that look alive ──────────────────────────────────────────

# A world must never look static or dead. Measured on phone renders (the
# share of the picture that visibly changes within a second): a wheat field 34%, a stream in the snow 11%,
# a misty pine forest 10%, a lavender field 7.5%, a salt flat's water 6%, a sea of clouds 5%, a canyon under
# the stars 4.3%; a rooftop at dusk 2.4%, dunes 0.4% and a canyon by day 0.2%, where only a sliver of sky drifts.
FALLING = {"rain", "snow", "leaves", "petals"}


def looks_dead(manifest: dict[str, Any], scene: dict[str, Any]) -> str | None:
    """Why a saved world would look dead on the phone, or None when something clearly moves all the time:
    water, a stream, a field or meadow, swaying crowns, falling rain, snow, leaves or petals, a moving
    subject; or a sky so wide (half the picture) or so starry (at night) that it carries the world alone."""
    living = manifest.get("living") or {}
    share = living.get("coverage") or {}
    named = {"water", "sky", "grass", "stream"} & set(scene.get("nature") or [])
    if named and not living.get("regions"):
        return "what moves in it was never found (" + ", ".join(sorted(named)) + ")"
    particles = life(scene, {k: bool(living.get(k)) for k in ("water", "sky", "grass", "stream")})["particles"]
    if (share.get("water", 0) >= 0.05 or living.get("stream") or share.get("grass", 0) >= 0.10
            or living.get("trees") or particles in FALLING
            or any(layer.get("motion") != "still" for layer in manifest.get("layers") or [])):
        return None
    sky = share.get("sky", 0)
    if sky >= 0.5 or (scene.get("time_of_day") == "night" and sky >= 0.25):
        return None
    return f"only a little sky moves ({sky:.0%})" if sky else "nothing moves"

