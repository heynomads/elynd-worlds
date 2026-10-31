"""The place a meditation needs, described by Llama 3.3 without the listener's words (31 Oct 2026).

A listener's meditation can repeat what they shared, their health included, and Meta's Model API takes no health
data: its processor terms allow no exception, with consent or without. So in the ELYND app a meditation never goes to
it. Llama 3.3, Meta's open model, which Amazon Bedrock runs in ELYND's own AWS account (where a listener's words may
go, as for Journeys), reads the meditation and describes the one calm place it needs, for a painter who never sees
the meditation: where, the light, the weather, up to three things in the foreground.

Code checks the description before anything goes on (`problems`): nothing about a person (no "you", no "I"), no word
about health, feelings or the body, no name taken from the meditation, no run of its words. One more ask with the
reasons, then none: the listener keeps their library world. Muse Spark designs from the description
(`Meditation(brief=True)`), never from the meditation, and the listener's mood is never sent on.

`chat(system, user) -> str` is any Llama 3.3 call: the engine's `journeys.llama.Llama`, or the server's own client.
"""
from __future__ import annotations

import re
from collections.abc import Callable

from .art_director import Meditation

MIN_WORDS, MAX_WORDS = 15, 90
COPIED_RUN = 7          # words in a row taken from the meditation: a sentence about the listener, not a place

PLACE_INSTRUCTIONS = """You help a painter who will never read the guided meditation below. Describe the one calm
place this meditation needs, in plain English, in 25 to 60 words:
- where: the place the meditation itself describes, as it describes it (an old garden with an apple tree, a long
  sandy beach at low tide, a cabin window at night), in English whatever the meditation's language; never the name
  of a town, house, street, person or animal;
- the time of day and the light, the weather and the air, as the meditation gives them;
- up to three things in the foreground that the meditation names (a wooden bench, a watering can, a rowboat).
An animal or a person only if the meditation itself brings one, and then only asleep or lying at rest (an old dog
asleep on the sand), or birds resting on water, or a butterfly.
Never write about the listener or anyone's life: no "you", no "I", no names, no feelings, no health, no body, no
memories, no worries. Do not copy any sentence or phrase of the meditation. If the meditation describes no place,
choose a natural place that suits its technique and mood. Answer with the description only."""

# Words that are about a person, not a place (the description is always English).
PERSON = frozenset("""you your yours yourself yourselves i me my mine myself we us our ours ourselves listener
meditator""".split())
HEALTH = frozenset("""anxiety anxious stress stressed panic depression depressed sad sadness grief grieving mourning
cancer illness ill sick sickness disease pain painful hurt hurting injury therapy therapist medication medicine
diagnosis trauma traumatic fear fears afraid scared worry worried worries lonely loneliness loss death dead dying died
divorce hospital doctor nurse surgery chemo chemotherapy insomnia addiction alcohol suicide heal healing recovery
symptom symptoms burnout exhausted overwhelmed heartbreak miscarriage pregnant pregnancy body chest shoulders lungs
breath breathe breathing heartbeat feeling feelings emotion emotions""".split())
WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")


def _words(text: str) -> list[str]:
    return [w.lower().strip("'") for w in WORD.findall(text)]


def _names(text: str) -> set[str]:
    """Capitalised words that do not begin a sentence: the names in a meditation."""
    found: set[str] = set()
    for sentence in re.split(r"(?<=[.!?:;])\s+|\n+", text):
        for word in WORD.findall(sentence)[1:]:
            if word[:1].isupper() and len(word) > 1:
                found.add(word.strip("'").lower())
    return found


def _capitalised(text: str) -> set[str]:
    """Every capitalised word, a sentence's first too: a name the description begins with is still a name."""
    return {w.strip("'").lower() for w in WORD.findall(text) if w[:1].isupper() and len(w) > 1}


def problems(text: str, meditation: str) -> list[str]:
    """Why a description cannot go on to Meta's models; [] when it can."""
    words = _words(text)
    found: list[str] = []
    if not MIN_WORDS <= len(words) <= MAX_WORDS:
        found.append(f"it has {len(words)} words, not {MIN_WORDS} to {MAX_WORDS}")
    person = sorted({w for w in words if w in PERSON})
    if person:
        found.append("it speaks of a person (" + ", ".join(person) + ")")
    health = sorted({w for w in words if w in HEALTH})
    if health:
        found.append("it names health, feelings or the body (" + ", ".join(health) + ")")
    names = sorted(_capitalised(text) & _names(meditation))
    if names:
        found.append("it takes a name from the meditation (" + ", ".join(names) + ")")
    source = _words(meditation)
    runs = {tuple(source[i:i + COPIED_RUN]) for i in range(len(source) - COPIED_RUN + 1)}
    if any(tuple(words[i:i + COPIED_RUN]) in runs for i in range(len(words) - COPIED_RUN + 1)):
        found.append(f"it copies {COPIED_RUN} or more words in a row from the meditation")
    return found


def _ask(meditation: Meditation) -> str:
    return (f"Technique: {meditation.technique or 'not given'}\n"
            f"Mood before the session: {meditation.mood or 'not given'}\n"
            f"Language of the meditation: {meditation.language}\n\n"
            f"Meditation:\n{meditation.script.strip()}")


def describe(chat: Callable[[str, str], str], meditation: Meditation) -> str | None:
    """The place the meditation needs, checked; asked once more with the reasons; else None."""
    user = _ask(meditation)
    reasons: list[str] = []
    for _ in range(2):
        ask = user if not reasons else (user + "\n\nYour last description cannot be used: " + "; ".join(reasons)
                                         + ". Write it again, following every rule.")
        text = " ".join((chat(PLACE_INSTRUCTIONS, ask) or "").split()).strip().strip('"')
        reasons = problems(text, meditation.script)
        if not reasons:
            return text
    return None


def for_designer(chat: Callable[[str, str], str], meditation: Meditation) -> Meditation | None:
    """What Muse Spark may read for this meditation: the place, its technique, length and language; no mood, no
    word of the meditation. None when no description passed: the listener keeps their library world."""
    text = describe(chat, meditation)
    if text is None:
        return None
    return Meditation(script=text, technique=meditation.technique, mood="", language=meditation.language,
                      minutes=meditation.minutes, brief=True)
