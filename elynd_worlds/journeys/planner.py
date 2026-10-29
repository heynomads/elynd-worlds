"""The plan of a 30-day journey: an arc (title, promise, the check-in question, four weeks, the journey's world), then
the thirty days, a week at a time, the four weeks at once.

Which day is what is fixed here, not left to the model: day 1 begins the journey, days 7, 14 and 21 look back on
their week and open the next, day 30 arrives; every other day is a practice. A moment the listener named (a talk, an
exam) is handled each day from its date (brief.py), so a skipped day never puts the rehearsal on the wrong day.

The journey's world is described here in English and in general terms only (a place, its light, how it changes), and
checked in code to carry none of the listener's own words: it is the only part of a journey that goes to Meta's API.
"""
from __future__ import annotations

import datetime as dt
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from .guide import LANGUAGES, Moment, _plain, first_json, forbidden_word
from .llama import Llama, LlamaError

DAYS = 30
WEEKS = ((1, 7), (8, 14), (15, 21), (22, 30))
LOOK_BACK = (7, 14, 21)
# The techniques ELYND's meditation writer knows (its session config's "technique").
TECHNIQUES = ("breath_awareness", "visualization", "body_scan", "gratitude_reflection", "mindfulness",
              "emotional_release", "sleep_journey", "affirmation", "focus", "reflection", "activation", "hypnosis")
FALLBACK_TECHNIQUE = {"begin": "visualization", "look_back": "reflection", "arrive": "reflection",
                      "practice": "mindfulness"}


class PlanError(Exception):
    """No usable plan. Never carries the listener's words."""


def kind(n: int) -> str:
    if n == 1:
        return "begin"
    if n == DAYS:
        return "arrive"
    return "look_back" if n in LOOK_BACK else "practice"


def week_of(n: int) -> int:
    return next(i for i, (first, last) in enumerate(WEEKS, 1) if first <= n <= last)


@dataclass
class Day:
    n: int
    kind: str
    title: str
    focus: str
    technique: str
    aim: str
    try_today: str


@dataclass
class Plan:
    title: str
    promise: str
    checkin_question: str
    weeks: list[dict]                 # [{"n", "theme", "line"}] x 4
    world: dict                       # {"scene": str, "stages": [str] x 5}, English, nothing of the listener's
    days: list[Day] = field(default_factory=list)
    usd: float = 0.0

    def to_json(self) -> dict:
        data = asdict(self)
        data.pop("usd")
        return data

    @classmethod
    def from_json(cls, data: dict) -> "Plan":
        days = [Day(**d) for d in data.get("days") or []]
        return cls(title=data["title"], promise=data["promise"], checkin_question=data["checkin_question"],
                   weeks=data["weeks"], world=data["world"], days=days)

    def day(self, n: int) -> Day:
        return self.days[n - 1]


ARC = """You plan a 30-day meditation journey at ELYND for one person, toward the goal below. Each day ELYND writes \
them a {minutes}-minute guided meditation from your plan. Write in {language}, except the "world", which is in English.

The weeks build: first noticing (when and how it shows up, without judging), then kindness toward themselves, then \
practising the moment itself (rehearsal, steadiness, the body's part), then carrying it into their life. Shape the \
themes to their goal; do not copy these words.

Never use the words heal, cure, therapy, treatment, disorder, symptom, patient, recover, cope or help, nor their \
equivalents in {language}. Never name a condition. This is meditation practice toward a personal intention, not advice.

Answer with one JSON object and nothing else:
{{"title": "<their journey's title, 2 to 6 words, in {language}>",
 "promise": "<one sentence to them, second person, what these thirty days are for>",
 "checkin_question": "<the question asked before each meditation about the time since the last one, with the one \
word for what they want more of, taken from their goal (confidence: confident; sleep: rested), in {language}. English: \
'Since yesterday, how confident did you feel?'; German: 'Wie ausgeruht hast du dich seit gestern gefühlt?'; Spanish \
and French with a noun, never an adjective that would assume their gender: '¿Cuánta confianza has sentido desde \
ayer?', 'Depuis hier, combien de calme as-tu ressenti ?'>",
 "weeks": [{{"theme": "<1 to 3 words>", "line": "<one sentence: what this week is for>"}}, (four in all)],
 "world": {{"scene": "<one English sentence: a real, peaceful landscape that can stand for this journey, with a path, \
a horizon or something that grows; no people, no animals, no buildings with signs, no text, nothing about the person, \
their life, their goal or their feelings>",
   "stages": ["<week 1, in English: the same place at its start, for example before dawn at the foot of the path>", \
"<week 2>", "<week 3>", "<week 4>", "<day 30: the same place at its fullest, for example in full light at the top>"]}}}}"""

WEEK = """You plan days {first} to {last} of a 30-day meditation journey at ELYND for one person. Each day ELYND writes \
them a {minutes}-minute guided meditation from your plan. Write in {language}.

Their goal, in their own words: {summary}
The journey: "{title}". {promise}
The four weeks: {weeks}
This is week {week}: "{theme}" ({line}).{moment}

The days of this week and what each is:
{days}

Kinds: "begin" opens the journey (says their goal back, sketches the thirty days, a first small practice); "practice" \
is one step of this week's theme; "look_back" looks back on the week that ends and opens the next week's theme; \
"arrive" closes the thirty days.
Vary the techniques; use sleep_journey only for a goal about sleep. Each day has one small, concrete thing to try in \
their day, different every day.
Never use the words heal, cure, therapy, treatment, disorder, symptom, patient, recover, cope or help, nor their \
equivalents in {language}.

Answer with one JSON object and nothing else:
{{"days": [{{"n": <day>, "title": "<2 to 6 words>", "focus": "<one sentence to them, second person, present \
tense, shown before they begin: what today's meditation is about, e.g. 'Today you notice what your body does when you \
hold back.'>", "technique": "<one of {techniques}>", "aim": "<one sentence: what they take from it>", \
"try_today": "<one sentence to them: one small thing to try today>"}}, ...]}}
Write titles in sentence case."""


def _personal_words(summary: str) -> set[str]:
    """The listener's distinctive words (5 letters or more), which the world's description may not repeat."""
    common = {"about", "after", "again", "every", "feel", "feels", "feeling", "their", "there", "these", "thing",
              "things", "would", "could", "should", "where", "which", "while", "being", "other", "really", "little",
              "light", "morning", "evening", "night", "today", "place", "quiet", "still", "calm", "peace", "peaceful"}
    return {w for w in re.findall(r"[^\W\d_]{5,}", summary.lower()) if w not in common}


def _world(data: object, summary: str) -> dict:
    if not isinstance(data, dict):
        raise ValueError("world")
    scene = _plain(data.get("scene"), 300)
    stages = [_plain(s, 200) for s in (data.get("stages") or [])][:5]
    if not scene or len(stages) != 5 or not all(stages):
        raise ValueError("world needs a scene and five stages")
    personal = _personal_words(summary)
    for text in [scene, *stages]:
        if set(re.findall(r"[^\W\d_]+", text.lower())) & personal or forbidden_word(text, "ENG"):
            raise ValueError("the world repeats the listener's words")
    return {"scene": scene, "stages": stages}


def parse_arc(text: str, language: str, summary: str) -> dict:
    data = first_json(text)
    weeks = data.get("weeks") or []
    if len(weeks) < 4:
        raise ValueError("four weeks")
    arc = {
        "title": _plain(data.get("title"), 60).rstrip("."),
        "promise": _plain(data.get("promise"), 220),
        "checkin_question": _plain(data.get("checkin_question"), 90),
        "weeks": [{"n": i, "theme": _plain(w.get("theme"), 40).rstrip("."), "line": _plain(w.get("line"), 160)}
                  for i, w in enumerate(weeks[:4], 1) if isinstance(w, dict)],
        "world": _world(data.get("world"), summary),
    }
    if not (arc["title"] and arc["promise"] and arc["checkin_question"].endswith("?")) or len(arc["weeks"]) != 4:
        raise ValueError("arc fields")
    if not all(w["theme"] and w["line"] for w in arc["weeks"]):
        raise ValueError("week fields")
    for value in [arc["title"], arc["promise"], arc["checkin_question"],
                  *(w["theme"] + " " + w["line"] for w in arc["weeks"])]:
        word = forbidden_word(value, language)
        if word:
            raise ValueError(f"a word it may not use: {word}")
    return arc


def parse_week(text: str, language: str, first: int, last: int) -> list[Day]:
    data = first_json(text)
    by_n = {}
    for item in data.get("days") or []:
        if isinstance(item, dict) and isinstance(item.get("n"), int):
            by_n.setdefault(item["n"], item)
    days = []
    for n in range(first, last + 1):
        item = by_n.get(n)
        if item is None:
            raise ValueError(f"day {n} missing")
        technique = str(item.get("technique") or "").strip().lower()
        day = Day(n=n, kind=kind(n), title=_plain(item.get("title"), 60).rstrip("."),
                  focus=_plain(item.get("focus"), 220), aim=_plain(item.get("aim"), 220),
                  technique=technique if technique in TECHNIQUES else FALLBACK_TECHNIQUE[kind(n)],
                  try_today=_plain(item.get("try_today"), 180))
        if not (day.title and day.focus and day.aim and day.try_today):
            raise ValueError(f"day {n} fields")
        for value in (day.title, day.focus, day.aim, day.try_today):
            word = forbidden_word(value, language)
            if word:
                raise ValueError(f"a word it may not use: {word}")
        days.append(day)
    return days


def _ask(llama: Llama, prompt: str, parse, max_tokens: int):
    """One answer, asked again once when it cannot be used. Returns (value, usd)."""
    spent, problem = 0.0, ""
    for attempt in range(3):
        try:
            reply = llama.chat("You plan meditation journeys for ELYND. You answer with one JSON object only.",
                               prompt + problem, max_tokens=max_tokens, temperature=0.7 if attempt == 0 else 0.4)
        except LlamaError as exc:
            raise PlanError(str(exc)) from None
        spent += reply.usd
        try:
            return parse(reply.text), spent
        except ValueError as exc:
            problem = f"\n\nYour last answer could not be used ({exc}). Answer again: one JSON object only."
    raise PlanError("no usable answer")


def make_plan(llama: Llama, language: str, title: str, summary: str, minutes: int = 10,
              moment: Moment | None = None) -> Plan:
    """The journey's plan. `summary` and `title` are the guide's, in the listener's words."""
    language = language if language in LANGUAGES else "ENG"
    lang_name = LANGUAGES[language]
    goal = f'"{title}". {summary}'
    arc, spent = _ask(llama, ARC.format(language=lang_name, minutes=minutes) + f"\n\nTheir goal, in their own words: "
                      f"{goal}" + (f"\nA moment coming up: {moment.label}" + (f" on {moment.date}" if moment.date
                                                                              else "") if moment else ""),
                      lambda text: parse_arc(text, language, summary), max_tokens=1200)
    weeks_line = "; ".join(f'week {w["n"]}: {w["theme"]} ({w["line"]})' for w in arc["weeks"])
    moment_line = ""
    if moment:
        moment_line = (f"\nA moment is coming up in their life: {moment.label}" +
                       (f" on {moment.date}" if moment.date else "") +
                       ". The days around it are shaped by its date, not by this plan; mention it only lightly.")

    def plan_week(index: int):
        first, last = WEEKS[index]
        week = arc["weeks"][index]
        prompt = WEEK.format(first=first, last=last, minutes=minutes, language=lang_name, summary=summary,
                             title=arc["title"], promise=arc["promise"], weeks=weeks_line, week=week["n"],
                             theme=week["theme"], line=week["line"], moment=moment_line,
                             days="\n".join(f"- day {n}: {kind(n)}" for n in range(first, last + 1)),
                             techniques=", ".join(TECHNIQUES))
        return _ask(llama, prompt, lambda text: parse_week(text, language, first, last), max_tokens=2600)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(plan_week, range(4)))
    days = [day for week_days, _ in results for day in week_days]
    spent += sum(usd for _, usd in results)
    return Plan(title=arc["title"], promise=arc["promise"], checkin_question=arc["checkin_question"],
                weeks=arc["weeks"], world=arc["world"], days=days, usd=spent)


def today_iso() -> str:
    return dt.date.today().isoformat()
