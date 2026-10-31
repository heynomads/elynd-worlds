"""The plan of a 30-day journey: an arc (what the days are for, the check-in question, four weeks), then the thirty
days, a week at a time, the four weeks at once.

Which day is what is fixed here, not left to the model: day 1 begins the journey, days 7, 14 and 21 look back on
their week and open the next, day 30 arrives; every other day is a practice. A moment the listener named (a talk, an
exam) is never written into a day: the planner is told today's date and the moment's, and that a day number is not a
date (people skip days); the days around the moment are shaped from its date, day by day (brief.py).

The journey's title is the guide's (the plan writes none), and the plan describes no world: a journey's world, when
there is one, is given by hand (stages.py). What the plan may never say is checked in code and asked again with the
exact reason, a week at a time: no word of healing, therapy, help or illness, no condition's name, no coaching or
productivity advice, no attention exercise while driving and no push into what they fear, no promise of a result, no
moment and no "tomorrow" in a day. A day title that repeats another is replaced by one small call for new titles (a
repeated title never fails a plan). Lengths are asked for and, on the last attempt, cut at a sentence end or a word
boundary, never inside a word; English titles are in sentence case; sentences end with a full stop. The listener's own
words are never checked by these lists.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable

from .guide import (FILLER, LANGUAGES, STOP, Moment, _plain, capitalised, condition_word, cut, first_json,
                    forbidden_word, sentence_case, words_of)
from .llama import Llama, LlamaError

DAYS = 30
WEEKS = ((1, 7), (8, 14), (15, 21), (22, 30))
LOOK_BACK = (7, 14, 21)
# The techniques ELYND's meditation writer knows (its session config's "technique").
TECHNIQUES = ("breath_awareness", "visualization", "body_scan", "gratitude_reflection", "mindfulness",
              "emotional_release", "sleep_journey", "affirmation", "focus", "reflection", "activation", "hypnosis")
FALLBACK_TECHNIQUE = {"begin": "visualization", "look_back": "reflection", "arrive": "reflection",
                      "practice": "mindfulness"}
# What is shown or read: asked for in the prompt; over it, asked again; on the last attempt, cut.
LIMITS = {"promise": 220, "line": 140, "theme": 40, "title": 60, "focus": 200, "aim": 200, "try_today": 180,
          "checkin_question": 90}
ATTEMPTS = 3
TEMPERATURES = (0.6, 0.4, 0.3)
SYSTEM = "You plan meditation journeys for ELYND. You answer with one JSON object only."

# The check-in before each meditation: about the time since the last one (people skip days), with one word for what
# they want more of. (the words that must be in it, the question when no right one comes)
CHECKIN = {
    "ENG": ("since your last meditation", "Since your last meditation, how calm have you felt?",
            "'Since your last meditation, how {word} have you felt?' with one word for what they want more of, from "
            "their goal (confident; rested; at ease), never a symptom"),
    "GER": ("seit deiner letzten meditation", "Wie ruhig hast du dich seit deiner letzten Meditation gefühlt?",
            "'Wie {Wort} hast du dich seit deiner letzten Meditation gefühlt?' with one word for what they want more "
            "of, from their goal (zuversichtlich; ausgeruht; gelassen), never a symptom"),
    "ES": ("desde tu última meditación", "¿Cuánta calma has sentido desde tu última meditación?",
           "'¿Cuánta {sustantivo} has sentido desde tu última meditación?' with a noun for what they want more of, "
           "never an adjective that would assume their gender (confianza; calma; '¿Cuánto descanso…?'), never a "
           "symptom"),
    "FR": ("depuis ta dernière méditation", "Depuis ta dernière méditation, combien de calme as-tu ressenti ?",
           "'Depuis ta dernière méditation, combien de {nom} as-tu ressenti ?' with a noun for what they want more "
           "of, never an adjective that would assume their gender (confiance; calme; repos), never a symptom"),
}
SYMPTOMS = re.compile(r"\b(?:anxious|stressed|tense|nervous|worried|overwhelmed|sad|depressed|scared|afraid|"
                      r"panicked|panicky|exhausted|tired|frustrated|angry|lonely)\b", re.I)
# Coaching and productivity advice instead of practice (any language: the words are borrowed). A word the listener
# used in their own goal is theirs to hear ("when a deadline lands, one breath").
COACHING = re.compile(r"\b(?:productiv\w*|prioriti\w*|multitask\w*|to-dos?|to-do list|to do list|deadlines?|"
                      r"notifications?|pomodoro|actionable)\b", re.I)
NERVOUS = re.compile(r"\bnervous system\b", re.I)
# An attention exercise while driving or doing anything that needs full attention, or a push into what they fear.
UNSAFE = re.compile(r"\b(?:(?:while|whilst|as|when)\s+(?:you(?:'re|\s+are)?\s+)?driv\w*|behind the wheel|at the wheel|"
                    r"during (?:your|the) drive|navigat\w*\s+(?:\w+\s+){0,2}(?:roads?|streets?|traffic|roundabouts?|"
                    r"junctions?|intersections?|motorways?|lanes?)|unfamiliar roads?|trust your instincts|"
                    r"(?:one|an) extra stop|push (?:through|yourself)|face your fears?|test yourself)\b", re.I)
# Something to try in a car, on the move: only before setting off, when parked or stopped, or after arriving.
IN_CAR = re.compile(r"\b(?:mirrors?|roundabouts?|steering|the wheel|pedals?|indicators?|junctions?|gearstick|gears|"
                    r"overtak\w*|motorways?)\b", re.I)
STILL = re.compile(r"\b(?:before|after|parked|park|car park|engine off|picture|imagine|visuali[sz]\w*|"
                   r"think(?:ing)? (?:of|about)|remember\w*|recall\w*|arriv\w*|noticed|write|writing|journal\w*|"
                   r"at home)\b", re.I)
# A result promised, or how their moment will go presumed.
PROMISE = re.compile(r"\b(?:you will|you'll|you're going to|you are going to)\b", re.I)
# The "then" of a small thing to try invites, never predicts ("when you're going to bed" predicts nothing).
PREDICTS = re.compile(r"\b(?:you will|you'll)\b", re.I)
PASSING = re.compile(r"\b(?:after|when|once)\s+(?:you\s+)?pass(?:ing|es)?\b", re.I)
# A day is a meditation, not a date: no "tomorrow", no "the day before".
DAY_RELATIVE = re.compile(r"\b(?:tomorrow\w*|the (?:day|evening|night|morning) (?:before|of your)|final preparations?|"
                          r"the big day)\b", re.I)
WRITING = re.compile(r"\b(?:writ(?:e|es|ing|ten)|journal\w*|jot\w*)\b", re.I)
_DETERMINERS = re.compile(r"^(?:(?:my|your|our|the|a|an|his|her|their|mein|meine|meinen|dein|deine|der|die|das|"
                          r"mi|tu|el|la|los|las|un|una|ma|mon|mes|ta|ton|tes|le|les|une)\s+)+", re.I)
_GENERIC = set("""calm calmer calmly calmness mind life goal journey meditation used usually mostly always never often
sometimes something everything anything nothing someone everyone able start starts right less better best full leave
leaves sit sits deal keep keeps take takes come comes around instead whole end ends part even though thing kind sort
moment moments stay stays big small said say says saying sure unsure okay yes know idea something anything
though although thought tell told answer answered test testing idk maybe actually""".split())
# The particle that makes a phrasal verb: "wound up" is tense, a "wound" is an injury.
_PARTICLES = {"up", "down", "out", "off", "away", "back", "through", "over"}


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
    title: str                        # the guide's title: the plan writes none
    promise: str
    checkin_question: str
    weeks: list[dict]                 # [{"n", "theme", "line"}] x 4
    world: dict | None = None         # none from the plan; a journey's world is given by hand (stages.py)
    days: list[Day] = field(default_factory=list)
    usd: float = 0.0

    def to_json(self) -> dict:
        data = asdict(self)
        data.pop("usd")
        return data

    @classmethod
    def from_json(cls, data: dict) -> "Plan":
        days = [Day(**d) for d in data.get("days") or []]
        return cls(title=data.get("title") or "", promise=data["promise"], checkin_question=data["checkin_question"],
                   weeks=data["weeks"], world=data.get("world"), days=days)

    def day(self, n: int) -> Day:
        return self.days[n - 1]


# ─── What the model is told ──────────────────────────────────────────────

RULES = """This is meditation practice toward their intention: never advice, never a condition's name, never causes \
or analysis. Never use the words heal, cure, therapy, treatment, disorder, symptom, patient, recover, cope, help, \
manage, trigger or "nervous system", nor their equivalents in {language}."""

ARC = """You plan a 30-day meditation journey at ELYND for one person, toward the goal below. Each day ELYND writes \
them a {minutes}-minute guided meditation from your plan. Write in {language}.

Their journey: "{title}". Their goal, in their own words: {summary}{moment}
Their own words: {words}.

The weeks build: first noticing (when and how it shows up, without judging), then kindness toward themselves, then \
practising the moment itself (rehearsal, steadiness, the body's part), then carrying it into their life. Shape the \
themes to their goal and their own situations; do not copy these words.

{rules} The promise says what these days are for, never a result: never "you will". Never assume how a coming moment \
will go.

Answer with one JSON object and nothing else:
{{"promise": "<one sentence to them, second person, 80 to 200 characters: what these thirty days are for, in their \
own words and situations>",
 "checkin_question": "<{checkin}>",
 "weeks": [{{"theme": "<1 to 3 words>", "line": "<one sentence to them, 60 to 140 characters, in one of their \
situations: what this week is for>"}}, (four in all)]}}"""

# Each day's small thing to try is an implementation intention, "when <situation>, <one small step>": tied to a
# situation they will meet, a plan in that form is carried out more often than an intention alone. Every day from the
# first: its step is the week's own (noticing, kindness, rehearsal, carrying it), so the form never asks for more.
WEEK = """You plan days {first} to {last} of a 30-day meditation journey at ELYND for one person. Each day ELYND writes \
them a {minutes}-minute guided meditation from your plan. Write in {language}.

Their journey: "{title}". Their goal, in their own words: {summary}
What the thirty days are for: {promise}
The four weeks: {weeks}
This is week {week}: "{theme}" ({line}).{moment}
Their own words: {words}. Shape most days around these situations, in their words, and vary them: a different \
moment and a different practice each day.

The days of this week and what each is:
{days}

Kinds: "begin" opens the journey (their goal said back, the thirty days sketched, a first small practice); "practice" \
is one step of this week's theme; "look_back" looks back on the week that ends and opens the next week's theme; \
"arrive" closes the thirty days: it honours what they practised, never says the goal is reached or how things went.
{rules}
Each day's small thing to try is a small when-then plan in one of the situations they named, different every day: \
"When <that situation, as it can come today>, <one small step of attention, breath or kindness that fits today's \
practice>", e.g. one slow breath, feet on the floor, a kind sentence to themselves. The "when" is something that \
happens (a place, a time, something they do), never a feeling they fear; in a car it is "Before you set off" or "When \
you have parked", never a moment of the drive itself. The "then" invites, in the present tense or as a gentle \
imperative, and never predicts: never "you will" or "you'll". Never a task, a list, a tool, a timer, a notification \
setting, work or productivity advice, or a rule for others. Writing at most once a week.
Never ask for attention exercises while driving or doing anything that needs full attention: anything to try in a car \
is only before setting off, when parked, or after arriving (a meditation may picture the drive). Never ask them to \
stay longer in, go further into, or test themselves against what they fear: invite noticing and small comforts, and \
anchor attention in what steadies (feet, breath, hands, something they enjoy), never in a racing heart or the feeling \
they fear. With a loss, never ask them to smile about, let go of or move on from the person: remembering has its \
place.
Vary the techniques; use sleep_journey only for a goal about sleep. A title names this day's own step, never a \
general one.

Answer with one JSON object and nothing else:
{{"days": [{{"n": <day>, "title": "<2 to 6 words, sentence case>", "focus": "<one sentence to them, second person, \
present tense, 50 to 200 characters: what today's meditation is about, e.g. 'Today you notice what your body does when \
you hold back.'>", "technique": "<one of {techniques}>", "aim": "<one sentence: what they take from it>", \
"try_today": "<one sentence to them, 40 to 180 characters: a small when-then plan for today in {language}, in one \
of their situations: 'When <situation>, <small step>.'>"}}, ...]}}"""

RETITLE = """Some days of a 30-day meditation journey at ELYND need a new title: each repeats another day's title. Write in \
{language}.

Titles in use (never use one of them again): {taken}

The days that need a new title:
{days}

Each new title: 2 to 6 words, sentence case, about that day's own step, different from every other title. {rules}
Answer with one JSON object and nothing else: {{"titles": {{"<day number>": "<new title>"}}}}"""


@dataclass
class _Context:
    language: str
    title: str
    summary: str
    keep: set[str]                    # words they capitalised: sentence case keeps them
    words: list[str]                  # their distinctive words, in their order
    label: re.Pattern | None          # the moment's label as a phrase


def their_words(summary: str, title: str = "", limit: int = 10) -> list[str]:
    """Up to `limit` of the listener's distinctive words (not the common ones, not the filler of someone who said
    little), in the order they come; a verb keeps its particle ("wound up", "switch off")."""
    out, seen = [], set()
    for text in (title, summary):
        for sentence in re.split(r"[.?!;:,]", text or ""):
            tokens = words_of(sentence)
            for i, word in enumerate(tokens):
                if len(word) < 3 or word in STOP or word in _GENERIC or word in FILLER:
                    continue
                if i + 1 < len(tokens) and tokens[i + 1] in _PARTICLES:
                    word = f"{word} {tokens[i + 1]}"
                if word not in seen:
                    seen.add(word)
                    out.append(word)
    return out[:limit]


def _stem(word: str) -> str:
    word = word.split()[0]
    return word[:max(4, len(word) - 3)]


def _label_pattern(moment: Moment | None) -> re.Pattern | None:
    if not moment or not moment.label:
        return None
    phrase = _DETERMINERS.sub("", " ".join(moment.label.lower().split())).strip(" .")
    return re.compile(rf"\b{re.escape(phrase)}\b", re.I) if len(phrase) >= 3 else None


def _moment_line(moment: Moment | None, today: dt.date, weeks: bool) -> str:
    if not moment or not moment.label:
        return ""
    line = (f"\nToday is {today.isoformat()} ({today.strftime('%A')}). Days are counted by meditations, and people "
            f"skip days: a day number is not a date.")
    when = None
    try:
        when = dt.date.fromisoformat(moment.date) if moment.date else None
    except ValueError:
        pass
    if when and when >= today:
        k = (when - today).days
        line += f' Their "{moment.label}" is on {when.isoformat()}, in {k} day{"s" if k != 1 else ""}.'
    else:
        line += f' Their "{moment.label}" is coming up.'
    line += " Never write it into a day's text: ELYND shapes the days around it from its date."
    return line + (" The weeks may prepare for it, lightly." if weeks else "")


def _sentence(text: str) -> str:
    text = text.strip()
    return text if not text or text[-1] in ".?!…\"'»”" else text + "."


def _field(value: object, limit: int, name: str, last: bool, hard: list[str], sentence: bool = True) -> str:
    """A field cleaned; over its limit it is a reason to ask again, or on the last attempt cut (never mid-word)."""
    text = " ".join(_plain(value, 2000).split())
    if len(text) > limit:
        if last:
            text = cut(text, limit)
        else:
            hard.append(f"{name} is longer than {limit} characters")
    return _sentence(text) if sentence and text else text


def _words_problem(text: str, ctx: _Context) -> str | None:
    """The rules every field of the plan keeps: the house words, condition names, "nervous system", coaching."""
    word = forbidden_word(text, ctx.language)
    if word:
        return f'a word ELYND never uses, in any form ("{word}"): say it another way'
    word = condition_word(text, ctx.language)
    if word:
        return f'it names a condition ("{word}"): write about their situation and practice instead'
    if NERVOUS.search(text):
        return 'it says "nervous system"'
    for found in COACHING.finditer(text):
        if not any(w.startswith(_stem(found.group(0).lower())) for w in words_of(ctx.summary)):
            return f'coaching or productivity advice ("{found.group(0)}"): a moment of practice instead'
    if PASSING.search(text):
        return f'it presumes how their moment will go ("{PASSING.search(text).group(0)}")'
    return None


def parse_arc(text: str, ctx: _Context, last: bool = False) -> tuple[dict, list[str], list[str]]:
    """(the arc, hard problems, soft problems). ValueError when it is no arc at all."""
    data = first_json(text)
    weeks = [w for w in data.get("weeks") or [] if isinstance(w, dict)]
    if len(weeks) < 4:
        raise ValueError("four weeks")
    hard: list[str] = []
    promise = _field(data.get("promise"), LIMITS["promise"], "the promise", last, hard)
    checkin = _plain(data.get("checkin_question"), 200)
    must, fallback, _ = CHECKIN[ctx.language]
    if (must not in checkin.lower() or "?" not in checkin or len(checkin) > LIMITS["checkin_question"]
            or SYMPTOMS.search(checkin) or _words_problem(checkin, ctx)):
        if last:
            checkin = fallback
        else:
            hard.append(f'the check-in question is not in the form {CHECKIN[ctx.language][2]}')
    out = []
    for i, week in enumerate(weeks[:4], 1):
        theme = _plain(week.get("theme"), LIMITS["theme"]).rstrip(".")
        if ctx.language == "ENG":
            theme = sentence_case(theme, ctx.keep)
        line = _field(week.get("line"), LIMITS["line"], f"week {i}'s line", last, hard)
        if not theme or not line:
            raise ValueError("week fields")
        for name, value in ((f"week {i}'s theme", theme), (f"week {i}'s line", line)):
            problem = _words_problem(value, ctx)
            if problem:
                hard.append(f"{name}: {problem}")
        if PROMISE.search(line):
            hard.append(f'week {i}\'s line: "{PROMISE.search(line).group(0)}" promises a result')
        out.append({"n": i, "theme": theme, "line": line})
    if not promise:
        raise ValueError("the promise")
    for name, value in (("the promise", promise), ("the check-in question", checkin)):
        problem = _words_problem(value, ctx)
        if problem:
            hard.append(f"{name}: {problem}")
    if PROMISE.search(promise):
        hard.append(f'the promise: "{PROMISE.search(promise).group(0)}" promises a result; say what the days are for')
    soft = []
    if _uses(ctx, [promise] + [w["line"] for w in out]) < 2 <= len({_stem(w) for w in ctx.words}):
        soft.append("the promise and the week lines use fewer than two of their own words: write them in the "
                    f"situations they named ({', '.join(ctx.words)})")
    return {"promise": promise, "checkin_question": checkin, "weeks": out}, hard, soft


def _title_key(title: str) -> str:
    return " ".join(re.findall(r"\w+", title.lower()))


def _uses(ctx: _Context, texts: list[str]) -> int:
    """How many of their distinctive words (by stem) the texts use."""
    stems = {_stem(w) for w in ctx.words}
    return len({s for s in stems for text in texts for w in words_of(text) if w.startswith(s)})


def parse_week(text: str, ctx: _Context, first: int, last_n: int,
               last: bool = False) -> tuple[list[Day], list[str], list[str]]:
    """(the week's days, hard problems, soft problems). ValueError when it is no week at all."""
    data = first_json(text)
    by_n = {}
    for item in data.get("days") or []:
        if isinstance(item, dict) and isinstance(item.get("n"), int):
            by_n.setdefault(item["n"], item)
    days, hard, soft = [], [], []
    for n in range(first, last_n + 1):
        item = by_n.get(n)
        if item is None:
            raise ValueError(f"day {n} missing")
        technique = str(item.get("technique") or "").strip().lower()
        title = _plain(item.get("title"), LIMITS["title"]).rstrip(".")
        if ctx.language == "ENG":
            title = sentence_case(title, ctx.keep)
        day = Day(n=n, kind=kind(n), title=title,
                  focus=_field(item.get("focus"), LIMITS["focus"], f"day {n} focus", last, hard),
                  technique=technique if technique in TECHNIQUES else FALLBACK_TECHNIQUE[kind(n)],
                  aim=_field(item.get("aim"), LIMITS["aim"], f"day {n} aim", last, hard),
                  try_today=_field(item.get("try_today"), LIMITS["try_today"], f"day {n} try_today", last, hard))
        if not (day.title and day.focus and day.aim and day.try_today):
            raise ValueError(f"day {n} fields")
        for name in ("title", "focus", "aim", "try_today"):
            value = getattr(day, name)
            problem = _words_problem(value, ctx)
            if problem:
                hard.append(f"day {n} {name}: {problem}")
            found = DAY_RELATIVE.search(value) or (ctx.label.search(value) if ctx.label else None)
            if found:
                hard.append(f'day {n} {name}: "{found.group(0)}": a day is a meditation, not a date; never write '
                            f"their moment or a day around it into a day")
            if name in ("focus", "try_today") and UNSAFE.search(value):
                hard.append(f'day {n} {name}: "{UNSAFE.search(value).group(0)}": never an attention exercise while '
                            f"driving, never a push into what they fear")
            elif name == "try_today" and IN_CAR.search(value) and not STILL.search(value):
                hard.append(f'day {n} try_today: "{IN_CAR.search(value).group(0)}" while driving: anything in a car is '
                            f'only before setting off, when parked, or after arriving (begin it "Before you set off," '
                            f'or "When you have parked,")')
            if name == "try_today" and PREDICTS.search(value):
                hard.append(f'day {n} try_today: "{PREDICTS.search(value).group(0)}" predicts: the "then" of a plan '
                            f"invites, in the present tense, and never says what will happen")
        days.append(day)
    if not WRITING.search(ctx.summary):
        writing = [d.n for d in days if WRITING.search(d.try_today) or WRITING.search(d.focus)]
        if len(writing) > 1:
            soft.append(f"days {', '.join(map(str, writing))} each ask them to write: writing at most once a week")
    if len({_stem(w) for w in ctx.words}) >= 2:
        if _uses(ctx, [d.try_today for d in days]) < 2:
            soft.append("the small things to try use fewer than two of their own words: place them in the "
                        f"situations they named ({', '.join(ctx.words)})")
    return days, hard, soft


def _ask(llama: Llama, prompt: str, parse: Callable[[str, bool], tuple], max_tokens: int):
    """One answer, asked again with the exact reasons while it breaks a rule (ATTEMPTS calls at most). A soft
    problem is asked again once and never fails. Returns (value, usd)."""
    spent, problem, best, soft_asked = 0.0, "", None, False
    for attempt in range(ATTEMPTS):
        last = attempt == ATTEMPTS - 1
        try:
            reply = llama.chat(SYSTEM, prompt + problem, max_tokens=max_tokens, temperature=TEMPERATURES[attempt])
        except LlamaError as exc:
            raise _plan_error(str(exc), spent) from None
        spent += reply.usd
        try:
            value, hard, soft = parse(reply.text, last)
        except ValueError as exc:
            problem = f"\n\nYour last answer could not be used ({exc}). Answer again: one JSON object only."
            continue
        if not hard:
            best = value
            if not soft or soft_asked or last:
                return value, spent
            soft_asked = True
        reasons = "; ".join(hard + ([] if soft_asked and hard else soft))
        problem = (f"\n\nYour last answer could not be used ({reasons}). Answer again: one JSON object only. Keep "
                   f"what was right and write anew every field marked REWRITE. Your last answer was:\n"
                   f"{_compact(reply.text, hard)}")
    if best is not None:
        return best, spent
    raise _plan_error("no usable answer", spent)


def _plan_error(reason: str, usd: float) -> PlanError:
    error = PlanError(reason)
    error.usd = usd   # what the calls cost before it
    return error


_DAY_FIELD = re.compile(r"^day (\d+) (title|focus|aim|try_today)\b")
_WEEK_FIELD = re.compile(r"^week (\d)'s (theme|line)\b")
REWRITE = "REWRITE"


def _compact(text: str, hard: list[str] = ()) -> str:
    """The last answer as it is shown when asked again, every field a hard reason names replaced by REWRITE: shown
    whole, a model tends to send the same field back unchanged."""
    try:
        data = first_json(text)
    except ValueError:
        return text[:4000]
    for reason in hard:
        day, week = _DAY_FIELD.match(reason), _WEEK_FIELD.match(reason)
        if day:
            for item in data.get("days") or []:
                if isinstance(item, dict) and item.get("n") == int(day.group(1)):
                    item[day.group(2)] = REWRITE
        elif week and isinstance(data.get("weeks"), list) and int(week.group(1)) <= len(data["weeks"]):
            target = data["weeks"][int(week.group(1)) - 1]
            if isinstance(target, dict):
                target[week.group(2)] = REWRITE
        elif reason.startswith("the promise"):
            data["promise"] = REWRITE
        elif reason.startswith("the check-in question"):
            data["checkin_question"] = REWRITE
    return json.dumps(data, ensure_ascii=False)


def make_plan(llama: Llama, language: str, title: str, summary: str, minutes: int = 10,
              moment: Moment | None = None, *, today: dt.date | None = None) -> Plan:
    """The journey's plan. `summary` and `title` are the guide's, in the listener's words; `today` is the listener's
    date (for the moment's distance)."""
    language = language if language in LANGUAGES else "ENG"
    lang_name = LANGUAGES[language]
    today = today or dt.date.today()
    title = " ".join(str(title or "").split())
    summary = " ".join(str(summary or "").split())
    ctx = _Context(language=language, title=title, summary=summary, keep=capitalised(summary) | capitalised(title),
                   words=their_words(summary, title), label=_label_pattern(moment))
    rules = RULES.format(language=lang_name)
    words = ", ".join(ctx.words) or "(few: use the situations in their goal)"
    arc, spent = _ask(llama, ARC.format(minutes=minutes, language=lang_name, title=title, summary=summary,
                                        moment=_moment_line(moment, today, weeks=True), words=words, rules=rules,
                                        checkin=CHECKIN[language][2]),
                      lambda text, last: parse_arc(text, ctx, last), max_tokens=700)
    weeks_line = "; ".join(f'week {w["n"]}: {w["theme"]} ({w["line"]})' for w in arc["weeks"])

    def plan_week(index: int):
        first, last_n = WEEKS[index]
        week = arc["weeks"][index]
        prompt = WEEK.format(first=first, last=last_n, minutes=minutes, language=lang_name, summary=summary,
                             title=title, promise=arc["promise"], weeks=weeks_line, week=week["n"],
                             theme=week["theme"], line=week["line"], moment=_moment_line(moment, today, weeks=False),
                             words=words, days="\n".join(f"- day {n}: {kind(n)}" for n in range(first, last_n + 1)),
                             rules=rules, techniques=", ".join(TECHNIQUES))
        return _ask(llama, prompt, lambda text, last: parse_week(text, ctx, first, last_n, last), max_tokens=2600)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(plan_week, range(4)))
    spent += sum(usd for _, usd in results)
    days = [day for week_days, _ in results for day in week_days]
    spent += _retitle(llama, ctx, days, lang_name, rules)
    plan = Plan(title=title, promise=arc["promise"], checkin_question=arc["checkin_question"], weeks=arc["weeks"],
                world=None, days=days, usd=spent)
    if language != "ENG":
        plan.usd += _proofread(llama, ctx, plan, lang_name)
    return plan


PROOFREAD = """You check the {language} of short texts from a meditation app, written to one person as "{you}". Fix \
only spelling mistakes, grammar mistakes and wording a native speaker would never use. Keep the meaning, the "{you}" \
form, the length and the style; leave every text that is already correct exactly as it is.

The texts, as JSON:
{texts}

Answer with one JSON object with exactly the same keys and the checked texts, and nothing else."""
YOU_FORM = {"GER": "du", "ES": "tú", "FR": "tu"}


def _proofread(llama: Llama, ctx: _Context, plan: Plan, lang_name: str) -> float:
    """German, French and Spanish: one call checks the texts the listener reads (the promise, the check-in question,
    the weeks, the day titles) for spelling and grammar, in place. A text comes back only when it still keeps every
    rule and its length; a failed or odd answer leaves the plan as it was. Returns the call's cost."""
    texts = {"promise": plan.promise, "checkin": plan.checkin_question}
    for week in plan.weeks:
        texts[f"w{week['n']}_theme"], texts[f"w{week['n']}_line"] = week["theme"], week["line"]
    for day in plan.days:
        texts[f"d{day.n}"] = day.title
    try:
        reply = llama.chat(SYSTEM, PROOFREAD.format(language=lang_name, you=YOU_FORM.get(ctx.language, "you"),
                                                     texts=json.dumps(texts, ensure_ascii=False, indent=0)),
                           max_tokens=1600, temperature=0.2)
        checked = first_json(reply.text)
    except Exception:          # the plan stands as made; a proofreading call never fails a plan
        return 0.0
    must = CHECKIN[ctx.language][0]

    def fixed(key: str, limit: int, title: bool = False) -> str | None:
        new = _plain(checked.get(key), limit).rstrip("." if title else "")
        old = texts[key]
        if not new or new == old or not 0.6 <= len(new) / max(1, len(old)) <= 1.5 or _words_problem(new, ctx):
            return None
        if title and (DAY_RELATIVE.search(new) or (ctx.label and ctx.label.search(new))):
            return None
        if key == "checkin" and must not in new.lower():
            return None
        return new

    plan.promise = fixed("promise", LIMITS["promise"]) or plan.promise
    plan.checkin_question = fixed("checkin", LIMITS["checkin_question"]) or plan.checkin_question
    for week in plan.weeks:
        week["theme"] = fixed(f"w{week['n']}_theme", LIMITS["theme"], title=True) or week["theme"]
        week["line"] = fixed(f"w{week['n']}_line", LIMITS["line"]) or week["line"]
    for day in plan.days:
        day.title = fixed(f"d{day.n}", LIMITS["title"], title=True) or day.title
    return reply.usd


def parse_titles(text: str, ctx: _Context, need: list[Day], taken: set[str],
                 last: bool = False) -> tuple[dict[int, str], list[str], list[str]]:
    """({day: new title}, hard problems, no soft ones) for the days whose titles repeat another's."""
    data = first_json(text)
    given = data.get("titles") if isinstance(data.get("titles"), dict) else {}
    out, hard, used = {}, [], set(taken)
    for day in need:
        new = _plain(given.get(str(day.n)) or given.get(day.n), LIMITS["title"]).rstrip(".")
        if not new:
            raise ValueError(f"day {day.n} has no new title")
        if ctx.language == "ENG":
            new = sentence_case(new, ctx.keep)
        key = _title_key(new)
        if key in used:
            hard.append(f'day {day.n}: "{new}" is already a title')
        problem = _words_problem(new, ctx)
        found = DAY_RELATIVE.search(new) or (ctx.label.search(new) if ctx.label else None)
        if problem:
            hard.append(f"day {day.n}: {problem}")
        elif found:
            hard.append(f'day {day.n}: "{found.group(0)}": a day is a meditation, not a date')
        used.add(key)
        out[day.n] = new
    return out, hard, []


def _retitle(llama: Llama, ctx: _Context, days: list[Day], lang_name: str, rules: str) -> float:
    """Days whose title repeats an earlier day's get new titles from one small call (in place). Returns its cost; a
    repeated title never fails a plan, so when no new titles come the repeats stay."""
    taken, need = set(), []
    for day in days:
        key = _title_key(day.title)
        if key in taken:
            need.append(day)
        else:
            taken.add(key)
    if not need:
        return 0.0
    in_use = "; ".join(f'"{d.title}"' for d in days if d not in need)
    lines = "\n".join(f'- day {d.n} (now "{d.title}"): {d.focus} Try: {d.try_today}' for d in need)
    try:
        titles, usd = _ask(llama, RETITLE.format(language=lang_name, taken=in_use, days=lines, rules=rules),
                           lambda text, last: parse_titles(text, ctx, need, taken, last), max_tokens=400)
    except PlanError as exc:
        return getattr(exc, "usd", 0.0)
    for day in need:
        day.title = titles[day.n]
    return usd


def today_iso() -> str:
    return dt.date.today().isoformat()
