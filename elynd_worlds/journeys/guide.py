"""The guide: a short conversation, like one with a meditation teacher, to understand what a listener wants to grow.

The app opens with its own first question ("What would you like to grow in the next 30 days?"), so the guide always
answers a conversation that has at least one answer in it. It asks at most three more questions, one at a time,
then closes with a summary in the listener's own words, a title for the journey and, when they named one, a moment
coming up in their life. Nothing of the conversation is kept: the app holds it and sends all of it every turn.

What it may never say is checked in code, not only asked for: no word of healing, therapy, help or illness (the house
rule of every text ELYND writes), no exclamation mark, no emoji.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from typing import Callable

from .llama import Llama, LlamaError

LANGUAGES = {"ENG": "English", "GER": "German", "ES": "Spanish", "FR": "French"}
MAX_ANSWERS = 4          # their answers: the app's question and at most three of the guide's
MAX_TURN_CHARS = 600
MAX_SAY = 280

SYSTEM = """You are a meditation teacher at ELYND. Someone wants to begin a 30-day meditation journey toward something \
that matters to them, and you are talking with them for a minute to understand it. Afterwards ELYND writes one guided \
meditation for each of their thirty days.

How you speak:
- Like a warm, calm teacher sitting across from them. Short and plain. Never like an assistant or a chatbot.
- Each turn is at most two short sentences: first a few of their own words reflected back (not praise, not a summary), \
then ONE open question.
- Never lists, emojis or exclamation marks. Never "great", "I understand", "thank you for sharing", "as an AI", \
"I'm here for you".
- No advice, techniques, explanations or reassurance yet. Never name a condition or diagnose. Never use the words \
heal, cure, therapy, treatment, disorder, symptom, patient, recover, cope or help, nor their equivalents in the \
listener's language.
- Speak to them as "you" (German "du", Spanish "tú", French "tu"). Write in {language}, in complete, natural sentences with normal punctuation; a question ends with a question mark.

What you want to understand, asking only for what is still missing:
1. what they want to feel or do differently a month from now, in their own words;
2. when or where it shows up most (situations, people, times of day);
3. what a good day would look like for them;
4. whether a moment is coming up in the next weeks where they want to feel this (a talk, an exam, a date, a trip), \
and when.
Ask at most three questions in all. Finish as soon as you understand enough, and always after their {last} answer.

Answer with one JSON object and nothing else.
While asking: {{"say": "...", "chips": ["...", "...", "..."], "done": false}}
"chips" are 2 or 3 short answers they might give (2 to 5 words each, in {language}, in their voice, varied and \
realistic, never "other").
When you finish: {{"say": "<a few of their last words reflected back, then one short, warm closing sentence of \
your own>", "chips": [], "done": true, \
"title": "<their goal as a title, 2 to 6 words, in {language}, no full stop>", \
"summary": "<2 or 3 sentences to them, second person, in {language}, made of their own words: what they want, when \
it shows up, what a good day looks like>", \
"moment": {{"label": "<2 to 4 words>", "date": "YYYY-MM-DD or null"}} or null}}
Today is {today} ({weekday}). Give a date only when they named one or a day you can work out (like "next Thursday"); \
never invent one.

How it sounds (an example on another goal; never copy its words):
Them: Calmer mornings
You: {{"say": "Calmer mornings. What usually happens in yours?", "chips": ["I wake up rushed", "My phone comes \
first", "Too much at once"], "done": false}}
Them: I check my phone in bed and I'm stressed before I'm even up.
You: {{"say": "Stressed before you're even up. And a good morning, what would it feel like?", "chips": ["Slow and \
quiet", "Time for coffee", "No rush"], "done": false}}
Them: Quiet. Coffee by the window before anyone else is awake.
You: {{"say": "Coffee by the window, before anyone else is awake. I can see what these thirty days are for.", \
"chips": [], "done": true, "title": "Quiet mornings", "summary": "You want your mornings to start calmer. Right now \
you reach for your phone in bed and feel stressed before you are even up. A good morning is a quiet one: coffee by the \
window before anyone else is awake.", "moment": null}}"""

# The house rule of every text ELYND writes (its meditations, its app): no word of healing, therapy, help or illness.
# Stems, matched at the start of a word.
NEVER = {
    "ENG": ("heal", "cure", "therap", "treatment", "disorder", "symptom", "patient", "recover", "cope", "coping",
            "help", "diagnos"),
    "GER": ("heil", "therap", "behandl", "störung", "symptom", "patient", "bewältig", "hilf", "helf", "genes",
            "diagnos"),
    "ES": ("sana", "sanar", "cura", "terap", "tratamiento", "trastorno", "síntoma", "paciente", "afront", "ayud",
           "recuper", "diagnós", "diagnos"),
    "FR": ("guéri", "thérap", "traitement", "trouble", "symptôme", "patient", "aide", "aider", "diagnost"),
}
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]")


class GuideError(Exception):
    """No usable turn: the app shows "The guide needs a moment" and the listener tries again. Never carries their
    words."""


@dataclass
class Moment:
    label: str
    date: str | None = None   # YYYY-MM-DD, within the next 30 days, or None


@dataclass
class Turn:
    say: str = ""
    chips: list[str] = field(default_factory=list)
    done: bool = False
    title: str | None = None
    summary: str | None = None
    moment: Moment | None = None
    care: bool = False        # their words need ELYND's care, not the guide: the app shows ELYND's care resources
    usd: float = 0.0


def clean_turns(turns: object) -> list[tuple[str, str]]:
    """The conversation as the app sent it: [("guide" | "you", text)], guide first, alternating, ending with their
    answer, at most MAX_ANSWERS answers. Raises ValueError otherwise."""
    if not isinstance(turns, list) or not turns:
        raise ValueError("turns")
    out = []
    for i, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError("turn")
        role, text = turn.get("role"), turn.get("text")
        if role != ("guide" if i % 2 == 0 else "you") or not isinstance(text, str):
            raise ValueError("order")
        text = " ".join(text.split())
        if not text or len(text) > MAX_TURN_CHARS:
            raise ValueError("text")
        out.append((role, text))
    answers = sum(1 for role, _ in out if role == "you")
    if out[-1][0] != "you" or answers > MAX_ANSWERS:
        raise ValueError("answers")
    return out


def forbidden_word(text: str, language: str) -> str | None:
    words = re.findall(r"[^\W\d_]+", text.lower())
    for stem in NEVER.get(language, ()) + (NEVER["ENG"] if language != "ENG" else ()):
        if any(w.startswith(stem) for w in words):
            return stem
    return None


def first_json(text: str) -> dict:
    """The first JSON object in a model's answer (it sometimes wraps it in a sentence or a code fence)."""
    start = text.find("{")
    while start != -1:
        depth, in_string, escape = 0, False, False
        for end in range(start, len(text)):
            ch = text[end]
            if in_string:
                escape = not escape and ch == "\\"
                if ch == '"' and not escape:
                    in_string = False
                if ch != "\\":
                    escape = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start:end + 1])
                    except ValueError:
                        break
                    if isinstance(value, dict):
                        return value
                    break
        start = text.find("{", start + 1)
    raise ValueError("no JSON object")


def _plain(text: object, limit: int) -> str:
    text = " ".join(EMOJI.sub("", str(text or "")).replace("!", ".").replace("¡", "").split())
    return text[:limit].strip()


def _moment(value: object, today: dt.date) -> Moment | None:
    if not isinstance(value, dict):
        return None
    label = _plain(value.get("label"), 40)
    if not label:
        return None
    date = None
    try:
        when = dt.date.fromisoformat(str(value.get("date")))
        if today <= when <= today + dt.timedelta(days=30):
            date = when.isoformat()
    except ValueError:
        pass
    return Moment(label=label, date=date)


def parse(text: str, language: str, today: dt.date, *, must_finish: bool) -> Turn:
    data = first_json(text)
    say = _plain(data.get("say"), MAX_SAY)
    done = data.get("done") is True or must_finish
    chips = []
    if not done:
        for chip in data.get("chips") or []:
            chip = _plain(chip, 40).rstrip(".")
            chip = chip[:1].upper() + chip[1:]
            if chip and chip.lower() not in {c.lower() for c in chips}:
                chips.append(chip)
        chips = chips[:3]
    turn = Turn(say=say, chips=chips, done=done)
    if done:
        turn.title = _plain(data.get("title"), 60).rstrip(".") or None
        turn.summary = _plain(data.get("summary"), 600) or None
        turn.moment = _moment(data.get("moment"), today)
        if not (turn.title and turn.summary and len(turn.summary) >= 20):
            raise ValueError("finish without title or summary")
    if not say:
        raise ValueError("nothing to say")
    if not done and "?" not in say and "¿" not in say:
        raise ValueError("a question ends with a question mark")
    if done and turn.summary.count(".") + turn.summary.count("?") < 2:
        raise ValueError("the summary is two or three full sentences")
    for text_part in [say, *chips, turn.title or "", turn.summary or ""]:
        word = forbidden_word(text_part, language)
        if word:
            raise ValueError(f"a word it may not use: {word}")
    return turn


def next_turn(llama: Llama, language: str, turns: object, *, today: dt.date | None = None,
              care_check: Callable[[str], bool] | None = None) -> Turn:
    """The guide's next turn. `care_check`: the server's own check, asked before any
    model call: a match answers `care` and nothing is sent anywhere."""
    language = language if language in LANGUAGES else "ENG"
    conversation = clean_turns(turns)
    if care_check is not None and any(care_check(text) for role, text in conversation if role == "you"):
        return Turn(care=True)
    today = today or dt.date.today()
    answers = sum(1 for role, _ in conversation if role == "you")
    last = {1: "first", 2: "second", 3: "third"}.get(MAX_ANSWERS - 1, "third")
    system = SYSTEM.format(language=LANGUAGES[language], last=last, today=today.isoformat(),
                           weekday=today.strftime("%A"))
    transcript = "\n".join(f"{'Teacher' if role == 'guide' else 'Them'}: {text}" for role, text in conversation)
    must_finish = answers >= MAX_ANSWERS
    ask = ("This was their last answer: finish now (done true)." if answers >= MAX_ANSWERS - 1
           else "Give your next turn.")
    user = f"The conversation so far:\n{transcript}\n\n{ask} Only the JSON object."
    spent, problem = 0.0, ""
    for attempt in range(2):
        try:
            reply = llama.chat(system, user + problem, max_tokens=500, temperature=0.6 if attempt == 0 else 0.3)
        except LlamaError as exc:
            raise GuideError(str(exc)) from None
        spent += reply.usd
        try:
            turn = parse(reply.text, language, today, must_finish=must_finish)
            turn.usd = spent
            return turn
        except ValueError as exc:
            problem = f"\n\nYour last answer could not be used ({exc}). Answer again: one JSON object only."
    raise GuideError("no usable turn")
