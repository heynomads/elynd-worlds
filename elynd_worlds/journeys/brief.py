"""The day's part of a journey, for the meditation's writer: what makes the thirty sessions one story.

Built in code from the plan, the days done, the listener's answers before each session and the notes they added
(no model call). Day one opens the journey and sketches the thirty days; a practice day picks up from the last
session and from what they said since, the way a teacher who knows them would; every seventh day looks back on the
week and opens the next; day thirty closes the circle with their first words and the note they wrote to themselves.
After a gap the session welcomes them back without judging and without counting the days; a hard day makes it
gentler, and a note is read for its tone; the day before a moment they named it rehearses it, on its day the session
follows their note (it may be behind or ahead of them), and the first session after it asks how it went. Every session
but the last closes by pointing to the next one ("Next time, …"), and the next one begins there. Their answers are
compared only on the look-back days and day thirty, and only when they rose. The plan is a sketch made on day one by a
smaller model: the writer keeps its step and technique and makes the meditation about their life, in their own words.

What the server knows besides comes as keyword arguments, each off by default (the block is then exactly as before):
what they heard last time (`heard`: the session's own title and closing line, and whether it was played: one made
but never played is never called heard), last time's small thing to try and the last titles (`last_try`,
`recent_titles`: never repeated), four openings in turn on practice days (`rotate_openings`), a light word at the
halfway day (`halfway`), the step held after two hard answers in a row (`hold_hard_days`), and the part of their day
(`local_hour`: from 21:00 to 05:00 nothing is "for today").

The block is in English, like the rest of the writer's prompt; the listener's words are quoted as they wrote them,
and the writer writes in the session's language.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Sequence

from .guide import Moment
from .planner import DAYS, Plan, kind, week_of

ANSWER_WORDS = {1: "not at all", 2: "a little", 3: "somewhat", 4: "quite", 5: "very"}
MAX_LETTER = 400   # their words for day thirty: at most 400 characters on every side, read whole
WEEK_SPAN = 10     # days: a part's sessions within this many days are "this week", else "these sessions"
KEPT_COMING = "name, simply, that they kept coming back and one thing they practised"
NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
                "seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three twenty-four twenty-five "
                "twenty-six twenty-seven twenty-eight twenty-nine thirty").split()


HALFWAY = 15       # the day that marks the middle of the thirty (with `halfway`)
NIGHT_FROM, NIGHT_UNTIL = 21, 5   # local hours: from 21:00 and before 05:00 nothing is "for today" (with `local_hour`)
OPENINGS = ("body", "their words", "theme", "thread")   # practice days in turn, by day number (with `rotate_openings`)


@dataclass
class Heard:
    """The last journey session as it was made: its own title and closing line (the writer's), and whether it was
    actually played. One made but never played is never called heard."""
    title: str
    closing: str | None = None
    played: bool = False


@dataclass
class Past:
    """One day of the journey so far. `answer` / `note`: what they said before that day's session (None if skipped).
    `done`: that day's session was made (False only for today's, before it exists)."""
    n: int
    date: dt.date
    answer: int | None = None
    note: str | None = None
    done: bool = True


def _when(days: int) -> str:
    """When the last meditation was, in words; from three days on never a number."""
    return {0: "earlier today", 1: "yesterday", 2: "two days ago"}.get(days, "some days ago")


def _sentence(text: str) -> str:
    text = text.strip()
    return text if not text or text[-1] in ".?!…\"'" else text + "."


def _quote(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


AFTER_WITHIN = 14   # days: the first session after their moment asks how it went, within this many days of it


def _moment_line(moment: Moment | None, today: dt.date, past: list[Past] = ()) -> tuple[str, str]:
    """(the fact, the instruction) for a moment in their life, by its date and by the sessions since: "after" is the
    first session on a date after it (within AFTER_WITHIN days), however many days were skipped."""
    if not moment or not moment.date:
        return "", ""
    try:
        when = dt.date.fromisoformat(moment.date)
    except ValueError:
        return "", ""
    days = (when - today).days
    label = moment.label
    if days == 1:
        # Rehearsing the steps (process simulation), not only picturing the result: picturing only success can sap the
        # effort a moment needs, while rehearsing how it unfolds goes with doing better and worrying less.
        return (f'Their "{label}" is tomorrow.',
                f'Make room to rehearse their "{label}" step by step: getting there, the first minute, one small '
                f"wobble, named in everyday words and met at once with what steadies them (their breath, their feet, "
                f"their hands), then getting through it in their own way. Rehearse the steps, not a picture of the "
                f"result (even if today's step pictures one): never assume how it will go, never "
                f'"you will". Leave them ready for it.')
    if days == 0:
        return (f'Today is their "{label}".',
                f'You do not know whether their "{label}" is behind or ahead of them today: follow their note if they '
                f"left one; otherwise let this meditation steady them for it, whenever it is.")
    if -AFTER_WITHIN <= days < 0 and not any(p.done and when < p.date < today for p in past):
        return (f'Their "{label}" was {_when(-days)}. This is the first meditation since.',
                f'Ask gently how their "{label}" went, without judging how it went; let them keep what went well and '
                f"set down the rest.")
    if 2 <= days <= 7:
        return (f'Their "{label}" is in {days} days.',
                f'You may mention their "{label}" once, lightly, as something this practice prepares them for.')
    return "", ""


def _said(text: str, limit: int) -> str:
    """Their or the writer's words in quotes, as one sentence: "…words." (never "…words.".)."""
    quoted = _quote(text, limit).strip()
    return f'"{quoted}"' + ("" if quoted[-1:] in ".?!…" else ".")


def _moment_today(moment: Moment | None, today: dt.date, past: list[Past]) -> bool:
    """Today is a moment day: the day before it, its day, or the first session after it (they keep their own
    opening)."""
    fact, _ = _moment_line(moment, today, past)
    return bool(fact) and not fact.endswith(" days.")


def _hard_in_a_row(past: list[Past], n: int) -> bool:
    """Today's answer and the one before it (by session) are both hard ("not at all" or "a little")."""
    answers = sorted((p for p in past if p.answer in ANSWER_WORDS and p.n <= n), key=lambda p: p.n)
    return len(answers) >= 2 and answers[-1].n == n and all(p.answer <= 2 for p in answers[-2:])


def day_block(plan: Plan, n: int, today: dt.date, summary: str, past: list[Past], moment: Moment | None = None,
              letter: str | None = None, world_changed: bool = False, *, heard: Heard | None = None,
              last_try: str | None = None, recent_titles: Sequence[str] = (), rotate_openings: bool = False,
              halfway: bool = False, hold_hard_days: bool = False, local_hour: int | None = None) -> str:
    """The block for day `n` (1 to 30). `past`: every day so far, today's included when they answered before it.
    `world_changed`: the journey's own place has moved on since the last look-back (False while no journey has a
    world of its own: the block then never speaks of one).

    Each of the rest is off by default, and the block is then the same, byte for byte:
    `heard`: the last session as made (Heard: its own title, closing line, played or not); `last_try`: last time's
    small thing to try; `recent_titles`: the last titles (up to three); `rotate_openings`: practice days open in turn
    with the body and breath, their words from today's check-in, the week's theme, last time's thread (look-backs,
    day one, day thirty and the moment days keep their own); `halfway`: day 15 is marked in one warm sentence;
    `hold_hard_days`: after two hard answers in a row a practice day holds the step; `local_hour`: their hour now
    (0-23), from 21:00 and before 05:00 the small thing to try is for tonight or tomorrow."""
    day = plan.day(n)
    week = week_of(n)
    theme = plan.weeks[week - 1]
    done = [p for p in past if p.done and p.n < n]
    last = done[-1] if done else None
    answered = [p for p in past if p.answer in ANSWER_WORDS]
    now = next((p for p in past if p.n == n), None)
    lines = [
        f"THIS MEDITATION IS DAY {n} OF THE LISTENER'S 30-DAY JOURNEY. Unlike a single meditation, it is the next "
        f"meeting with a teacher who knows them and remembers the days before.",
        f'Their goal, in their own words: "{_quote(summary, 600)}"',
        f'The journey: "{plan.title}". {plan.promise}',
        f'Week {week} of 4: "{theme["theme"]}": {theme["line"]}',
        f'Today, day {n}: "{day.title}". {_sentence(day.focus)} What they take from it: {_sentence(day.aim)} One '
        f"small thing to try today: {_sentence(day.try_today)}",
    ]
    night = local_hour is not None and (local_hour >= NIGHT_FROM or local_hour < NIGHT_UNTIL)
    if night:
        lines[-1] = lines[-1].replace(" One small thing to try today: ", " One small thing to try (it is late in "
                                      "their day: for tonight before sleep, or for tomorrow): ")
    unheard = heard is not None and not heard.played
    if last is not None and heard is None:
        previous = plan.day(last.n)
        lines.append(f'Last time (day {last.n}, {_when((today - last.date).days)}): "{previous.title}": '
                     f"{_sentence(previous.focus)} It closed by pointing to today's step: begin where it pointed.")
    elif last is not None and heard.played:
        previous = plan.day(last.n)
        said = (f'"{heard.title}", which closed with: {_said(heard.closing, 200)}'
                if heard.closing and heard.closing.strip() else _said(heard.title, 120))
        lines.append(f'Last time (day {last.n}, {_when((today - last.date).days)}) they heard {said} Its step: '
                     f'"{previous.title}": {_sentence(previous.focus)} Pick up from it '
                     f'naturally, as a teacher who was there would ("Last time we stayed with the breath before you '
                     f'speak"), never quoting it word for word; it pointed to today\'s step: begin where it pointed.')
    elif last is not None:
        lines.append(f'The last meditation (day {last.n}, made {_when((today - last.date).days)}), "{heard.title}", '
                     f"was never played: they have not heard it. Never say or suggest that they heard it, practised "
                     f"it or know what it said; begin from today's step as it is.")
    if last_try and last_try.strip():
        lines.append(f"Last time's small thing to try: {_said(last_try, 180)} Never offer it again, nor in other "
                     f"words.")
    titles = [t.strip() for t in recent_titles if t and t.strip()][-3:]
    if titles:
        lines.append("The last meditations were called " + ", ".join(f'"{t}"' for t in titles) + ": give this one a "
                     "title of its own, never one of these nor close to them.")
    if now is not None and now.answer in ANSWER_WORDS:
        lines.append(f'Before this meditation they were asked "{plan.checkin_question}" and answered '
                     f'"{ANSWER_WORDS[now.answer]}".')
    if now is not None and now.note:
        lines.append(f'They added: "{_quote(now.note)}"')
    fact, moment_rule = _moment_line(moment, today, past)
    if fact:
        lines.append(fact)

    rules = []
    words = NUMBER_WORDS[n]
    k = kind(n)
    if k == "begin":
        names = ", then ".join(f'"{w["theme"]}"' for w in plan.weeks)
        rules.append(f"Open by welcoming them to the first day of their journey. Say their goal back to them in one "
                     f"or two sentences, close to their own words. Then tell them simply, as a teacher would and "
                     f"without listing, how the thirty days will go: {names}. Then the first practice.")
    elif k == "look_back":
        week_answers = [p for p in answered if plan_week_of(p.n) == week]
        dates = [p.date for p in past if plan_week_of(p.n) == week] + [today]
        a_week = (today - min(dates)).days <= WEEK_SPAN
        span = "this week" if a_week else "over these sessions"
        moved = f"; {KEPT_COMING}"
        if len(week_answers) >= 2 and week_answers[-1].answer > week_answers[0].answer:
            moved = (f', how their answers rose {span} (from "{ANSWER_WORDS[week_answers[0].answer]}" to '
                     f'"{ANSWER_WORDS[week_answers[-1].answer]}")')
        notes = [p.note for p in past if p.note and plan_week_of(p.n) == week][-3:]
        heard = ""
        if notes:
            heard = f", and a few of their own words {span}: " + " / ".join(f'"{_quote(t, 120)}"' for t in notes)
        upcoming = plan.weeks[week] if week < 4 else None
        opening = (f"This is the last day of week {week}. Open by looking back on the week with them" if a_week else
                   f"This is the last meditation of part {week} of 4, whose sessions took more than a week: never "
                   f'call it "this week". Open by looking back on these sessions with them')
        rules.append(f'{opening}: what it was about ("{theme["theme"]}"){moved}{heard}. Name one small win, '
                     f"honestly. "
                     + (f'Then open the next part: "{upcoming["theme"]}": {upcoming["line"]} ' if upcoming else "")
                     + "Then today's practice."
                     + (" Mention that the place they are in has moved on a little, like them." if world_changed
                        else ""))
    elif k == "arrive":
        start = answered[0] if answered else None
        end = answered[-1] if answered else None
        changed = f" Then {KEPT_COMING} on the way."
        if start and end and start.n != end.n and end.answer > start.answer:
            changed = (f' Say, gently, what changed: on day {start.n} they answered "{ANSWER_WORDS[start.answer]}", '
                       f'today "{ANSWER_WORDS[end.answer]}".')
        rules.append(f"This is the last day of the thirty. Open by honouring the journey. Remind them where they "
                     f'started, close to their first words.{changed} Name what seemed to carry them. Then a practice '
                     f"that gathers the whole journey, and how to carry it on after today.")
        if letter and letter.strip():
            rules.append(f"On the first day they wrote a few words to themselves for today. Read them aloud, word "
                         f'for word and whole, as a quiet moment of their own: "{" ".join(letter.split())}"')
    else:
        told = now is not None and (now.answer in ANSWER_WORDS or bool(now.note))
        opening = None
        if rotate_openings and not _moment_today(moment, today, past):
            opening = OPENINGS[(n - 2) % len(OPENINGS)]
            if opening == "their words" and not told:
                opening = "body"
            if opening == "thread" and (last is None or unheard):
                opening = "theme"
        thread = "what they practised last time" if last is not None and not unheard else ""
        if opening == "body":
            rules.append(f"Open with the body and the breath, before any word about the journey; greet them on day "
                         f"{words} a little later, within the first minute"
                         + (f", and say in a sentence {thread}" if thread else "")
                         + ". Then today's practice, as a step of this week's theme.")
        elif opening == "their words":
            rules.append(f"Open with what they told you before this meditation, said back as a teacher would (\"you "
                         f"said it felt a little easier\"), never as a rating; greet them on day {words} within the "
                         f"first minute. Then today's practice, as a step of this week's theme.")
        elif opening == "theme":
            rules.append(f'Open with this week\'s theme, "{theme["theme"]}", and where today\'s step sits in it; greet '
                         f"them on day {words} within the first minute. Then today's practice.")
        elif opening == "thread":
            rules.append(f"Open by picking up the thread of last time in one or two natural sentences ({thread}), "
                         f"then greet them on day {words}. Then today's practice, as a step of this week's theme.")
        elif unheard:
            rules.append(f"Open by greeting them on day {words} and begin today's practice, as a step of this week's "
                         f"theme" + (", with what they told you before this meditation, said as a teacher would "
                                     "(\"you said it felt a little easier\"), never as a rating" if told else "")
                         + ".")
        else:
            picked_up = "what they practised"
            if told:
                picked_up += " and what they told you before this meditation, said as a teacher would (\"you said " \
                             "it felt a little easier\"), never as a rating"
            rules.append(f"Open by greeting them on day {words} and, in one or two natural sentences, pick up from "
                         f"last time: {picked_up}. Then today's practice, as a step of this week's theme.")
        if halfway and n == HALFWAY:
            rules.append("They are halfway through the thirty: mark it in one light, warm sentence, never as a score, "
                         "a count or a percentage.")
        if hold_hard_days and last is not None and _hard_in_a_row(past, n):
            held = plan.day(last.n)
            rules.append(f'Hold the step today: instead of moving on, return to the gentler practice of last time '
                         f'("{held.title}": {_sentence(held.focus)}), slower and kinder, and leave today\'s new step '
                         f"for another day. Never name what is hard in clinical words and never mention their "
                         f"answers.")
    if last is not None and (today - last.date).days >= 3:
        rules.append("Some days have passed since their last meditation: welcome them back warmly, without any "
                     "judgment, and never say how many days. Simply go on from where they were.")
    if now is not None and now.answer in (1, 2):
        rules.append("Today may be a harder day: be gentler than the plan, slower, more kindness, less push.")
    elif now is not None and now.note:
        rules.append("Read the tone of what they added: if it sounds heavy, be gentler than the plan, slower, more "
                     "kindness, less push; if it sounds light, meet it.")
    if moment_rule:
        rules.append(moment_rule)
    if k != "arrive":
        upcoming_day = plan.day(n + 1)
        offer = ("the one small thing to try, offered for tonight before sleep or for tomorrow, never as something "
                 "for today" if night else "today's one small thing to try")
        never = ('Never call the next meditation "tomorrow": it may be days away.' if night else
                 'Never say "tomorrow": the next meditation may be days away.')
        rules.append(f'Close with {offer}, said simply, then one sentence that points to the '
                     f'next meditation, beginning "Next time, …": its step is "{upcoming_day.title}": '
                     f'{_sentence(upcoming_day.focus)} {never}')
    rules += [
        "The plan above is a sketch made on day one. Keep its step and technique; make the meditation about their "
        "life, using the situations in their own words. Meditation, never advice.",
        f"Say the day in words once, near the start (day {words}).",
        "Never say the app, the plan, the program, data, rating, scale, score or AI, and never turn their answers "
        "into numbers. Quote their words sparingly, a short phrase at most.",
        "Everything else about this meditation (its length, its voice, its structure, its closing) stays as the rest "
        "of this prompt asks.",
    ]
    return "\n".join(lines) + "\nHOW TO SHAPE IT:\n" + "\n".join(f"- {rule}" for rule in rules)


def plan_week_of(n: int) -> int:
    return week_of(n) if 1 <= n <= DAYS else 0
