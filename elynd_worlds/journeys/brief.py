"""The day's part of a journey, for the meditation's writer: what makes the thirty sessions one story.

Built in code from the plan, the days done, the listener's answers before each session and the notes they added
(no model call). Day one opens the journey and sketches the thirty days; a practice day picks up from the last
session and from what they said since, the way a teacher who knows them would; every seventh day looks back on the
week and opens the next; day thirty closes the circle with their first words and the note they wrote to themselves.
After a gap the session welcomes them back without judging; a hard day makes it gentler; the day before a moment they
named it rehearses it, the day after it asks how it went.

The block is in English, like the rest of the writer's prompt; the listener's words are quoted as they wrote them,
and the writer writes in the session's language.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .guide import Moment
from .planner import DAYS, Plan, kind, week_of

ANSWER_WORDS = {1: "not at all", 2: "a little", 3: "somewhat", 4: "quite", 5: "very"}
NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
                "seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three twenty-four twenty-five "
                "twenty-six twenty-seven twenty-eight twenty-nine thirty").split()


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
    return {0: "earlier today", 1: "yesterday", 2: "two days ago"}.get(days, f"{days} days ago")


def _sentence(text: str) -> str:
    text = text.strip()
    return text if not text or text[-1] in ".?!…\"'" else text + "."


def _quote(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _moment_line(moment: Moment | None, today: dt.date) -> tuple[str, str]:
    """(the fact, the instruction) for a moment in their life, by its date."""
    if not moment or not moment.date:
        return "", ""
    try:
        days = (dt.date.fromisoformat(moment.date) - today).days
    except ValueError:
        return "", ""
    label = moment.label
    if days == 1:
        return (f'Their "{label}" is tomorrow.',
                f'Make room to rehearse their "{label}": let them picture it going well, step by step, steady in '
                f"the body, and leave them ready for it.")
    if days == 0:
        return f'Today is their "{label}".', f'Let this meditation steady them for their "{label}" today.'
    if days in (-1, -2):
        return (f'Their "{label}" was {_when(-days)}.',
                f'Ask gently how their "{label}" went, without judging how it went; let them keep what went well and '
                f"set down the rest.")
    if 2 <= days <= 7:
        return (f'Their "{label}" is in {days} days.',
                f'You may mention their "{label}" once, lightly, as something this practice prepares them for.')
    return "", ""


def day_block(plan: Plan, n: int, today: dt.date, summary: str, past: list[Past], moment: Moment | None = None,
              letter: str | None = None) -> str:
    """The block for day `n` (1 to 30). `past`: every day so far, today's included when they answered before it."""
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
    if last is not None:
        previous = plan.day(last.n)
        lines.append(f'Last time (day {last.n}, {_when((today - last.date).days)}): "{previous.title}": '
                     f"{_sentence(previous.focus)}")
    if now is not None and now.answer in ANSWER_WORDS:
        first = answered[0] if answered and answered[0].n != n else None
        trend = f' (on day {first.n} they answered "{ANSWER_WORDS[first.answer]}")' if first else ""
        lines.append(f'Before this meditation they were asked "{plan.checkin_question}" and answered '
                     f'"{ANSWER_WORDS[now.answer]}"{trend}.')
    if now is not None and now.note:
        lines.append(f'They added: "{_quote(now.note)}"')
    fact, moment_rule = _moment_line(moment, today)
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
        moved = ""
        if len(week_answers) >= 2:
            moved = (f', how their answers moved this week (from "{ANSWER_WORDS[week_answers[0].answer]}" to '
                     f'"{ANSWER_WORDS[week_answers[-1].answer]}")')
        notes = [p.note for p in past if p.note and plan_week_of(p.n) == week][-3:]
        heard = ""
        if notes:
            heard = ", and a few of their own words this week: " + " / ".join(f'"{_quote(t, 120)}"' for t in notes)
        upcoming = plan.weeks[week] if week < 4 else None
        rules.append(f'This is the last day of week {week}. Open by looking back on the week with them: what it was '
                     f'about ("{theme["theme"]}"){moved}{heard}. Name one small win, honestly. '
                     + (f'Then open next week: "{upcoming["theme"]}": {upcoming["line"]} ' if upcoming else "")
                     + "Then today's practice. Mention that the place they are in has moved on a little, like them.")
    elif k == "arrive":
        start = answered[0] if answered else None
        end = answered[-1] if answered else None
        changed = ""
        if start and end and start.n != end.n:
            changed = (f' Say, gently, what changed: on day {start.n} they answered "{ANSWER_WORDS[start.answer]}", '
                       f'today "{ANSWER_WORDS[end.answer]}".')
        rules.append(f"This is the last day of the thirty. Open by honouring the journey. Remind them where they "
                     f'started, close to their first words.{changed} Name what seemed to carry them. Then a practice '
                     f"that gathers the whole journey, and how to carry it on after today.")
        if letter and letter.strip():
            rules.append(f"On the first day they wrote a few words to themselves for today. Read them aloud, word "
                         f'for word, as a quiet moment of their own: "{_quote(letter, 400)}"')
    else:
        picked_up = "what they practised"
        if now is not None and (now.answer in ANSWER_WORDS or now.note):
            picked_up += " and what they told you before this meditation, said as a teacher would (\"you said it " \
                         "felt a little easier\"), never as a rating"
        rules.append(f"Open by greeting them on day {words} and, in one or two natural sentences, pick up from last "
                     f"time: {picked_up}. Then today's practice, as a step of this week's theme.")
    if last is not None and (today - last.date).days >= 3:
        rules.append(f"It has been {(today - last.date).days} days since their last meditation: welcome them back "
                     f"warmly, without any judgment, and simply go on from where they were.")
    if now is not None and now.answer in (1, 2) or (now is not None and now.note and k == "practice"
                                                    and now.answer is None):
        rules.append("Today may be a harder day: be gentler than the plan, slower, more kindness, less push.")
    if moment_rule:
        rules.append(moment_rule)
    if k != "arrive":
        rules.append("Close with today's one small thing to try, said simply.")
    rules += [
        f"Say the day in words once, near the start (day {words}).",
        "Never say the app, the plan, the program, data, rating, scale, score or AI, and never turn their answers "
        "into numbers. Quote their words sparingly, a short phrase at most.",
        "Everything else about this meditation (its length, its voice, its structure, its closing) stays as the rest "
        "of this prompt asks.",
    ]
    return "\n".join(lines) + "\nHOW TO SHAPE IT:\n" + "\n".join(f"- {rule}" for rule in rules)


def plan_week_of(n: int) -> int:
    return week_of(n) if 1 <= n <= DAYS else 0
