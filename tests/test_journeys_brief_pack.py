"""The day's block with what the server pack adds (keyword arguments, each off by default): what they heard last time,
the last small thing to try and titles, the four openings, the halfway day, the held step after hard days, and the
part of their day. The defaults reproduce the blocks recorded before them, byte for byte (tests/brief_goldens.json)."""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import pytest

from elynd_worlds.journeys import brief, guide
from elynd_worlds.journeys.brief import Heard, Past
from elynd_worlds.journeys.guide import Moment
from elynd_worlds.journeys.planner import Plan

GOLDENS = json.loads((Path(__file__).parent / "brief_goldens.json").read_text())
PLAN = Plan.from_json(GOLDENS["plan"])
TODAY = dt.date.fromisoformat(GOLDENS["today"])
SUMMARY = GOLDENS["summary"]


def past_of(rows):
    return [Past(n=r[0], date=dt.date.fromisoformat(r[1]), answer=r[2], note=r[3], done=r[4]) for r in rows]


def block(case, **extra):
    moment = Moment(*case["moment"]) if case.get("moment") else None
    return brief.day_block(PLAN, case["n"], TODAY, SUMMARY, past_of(case["past"]), moment, letter=case.get("letter"),
                           world_changed=case.get("world_changed", False), **extra)


def day(n, *rows):
    """A practice day n with yesterday's session made (and the given check-in rows, today's last)."""
    return [Past(n - 1, TODAY - dt.timedelta(days=1), 3), *rows]


# ─── The defaults: today's block, byte for byte ─────────────────────────

@pytest.mark.parametrize("case", GOLDENS["cases"], ids=lambda c: f'day{c["n"]}-{len(c["past"])}')
def test_the_defaults_reproduce_todays_block_byte_for_byte(case):
    assert block(case) == case["block"]
    assert block(case, heard=None, last_try=None, recent_titles=(), rotate_openings=False, halfway=False,
                 hold_hard_days=False, local_hour=None) == case["block"]


def test_a_day_hour_changes_nothing():
    for case in GOLDENS["cases"]:
        assert block(case, local_hour=11) == case["block"]


# ─── What they heard ─────────────────────────────────────────────────────

def test_a_played_session_is_picked_up_naturally_never_quoted_robotically():
    rows = day(9, Past(9, TODAY, 3, done=False))
    text = brief.day_block(PLAN, 9, TODAY, SUMMARY, rows,
                           heard=Heard("The breath before you speak", "Let your breath arrive before your words.",
                                       played=True))
    assert 'Last time (day 8, yesterday) they heard "The breath before you speak", which closed with: "Let your ' \
           'breath arrive before your words." Its step: "Step 8 of the voice"' in text
    untold = brief.day_block(PLAN, 9, TODAY, SUMMARY, rows, heard=Heard("The breath before you speak", played=True))
    assert 'they heard "The breath before you speak". Its step: "Step 8 of the voice"' in untold
    assert "Pick up from it naturally, as a teacher who was there would" in text and "never quoting it word for word" \
        in text
    assert "pick up from last time: what they practised" in text


@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize("n", range(2, 30))
def test_a_session_made_but_never_played_is_never_called_heard(n, rotate):
    if brief.kind(n) != "practice":
        return
    rows = day(n, Past(n, TODAY, 4, "Felt calmer", done=False))
    text = brief.day_block(PLAN, n, TODAY, SUMMARY, rows, heard=Heard("Steady at the door", "Go gently.",
                                                                       played=False), rotate_openings=rotate)
    assert "was never played: they have not heard it. Never say or suggest that they heard it" in text
    assert 'they heard "' not in text and "Go gently" not in text   # its closing line is never given to the writer
    assert [line for line in text.split("\n") if "Steady at the door" in line] == [
        line for line in text.split("\n") if "was never played" in line]
    assert "Last time (day" not in text and "begin where it pointed" not in text
    assert "what they practised" not in text and "thread of last time" not in text


def test_last_times_small_thing_and_the_last_titles_are_never_repeated():
    text = brief.day_block(PLAN, 9, TODAY, SUMMARY, day(9), last_try="Before the call, one slow breath.",
                           recent_titles=["Old one", "Breath first", "Feet on the floor", "Steady at the door"])
    assert 'Last time\'s small thing to try: "Before the call, one slow breath." Never offer it again' in text
    assert ('The last meditations were called "Breath first", "Feet on the floor", "Steady at the door": give this '
            "one a title of its own") in text and "Old one" not in text


# ─── The four openings ───────────────────────────────────────────────────

def opening(text):
    return next(line for line in text.split("\n") if line.startswith("- Open"))


def test_practice_days_open_in_turn_with_body_their_words_the_theme_and_the_thread():
    seen = {}
    for n in (2, 3, 4, 5, 6):
        rows = day(n, Past(n, TODAY, 3, "Spoke once in the meeting", done=False))
        seen[n] = opening(brief.day_block(PLAN, n, TODAY, SUMMARY, rows, rotate_openings=True))
    assert seen[2].startswith("- Open with the body and the breath, before any word about the journey")
    assert seen[3].startswith("- Open with what they told you before this meditation")
    assert seen[4].startswith('- Open with this week\'s theme, "Noticing"')
    assert seen[5].startswith("- Open by picking up the thread of last time")
    assert seen[6] == seen[2].replace("day two", "day six")
    # Their words need a check-in: without one, the body and breath instead.
    assert opening(brief.day_block(PLAN, 3, TODAY, SUMMARY, day(3), rotate_openings=True)).startswith(
        "- Open with the body and the breath")


@pytest.mark.parametrize("n", [1, 7, 14, 21, 30])
def test_look_backs_day_one_and_day_thirty_keep_their_own_openings(n):
    rows = [] if n == 1 else day(n, Past(n, TODAY, 3, done=False))
    assert brief.day_block(PLAN, n, TODAY, SUMMARY, rows, rotate_openings=True) == \
        brief.day_block(PLAN, n, TODAY, SUMMARY, rows)


@pytest.mark.parametrize("days", [1, 0, -2])
def test_the_moment_days_keep_their_own_opening(days):
    moment = Moment("My presentation", (TODAY + dt.timedelta(days=days)).isoformat())
    rows = [Past(4, TODAY - dt.timedelta(days=3)), Past(5, TODAY, 3, done=False)]
    assert brief.day_block(PLAN, 5, TODAY, SUMMARY, rows, moment, rotate_openings=True) == \
        brief.day_block(PLAN, 5, TODAY, SUMMARY, rows, moment)
    # Five days ahead is no moment day: the openings go on.
    far = Moment("My presentation", (TODAY + dt.timedelta(days=5)).isoformat())
    assert brief.day_block(PLAN, 5, TODAY, SUMMARY, rows, far, rotate_openings=True) != \
        brief.day_block(PLAN, 5, TODAY, SUMMARY, rows, far)


# ─── Halfway, the held step, the part of the day ─────────────────────────

def test_day_fifteen_marks_halfway_in_one_warm_sentence_never_a_score():
    text = brief.day_block(PLAN, 15, TODAY, SUMMARY, day(15), halfway=True)
    assert "They are halfway through the thirty: mark it in one light, warm sentence, never as a score" in text
    assert "halfway" not in brief.day_block(PLAN, 16, TODAY, SUMMARY, day(16), halfway=True)


def test_two_hard_answers_in_a_row_hold_the_step():
    hard = [Past(8, TODAY - dt.timedelta(days=1), 2), Past(9, TODAY, 1, done=False)]
    text = brief.day_block(PLAN, 9, TODAY, SUMMARY, hard, hold_hard_days=True)
    assert 'Hold the step today: instead of moving on, return to the gentler practice of last time ("Step 8 of the ' \
           'voice"' in text
    assert "never mention their answers" in text and "Today may be a harder day" in text
    # One hard answer, or a lighter one between, moves on as planned.
    for rows in ([Past(8, TODAY - dt.timedelta(days=1), 4), Past(9, TODAY, 1, done=False)],
                 [Past(7, TODAY - dt.timedelta(days=2), 1), Past(8, TODAY - dt.timedelta(days=1), 3),
                  Past(9, TODAY, 2, done=False)],
                 [Past(8, TODAY - dt.timedelta(days=1), 2), Past(9, TODAY, None, "tired", done=False)]):
        assert "Hold the step" not in brief.day_block(PLAN, 9, TODAY, SUMMARY, rows, hold_hard_days=True)


@pytest.mark.parametrize("hour, night", [(20, False), (21, True), (23, True), (0, True), (4, True), (5, False)])
def test_late_in_their_day_nothing_is_for_today(hour, night):
    text = brief.day_block(PLAN, 9, TODAY, SUMMARY, day(9), local_hour=hour)
    assert ("for tonight before sleep or for tomorrow, never as something for today" in text) is night
    assert ("One small thing to try today:" in text) is not night
    assert ("Close with today's one small thing to try" in text) is not night
    if night:
        assert 'Never call the next meditation "tomorrow"' in text


# ─── House rules in everything the block adds ────────────────────────────

def test_what_the_block_adds_keeps_the_house_rules():
    hard = [Past(8, TODAY - dt.timedelta(days=1), 2), Past(9, TODAY, 1, "rough", done=False)]
    full = brief.day_block(PLAN, 9, TODAY, SUMMARY, hard, heard=Heard("Breath first", "Go gently.", played=True),
                           last_try="One breath.", recent_titles=["A", "B"], rotate_openings=True, halfway=True,
                           hold_hard_days=True, local_hour=22)
    added = set(full.split("\n")) - set(brief.day_block(PLAN, 9, TODAY, SUMMARY, hard).split("\n"))
    for line in added:
        assert not guide.forbidden_word(line, "ENG") and not guide.condition_word(line, "ENG"), line
        assert not re.search(r"\b(?:missed|streak|\d+ days)\b", line), line
