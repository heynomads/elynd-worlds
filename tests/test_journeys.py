"""Journeys: the guide, the plan and the day's block for the writer, against a fake Llama (no network, no key)."""
from __future__ import annotations

import datetime as dt
import json
import threading

import pytest

from elynd_worlds.journeys import brief, guide, planner
from elynd_worlds.journeys.guide import GuideError, Moment
from elynd_worlds.journeys.llama import Llama, LlamaError, Reply
from elynd_worlds.journeys.planner import Plan, PlanError

TODAY = dt.date(2026, 10, 31)


class FakeLlama:
    """Answers from a script: a list of texts, or a function of (system, user) for answers that depend on the ask."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []
        self.lock = threading.Lock()

    def chat(self, system, user, max_tokens=600, temperature=0.6):
        with self.lock:
            self.calls.append((system, user))
            text = self.answers(system, user) if callable(self.answers) else self.answers.pop(0)
        if isinstance(text, Exception):
            raise text
        return Reply(text, 1000, 100)


def turns(*answers):
    out = [{"role": "guide", "text": "What would you like to grow in the next 30 days?"}]
    for i, answer in enumerate(answers):
        if i:
            out.append({"role": "guide", "text": f"Question {i + 1}?"})
        out.append({"role": "you", "text": answer})
    return out


ASK = json.dumps({"say": "Confidence when you speak up. When do you notice it slipping the most?",
                  "chips": ["In meetings", "With new people", "When I'm criticised"], "done": False})
FINISH = json.dumps({"say": "Thank you, I have what I need to shape your thirty days.", "chips": [], "done": True,
                     "title": "Steady in my own voice",
                     "summary": "You want to feel confident when you speak up at work, especially in meetings with "
                                "your manager. A good day is one where you say what you think without rehearsing it.",
                     "moment": {"label": "My presentation", "date": "2026-11-12"}})


# ─── The guide ────────────────────────────────────────────────────────────

def test_the_guide_asks_one_question_with_answers_to_tap():
    llama = FakeLlama([ASK])
    turn = guide.next_turn(llama, "ENG", turns("More confident"), today=TODAY)
    assert (turn.done, turn.say[:10], turn.chips) == (False, "Confidence", ["In meetings", "With new people",
                                                                             "When I'm criticised"])
    system, user = llama.calls[0]
    assert "Write in English" in system and "Today is 2026-10-31 (Saturday)" in system
    assert "Them: More confident" in user and "Teacher: What would you like" in user


def test_words_that_need_care_go_to_care_before_anything_is_sent():
    llama = FakeLlama([])
    turn = guide.next_turn(llama, "ENG", turns("words the server flags"), care_check=lambda t: "server flags" in t)
    assert turn.care and not turn.say and llama.calls == []


def test_the_conversation_must_be_well_formed():
    llama = FakeLlama([ASK] * 9)
    for bad in ([], [{"role": "you", "text": "hi"}], turns("a") + [{"role": "guide", "text": "q"}],
                turns("a", "b", "c", "d", "e"), [{"role": "guide", "text": "q"}, {"role": "you", "text": "x" * 601}]):
        with pytest.raises(ValueError):
            guide.next_turn(llama, "ENG", bad)


def test_after_the_third_answer_it_is_told_to_finish_and_after_the_fourth_it_must():
    llama = FakeLlama([FINISH])
    turn = guide.next_turn(llama, "ENG", turns("More confident", "In meetings", "Saying what I think"), today=TODAY)
    assert "finish now" in llama.calls[0][1]
    assert (turn.done, turn.title, turn.moment) == (True, "Steady in my own voice",
                                                   Moment("My presentation", "2026-11-12"))
    assert turn.summary.startswith("You want to feel confident")
    # A fourth answer finishes even when the model forgets to say so; without a title it is asked again.
    no_title = json.dumps({"say": "Thank you.", "done": False, "chips": ["x"]})
    llama = FakeLlama([no_title, FINISH])
    turn = guide.next_turn(llama, "ENG", turns("a", "b", "c", "d"), today=TODAY)
    assert turn.done and turn.chips == [] and len(llama.calls) == 2


def test_words_it_may_never_use_are_asked_again_then_refused():
    helping = json.dumps({"say": "I can help you with that. When does it happen?", "chips": ["At work"], "done": False})
    healing = json.dumps({"say": "Lass uns an deiner Heilung arbeiten. Wann passiert es?", "chips": [], "done": False})
    llama = FakeLlama([helping, ASK])
    assert guide.next_turn(llama, "ENG", turns("More confident")).say.startswith("Confidence")
    assert "could not be used (a word it may not use: help)" in llama.calls[1][1]
    with pytest.raises(GuideError):
        guide.next_turn(FakeLlama([healing, healing]), "GER", turns("Mehr Selbstvertrauen"))


def test_no_emoji_no_exclamation_and_a_date_only_when_real():
    cheerful = json.dumps({"say": "Wonderful! 🌅 When does it show up?", "chips": ["Mornings! ✨"], "done": False})
    turn = guide.next_turn(FakeLlama([cheerful]), "ENG", turns("Sleep better"))
    assert (turn.say, turn.chips) == ("Wonderful. When does it show up?", ["Mornings"])
    far = dict(json.loads(FINISH), moment={"label": "Wedding", "date": "2027-06-01"})
    turn = guide.next_turn(FakeLlama([json.dumps(far)]), "ENG", turns("a", "b", "c"), today=TODAY)
    assert turn.moment == Moment("Wedding", None)


def test_json_is_found_inside_prose_or_a_code_fence():
    assert guide.first_json('Sure:\n```json\n{"say": "a {b}", "done": false}\n```') == {"say": "a {b}",
                                                                                         "done": False}
    with pytest.raises(ValueError):
        guide.first_json("no object here")


def test_a_failing_model_is_a_guide_error_that_names_no_words():
    with pytest.raises(GuideError) as raised:
        guide.next_turn(FakeLlama([LlamaError("ThrottlingException")]), "ENG", turns("More confident"))
    assert str(raised.value) == "ThrottlingException"


def test_the_bedrock_client_sends_one_message_and_counts_its_cost():
    class Bedrock:
        def converse(self, **kwargs):
            self.kwargs = kwargs
            return {"output": {"message": {"content": [{"text": " ready "}]}},
                    "usage": {"inputTokens": 1000, "outputTokens": 500}}
    bedrock = Bedrock()
    llama = Llama(client=bedrock)
    reply = llama.chat("system", "user", max_tokens=5)
    assert reply.text == "ready" and round(llama.spent, 6) == round(1500 * 0.72 / 1e6, 6)
    assert bedrock.kwargs["modelId"] == "us.meta.llama3-3-70b-instruct-v1:0"
    assert bedrock.kwargs["messages"] == [{"role": "user", "content": [{"text": "user"}]}]

    class Broken:
        def converse(self, **kwargs):
            raise RuntimeError("the request said: my private words")
    with pytest.raises(LlamaError) as raised:
        Llama(client=Broken()).chat("s", "u")
    assert "private" not in str(raised.value)


# ─── The plan ─────────────────────────────────────────────────────────────

ARC = {"title": "Steady in my own voice", "promise": "Thirty days to feel steady when you speak up.",
       "checkin_question": "Since yesterday, how confident did you feel?",
       "weeks": [{"theme": "Noticing", "line": "See when your voice tightens."},
                 {"theme": "Kindness", "line": "Meet the inner critic gently."},
                 {"theme": "Rehearsal", "line": "Practise the moment itself."},
                 {"theme": "Carrying it", "line": "Take it into your days."}],
       "world": {"scene": "A mountain path through pine woods toward a summit above a valley.",
                 "stages": ["Before dawn at the foot of the path", "Early morning on the lower slopes",
                            "Mid-morning above the trees", "Late morning near the ridge",
                            "Full light at the summit"]}}


def week_answer(system, user):
    if "plan days" not in user:
        return json.dumps(ARC)
    first = int(user.split("plan days ")[1].split(" ")[0])
    last = int(user.split(" to ")[1].split(" ")[0])
    return json.dumps({"days": [{"n": n, "title": f"Day {n} step", "focus": "A short practice.",
                                 "technique": "visualization" if n % 2 else "made_up_technique",
                                 "aim": "A little steadier.", "try_today": "Pause before you speak."}
                                for n in range(first, last + 1)]})


def test_a_plan_has_thirty_days_and_their_kinds_are_fixed_in_code():
    llama = FakeLlama(week_answer)
    plan = planner.make_plan(llama, "ENG", "Steady in my own voice", "You want to feel confident in meetings.",
                             minutes=10, moment=Moment("My presentation", "2026-11-12"))
    assert [d.n for d in plan.days] == list(range(1, 31))
    assert {d.n: d.kind for d in plan.days if d.kind != "practice"} == {1: "begin", 7: "look_back", 14: "look_back",
                                                                        21: "look_back", 30: "arrive"}
    assert plan.day(2).technique == "mindfulness" and plan.day(3).technique == "visualization"   # unknown: fallback
    assert plan.day(14).technique == "reflection"   # a look-back day's fallback
    assert len(llama.calls) == 5 and round(plan.usd, 6) == round(5 * 1100 * 0.72 / 1e6, 6)
    assert "My presentation on 2026-11-12" in llama.calls[1][1] or "My presentation on 2026-11-12" in llama.calls[2][1]
    assert Plan.from_json(json.loads(json.dumps(plan.to_json()))).to_json() == plan.to_json()


def test_the_world_never_carries_the_listeners_words_and_is_asked_again():
    personal = dict(ARC, world=dict(ARC["world"], scene="A meeting room where my manager listens."))
    answers = iter([json.dumps(personal), json.dumps(ARC)])
    llama = FakeLlama(lambda s, u: next(answers) if "plan days" not in u else week_answer(s, u))
    plan = planner.make_plan(llama, "ENG", "Steady", "You want to feel confident with your manager in meetings.")
    assert plan.world["scene"].startswith("A mountain path")
    assert "the world repeats the listener's words" in llama.calls[1][1]


def test_a_plan_that_never_comes_right_is_a_plan_error():
    bad = dict(ARC, promise="Thirty days to heal your fear.")
    with pytest.raises(PlanError):
        planner.make_plan(FakeLlama(lambda s, u: json.dumps(bad)), "ENG", "Steady", "You want to feel confident.")


# ─── The day's block for the writer ──────────────────────────────────────

def plan():
    return planner.make_plan(FakeLlama(week_answer), "ENG", "Steady in my own voice",
                             "You want to feel confident when you speak up in meetings.")


SUMMARY = "You want to feel confident when you speak up in meetings."


def test_day_one_opens_the_journey_and_sketches_the_thirty_days():
    text = brief.day_block(plan(), 1, TODAY, SUMMARY, [])
    assert "DAY 1 OF THE LISTENER'S 30-DAY JOURNEY" in text and f'"{SUMMARY}"' in text
    assert 'how the thirty days will go: "Noticing", then "Kindness", then "Rehearsal", then "Carrying it"' in text
    assert "Last time" not in text and "(day one)" in text


def test_a_practice_day_picks_up_from_the_last_session_and_what_they_said():
    past = [brief.Past(1, TODAY - dt.timedelta(days=2), None), brief.Past(2, TODAY - dt.timedelta(days=1), 1),
            brief.Past(3, TODAY, 3, "The meeting went a bit better", done=False)]
    text = brief.day_block(plan(), 3, TODAY, SUMMARY, past)
    assert 'Last time (day 2, yesterday): "Day 2 step"' in text
    assert ('asked "Since yesterday, how confident did you feel?" and answered "somewhat" (on day 2 they answered '
            '"not at all")') in text
    assert 'They added: "The meeting went a bit better"' in text
    assert "greeting them on day three" in text and "never as a rating" in text
    assert "welcome them back" not in text and "harder day" not in text


def test_a_gap_is_welcomed_back_and_a_hard_day_is_gentler():
    past = [brief.Past(4, TODAY - dt.timedelta(days=5), 3), brief.Past(5, TODAY, 2, done=False)]
    text = brief.day_block(plan(), 5, TODAY, SUMMARY, past)
    assert "It has been 5 days since their last meditation: welcome them back warmly" in text
    assert "Today may be a harder day: be gentler" in text


@pytest.mark.parametrize("days, says", [
    (1, 'Make room to rehearse their "My presentation"'),
    (0, 'Let this meditation steady them for their "My presentation" today.'),
    (-1, 'Ask gently how their "My presentation" went'),
    (5, 'You may mention their "My presentation" once, lightly'),
    (12, None),
])
def test_a_moment_in_their_life_shapes_the_days_around_it(days, says):
    moment = Moment("My presentation", (TODAY + dt.timedelta(days=days)).isoformat())
    text = brief.day_block(plan(), 9, TODAY, SUMMARY, [brief.Past(8, TODAY - dt.timedelta(days=1))], moment)
    if says:
        assert says in text
    else:
        assert "My presentation" not in text


def test_every_seventh_day_looks_back_on_the_week_and_opens_the_next():
    start = TODAY - dt.timedelta(days=14)
    past = [brief.Past(n, start + dt.timedelta(days=n), a, note) for n, a, note in
            [(8, 2, None), (9, 2, "Hard morning"), (11, 3, None), (12, 4, "Spoke up once"), (13, 4, None)]]
    past.append(brief.Past(14, TODAY, 4, None, done=False))
    text = brief.day_block(plan(), 14, TODAY, SUMMARY, past)
    assert 'This is the last day of week 2.' in text and '("Kindness")' in text
    assert 'how their answers moved this week (from "a little" to "quite")' in text
    assert '"Hard morning" / "Spoke up once"' in text
    assert 'Then open next week: "Rehearsal": Practise the moment itself.' in text


def test_day_thirty_closes_the_circle_with_their_first_words_and_their_note():
    start = TODAY - dt.timedelta(days=30)
    past = [brief.Past(n, start + dt.timedelta(days=n), None if n == 1 else 1 if n == 2 else 4) for n in range(1, 30)]
    past.append(brief.Past(30, TODAY, 5, done=False))
    text = brief.day_block(plan(), 30, TODAY, SUMMARY, past, letter="You were braver than you thought.")
    assert "This is the last day of the thirty." in text
    assert 'on day 2 they answered "not at all", today "very"' in text
    assert 'Read them aloud, word for word, as a quiet moment of their own: "You were braver than you thought."' in text
    assert "Close with today's one small thing" not in text
