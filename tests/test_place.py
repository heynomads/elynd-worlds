"""The place a meditation needs, described by Llama 3.3 without the listener's words (place.py)."""
from elynd_worlds import place
from elynd_worlds.art_director import Meditation

MEDITATION = Meditation(
    script=("Since the diagnosis you have felt tired and afraid, and tonight the worry about Anna keeps you awake.\n"
            "Imagine you are lying on a quiet pebble beach at dusk with Bruno, your old dog, asleep beside you.\n"
            "Breathe in slowly, and let the sound of the small waves carry the day away."),
    technique="sleep", mood="anxious", language="en", minutes=10)
GOOD = ("A quiet pebble beach at dusk, the last warm light low over a calm sea, small waves folding onto smooth grey "
        "stones. An old golden dog lies asleep on a worn blanket near a driftwood log, and the air is still and soft.")


class Chat:
    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def __call__(self, system, user):
        self.calls.append((system, user))
        return self.answers.pop(0)


def test_a_place_described_without_the_listener_passes():
    assert place.problems(GOOD, MEDITATION.script) == []


def test_a_description_about_the_listener_or_their_health_never_goes_on():
    reasons = place.problems("You lie on a quiet pebble beach at dusk, tired and afraid after the diagnosis, as small "
                             "waves fold onto smooth grey stones and your old dog sleeps.", MEDITATION.script)
    assert any("speaks of a person (you, your)" in r for r in reasons)
    assert any("health, feelings or the body (afraid, diagnosis" in r for r in reasons)


def test_a_name_from_the_meditation_or_a_run_of_its_words_never_goes_on():
    named = GOOD.replace("An old golden dog", "Bruno, an old golden dog,")
    assert any("takes a name from the meditation (bruno)" in r for r in place.problems(named, MEDITATION.script))
    copied = GOOD + " Let the sound of the small waves carry the day away."
    assert any("copies 7 or more words in a row" in r for r in place.problems(copied, MEDITATION.script))
    # A place word shared with the meditation is no copy: "a quiet pebble beach at dusk" is five words.
    assert place.problems(GOOD, MEDITATION.script) == []


def test_too_short_or_too_long_is_asked_again():
    assert any("not 15 to 90" in r for r in place.problems("A beach at dusk.", MEDITATION.script))
    assert any("not 15 to 90" in r for r in place.problems("calm sea " * 60, MEDITATION.script))


def test_llama_reads_the_meditation_and_mood_and_one_retry_carries_the_reasons():
    chat = Chat(["You rest on a quiet beach, afraid, while the waves fold onto stones near your old dog asleep.", GOOD])
    assert place.describe(chat, MEDITATION) == GOOD
    first, second = chat.calls
    assert "Mood before the session: anxious" in first[1] and MEDITATION.script in first[1]
    assert "Your last description cannot be used: it speaks of a person" in second[1]


def test_two_failed_descriptions_mean_no_personal_world():
    chat = Chat(["Your beach.", "Your beach again."])
    assert place.describe(chat, MEDITATION) is None and len(chat.calls) == 2
    assert place.for_designer(Chat(["Your beach.", "Your beach again."]), MEDITATION) is None


def test_muse_spark_reads_only_the_place_never_the_meditation_or_the_mood():
    brief = place.for_designer(Chat([GOOD]), MEDITATION)
    message = brief.as_message()
    assert brief.brief and brief.mood == "" and brief.technique == "sleep" and brief.minutes == 10
    assert "described for you without the listener's words:\n" + GOOD in message
    assert "Mood" not in message and "anxious" not in message
    for line in MEDITATION.script.splitlines():
        assert line not in message
    for word in ("diagnosis", "Anna", "Bruno", "afraid", "worry"):
        assert word not in message


def test_a_meditation_message_is_unchanged():
    assert MEDITATION.as_message().startswith("Technique: sleep\nMood before the session: anxious\nLength: 10 minutes")
    assert MEDITATION.as_message().endswith("Meditation:\n" + MEDITATION.script)
