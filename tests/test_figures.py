"""A world is the place alone unless its meditation brings an animal or a person (29 Oct 2026).

Told how to paint figures at rest in any world, Muse Image painted one nobody designed into every painting checked
(a sitting person, then a lying figure) and refused half the requests of the latest wording; told "no people and no
animals", it painted none. So the designer says what each subject is (`kind`), the painter names only the animals
and people a design brings, and the critic is told which ones belong and turns away any other.
"""
from __future__ import annotations

import json

from elynd_worlds import Meditation, Meter, Settings, art_director, critic, painter
from elynd_worlds.director import make_world

from .fakes import DOG, PAINTING, PASS, SCENE, WOMAN, FakeClient

STONE = {"name": "smooth stone", "kind": "thing", "depth": "near", "attached": "ground", "motion": "still",
         "glows": False, "where": "left"}
# The painter's line on figures before 29 Oct 2026 (98ad93c): it must never reach Muse Image again.
GENERIC = ("any animal asleep or person lying at rest", "any animal or person")
AT_REST = ("asleep or lying at rest, eyes closed or turned away, painted naturally (never standing, sitting up or "
           "walking)")
END = "no text, no letters, no logos, no symbols."


def design(*subjects: dict) -> dict:
    """SCENE with these subjects, checked as the art director's answer is (a copy: check changes what it is given)."""
    return art_director.check(json.loads(json.dumps({**SCENE, "subjects": list(subjects)})))


def strict(schema: dict, where: str = "") -> list[str]:
    """What JSON-schema mode refuses in a schema: an object that allows more or leaves a property optional, an enum
    that is not of distinct strings."""
    faults = []
    if schema.get("type") == "object":
        if schema.get("additionalProperties") is not False:
            faults.append(f"{where}: allows more")
        if sorted(schema.get("required") or []) != sorted(schema["properties"]):
            faults.append(f"{where}: not every property required")
        for name, prop in schema["properties"].items():
            faults += strict(prop, f"{where}.{name}")
    if schema.get("type") == "array":
        faults += strict(schema["items"], f"{where}[]")
    enum = schema.get("enum")
    if enum is not None and (not all(isinstance(v, str) for v in enum) or len(set(enum)) != len(enum)):
        faults.append(f"{where}: enum")
    return faults


# ─── the design ─────────────────────────────────────────────────────────

def test_a_subject_says_what_it_is_and_the_schemas_stay_strict():
    kind = art_director.SUBJECT_SCHEMA["properties"]["kind"]
    assert kind["type"] == "string" and kind["enum"] == ["thing", "animal", "person"]
    assert "kind" in art_director.SUBJECT_SCHEMA["required"] and "meditation itself brings" in kind["description"]
    for schema in (art_director.SUBJECT_SCHEMA, art_director.SCENE_SCHEMA, art_director.STAGE_SCHEMA):
        assert strict(schema) == [] and json.loads(json.dumps(schema)) == schema
    for schema in (art_director.SCENE_SCHEMA, art_director.STAGE_SCHEMA):   # a stage's subjects say it too
        assert schema["properties"]["subjects"]["items"] is art_director.SUBJECT_SCHEMA


async def test_the_designer_is_asked_for_each_subjects_kind_and_told_the_world_is_the_place_alone():
    client = FakeClient()
    await art_director.design(client, Meditation(script="Picture your old dog asleep on the warm sand."), Meter())
    request = client.requests[0]
    assert request["response_format"]["json_schema"]["schema"]["properties"]["subjects"]["items"]["required"][:2] == [
        "name", "kind"]
    rules = " ".join(request["messages"][0]["content"].split())
    assert "3. The world is the place alone, as the listener sees it (they are never in it)." in rules
    assert ("An animal or a person is in it only when the meditation itself brings one: it names or describes it "
            "(their dog, a child, swans on the lake they picture). Never add one of your own, not even small or far "
            "away.") in rules
    assert 'Each one it brings is a subject of kind "animal" or "person" (rule 5)' in rules
    # The at-rest rule as it was.
    assert "shown only ASLEEP or LYING DOWN AT REST, eyes closed or turned away" in rules
    assert "Never standing, sitting up, looking out, walking, running or jumping" in rules
    assert "An animal never looks at the viewer." in rules and "never a name" in rules
    # A person rests where people rest, never on the ground or in water; birds in flight are the app's.
    assert ("A person is small, at a distance or seen from behind, never a face up close, asleep where people rest "
            "(in a hammock, on a blanket or a beach towel), never lying on bare ground or in water.") in rules
    assert "someone lying in the grass" not in rules
    assert "Birds in flight are added by the app: the painting never has birds in the sky." in rules
    # Rule 5: a butterfly is an animal too; things are things.
    assert ('An animal or a person the meditation brings (rule 3) is one subject too ("sleeping dog", "sleeping '
            'monkey", "white swan", "white butterfly").') in rules
    assert 'kind: "animal" or "person" only for one the meditation brings (rule 3), "thing" for every other.' in rules
    assert '"ceramic teacup"), never a texture' in rules
    assert "Name no animal or person in it that is not a subject." in rules


def test_a_subject_without_a_kind_is_a_thing():
    # A design from before 29 Oct 2026, or a library world's stored one: its subjects have no kind.
    older = {**SCENE, "subjects": [{"name": "sleeping dog", "depth": "mid", "attached": "ground", "motion": "still"}]}
    assert art_director.figures_in(older) == [] and art_director.figures_in({"layers": []}) == []
    assert painter.painting_prompt(older).endswith(painter.PLACE_ALONE_STYLE)
    assert critic.figures_rule(older) == critic.PLACE_ALONE
    # Checked, a missing or unknown kind becomes "thing"; a known one is kept, whatever its case.
    checked = art_director.check(json.loads(json.dumps({**older, "subjects": [
        older["subjects"][0], {**STONE, "kind": "rock"}, {**DOG, "name": "Old Dog", "kind": " Animal "}]})))
    assert [s["kind"] for s in checked["subjects"]] == ["thing", "thing", "animal"]
    assert art_director.figures_in(checked) == ["old dog"]
    assert art_director.figures_in(design(STONE, DOG, WOMAN)) == ["sleeping dog", "sleeping woman"]
    assert art_director.figures_in(design(STONE, DOG, WOMAN), ("person",)) == ["sleeping woman"]


async def test_a_revision_keeps_every_subjects_kind():
    scene = design(STONE, DOG)
    flipped = {**SCENE, "subjects": [{**STONE, "name": "Smooth  Stone", "kind": "animal"}, {**DOG, "kind": "thing"},
                                     {**STONE, "name": "seashell", "kind": "thing"}]}
    client = FakeClient(scenes=[flipped])
    revised = await art_director.revise(client, scene, {"figures_wrong": True, "fix": "Curl the dog up asleep."},
                                        Meter())
    assert [(s["name"], s["kind"]) for s in revised["subjects"]] == [
        ("smooth stone", "thing"), ("sleeping dog", "animal"), ("seashell", "thing")]
    system = " ".join(client.requests[0]["messages"][0]["content"].split())
    assert ("Keep every subject's kind, and never add an animal or a person: only the meditation brings them, and "
            "you do not see it again.") in system


async def test_a_stage_keeps_the_first_designs_kinds_and_is_asked_with_the_same_subject_schema():
    first = design(STONE, DOG)
    answer = {**SCENE, "title": "Bright Lake", "change": "Noon light.",
              "subjects": [{**STONE, "kind": "animal"}, {**DOG, "kind": "thing"}]}
    client = FakeClient(stages=[answer])
    designed = await art_director.stage(client, first, ["dawn", "noon", "dusk", "night", "stars"], 2, Meter())
    assert [(s["name"], s["kind"]) for s in designed["subjects"]] == [
        ("smooth stone", "thing"), ("sleeping dog", "animal")]   # as in the first design, whatever the answer
    schema = client.requests[0]["response_format"]["json_schema"]["schema"]
    assert schema is art_director.STAGE_SCHEMA and "kind" in schema["properties"]["subjects"]["items"]["required"]


# ─── the painter ────────────────────────────────────────────────────────

def test_the_painter_paints_a_design_of_things_as_the_place_alone():
    prompt = painter.painting_prompt(design(STONE, {**STONE, "name": "wooden rowboat", "motion": "float"}))
    assert prompt.endswith("; no people and no animals; " + END)
    assert prompt.endswith(painter.PLACE_ALONE_STYLE)
    assert "at rest" not in prompt and not any(line in prompt for line in GENERIC)


def test_the_painter_names_the_sleeping_dog_the_meditation_brings_and_no_other_figure():
    prompt = painter.painting_prompt(design(STONE, DOG))
    assert prompt.endswith(f"; the sleeping dog {AT_REST}; no people and no other animals; {END}")
    assert not any(line in prompt for line in GENERIC) and "person" not in prompt and "face" not in prompt
    assert not prompt.endswith(painter.PLACE_ALONE_STYLE)


def test_the_painter_paints_a_person_the_meditation_brings_small_and_asleep_where_people_rest():
    person = ("small, at a distance or seen from behind, asleep where people rest (in a hammock, on a blanket or a "
              "beach towel), never on bare ground or in water, no face up close")
    prompt = painter.painting_prompt(design(STONE, WOMAN))
    assert prompt.endswith(f"; the sleeping woman {AT_REST}; the sleeping woman {person}; no other people and no "
                           f"animals; {END}")
    both = painter.painting_prompt(design(DOG, WOMAN))
    assert both.endswith(f"; the sleeping dog and the sleeping woman {AT_REST}; the sleeping woman {person}; no other "
                         f"people and no other animals; {END}")
    assert not any(line in prompt + both for line in GENERIC)


def test_the_painter_paints_the_place_alone_with_figures_off_even_when_the_meditation_brings_a_dog():
    # A journey's world (Settings.figures off): no people and no animals, whatever its design says.
    prompt = painter.painting_prompt(design(STONE, DOG), figures=False)
    assert prompt.endswith(painter.PLACE_ALONE_STYLE) and "dog" not in prompt


async def test_the_painter_is_never_told_of_figures_in_general():
    things, dog = FakeClient(), FakeClient(scenes=[{**SCENE, "subjects": [STONE, DOG]}])
    await make_world(things, Meditation(script="Picture a quiet lake at dawn."))
    await make_world(dog, Meditation(script="Picture your old dog asleep by the lake."))
    (alone,), (brought,) = ([r["prompt"] for r in c.requests if r.get("size") and "image" not in r]
                            for c in (things, dog))   # the paintings, not the plate's edit
    assert alone.endswith(painter.PLACE_ALONE_STYLE) and "the sleeping dog asleep or lying at rest" in brought
    assert not any(line in alone + brought for line in GENERIC)


# ─── the critic ─────────────────────────────────────────────────────────

async def test_the_critic_is_told_which_animals_and_people_belong_or_that_the_world_is_the_place_alone():
    only_dog = ("The only animals or people that belong in this world: the sleeping dog. Any other person or animal, "
                "at rest or not, makes figures_wrong true.")
    for scene, settings, told in (
            (design(STONE), Settings(), critic.PLACE_ALONE),                  # a design of things
            (design(STONE, DOG), Settings(), only_dog),                       # the dog the meditation brings
            (design(STONE, DOG), Settings(figures=False), critic.PLACE_ALONE),  # a journey's world
    ):
        meta, open_model = FakeClient(), FakeClient(verdicts=[PASS])
        await critic.critique(meta, PAINTING, scene, Meter(), settings)
        await critic.critique(FakeClient(), PAINTING, scene, Meter(), settings, open_client=open_model)
        for system in (meta.requests[0]["messages"][0]["content"], open_model.requests[0]["messages"][0]["content"]):
            assert system.endswith(critic.CRITIC_INSTRUCTIONS + "\n" + told)
            assert "awake and upright" in system   # on top of the at-rest rule
    both = critic.figures_rule(design(DOG, WOMAN))
    assert both == critic.ONLY_THESE.format("the sleeping dog and the sleeping woman")
