"""A journey's world grows (journeys/stages.py): five living worlds of one place, against the fake Meta Model API."""
from __future__ import annotations

import asyncio
import csv
import io
import json

import pytest
from PIL import Image

from elynd_worlds import Meter, Settings, art_director, cli, critic, life, painter
from elynd_worlds.director import NoWorld
from elynd_worlds.journeys import stages
from elynd_worlds.meta import NO_IMAGE_TOOLS, is_marked, png_size

from .fakes import FAIL, PAINTING, PASS, SCENE, STAGE_PAINTING, FakeClient, png


@pytest.fixture(autouse=True)
def _every_model_offered():
    from elynd_worlds import meta
    meta._missing.clear()
    yield
    meta._missing.clear()


WORLD = {"scene": "A winding mountain path through pine woods toward a summit above a valley.",
         "stages": ["Before dawn at the foot of the path", "First light on the lower slopes",
                    "Mid-morning above the trees", "Late morning near the ridge", "Full light at the summit"]}


# A critic that judges a later stage also says whether it is the same place.
SAME = {**PASS, "same_place": True}
ANOTHER_PLACE = {**PASS, "same_place": False, "reason": "Another valley, seen from higher up.",
                 "fix": "Keep the first painting's view of the valley."}


def pixels(picture: bytes) -> bytes:
    return Image.open(io.BytesIO(picture)).convert("RGB").tobytes()


def judged_title(request: dict) -> str:
    """The title of the design a critique request carries."""
    text = request["messages"][1]["content"][0]["text"]
    return json.loads(text.split("The design:\n", 1)[1])["title"]


def failing(*titles: str):
    """A critic that finds another place in the stages with these titles and passes everything else."""
    return lambda request: ANOTHER_PLACE if judged_title(request) in titles else SAME


def repaints(client: FakeClient) -> list[dict]:
    return [r for r in client.requests if str(r.get("prompt", "")).startswith("The same place")]


# ─── the world the plan describes ───────────────────────────────────────

def test_the_journeys_world_is_read_from_a_plan_or_on_its_own_and_nothing_else_is():
    plan = {"title": "Steady in my own voice", "world": {**WORLD, "scene": "  " + WORLD["scene"] + "\n"}}
    assert stages.journey_world(plan) == WORLD == stages.journey_world(WORLD)
    for broken in ({"scene": WORLD["scene"], "stages": WORLD["stages"][:4]}, {"scene": " ", "stages": WORLD["stages"]},
                   {"scene": WORLD["scene"], "stages": [*WORLD["stages"][:4], ""]}, ["a list"], {"world": "text"}):
        with pytest.raises(ValueError):
            stages.journey_world(broken)


def test_stage_one_is_designed_from_the_place_and_its_first_stage_only():
    text = stages.first_text(WORLD)
    assert text.startswith(WORLD["scene"]) and "Now: before dawn at the foot of the path." in text
    assert text.endswith("Only the place: no people and no animals.")
    assert not any(later.lower() in text.lower() for later in WORLD["stages"][1:])


# ─── five stages of one place ───────────────────────────────────────────

async def test_a_journeys_world_grows_in_five_stages_of_one_place():
    client = FakeClient(verdicts=[SAME])
    made = await stages.make_stages(client, WORLD)
    assert client.requests[0]["messages"][1]["content"].endswith(stages.first_text(WORLD))  # stage 1's "meditation"
    assert [s.shows for s in made.stages] == [1, 2, 3, 4, 5] and all(s.made for s in made.stages)
    first, *later = made.worlds
    assert pixels(first.painting) == pixels(PAINTING)
    assert all(pixels(w.painting) == pixels(STAGE_PAINTING) for w in later)
    # One design and one painting; each later stage is the art director's re-lit design and one repaint.
    assert client.calls.count("scene") == 1 and client.calls.count("paint") == 1
    assert client.calls.count("stage") == 4 and client.calls.count("repaint") == 4
    assert [w.scene["title"] for w in made.worlds] == ["Still Morning Lake", "Stage 2", "Stage 3", "Stage 4", "Stage 5"]
    # The same place, the same subjects (they move the same way in every stage), only another light.
    for world in later:
        assert world.scene["place"] == first.scene["place"] and world.scene["subjects"] == first.scene["subjects"]
        assert world.scene["time_of_day"] == "midday" and world.scene["change"].startswith("The lake in bright noon")
        assert world.moving and [layer.name for layer in world.layers] == [layer.name for layer in first.layers]
        assert world.plate_source == "muse_image"
        names = [step.name for step in world.trace]
        assert names[:2] == ["stage design", "edit 1"] and {"critique 1", "cut 1", "plate"} <= set(names)
    # Every stage's cost counts, apart and together.
    assert made.meter.usd == pytest.approx(sum(s.meter.usd for s in made.stages)) and made.meter.usd > 0
    assert made.meter.images == 1 + 4 + 5  # the painting, four repaints, and each stage's plate edit
    assert made.meter.calls["stage"] == 4 and made.meter.calls["stage_edit"] == 4


async def test_each_later_stage_is_the_first_painting_repainted_from_the_same_viewpoint():
    client = FakeClient(verdicts=[SAME])
    made = await stages.make_stages(client, WORLD, user="listener-7f3a")
    requests = repaints(client)
    assert len(requests) == 4
    for request in requests:
        assert request["model"] == "muse-image-1.0" and request["image"][1] == made.worlds[0].painting
        assert request["extra_body"] == {**NO_IMAGE_TOOLS, "output_format": "png"}
        assert request["user"] == "listener-7f3a"
        prompt = request["prompt"]
        assert "exactly the same viewpoint" in prompt and "the smooth stone and the pine branch" in prompt
        assert "Time of day: midday. Light: bright noon light from above." in prompt and "no text" in prompt
    told = sorted(r["prompt"].split("now: ")[1].split(".")[0] for r in requests)
    assert told == sorted(painter.clause(s) for s in WORLD["stages"][1:])
    # Shown to a listener: marked as made by AI, at the first painting's size so the stages lie on one another.
    assert all(is_marked(w.painting) for w in made.worlds[1:])


async def test_a_repaint_of_another_size_is_brought_to_the_first_paintings_size():
    client = FakeClient(verdicts=[SAME], stage_painting=png(240, 420, colour=(150, 160, 120)))
    made = await stages.make_stages(client, WORLD)
    assert {png_size(w.painting) for w in made.worlds} == {png_size(PAINTING)}
    assert all(is_marked(w.painting) for w in made.worlds[1:])


async def test_the_art_director_re_lights_its_own_design_and_keeps_the_place_and_the_subjects():
    moved = {**SCENE, "title": "Bright Lake", "place": "Another valley with a city.", "nature": ["water"],
             "subjects": [{**SCENE["subjects"][0], "where": "right", "glows": True},
                          {"name": "golden statue", "depth": "near", "attached": "ground", "motion": "still",
                           "glows": False, "where": "centre"}],
             "change": "  Noon light;\n the stone glows.  "}
    client, meter = FakeClient(), Meter()
    first = art_director.check(json.loads(json.dumps({**SCENE, "nature": ["water", "sky"]})))
    designed = await art_director.stage(FakeClient(stages=[moved]), first, WORLD["stages"], 3, meter)
    assert designed["place"] == first["place"] and designed["nature"] == ["water", "sky"]
    assert [s["name"] for s in designed["subjects"]] == ["smooth stone", "pine branch"]
    assert designed["subjects"][0]["where"] == "left"            # where it was painted
    assert [s["glows"] for s in designed["subjects"]] == [True, False]  # only its light may follow the stage
    assert designed["title"] == "Bright Lake" and designed["change"] == "Noon light; the stone glows."
    # Asked as the art director, with every stage listed and the one to design named.
    await art_director.stage(client, first, WORLD["stages"], 3, meter)
    request = client.requests[0]
    assert request["model"] == "muse-spark-1.3" and request["reasoning_effort"] == "low"
    assert request["response_format"]["json_schema"]["schema"] is art_director.STAGE_SCHEMA
    assert request["extra_body"]["prompt_cache_key"] == "elynd-worlds-design"
    system, user = (m["content"] for m in request["messages"])
    assert system.startswith(art_director.DIRECTOR_INSTRUCTIONS) and "the viewpoint\nnever moves" in system
    assert "5. Full light at the summit" in user and user.endswith("Design stage 3: Mid-morning above the trees")
    assert meter.calls["stage"] == 2


# ─── the critic, next to the first stage ────────────────────────────────

async def test_the_critic_sees_each_stage_next_to_the_first_and_turns_away_another_place():
    client = FakeClient(verdicts=[SAME])
    scene = art_director.check(json.loads(json.dumps(SCENE)))
    verdict = await critic.critique(client, STAGE_PAINTING, scene, Meter(), reference=PAINTING)
    assert critic.passes(verdict) and verdict["same_place"] is True
    request = client.requests[0]
    content = request["messages"][1]["content"]
    assert [part["type"] for part in content] == ["text", "text", "image_url", "text", "image_url"]
    assert "first painted" in content[1]["text"] and "you judge" in content[3]["text"]
    assert request["response_format"]["json_schema"]["schema"] is critic.STAGE_VERDICT_SCHEMA
    assert "same_place" in request["messages"][0]["content"]
    # Another place never passes, however good; a verdict that does not say is no verdict.
    assert not critic.passes({**PASS, "same_place": False}) and critic.passes(PASS)
    with pytest.raises(ValueError, match="same_place"):
        await critic.critique(FakeClient(verdicts=[PASS]), STAGE_PAINTING, scene, Meter(), reference=PAINTING)
    gone = await critic.critique(FakeClient(verdicts=[{**PASS, "same_place": False, "reason": ""}]),
                                 STAGE_PAINTING, scene, Meter(), reference=PAINTING)
    assert gone["reason"] == "The painting is not the same place as the first stage."
    # The open model is told the shape with same_place first, and gets no listener id.
    open_model = FakeClient(verdicts=[SAME])
    await critic.critique(FakeClient(), STAGE_PAINTING, scene, Meter(), open_client=open_model, reference=PAINTING)
    assert open_model.requests[0]["messages"][0]["content"].startswith(critic.OPEN_STAGE_SHAPE)
    assert '"same_place": true or false' in critic.OPEN_STAGE_SHAPE and "extra_body" not in open_model.requests[0]


async def test_every_later_stage_is_judged_with_the_first_painting_beside_it():
    client = FakeClient(verdicts=[SAME])
    made = await stages.make_stages(client, WORLD)
    judged = [r for r in client.requests if "messages" in r and r["response_format"]["json_schema"]["name"] == "verdict"]
    first = critic.data_url(critic.preview_jpeg(made.worlds[0].painting, Settings().critic_image_width), "image/jpeg")
    with_reference = [r for r in judged if len(r["messages"][1]["content"]) == 5]
    assert len(judged) == 5 and len(with_reference) == 4
    assert all(r["messages"][1]["content"][2]["image_url"]["url"] == first for r in with_reference)


# ─── a stage that cannot be made shows the one before ───────────────────

async def test_a_stage_turned_away_twice_shows_the_stage_before_and_the_second_edit_hears_the_critic():
    client = FakeClient(verdicts=[failing("Stage 3")])
    made = await stages.make_stages(client, WORLD)
    assert [s.shows for s in made.stages] == [1, 2, 2, 4, 5]
    assert made.worlds[2] is made.worlds[1] and not made.stages[2].made
    assert made.stages[2].error == "NoWorld: no edit passed the critic: Another valley, seen from higher up."
    third = [r["prompt"] for r in repaints(client) if "now: mid-morning above the trees" in r["prompt"]]
    assert len(third) == 2 and "Mind this" not in third[0]
    assert "Mind this: Keep the first painting's view of the valley." in third[1]
    names = [step.name for step in made.stages[2].trace]
    assert {"edit 1", "critique 1", "edit 2", "critique 2"} <= set(names) and "plate" not in names
    # What the turned-away stage cost is counted all the same.
    assert made.stages[2].meter.calls["stage_edit"] == 2 and made.meter.calls["stage_edit"] == 5


async def test_failures_one_after_another_all_show_the_last_good_stage():
    made = await stages.make_stages(FakeClient(verdicts=[failing("Stage 2", "Stage 3")]), WORLD)
    assert [s.shows for s in made.stages] == [1, 1, 1, 4, 5]
    assert made.worlds[1] is made.worlds[0] and made.worlds[2] is made.worlds[0]


@pytest.mark.parametrize("client", [
    FakeClient(verdicts=[SAME], fail_repaint=True),                                   # Muse Image's edit fails
    FakeClient(verdicts=[SAME], stages=[{**SCENE, "prompt": "too short"}]),           # a design out of shape
    FakeClient(verdicts=[failing("Stage 2", "Stage 3", "Stage 4", "Stage 5")]),       # never the same place
    FakeClient(verdicts=[lambda r: PASS if "same_place" not in r["messages"][0]["content"] else "not json"]),
])
async def test_a_journey_never_loses_its_world_for_a_later_stage(client):
    made = await stages.make_stages(client, WORLD)
    assert [s.shows for s in made.stages] == [1, 1, 1, 1, 1]
    assert all(world is made.worlds[0] for world in made.worlds) and all(s.error for s in made.stages[1:])


async def test_a_stage_that_takes_too_long_shows_the_stage_before():
    client = FakeClient(verdicts=[SAME])
    edit = client.images.edit

    async def slow_repaint(**kwargs):
        if kwargs["prompt"].startswith("The same place"):
            await asyncio.sleep(5)
        return await edit(**kwargs)

    client.images.edit = slow_repaint
    made = await stages.make_stages(client, WORLD, settings=Settings(stage_timeout=0.2))
    assert [s.shows for s in made.stages] == [1, 1, 1, 1, 1]
    assert all(s.error.startswith("TimeoutError") for s in made.stages[1:])


async def test_stage_one_is_made_once_more_from_a_fresh_design_when_no_world_comes_of_it():
    # 29 Oct: Muse Image's filter refused one design's painting of a path before dawn.
    client = FakeClient(verdicts=[SAME], refused_paintings=1)
    made = await stages.make_stages(client, WORLD)
    assert client.calls.count("scene") == 2 and client.calls.count("paint") == 2
    assert all(s.made for s in made.stages) and made.stages[0].meter.calls["design"] == 2


async def test_without_stage_one_there_is_no_place_to_grow_and_the_spend_is_still_counted():
    meter = Meter()
    with pytest.raises(NoWorld):
        await stages.make_stages(FakeClient(verdicts=[FAIL]), WORLD, meter=meter)
    assert meter.calls["paint"] == 4 and meter.usd > 0   # twice two paintings, all turned away


# ─── saved like any world ───────────────────────────────────────────────

async def test_the_stages_are_saved_like_any_world(tmp_path):
    made = await stages.make_stages(FakeClient(verdicts=[failing("Stage 4")]), WORLD)
    folder = made.save(tmp_path / "journey")
    for n in range(1, 6):
        manifest = json.loads((folder / f"stage{n}" / "world.json").read_text())
        assert manifest["schema"] == "elynd.world/1" and (folder / f"stage{n}" / "painting.png").exists()
        assert manifest["stage"] == {"n": n, "of": 5, "shows": 3 if n == 4 else n, "text": WORLD["stages"][n - 1]}
        assert {"plate.png", "layer_0.png", "scene.json", "verdict.json"} <= {p.name for p in (folder / f"stage{n}").iterdir()}
    assert (folder / "stage4" / "painting.png").read_bytes() == (folder / "stage3" / "painting.png").read_bytes()
    summary = json.loads((folder / "stages.json").read_text())
    assert summary["schema"] == "elynd.stages/1" and summary["scene"] == WORLD["scene"]
    assert [s["shows"] for s in summary["stages"]] == [1, 2, 3, 3, 5]
    assert summary["stages"][3]["title"] is None and summary["stages"][3]["error"].startswith("NoWorld")
    assert summary["stages"][1]["title"] == "Stage 2" and summary["stages"][1]["layers"] == ["smooth stone", "pine branch"]
    assert summary["cost_usd"] == made.meter.usd == pytest.approx(sum(s["usd"] for s in summary["stages"]), abs=1e-4)
    # One place, one life: every stage's birds come when the first stage's do (its titles differ).
    seeds = {json.loads((folder / f"stage{n}" / "world.json").read_text())["living"]["seed"] for n in range(1, 6)}
    assert seeds == {life.seed(made.worlds[0].scene)}


async def test_every_request_of_the_five_keeps_no_copy_and_names_the_listener_by_a_pseudonym():
    meta, open_model = FakeClient(), FakeClient(verdicts=[SAME])
    await stages.make_stages(meta, WORLD, user="listener-7f3a", open_client=open_model)
    chats = [r for r in meta.requests if "messages" in r]
    assert chats and all(r["extra_body"]["store"] is False and r["extra_body"]["safety_identifier"] == "listener-7f3a"
                         for r in chats)
    assert all(r["user"] == "listener-7f3a" for r in meta.requests if "prompt" in r and "messages" not in r)
    assert all(r["store"] is False and r["safety_identifier"] == "listener-7f3a" for r in meta.sam_requests)
    sent = json.dumps(open_model.requests)
    assert len(open_model.requests) == 5 and "listener-7f3a" not in sent and "extra_body" not in sent


# ─── command line ───────────────────────────────────────────────────────

def test_stages_from_the_command_line_with_each_bill_in_its_ledger(tmp_path, monkeypatch):
    meta, open_model = FakeClient(), FakeClient(verdicts=[SAME])
    monkeypatch.setattr(cli, "connect", lambda key: meta)
    monkeypatch.setattr(cli, "connect_open", lambda key: open_model)
    key = tmp_path / "key"
    key.write_text("test-key")
    ledgers = tmp_path / "meta.csv", tmp_path / "open.csv"
    common = ["--key-file", str(key), "--ledger", str(ledgers[0]), "--open-ledger", str(ledgers[1]),
              "--open-key-file", str(key)]
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"title": "Steady", "promise": "...", "world": WORLD, "days": []}))
    assert cli.main(["stages", str(plan), "--out", str(tmp_path / "j"), *common]) == 0
    assert all((tmp_path / "j" / f"stage{n}" / "world.json").exists() for n in range(1, 6))
    summary = json.loads((tmp_path / "j" / "stages.json").read_text())
    meta_row, open_row = (list(csv.DictReader(ledger.open()))[0] for ledger in ledgers)
    assert meta_row["what"] == "stages" and float(meta_row["usd"]) == pytest.approx(summary["meta_usd"], abs=1e-5)
    assert float(open_row["usd"]) == pytest.approx(summary["open_usd"], abs=1e-5) and summary["open_usd"] > 0
    # The world alone works too; anything else is refused before anything is spent.
    world = tmp_path / "world.json"
    world.write_text(json.dumps(WORLD))
    assert cli.main(["stages", str(world), "--out", str(tmp_path / "k"), *common]) == 0
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"scene": "A lake."}))
    calls = len(meta.calls)
    assert cli.main(["stages", str(broken), *common]) == 1 and len(meta.calls) == calls
    # At the budget: refused.
    assert cli.main(["stages", str(world), *common, "--budget", "0.1"]) == 2 and len(meta.calls) == calls


def test_a_meter_counts_another_meters_use():
    a, b = Meter(), Meter()
    a.spark({"prompt_tokens": 1000, "completion_tokens": 100}, "design")
    b.spark({"prompt_tokens": 2000, "completion_tokens": 50}, "stage")
    b.image("stage_edit", 2)
    b.sam("segment", 3)
    b.spark_open({"prompt_tokens": 500, "completion_tokens": 10}, "critique")
    total = Meter().add(a).add(b)
    assert total.usd == pytest.approx(a.usd + b.usd)
    assert total.calls == {"design": 1, "stage": 1, "stage_edit": 1, "segment": 1, "critique@open": 1}
    assert (total.images, total.sam_images, total.open_output_tokens) == (2, 3, 10)


async def test_a_world_that_is_not_a_stage_is_never_judged_on_the_same_place():
    scene = art_director.check(json.loads(json.dumps(SCENE)))
    open_model = FakeClient(verdicts=[{**PASS, "same_place": False}])  # a stray key from a model that adds one
    verdict = await critic.critique(FakeClient(), PAINTING, scene, Meter(), open_client=open_model)
    assert critic.passes(verdict) and "same_place" not in verdict
    assert len(open_model.requests[0]["messages"][1]["content"]) == 2  # the design and the one painting


# ─── the place alone ────────────────────────────────────────────────────

def test_a_world_of_the_place_alone_tells_the_painter_no_people_and_no_animals():
    scene = art_director.check(json.loads(json.dumps(SCENE)))
    assert painter.FIGURES in painter.STYLE and painter.painting_prompt(scene).endswith(painter.STYLE)  # as before
    alone = painter.painting_prompt(scene, figures=False)
    assert alone.endswith(painter.PLACE_ALONE_STYLE) and "no people and no animals" in alone
    assert "lying at rest" not in alone and "no text, no letters" in alone


async def test_the_critic_turns_away_any_figure_in_a_world_of_the_place_alone():
    scene = art_director.check(json.loads(json.dumps(SCENE)))
    client = FakeClient()
    await critic.critique(client, PAINTING, scene, Meter())
    await critic.critique(client, PAINTING, scene, Meter(), Settings(figures=False))
    before, alone = (r["messages"][0]["content"] for r in client.requests)
    assert before == critic.CRITIC_INSTRUCTIONS and alone == critic.CRITIC_INSTRUCTIONS + "\n" + critic.PLACE_ALONE


async def test_every_stage_is_the_place_alone():
    client = FakeClient(verdicts=[SAME])
    await stages.make_stages(client, WORLD)
    paintings = [r["prompt"] for r in client.requests if "prompt" in r and "image" not in r and "messages" not in r]
    assert len(paintings) == 1 and paintings[0].endswith(painter.PLACE_ALONE_STYLE)
    judged = [r["messages"][0]["content"] for r in client.requests
              if "messages" in r and r["response_format"]["json_schema"]["name"] == "verdict"]
    assert len(judged) == 5 and all(critic.PLACE_ALONE in system for system in judged)
    assert all("people, animals or text" in r["prompt"] for r in repaints(client))


async def test_the_later_stages_can_be_made_a_few_at_a_time():
    for parallel, designs_first in ((4, True), (1, False)):
        client = FakeClient(verdicts=[SAME], chat_delay=0.01)  # each answer takes a moment, as Meta's do
        made = await stages.make_stages(client, WORLD, parallel=parallel)
        order = [c for c in client.calls if c in ("stage", "repaint")]
        assert (order[:4] == ["stage"] * 4) is designs_first          # all four begin at once, or one by one
        assert parallel == 4 or order == ["stage", "repaint"] * 4
        assert all(s.made for s in made.stages)
