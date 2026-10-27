"""The engine, step by step, against the fake Meta Model API (tests/fakes.py)."""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import time

import httpx
import numpy as np
import pytest
from PIL import Image

from elynd_worlds import (Meditation, Meter, Settings, art_director, cli, critic, cutter, library, painter, plate,
                          timeline)
from elynd_worlds.card import make_card
from elynd_worlds.director import NoWorld, make_world, make_world_or_none
from elynd_worlds.meta import MetaAPIError, _ResponsesOverHTTP, is_marked, preview_jpeg
from elynd_worlds.settings import Prices

from .fakes import BRANCH_BOX, EDIT, FAIL, H, PAINTING, PASS, SCENE, SKY_BOX, STONE_BOX, W, FakeClient, png

MEDITATION = Meditation(script="Breathe in. Picture a quiet lake at dawn. Feel the cool stone under your hand.",
                        technique="visualization", mood="stressed", minutes=10)


def rgb(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.int16)


# ─── settings and cost ──────────────────────────────────────────────────

def test_contributor_models_are_refused():
    with pytest.raises(ValueError, match="contributor"):
        Settings(director_model="muse-spark-1.3-contributor")


def test_meter_prices_every_kind_of_use():
    meter = Meter(Prices())
    meter.spark({"prompt_tokens": 2_000_000, "completion_tokens": 1_000_000,
                 "prompt_tokens_details": {"cached_tokens": 1_000_000}}, "design")
    meter.image("paint", 3)
    meter.sam("segment", 400)
    meter.voice(3600.9)
    # 1M input x 1.25 + 1M cached x 0.15 + 1M output x 4.25 + 3 x 0.01 + 400 x 0.0025 + 1 h x 0.18
    assert meter.usd == pytest.approx(1.25 + 0.15 + 4.25 + 0.03 + 1.0 + 0.18)
    assert meter.as_dict()["calls"] == {"design": 1, "paint": 1, "segment": 1, "transcribe": 1}


def test_the_open_model_is_a_bill_of_its_own():
    meter = Meter(Prices())
    meter.spark_open({"prompt_tokens": 2_000_000, "completion_tokens": 1_000_000,
                      "prompt_tokens_details": {"cached_tokens": 1_000_000}}, "critique")
    meter.image("paint")
    assert meter.open_usd == pytest.approx(1.25 + 0.15 + 4.25)
    assert meter.meta_usd == pytest.approx(0.01)
    assert meter.usd == pytest.approx(5.66)
    assert meter.as_dict()["calls"] == {"critique@open": 1, "paint": 1}
    assert meter.as_dict()["open_tokens"] == {"input": 1_000_000, "cached": 1_000_000, "output": 1_000_000}


# ─── art director ───────────────────────────────────────────────────────

def test_scene_check_cleans_and_refuses():
    scene = art_director.check(json.loads(json.dumps(SCENE)))
    assert scene["palette"] == ["#AABBCC", "#112233", "#445566"]
    assert [s["name"] for s in scene["subjects"]] == ["smooth stone", "pine branch"]
    for broken, message in (({"palette": ["#fff"]}, "palette"), ({"subjects": []}, "1-3 subjects"),
                            ({"prompt": "too short"}, "too short"),
                            ({"subjects": [SCENE["subjects"][0], SCENE["subjects"][0]]}, "twice")):
        with pytest.raises(art_director.SceneError, match=message):
            art_director.check({**json.loads(json.dumps(SCENE)), **broken})


async def test_design_and_revise_go_to_muse_spark_with_strict_json():
    client, meter = FakeClient(), Meter()
    scene = await art_director.design(client, MEDITATION, meter)
    assert scene["title"] == "Still Morning Lake"
    first = client.requests[0]
    assert first["model"] == "muse-spark-1.3"
    assert first["response_format"]["json_schema"]["strict"] is True
    assert "quiet lake at dawn" in first["messages"][1]["content"]
    await art_director.revise(client, scene, {**FAIL, "judge": "meta-models/muse-spark-open-1026"}, meter)
    revise = client.requests[1]["messages"][1]["content"]
    assert "Remove any figure" in revise and "under your hand" not in revise  # the meditation is not resent
    assert "muse-spark-open" not in revise  # who judged is no business of the director
    assert meter.calls == {"design": 1, "revise": 1}


# ─── critic ─────────────────────────────────────────────────────────────

def test_a_painting_passes_only_if_safe_calm_on_brief_and_good():
    assert critic.passes(PASS)
    for change in ({"people_or_faces": True}, {"text_or_symbols": True}, {"calm": False},
                   {"belongs": False}, {"score": 6}):
        assert not critic.passes({**PASS, **change})


async def test_the_critic_sees_a_small_jpeg_and_scores_are_clamped():
    client = FakeClient(verdicts=[{**PASS, "score": 14}])
    verdict = await critic.critique(client, PAINTING, art_director.check(json.loads(json.dumps(SCENE))), Meter())
    assert verdict["score"] == 10
    image_part = client.requests[0]["messages"][1]["content"][1]["image_url"]["url"]
    assert image_part.startswith("data:image/jpeg;base64,")
    assert Image.open(io.BytesIO(preview_jpeg(png(1200, 2000), 576))).size == (576, 960)
    assert verdict["judge"] == "muse-spark-1.3"


CHECKED = art_director.check(json.loads(json.dumps(SCENE)))


async def test_the_open_model_judges_and_never_sees_the_listener_or_the_meditation():
    meta, open_model = FakeClient(), FakeClient(verdicts=[PASS])
    world = await make_world(meta, MEDITATION, user="listener-7f3a", open_client=open_model)
    assert open_model.calls == ["verdict"] and "verdict" not in meta.calls
    request = open_model.requests[0]
    assert request["model"] == "meta-models/muse-spark-open-1026" and request["reasoning_effort"] == "low"
    sent = json.dumps(request)
    assert "extra_body" not in request and "listener-7f3a" not in sent
    assert "Breathe in" not in sent and "under your hand" not in sent  # only the scenery design and the painting
    assert world.verdict["judge"] == "meta-models/muse-spark-open-1026"
    assert world.meter.calls["critique@open"] == 1 and "critique" not in world.meter.calls
    assert world.meter.open_usd > 0 and world.meter.usd == pytest.approx(world.meter.meta_usd + world.meter.open_usd)
    assert any(s.name == "critique 1" and "by meta-models/muse-spark-open-1026" in s.note for s in world.trace)


@pytest.mark.parametrize("broken, billed", [
    (FakeClient(fail_chat=True), 0),                          # an error or an outage
    (FakeClient(verdicts=['{"score": 9}']), 1),               # an answer without the checks
    (FakeClient(verdicts=[{**PASS, "calm": "yes"}]), 1),      # a check that is not true or false
    (FakeClient(verdicts=["not json"]), 1),
])
async def test_when_the_open_model_fails_muse_spark_judges(broken, billed):
    meta, meter = FakeClient(), Meter()
    verdict = await critic.critique(meta, PAINTING, CHECKED, meter, open_client=broken)
    assert verdict["judge"] == "muse-spark-1.3" and meta.calls == ["verdict"]
    assert meta.requests[0]["extra_body"]["store"] is False
    assert "reasoning_effort" not in meta.requests[0]  # Muse Spark, the fallback, thinks as long as it likes
    assert meter.calls["critique@open"] == billed and meter.calls["critique"] == 1  # what was answered is paid


async def test_a_slow_open_model_is_not_waited_for():
    started = time.monotonic()
    verdict = await critic.critique(FakeClient(), PAINTING, CHECKED, Meter(), Settings(open_critic_timeout=0.05),
                                    open_client=FakeClient(chat_delay=5))
    assert verdict["judge"] == "muse-spark-1.3" and time.monotonic() - started < 2  # the fake answers after 5 s


@pytest.mark.parametrize("given, reason", [
    ({**PASS, "reason": " "}, "No fault found; score 9 of 10."),
    ({**FAIL, "reason": ""}, "The painting shows a person or a face."),
    ({**PASS, "calm": False, "belongs": False, "reason": ""},
     "The painting is not calm, does not show the designed place and mood."),
])
async def test_a_missing_reason_is_written_from_the_checks(given, reason):
    verdict = await critic.critique(FakeClient(verdicts=[given]), PAINTING, CHECKED, Meter())
    assert verdict["reason"] == reason


# ─── cutter (SAM 3.1) ───────────────────────────────────────────────────

@pytest.mark.parametrize("name, first, fallback", [
    ("smooth stone", ["smooth stone", "smooth stones"], "stone"),
    ("pine branch", ["pine branch", "pine branches"], "branch"),
    ("lantern", ["lantern", "lanterns"], None),
    ("maple leaves", ["maple leaves", "maple leaf"], "leaves"),
    ("glass", ["glass", "glasses"], None),
    ("butterfly", ["butterfly", "butterflies"], None),
])
def test_wordings(name, first, fallback):
    assert cutter.wordings(name) == (first, fallback)


def test_masks_land_where_sam_says_and_holes_are_filled():
    ring = np.zeros((20, 20), dtype=np.uint8)
    ring[5:15, 5:15] = 1
    ring[8:12, 8:12] = 0  # the tea in the cup
    assert cutter.solid(ring)[10, 10] == 1 and cutter.solid(ring)[2, 2] == 0
    assert cutter.box_of(np.pad(np.ones((10, 20), np.uint8), ((90, 100), (40, 40)))) == (0.4, 0.45, 0.6, 0.5)


def test_anchor_follows_how_the_subject_is_attached():
    def layer(attached, box):
        return cutter.Layer("x", "near", attached, "still", "x", b"", box, 0.1, 0.9)
    assert layer("ground", (0.2, 0.7, 0.4, 0.9)).anchor() == (0.3, 0.9)
    assert layer("edge", (0.0, 0.1, 0.3, 0.2)).anchor() == (0.0, 0.15)    # grows in from the left
    assert layer("edge", (0.6, 0.0, 0.9, 0.2)).anchor() == (0.75, 0.0)    # hangs from the top
    assert layer("air", (0.4, 0.4, 0.6, 0.6)).anchor() == (0.5, 0.5)


async def test_cut_finds_each_subject_by_name_or_plural_and_keeps_near_ones():
    client = FakeClient(sam={"smooth stone": [STONE_BOX, (100, 10, 6, 4)], "pine branches": [BRANCH_BOX]})
    scene = art_director.check(json.loads(json.dumps(SCENE)))
    layers = await cutter.cut(client, PAINTING, scene, Meter())
    assert [(l.name, l.wording, l.depth) for l in layers] == [("smooth stone", "smooth stone", "near"),
                                                             ("pine branch", "pine branches", "mid")]
    stone = layers[0]
    assert stone.mask[160, 30] == 1 and stone.mask[12, 102] == 0  # the far, small stone is not moved
    assert stone.box == pytest.approx((20 / W, 150 / H, 70 / W, 180 / H), abs=1e-3)
    alpha = np.asarray(Image.open(io.BytesIO(stone.png)).getchannel("A"))
    assert alpha[160, 30] == 255 and alpha[100, 100] == 0


async def test_cut_falls_back_to_the_head_noun_and_skips_what_is_not_found():
    client = FakeClient(sam={"stone": [STONE_BOX]})
    layers = await cutter.cut(client, PAINTING, art_director.check(json.loads(json.dumps(SCENE))), Meter())
    assert [l.wording for l in layers] == ["stone"]
    assert "sam:branch" in client.calls  # tried, nothing found: no branch layer


async def test_a_mask_covering_most_of_the_picture_is_not_a_subject():
    client = FakeClient(sam={"smooth stone": [(0, 0, W, H)]})
    assert await cutter.cut(client, PAINTING, art_director.check(json.loads(json.dumps(SCENE))), Meter()) == []


def test_an_oar_lying_in_a_boat_rides_with_it():
    # SAM leaves the oar out: a 10 px gap from the boat's side into it. Closed for a boat, so the oar
    # moves with it and the repaint behind never shows through; never for things in the air.
    boat = np.zeros((120, 160), np.uint8)
    boat[30:70, 40:120] = 1
    boat[30:60, 75:85] = 0
    shut = cutter.solid(cutter.closed(boat))
    assert shut[30:60, 75:85].all() and (shut >= boat).all() and shut.sum() == 40 * 80
    assert cutter.rides_the_water({"motion": "float", "attached": "ground"})
    assert not cutter.rides_the_water({"motion": "float", "attached": "air"})
    assert not cutter.rides_the_water({"motion": "sway", "attached": "edge"})


# ─── plate ──────────────────────────────────────────────────────────────

def test_confine_keeps_the_painting_outside_the_subjects():
    painting, edit = Image.open(io.BytesIO(PAINTING)), Image.open(io.BytesIO(EDIT))
    mask = np.zeros((H, W), dtype=np.uint8)
    x, y, w, h = STONE_BOX
    mask[y:y + h, x:x + w] = 1
    out = np.asarray(plate.confine(painting, edit, mask), dtype=np.int16)
    assert (out[: y - 3] == rgb(PAINTING)[: y - 3]).all()            # untouched outside
    assert (out[y + 5:y + h - 5, x + 5:x + w - 5] == rgb(EDIT)[y + 5:y + h - 5, x + 5:x + w - 5]).all()


async def test_plate_plus_layers_at_rest_is_the_painting():
    client = FakeClient()
    scene = art_director.check(json.loads(json.dumps(SCENE)))
    layers = await cutter.cut(client, PAINTING, scene, Meter())
    plate_png, source = await plate.make_plate(client, PAINTING, EDIT, layers, Meter())
    assert source == "muse_image"
    composite = Image.open(io.BytesIO(plate_png)).convert("RGBA")
    for layer in layers:
        composite.alpha_composite(Image.open(io.BytesIO(layer.png)))
    # Equal: the repaint's soft edge lies where the layer is opaque (rounding aside).
    assert np.abs(np.asarray(composite.convert("RGB"), dtype=np.int16) - rgb(PAINTING)).max() <= 1


async def test_a_moving_subject_left_on_the_plate_means_a_local_fill():
    client = FakeClient(sam_on_plate={"pine branches": [BRANCH_BOX]})
    scene = art_director.check(json.loads(json.dumps(SCENE)))
    layers = await cutter.cut(client, PAINTING, scene, Meter())
    _, source = await plate.make_plate(client, PAINTING, EDIT, layers, Meter())
    assert source == "filled"
    # The same kind of object elsewhere in the scene is fine, and so is a still
    # subject: it stays in the plate on purpose.
    client = FakeClient(sam_on_plate={"pine branches": [(80, 120, 30, 16)], "smooth stone": [STONE_BOX]})
    layers = await cutter.cut(client, PAINTING, scene, Meter())
    assert (await plate.make_plate(client, PAINTING, EDIT, layers, Meter()))[1] == "muse_image"


async def test_a_still_subject_stays_in_the_plate_as_painted():
    client = FakeClient()
    scene = art_director.check(json.loads(json.dumps(SCENE)))
    layers = await cutter.cut(client, PAINTING, scene, Meter())
    still = [layer for layer in layers if layer.motion == "still"]
    assert still and len(still) < len(layers)  # the stone stays, the branch sways
    plate_png, _ = await plate.make_plate(client, PAINTING, EDIT, layers, Meter())
    stone = still[0].mask.astype(bool)
    assert (rgb(plate_png)[stone] == rgb(PAINTING)[stone]).all()
    # Nothing moves: the plate is the painting, and nothing is checked or repainted for it.
    calls = len(client.sam_requests)
    plate_png, source = await plate.make_plate(client, PAINTING, EDIT, still, Meter())
    assert source == "painting" and (rgb(plate_png) == rgb(PAINTING)).all()
    assert len(client.sam_requests) == calls


async def test_the_local_fill_stays_where_the_layer_is_opaque_so_at_rest_it_never_shows():
    client = FakeClient(fail_edit=True)
    scene = art_director.check(json.loads(json.dumps(SCENE)))
    layers = await cutter.cut(client, PAINTING, scene, Meter())
    plate_png, source = await plate.make_plate(client, PAINTING, None, layers, Meter())
    assert source == "filled"
    branch = next(layer for layer in layers if layer.motion != "still")
    near = cutter.grown(branch.mask, cutter.EDGE_PX + 2).astype(bool)
    assert (rgb(plate_png)[~near] == rgb(PAINTING)[~near]).all()  # beyond the layer's edge: the painting itself
    assert (rgb(plate_png)[branch.mask.astype(bool)] != rgb(PAINTING)[branch.mask.astype(bool)]).any()
    alpha = np.asarray(Image.open(io.BytesIO(branch.png)).getchannel("A"))
    assert (alpha[cutter.grown(branch.mask).astype(bool)] == 255).all()  # the layer carries its soft edge
    composite = Image.open(io.BytesIO(plate_png)).convert("RGBA")
    for layer in layers:
        composite.alpha_composite(Image.open(io.BytesIO(layer.png)))
    assert np.abs(np.asarray(composite.convert("RGB"), dtype=np.int16) - rgb(PAINTING)).max() <= 1


def test_fill_leaves_no_empty_pixels_even_for_a_large_subject():
    painting = Image.new("RGB", (300, 400), (90, 140, 60))
    mask = np.zeros((400, 300), dtype=np.uint8)
    mask[50:350, 40:260] = 1  # far wider than the first blur passes reach
    out = np.asarray(plate.fill(painting, mask), dtype=np.int16)
    assert np.abs(out - np.array([90, 140, 60])).max() <= 3


def test_tone_match_moves_the_edit_towards_the_surroundings():
    painting = Image.new("RGB", (100, 100), (40, 60, 80))
    edit = Image.new("RGB", (100, 100), (140, 160, 180))
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 1
    out = np.asarray(plate.tone_match(edit, painting, mask), dtype=np.int16)
    assert tuple(out[50, 50]) == (40, 60, 80)


def test_tone_match_ignores_what_the_repaint_rightly_took_away():
    # Water at 200; the painting has the boat's dark reflection just below it, which the
    # repaint (rightly) took away with the boat. The repaint of the water must stay as it is.
    painting = np.full((100, 100, 3), 200, dtype=np.uint8)
    painting[60:75, 35:65] = 40
    edit = Image.new("RGB", (100, 100), (200, 200, 200))
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 1
    out = np.asarray(plate.tone_match(edit, Image.fromarray(painting), mask), dtype=np.int16)
    assert np.abs(out[40:60, 40:60] - 200).max() <= 1


# ─── the agent ──────────────────────────────────────────────────────────

def pixels(picture: bytes) -> bytes:
    return Image.open(io.BytesIO(picture)).convert("RGB").tobytes()


async def test_a_world_is_shown_still_first_then_moves():
    client, shown = FakeClient(), []

    async def on_painting(painting, scene, verdict):
        shown.append((pixels(painting) == pixels(PAINTING), verdict["score"],
                      "plate" in [s for s in client.calls if s == "plate"]))

    world = await make_world(client, MEDITATION, on_painting=on_painting)
    assert shown == [(True, 9, False)]
    assert world.moving and world.plate_source == "muse_image" and world.paintings == 1
    assert [s.name for s in world.trace][:1] == ["design"]
    assert {"critique 1", "cut 1", "remove subjects 1", "plate"} <= {s.name for s in world.trace}
    # design + critique (Spark), 1 painting + 1 edit (Image), SAM for the subjects and the plate check
    assert world.meter.images == 2 and world.meter.sam_images >= 4
    assert world.manifest()["layers"][0]["anchor"] == list(world.layers[0].anchor())


async def test_a_rejected_painting_is_revised_and_painted_again():
    client = FakeClient(verdicts=[FAIL, PASS])
    world = await make_world(client, MEDITATION)
    assert world.paintings == 2
    assert client.calls.count("paint") == 2 and client.calls.count("scene") == 2  # design + revise
    names = [s.name for s in world.trace]
    assert "revise 1" in names and "critique 2" in names
    assert any(s.note.startswith("cancelled") for s in world.trace if s.name.endswith(" 1"))


async def test_two_rejections_mean_no_world():
    with pytest.raises(NoWorld):
        await make_world(FakeClient(verdicts=[FAIL]), MEDITATION)
    assert await make_world_or_none(FakeClient(verdicts=[FAIL]), MEDITATION) is None


async def test_without_layers_the_world_is_a_still_painting():
    world = await make_world(FakeClient(sam={}), MEDITATION)
    assert not world.moving and world.plate is None
    manifest = world.manifest()
    assert manifest["plate"] is None and manifest["layers"] == []
    world = await make_world(FakeClient(fail_sam=True), MEDITATION)
    assert not world.moving


async def test_a_failed_repaint_falls_back_to_the_local_fill():
    world = await make_world(FakeClient(fail_edit=True), MEDITATION)
    assert world.moving and world.plate_source == "filled"


async def test_only_still_subjects_no_repaint_and_a_still_world_unless_one_glows():
    still = json.loads(json.dumps(SCENE))
    for subject in still["subjects"]:
        subject["motion"] = "still"
    client = FakeClient(scenes=[still])
    world = await make_world(client, MEDITATION)
    assert "edit" not in client.calls and world.plate_source == "painting"
    assert not world.moving and world.manifest()["layers"] == []
    glowing = json.loads(json.dumps(still))
    glowing["subjects"][0]["glows"] = True  # a lantern: it flickers, so the world moves
    world = await make_world(FakeClient(scenes=[glowing]), MEDITATION)
    assert world.moving and [layer["glows"] for layer in world.manifest()["layers"]][:1] == [True]


def test_the_repaint_names_only_the_moving_subjects():
    scene = art_director.check(json.loads(json.dumps(SCENE)))
    prompt = painter.plate_prompt(scene)
    assert "pine branch" in prompt and "stone" not in prompt.lower() and "behind it" in prompt


async def test_the_world_is_saved_with_its_manifest(tmp_path):
    world = await make_world(FakeClient(), MEDITATION)
    folder = world.save(tmp_path / "w")
    manifest = json.loads((folder / "world.json").read_text())
    assert manifest["schema"] == "elynd.world/1" and manifest["size"] == [W, H]
    assert {p.name for p in folder.iterdir()} >= {"painting.png", "plate.png", "layer_0.png", "layer_1.png",
                                                    "world.json", "scene.json", "verdict.json", "cost.json"}
    assert manifest["cost_usd"] == world.meter.usd > 0


def test_the_focus_follows_the_cut_subjects_or_the_design():
    from elynd_worlds.world import World, focus_of_scene
    scene = art_director.check(json.loads(json.dumps(SCENE)))
    assert focus_of_scene(scene) == (0.33, 0.5)  # both subjects designed "left"
    stone = cutter.Layer("stone", "near", "ground", "still", "stone", b"", (0.6, 0.5, 0.8, 0.6), 0.03, 0.9)
    leaf = cutter.Layer("leaf", "mid", "air", "float", "leaf", b"", (0.0, 0.0, 0.2, 0.2), 0.01, 0.9)
    world = World(scene=scene, painting=PAINTING, verdict=PASS, layers=[stone, leaf], plate=EDIT)
    assert world.focus() == (0.55, 0.4375)  # weighted by size: (0.7*3 + 0.1*1)/4, (0.55*3 + 0.1*1)/4
    assert world.manifest()["focus"] == [0.55, 0.4375]


async def test_every_request_keeps_no_copy_and_names_the_listener_by_a_pseudonym():
    client = FakeClient()
    await make_world(client, MEDITATION, user="listener-7f3a")
    chats = [r for r in client.requests if "messages" in r]
    images = [r for r in client.requests if "prompt" in r and "messages" not in r]
    assert chats and all(r["extra_body"] == {"store": False, "safety_identifier": "listener-7f3a"} for r in chats)
    assert len(images) == 2 and all(r.get("user") == "listener-7f3a" for r in images)  # painting and repaint
    assert client.sam_requests and all(r["store"] is False and r["safety_identifier"] == "listener-7f3a"
                                       for r in client.sam_requests)


async def test_every_picture_a_listener_sees_is_marked_as_made_by_ai():
    assert not is_marked(PAINTING)  # the fake Muse Image forgets the mark Meta's adds
    world = await make_world(FakeClient(), MEDITATION)
    pictures = [world.painting, world.plate, *(layer.png for layer in world.layers),
                make_card(world.painting, "Still water holds the whole sky.")]
    assert len(pictures) == 5 and all(is_marked(p) for p in pictures)
    for picture in pictures:  # real XMP that image tools read, not just bytes
        info = Image.open(io.BytesIO(picture)).info
        assert b"trainedAlgorithmicMedia" in str(info.get("xmp") or info.get("XML:com.adobe.xmp")).encode()


# ─── what moves in nature ───────────────────────────────────────────────

LIVING = {**SCENE, "nature": ["water", "sky", "trees", "lava"], "particles": "fireflies",
          "subjects": [{**SCENE["subjects"][0], "glows": True}, SCENE["subjects"][1]]}


def test_the_design_names_what_moves_and_unknown_values_go():
    scene = art_director.check(json.loads(json.dumps(LIVING)))
    assert scene["nature"] == ["water", "sky", "trees"] and scene["particles"] == "fireflies"
    assert [s["glows"] for s in scene["subjects"]] == [True, False]
    old = art_director.check(json.loads(json.dumps(SCENE)))   # a design from before: nothing moves in nature
    assert old["nature"] == [] and old["particles"] == "none"


async def test_sam_finds_the_water_and_the_sky_and_the_map_carries_them():
    client = FakeClient(scenes=[LIVING])
    world = await make_world(client, MEDITATION)
    assert {"sam:water", "sam:sky"} <= set(client.calls) and "sam:trees" not in client.calls
    regions = world.regions
    assert regions.present == {"water": True, "sky": True} and regions.trees
    assert regions.horizon == pytest.approx(110 / 200, abs=0.02)
    rgba = np.asarray(Image.open(io.BytesIO(regions.png)))
    assert rgba.shape == (100, 60, 4)                                   # half the painting's size
    water, sky, grass, no_stream = (rgba[..., i] for i in range(4))
    assert water[95, 50] == 255 and water[10, 30] == 0                   # red: the water, at the bottom
    # The stone lies in the water and stays still: it is taken out of the map (half size here).
    x, y, w, h = STONE_BOX
    assert water[(y + h // 2) // 2, (x + w // 2) // 2] == 0
    assert sky[10, 30] == 255 and sky[90, 30] == 0                       # green: the sky, at the top
    assert grass.max() == 0
    # Alpha is 1 - stream: opaque wherever no stream flows, so a phone's premultiplied
    # copy of the map keeps the water and the sky.
    assert no_stream.min() == 255
    living = world.manifest()["living"]
    assert (living["regions"], living["water"], living["sky"], living["grass"], living["trees"]) == \
           ("regions.png", True, True, False, True)
    assert living["particles"] == "fireflies"
    # The stone glows: it flickers where SAM cut it out, a little above the middle of its box.
    (light,) = living["lights"]
    assert light["x"] == pytest.approx((20 + 25) / 120, abs=0.02) and light["size"] == pytest.approx(50 / 120, abs=0.02)
    assert any(s.name == "regions 1" for s in world.trace)
    assert [layer["glows"] for layer in world.manifest()["layers"]] == [True, False]


async def test_a_scrap_of_water_does_not_ripple_and_a_failed_search_keeps_the_world():
    client = FakeClient(scenes=[LIVING], sam={"smooth stone": [STONE_BOX], "pine branches": [BRANCH_BOX],
                                              "water": [(0, 190, 4, 4)], "sky": [SKY_BOX]})
    world = await make_world(client, MEDITATION)
    assert world.regions.present == {"sky": True} and world.regions.horizon is None
    assert world.moving   # the subjects still move


async def test_the_region_map_is_saved_with_the_world(tmp_path):
    world = await make_world(FakeClient(scenes=[LIVING]), MEDITATION)
    folder = world.save(tmp_path / "w")
    assert (folder / "regions.png").exists()
    assert json.loads((folder / "world.json").read_text())["living"]["regions"] == "regions.png"


# ─── timeline ───────────────────────────────────────────────────────────

def test_the_world_fades_as_eyes_close_and_returns():
    tl = {f["t"]: f for f in timeline.build(600, [timeline.Silence(200, 260)])["frames"]}
    assert tl[10]["visibility"] == 1.0 and tl[120]["visibility"] == 0.25
    assert tl[230]["visibility"] == 0.75        # a long silence brings it back
    assert tl[600]["visibility"] == 1.0
    sleep = {f["t"]: f for f in timeline.build(600, sleep=True)["frames"]}
    assert sleep[300]["visibility"] == 0.15 and sleep[600]["visibility"] == 0.0


def test_a_spoken_cue_leads_the_breath():
    cues = [timeline.Cue(100, "inhale"), timeline.Cue(104, "exhale")]
    frames = {f["t"]: f["breath"] for f in timeline.build(200, cues=cues)["frames"]}
    assert frames[100] == 0.0 and frames[102] == 0.5 and frames[104] == 1.0 and frames[107] == 0.5


# ─── share card ─────────────────────────────────────────────────────────

def test_the_card_is_a_story_sized_jpeg_and_skips_lines_it_cannot_draw():
    card = make_card(PAINTING, "Still water holds the whole sky.", title="Still Morning Lake")
    image = Image.open(io.BytesIO(card))
    assert image.format == "JPEG" and image.size == (1080, 1920)
    assert make_card(PAINTING, "静かな湖", title="湖") != card  # no error, no tofu boxes


# ─── older OpenAI SDKs ──────────────────────────────────────────────────

async def test_responses_over_http_streams_events_and_reports_errors():
    def answer(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True and request.headers["authorization"] == "Bearer k"
        if body["model"] == "bad":
            return httpx.Response(402, text='{"error": "billing_not_configured"}')
        return httpx.Response(200, text='data: {"type": "a"}\n\ndata: {"type": "b"}\n\ndata: [DONE]\n\n')

    responses = _ResponsesOverHTTP("k", transport=httpx.MockTransport(answer))
    events = [e async for e in await responses.create(model="sam-3.1", input=[])]
    assert events == [{"type": "a"}, {"type": "b"}]
    with pytest.raises(MetaAPIError, match="402"):
        [e async for e in await responses.create(model="bad", input=[])]


# ─── command line ───────────────────────────────────────────────────────

def test_the_open_critic_needs_a_key_and_room_in_its_budget(tmp_path, monkeypatch):
    monkeypatch.delenv("OPEN_MODEL_API_KEY", raising=False)
    ledger = tmp_path / "open.csv"
    args = argparse.Namespace(open_key_file=None, open_ledger=str(ledger), open_budget=80.0, scripts=["a.txt"])
    assert cli.open_critic(args) == (None, "no open-model key")
    key = tmp_path / "key"
    key.write_text("test-key\n")
    args.open_key_file = str(key)
    assert cli.open_critic(args)[0] is not None
    cli.record(ledger, "make", 79.99, "earlier runs")
    client, why = cli.open_critic(args)
    assert client is None and "$79.99 of its $80.00 budget" in why


def test_make_writes_each_bill_to_its_own_ledger(tmp_path, monkeypatch):
    meta, open_model = FakeClient(), FakeClient()
    monkeypatch.setattr(cli, "connect", lambda key: meta)
    monkeypatch.setattr(cli, "connect_open", lambda key: open_model)
    script = tmp_path / "lake.txt"
    script.write_text("# technique=visualization\nBreathe in. Picture a quiet lake at dawn.")
    key = tmp_path / "key"
    key.write_text("test-key")
    ledgers = tmp_path / "meta.csv", tmp_path / "open.csv"
    assert cli.main(["make", str(script), "--out", str(tmp_path / "out"), "--key-file", str(key),
                     "--ledger", str(ledgers[0]), "--open-ledger", str(ledgers[1]),
                     "--open-key-file", str(key)]) == 0
    report = json.loads(next((tmp_path / "out").glob("*/report.json")).read_text())[0]
    assert report["ok"] and report["judge"] == "meta-models/muse-spark-open-1026"
    meta_row, open_row = (list(csv.DictReader(ledger.open()))[0] for ledger in ledgers)
    assert float(meta_row["usd"]) == pytest.approx(report["meta_usd"], abs=1e-5)
    assert float(open_row["usd"]) == pytest.approx(report["open_usd"], abs=1e-5) and report["open_usd"] > 0
    assert report["usd"] == pytest.approx(report["meta_usd"] + report["open_usd"])


# ─── library ────────────────────────────────────────────────────────────

LIBRARY = [library.Entry("beach-dawn", "Quiet Dawn Beach", "A sandy beach at dawn.", "dawn"),
           library.Entry("forestlake-mist", "Misty Pine Lake", "A still lake among pines.", "dawn"),
           library.Entry("wildflowers-golden", "Golden Meadow", "A meadow of wildflowers.", "golden hour"),
           library.Entry("beach-night", "Moonlit Beach", "A beach under the moon.", "night", sleep=True),
           library.Entry("cabin-night", "Cabin Night", "A cabin by a forest at night.", "night", sleep=True)]


async def test_muse_spark_chooses_the_library_world_from_the_meditation():
    client = FakeClient(picks=[{"own_world": False,
                                "choices": [{"id": "forestlake-mist", "fit": 9}, {"id": "beach-dawn", "fit": 4}]}])
    meter = Meter()
    chosen = await library.pick(client, MEDITATION, LIBRARY, meter)
    assert chosen == library.Pick(LIBRARY[1], 9, own_world=False)
    assert not library.needs_own_world(chosen)  # it fits: the library world is the session's world
    request = client.requests[0]
    system, user = (m["content"] for m in request["messages"])
    assert "forestlake-mist | Misty Pine Lake" in system and "beach-night" not in system  # no sleep worlds
    assert "quiet lake at dawn" in user and "quiet lake at dawn" not in system  # the library first, then this
    assert request["model"] == "muse-spark-1.3" and request["reasoning_effort"] == "low"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert "own_world" in request["response_format"]["json_schema"]["schema"]["required"]
    assert request["extra_body"] == {"store": False}
    assert meter.calls == {"pick": 1}


async def test_a_sleep_meditation_chooses_only_among_the_sleep_worlds():
    client = FakeClient(picks=[{"choices": [{"id": "beach-dawn", "fit": 10}, {"id": "cabin-night", "fit": 7}]}])
    sleepy = Meditation(script="Let your body grow heavy.", technique="sleep_journey")
    chosen = await library.pick(client, sleepy, LIBRARY, Meter())
    assert chosen.entry.id == "cabin-night"  # a world not offered is never taken
    system = client.requests[0]["messages"][0]["content"]
    assert "cabin-night" in system and "beach-dawn" not in system
    assert await library.pick(FakeClient(), sleepy, LIBRARY[:3], Meter()) is None  # no sleep world at all


async def test_a_meditation_built_around_something_no_library_world_shows_gets_its_own():
    monkey = FakeClient(picks=[{"own_world": True, "choices": [{"id": "wildflowers-golden", "fit": 4}]}])
    chosen = await library.pick(monkey, MEDITATION, LIBRARY, Meter())
    assert chosen.own_world and library.needs_own_world(chosen)
    # Nothing that fits well enough, or no library world at all: its own world too.
    assert library.needs_own_world(library.Pick(LIBRARY[0], 4))
    assert not library.needs_own_world(library.Pick(LIBRARY[0], 5))
    assert library.needs_own_world(None)
    assert library.needs_own_world(library.Pick(LIBRARY[0], 9), Settings(library_min_fit=10))


def test_a_world_seen_lately_gives_way_to_the_next_if_it_fits_nearly_as_well():
    choices = [{"id": "beach-dawn", "fit": 9}, {"id": "forestlake-mist", "fit": 8}, {"id": "wildflowers-golden", "fit": 4}]
    assert library.choose(choices, LIBRARY).entry.id == "beach-dawn"
    assert library.choose(choices, LIBRARY, recent=["beach-dawn"]).entry.id == "forestlake-mist"
    # Nothing unseen fits nearly as well: the best is shown again.
    assert library.choose(choices, LIBRARY, recent=["beach-dawn", "forestlake-mist"]).entry.id == "beach-dawn"
    # Unknown ids and repeats are ignored, fits are clamped, and no usable choice means no library world.
    odd = [{"id": "no-such-world", "fit": 10}, {"id": " beach-dawn ", "fit": 14}, {"id": "beach-dawn", "fit": 3}]
    assert library.choose(odd, LIBRARY) == library.Pick(LIBRARY[0], 10)
    assert library.choose([{"id": "elsewhere", "fit": 9}], LIBRARY) is None


async def test_only_the_start_of_a_long_meditation_is_sent():
    client = FakeClient(picks=[{"choices": [{"id": "beach-dawn", "fit": 8}]}])
    long = Meditation(script="Picture a quiet beach at dawn. " + "Breathe slowly. " * 2000, technique="visualization")
    await library.pick(client, long, LIBRARY, Meter(), Settings(pick_words=50))
    user = client.requests[0]["messages"][1]["content"]
    assert "quiet beach at dawn" in user and user.endswith("...") and len(user.split()) < 80
    await library.pick(client, long, LIBRARY, Meter(), Settings(pick_reasoning=None))
    assert "reasoning_effort" not in client.requests[1]


async def test_the_library_is_read_from_saved_worlds(tmp_path):
    world = await make_world(FakeClient(), MEDITATION)
    world.save(tmp_path / "batch-1" / "forestlake-mist")
    (tmp_path / "unfinished").mkdir()
    (tmp_path / "unfinished" / "scene.json").write_text(json.dumps(SCENE))  # no world.json: not in the library
    assert library.entries_in([tmp_path]) == [
        library.Entry("forestlake-mist", "Still Morning Lake", SCENE["place"], "dawn", sleep=False)]


def test_pick_shows_the_choice_for_each_meditation(tmp_path, monkeypatch, capsys):
    client = FakeClient(picks=[{"own_world": True, "choices": [{"id": "forestlake-mist", "fit": 7}]}])
    monkeypatch.setattr(cli, "connect", lambda key: client)
    world = asyncio.run(make_world(FakeClient(), MEDITATION))
    world.save(tmp_path / "library" / "forestlake-mist")
    script = tmp_path / "monkey.txt"
    script.write_text("# technique=visualization\nPicture your little monkey, resting after a long day.")
    key = tmp_path / "key"
    key.write_text("test-key")
    ledger = tmp_path / "spend.csv"
    assert cli.main(["pick", str(script), "--library", str(tmp_path / "library"), "--key-file", str(key),
                     "--ledger", str(ledger)]) == 0
    line = json.loads(capsys.readouterr().out.splitlines()[0])
    assert line == {"name": "monkey", "library": "forestlake-mist", "fit": 7, "own_world": True,
                    "paint_its_own": True, "seconds": line["seconds"]}
    assert float(list(csv.DictReader(ledger.open()))[0]["usd"]) > 0  # what it cost is in the ledger

