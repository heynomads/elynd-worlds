"""A journey's world grows: five living worlds of one place, from the plan's non-personal description.

The plan (planner.py, `Plan.world`) describes the journey's world in English and in general terms only: a place
(`scene`) and five stages of it (`stages`: weeks one to four and day thirty), for example before dawn at the foot of
a path, then first light on the lower slopes, ... full light at the summit. It never holds the listener's words
(planner checks that), so it may go to Meta's API like any world's design.

* Stage 1 is a world like any other (director.make_world): the art director designs it from a short text made of
  the place and its first stage only (`first_text`), Muse Image paints it, the critic checks it, SAM 3.1 cuts its
  subjects and finds what moves, and the plate is made behind the moving ones. It is a world of the place alone
  (`Settings.figures` off): the painter is told no people and no animals, and the critic turns away any. When no
  world comes of it (two
  paintings turned away, or a request refused: on 29 Oct Muse Image's filter refused one design's painting of a
  path before dawn), it is made once more from a fresh design; only then is there no journey world.
* Stages 2-5 keep stage 1's composition: the same viewpoint, landmarks and subjects; only the light, the season and
  what has grown change. The art director re-lights its own first design for each (art_director.stage: time of day,
  light, air, palette, season; the place, the subjects and what moves in nature are kept in code), and Muse Image's
  edit repaints stage 1's painting into it (painter.paint_stage). The critic judges each next to stage 1's painting
  and also checks that it shows the same place from the same viewpoint (critic.critique, `reference`). Each stage
  gets its own layers, region map and plate, of the same subjects, so the same things move in every stage, and
  the first stage's life seed, so its birds cross at the same moments.
* An edit the critic turns away is made once more with the critic's note; a stage turned away twice, or that fails
  in any other way or takes longer than `Settings.stage_timeout`, shows the stage before it. The journey always
  shows its last good stage: one stage never loses the set.

Stages 2-5 are made at the same time (`parallel` at once), each from stage 1's painting, never from one another, so
no stage drifts further from the place than one edit. Judged with two pictures and four at once, Muse Spark Open took
9-32 s a stage and twice more than 45 s (29 Oct 2026); where only it may judge, two at a time keeps a stage from
falling back for want of a verdict.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import art_director, painter
from ..art_director import Meditation
from ..director import NoWorld, OnPainting, Run, finish, look, make_world
from ..life import seed
from ..meta import Meter, listener as listener_id
from ..settings import DEFAULT, Settings
from ..world import Step, World

log = logging.getLogger("elynd_worlds")

STAGES = 5
SCHEMA = "elynd.stages/1"


def journey_world(data: Any) -> dict[str, Any]:
    """The journey's world, {"scene": one sentence, "stages": five}, from a plan (`Plan.to_json()`, its "world") or
    from the world alone. ValueError when it is not one."""
    if isinstance(data, dict) and isinstance(data.get("world"), dict):
        data = data["world"]
    if not isinstance(data, dict):
        raise ValueError("a journey's world is a JSON object with a scene and five stages")
    scene = " ".join(str(data.get("scene") or "").split())
    stages = data.get("stages")
    stages = [" ".join(str(s or "").split()) for s in stages] if isinstance(stages, list) else []
    if not scene or len(stages) != STAGES or not all(stages):
        raise ValueError(f"a journey's world needs a scene and {STAGES} stages")
    return {"scene": scene, "stages": stages}


def first_text(world: dict[str, Any]) -> str:
    """What the art director designs stage 1 from, in place of a meditation: the place and its first stage."""
    return (f"{world['scene'].rstrip('.')}. Now: {painter.clause(world['stages'][0])}. This is the first of five "
            "stages of one place: later the same view is painted again as the light, the season and what grows "
            "there change, so show the whole place in one calm view. Only the place: no people and no animals.")


@dataclass
class Stage:
    """One stage of a journey's world."""

    n: int
    text: str                    # the plan's words for it
    world: World                 # its own world, or, when it could not be made, the world of the stage before
    shows: int                   # the stage whose world it shows: `n` when it was made
    meter: Meter                 # what making it cost, made or not
    trace: list[Step] = field(default_factory=list)
    error: str = ""              # why it shows the stage before

    @property
    def made(self) -> bool:
        return self.shows == self.n


@dataclass
class Stages:
    """A journey's world: five stages, the first painted, the others repainted from it."""

    world: dict[str, Any]        # the plan's {"scene", "stages"}
    stages: list[Stage]
    meter: Meter                 # all five together
    seconds: float = 0.0

    @property
    def worlds(self) -> list[World]:
        """The five worlds, stage 1 first; a stage that could not be made holds the world of the stage before."""
        return [stage.world for stage in self.stages]

    def summary(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "scene": self.world["scene"],
            "stages": [{
                "n": s.n, "text": s.text, "folder": f"stage{s.n}", "shows": s.shows,
                "title": s.world.scene["title"] if s.made else None,
                "score": s.world.verdict.get("score") if s.made else None,
                "judge": s.world.verdict.get("judge") if s.made else None,
                "paintings": s.world.paintings if s.made else None,
                "plate": s.world.plate_source if s.made else None,
                "layers": [layer.name for layer in s.world.layers] if s.made and s.world.moving else [],
                "usd": s.meter.usd, "error": s.error,
                "trace": [{"step": t.name, "seconds": round(t.seconds, 2), "note": t.note} for t in s.trace],
            } for s in self.stages],
            "cost_usd": self.meter.usd, "meta_usd": self.meter.meta_usd, "open_usd": self.meter.open_usd,
            "seconds": round(self.seconds, 2),
        }

    def save(self, folder: str | Path) -> Path:
        """Each stage's world (world.json and its pictures) in `folder`/stage1 ... stage5, and stages.json."""
        folder = Path(folder)
        for s in self.stages:
            s.world.save(folder / f"stage{s.n}", extra={"stage": {"n": s.n, "of": STAGES, "shows": s.shows,
                                                                  "text": s.text}})
        (folder / "stages.json").write_text(json.dumps(self.summary(), indent=2, ensure_ascii=False))
        return folder


async def _later(client, first: World, world: dict[str, Any], n: int, run: Run, settings: Settings,
                 open_client) -> World:
    """Stage `n` (2-5): the art director re-lights the first design, Muse Image repaints the first painting into it,
    the critic judges it next to the first (once more with the critic's note), and it is cut and brought to life."""
    text = world["stages"][n - 1]
    scene = await run.step("stage design",
                           art_director.stage(client, first.scene, world["stages"], n, run.meter, settings,
                                              open_client=open_client),
                           lambda s: f"{s['title']}; {s['time_of_day']}: {s.get('change') or s['light']}")
    fix, verdict = "", {}
    for attempt in range(1, settings.paintings + 1):
        painting = await run.step(f"edit {attempt}",
                                  painter.paint_stage(client, scene, first.painting, text, run.meter, settings, fix))
        verdict, parts = await look(client, run, painting, scene, attempt, settings=settings,
                                    open_client=open_client, reference=first.painting)
        if parts is not None:
            world = await finish(client, run, painting, scene, verdict, parts, attempt, settings)
            world.life_seed = seed(first.scene)  # its birds come when the first stage's do
            return world
        fix = verdict.get("fix") or verdict.get("reason") or ""
    raise NoWorld(f"no edit passed the critic: {verdict.get('reason')}")


async def make_stages(client, world: dict[str, Any], *, settings: Settings = DEFAULT, meter: Meter | None = None,
                      open_client=None, user: str | None = None, on_painting: OnPainting | None = None,
                      parallel: int = 4) -> Stages:
    """The five stages of a journey's world (`journey_world`). `meter`, if given, counts all of it, also when stage 1
    fails; `on_painting` fires when stage 1's painting is accepted (director.make_world). Without stage 1 (twice)
    there is no place to grow: NoWorld or the first world's own error, and the journey keeps the world it has."""
    world = journey_world(world)
    # The place alone, in every stage: the plan's world never holds people or animals.
    settings = dataclasses.replace(settings, figures=False)
    started = time.monotonic()
    total = meter or Meter(settings.prices)
    meters = [Meter(settings.prices) for _ in range(STAGES)]
    if user:
        listener_id.set(user)  # every request of the five carries it (meta.privacy_fields)
    try:
        for attempt in (1, 2):
            try:
                first = await make_world(client, Meditation(script=first_text(world)), settings=settings,
                                         meter=meters[0], on_painting=on_painting, open_client=open_client)
                break
            except Exception as error:  # noqa: BLE001 - once more from a fresh design, then the caller hears it
                if attempt == 2:
                    raise
                log.warning("stage 1 once more: %s: %s", type(error).__name__, error)
        runs = {n: Run(meters[n - 1]) for n in range(2, STAGES + 1)}
        gate = asyncio.Semaphore(max(1, parallel))

        async def later(n: int) -> World | Exception:
            async with gate:
                runs[n].started = time.monotonic()
                try:
                    return await asyncio.wait_for(_later(client, first, world, n, runs[n], settings, open_client),
                                                  settings.stage_timeout)
                except Exception as error:  # noqa: BLE001 - this stage shows the one before
                    log.warning("stage %d shows the stage before: %s: %s", n, type(error).__name__, error)
                    return error

        results = await asyncio.gather(*(later(n) for n in range(2, STAGES + 1)))
        stages = [Stage(1, world["stages"][0], first, 1, meters[0], first.trace)]
        for n, result in zip(range(2, STAGES + 1), results):
            text, before = world["stages"][n - 1], stages[-1]
            if isinstance(result, World):
                stages.append(Stage(n, text, result, n, meters[n - 1], runs[n].trace))
            else:
                stages.append(Stage(n, text, before.world, before.shows, meters[n - 1], runs[n].trace,
                                    error=f"{type(result).__name__}: {result}"[:300]))
        return Stages(world=world, stages=stages, meter=total, seconds=time.monotonic() - started)
    finally:
        for part in meters:
            total.add(part)
