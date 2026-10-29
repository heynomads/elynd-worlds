"""The agent that makes one world: design, paint, check, cut, and revise when needed.

    design (Muse Spark)
      └─ paint (Muse Image)
           ├─ critique (Muse Spark Open, vision; ┐
           │   Muse Spark if it fails)          │
           ├─ cut layers (SAM 3.1)              │
           ├─ find what moves (SAM 3.1)         ├─ run at the same time
           └─ paint without subjects (Image) ───┘
         rejected → the art director revises its design from the critic's note
                    → paint again (at most `Settings.paintings` paintings)
         accepted → `on_painting` shows the still painting at once,
                    then the clean plate is made and checked (SAM 3.1),
                    and the world can move.

The four steps after a painting only need the painting, and the critic
passes most of them, so they run together; a rejection cancels the other three.
Every step reports its cost to the world's `Meter` and its time to the trace.

`look` (the four steps on one painting) and `finish` (the plate and the living world) are shared with a
journey's later stages (journeys/stages.py), whose paintings come from an edit of the first instead.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from . import art_director, critic, cutter, painter, plate, regions as nature
from .art_director import Meditation
from .meta import Meter, listener as listener_id
from .settings import DEFAULT, Settings
from .world import Step, World

log = logging.getLogger("elynd_worlds")

OnPainting = Callable[[bytes, dict[str, Any], dict[str, Any]], Awaitable[None]]


class NoWorld(RuntimeError):
    """No painting passed the critic within the allowed attempts."""


async def _nothing() -> None:
    return None


class Run:
    """One world in the making: what it costs (its `Meter`), each step's time and note (its trace), and its clock."""

    def __init__(self, meter: Meter):
        self.meter = meter
        self.trace: list[Step] = []
        self.started = time.monotonic()

    async def step(self, name: str, work: Awaitable, note: Callable[[Any], str] = lambda _: "") -> Any:
        t0 = time.monotonic()
        try:
            result = await work
        except asyncio.CancelledError:
            self.trace.append(Step(name, time.monotonic() - t0, "cancelled: the painting was rejected"))
            raise
        except BaseException as error:
            self.trace.append(Step(name, time.monotonic() - t0, f"failed: {type(error).__name__}"))
            raise
        self.trace.append(Step(name, time.monotonic() - t0, note(result)))
        return result


@dataclass
class Parts:
    """What the steps after an accepted painting made: each one, or the error it failed with."""

    layers: Any
    edit: Any
    regions: Any


async def look(client, run: Run, painting: bytes, scene: dict[str, Any], attempt: int, *,
               settings: Settings = DEFAULT, open_client=None, on_painting: OnPainting | None = None,
               reference: bytes | None = None) -> tuple[dict[str, Any], Parts | None]:
    """(verdict, parts): the critique, the cutting, the repaint without the moving subjects and the search for what
    moves in nature, all on one painting at once. `parts` is None when the critic rejects the painting (the other
    three are then cancelled). With `reference` (a journey's first stage), the critic also checks that the painting
    shows the same place."""
    meter = run.meter
    checking = asyncio.ensure_future(run.step(f"critique {attempt}",
                                              critic.critique(client, painting, scene, meter, settings, open_client,
                                                              reference=reference),
                                              lambda v: f"score {v['score']} by {v['judge']}: {v['reason']}"))
    cutting = asyncio.ensure_future(run.step(f"cut {attempt}", cutter.cut(client, painting, scene, meter, settings),
                                             lambda ls: ", ".join(layer.wording for layer in ls) or "no layers"))
    # Only a moving subject needs what lies behind it painted (plate.py); with none, no repaint.
    removing = asyncio.ensure_future(run.step(f"remove subjects {attempt}",
                                              painter.paint_without_subjects(client, scene, painting, meter, settings))
                                     if painter.moving_subjects(scene) else _nothing())
    finding = asyncio.ensure_future(run.step(f"regions {attempt}",
                                             nature.find_regions(client, painting, scene, meter, settings),
                                             nature.note))
    try:
        verdict = await checking
    except BaseException:
        for task in (cutting, removing, finding):
            task.cancel()
        raise
    if critic.passes(verdict, settings):
        if on_painting is not None:
            try:
                await on_painting(painting, scene, verdict)
            except Exception as error:  # noqa: BLE001 - showing early is a bonus, never a failure
                log.info("on_painting failed: %s", error)
        layers, edit, regions = await asyncio.gather(cutting, removing, finding, return_exceptions=True)
        return verdict, Parts(layers, edit, regions)
    for task in (cutting, removing, finding):
        task.cancel()
    await asyncio.gather(cutting, removing, finding, return_exceptions=True)
    return verdict, None


async def finish(client, run: Run, painting: bytes, scene: dict[str, Any], verdict: dict[str, Any], parts: Parts,
                 paintings: int, settings: Settings = DEFAULT) -> World:
    """The living world of an accepted painting: its layers, the clean plate behind the moving ones, the region map
    without the still ones, and its lights. A part that failed leaves the world stiller, never without a world."""
    layers, edit, regions = parts.layers, parts.edit, parts.regions
    if isinstance(layers, BaseException):
        log.info("cutting failed, the world stays a still painting: %s", layers)
        layers = []
    if isinstance(edit, BaseException):
        log.info("the repaint without subjects failed, the plate is filled locally: %s", edit)
        edit = None
    if isinstance(regions, BaseException):
        log.info("finding what moves failed, only the subjects move: %s", regions)
        regions = nature.Regions(trees="trees" in (scene.get("nature") or []))
    # A still subject never ripples or drifts with the water, grass or sky around it.
    regions = nature.without(regions, [layer.mask for layer in layers
                                       if not plate.moves(layer) and layer.mask is not None])
    plate_png, source = await run.step("plate", plate.make_plate(client, painting, edit, layers, run.meter, settings),
                                       lambda result: result[1])
    if plate_png is None:
        layers = []
    return World(scene=scene, painting=painting, verdict=verdict, layers=layers, plate=plate_png,
                 plate_source=source, paintings=paintings, trace=run.trace, meter=run.meter,
                 seconds=time.monotonic() - run.started, regions=regions,
                 lights=nature.lights_from(layers, scene))


async def make_world(client, meditation: Meditation, *, settings: Settings = DEFAULT, meter: Meter | None = None,
                     on_painting: OnPainting | None = None, user: str | None = None,
                     open_client=None) -> World:
    """Run the agent for one meditation. `on_painting(painting, scene, verdict)` fires as soon as a
    painting is accepted, before the layers and the plate are ready. With `open_client`
    (meta.connect_open), Muse Spark Open is the critic."""
    run = Run(meter or Meter(settings.prices))
    if user:
        listener_id.set(user)  # every request of this world carries it (meta.privacy_fields)

    scene = await run.step("design", art_director.design(client, meditation, run.meter, settings,
                                                         open_client=open_client),
                           lambda s: f"{s['title']}; subjects: {', '.join(x['name'] for x in s['subjects'])}")
    for attempt in range(1, settings.paintings + 1):
        painting = await run.step(f"paint {attempt}", painter.paint(client, scene, run.meter, settings))
        verdict, parts = await look(client, run, painting, scene, attempt, settings=settings,
                                    open_client=open_client, on_painting=on_painting)
        if parts is not None:
            return await finish(client, run, painting, scene, verdict, parts, attempt, settings)
        if attempt == settings.paintings:
            raise NoWorld(f"no painting passed the critic: {verdict.get('reason')}")
        scene = await run.step(f"revise {attempt}", art_director.revise(client, scene, verdict, run.meter, settings,
                                                                        open_client=open_client),
                               lambda s: verdict.get("fix") or "")
    raise NoWorld("no painting was made")  # only with settings.paintings < 1


async def make_world_or_none(client, meditation: Meditation, *, settings: Settings = DEFAULT,
                             meter: Meter | None = None, on_painting: OnPainting | None = None,
                             user: str | None = None, open_client=None) -> World | None:
    """`make_world` within `Settings.world_timeout`; None on any failure (a world must never break a meditation)."""
    try:
        return await asyncio.wait_for(
            make_world(client, meditation, settings=settings, meter=meter, on_painting=on_painting, user=user,
                       open_client=open_client),
            settings.world_timeout)
    except Exception as error:  # noqa: BLE001
        log.warning("no world: %s: %s", type(error).__name__, error)
        return None
