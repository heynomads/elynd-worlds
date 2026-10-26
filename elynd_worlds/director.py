"""The agent that makes one world: design, paint, check, cut, and revise when needed.

    design (Muse Spark)
      └─ paint (Muse Image)
           ├─ critique (Muse Spark, vision) ──┐
           ├─ cut layers (SAM 3.1)            │
           ├─ find what moves (SAM 3.1)       ├─ run at the same time
           └─ paint without subjects (Image) ─┘
         rejected → the art director revises its design from the critic's note
                    → paint again (at most `Settings.paintings` paintings)
         accepted → `on_painting` shows the still painting at once,
                    then the clean plate is made and checked (SAM 3.1),
                    and the world can move.

The four steps after a painting only need the painting, and the critic
passes most of them, so they run together; a rejection cancels the other three.
Every step reports its cost to the world's `Meter` and its time to the trace.
"""
from __future__ import annotations

import asyncio
import logging
import time
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


async def make_world(client, meditation: Meditation, *, settings: Settings = DEFAULT, meter: Meter | None = None,
                     on_painting: OnPainting | None = None, user: str | None = None) -> World:
    """Run the agent for one meditation. `on_painting(painting, scene, verdict)` fires as soon as a
    painting is accepted, before the layers and the plate are ready."""
    meter = meter or Meter(settings.prices)
    trace: list[Step] = []
    started = time.monotonic()
    if user:
        listener_id.set(user)  # every request of this world carries it (meta.privacy_fields)

    async def step(name: str, work: Awaitable, note: Callable[[Any], str] = lambda _: "") -> Any:
        t0 = time.monotonic()
        try:
            result = await work
        except asyncio.CancelledError:
            trace.append(Step(name, time.monotonic() - t0, "cancelled: the painting was rejected"))
            raise
        except BaseException as error:
            trace.append(Step(name, time.monotonic() - t0, f"failed: {type(error).__name__}"))
            raise
        trace.append(Step(name, time.monotonic() - t0, note(result)))
        return result

    scene = await step("design", art_director.design(client, meditation, meter, settings),
                       lambda s: f"{s['title']}; subjects: {', '.join(x['name'] for x in s['subjects'])}")
    verdict: dict[str, Any] = {}
    painting = b""
    for attempt in range(1, settings.paintings + 1):
        painting = await step(f"paint {attempt}", painter.paint(client, scene, meter, settings))
        checking = asyncio.ensure_future(step(f"critique {attempt}", critic.critique(client, painting, scene, meter, settings),
                                              lambda v: f"score {v['score']}: {v['reason']}"))
        cutting = asyncio.ensure_future(step(f"cut {attempt}", cutter.cut(client, painting, scene, meter, settings),
                                             lambda ls: ", ".join(layer.wording for layer in ls) or "no layers"))
        removing = asyncio.ensure_future(step(f"remove subjects {attempt}",
                                              painter.paint_without_subjects(client, scene, painting, meter, settings)))
        finding = asyncio.ensure_future(step(f"regions {attempt}",
                                             nature.find_regions(client, painting, scene, meter, settings),
                                             lambda r: ", ".join(k for k, v in r.present.items() if v) or "none"))
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
            break
        for task in (cutting, removing, finding):
            task.cancel()
        await asyncio.gather(cutting, removing, finding, return_exceptions=True)
        if attempt == settings.paintings:
            raise NoWorld(f"no painting passed the critic: {verdict.get('reason')}")
        scene = await step(f"revise {attempt}", art_director.revise(client, scene, verdict, meter, settings),
                           lambda s: verdict.get("fix") or "")

    if isinstance(layers, BaseException):
        log.info("cutting failed, the world stays a still painting: %s", layers)
        layers = []
    if isinstance(edit, BaseException):
        log.info("the repaint without subjects failed, the plate is filled locally: %s", edit)
        edit = None
    if isinstance(regions, BaseException):
        log.info("finding what moves failed, only the subjects move: %s", regions)
        regions = nature.Regions(trees="trees" in (scene.get("nature") or []))
    plate_png, source = await step("plate", plate.make_plate(client, painting, edit, layers, meter, settings),
                                   lambda result: result[1])
    if plate_png is None:
        layers = []
    return World(scene=scene, painting=painting, verdict=verdict, layers=layers, plate=plate_png,
                 plate_source=source, paintings=attempt, trace=trace, meter=meter,
                 seconds=time.monotonic() - started, regions=regions,
                 lights=nature.lights_from(layers, scene))


async def make_world_or_none(client, meditation: Meditation, *, settings: Settings = DEFAULT,
                             meter: Meter | None = None, on_painting: OnPainting | None = None,
                             user: str | None = None) -> World | None:
    """`make_world` within `Settings.world_timeout`; None on any failure (a world must never break a meditation)."""
    try:
        return await asyncio.wait_for(
            make_world(client, meditation, settings=settings, meter=meter, on_painting=on_painting, user=user),
            settings.world_timeout)
    except Exception as error:  # noqa: BLE001
        log.warning("no world: %s: %s", type(error).__name__, error)
        return None
