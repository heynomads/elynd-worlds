"""A fake Meta Model API for the tests: no network, no cost.

The painting is a plain background with one coloured block per subject; SAM
answers with the blocks it is told about; the critic answers from a queue.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
from types import SimpleNamespace

import httpx
from openai import APITimeoutError, NotFoundError

from meta_sam_parser._mask_codec import _encode_raster
from PIL import Image, ImageDraw

from elynd_worlds.cutter import sam_image_url

W, H = 120, 200
BACKGROUND = (70, 110, 150)
STONE_BOX = (20, 150, 50, 30)     # x, y, w, h
BRANCH_BOX = (0, 20, 40, 16)
WATER_BOX = (0, 110, 120, 90)     # what moves in nature (regions.py)
SKY_BOX = (0, 0, 120, 70)

SCENE = {
    "title": "Still Morning Lake",
    "place": "A quiet lake at dawn with a stone on the shore.",
    "time_of_day": "dawn",
    "palette": ["#AABBCC", "#112233", "#445566", "not a colour"],
    "light": "soft, low from the left",
    "air": "thin mist over still water",
    "subjects": [
        {"name": "Smooth Stone", "depth": "near", "attached": "ground", "motion": "still", "where": "left"},
        {"name": "pine branch", "depth": "mid", "attached": "edge", "motion": "sway", "where": "left"},
    ],
    "prompt": ("A calm lake at dawn, mist over still water, a smooth stone on the shore in the lower left, "
               "a pine branch entering from the upper left edge, soft pastel light, painterly."),
    "sleep": False,
}
PASS = {"belongs": True, "calm": True, "figures_wrong": False, "text_or_symbols": False,
        "subjects_seen": ["smooth stone", "pine branch"], "score": 9, "reason": "Calm and on brief.", "fix": ""}
FAIL = {**PASS, "figures_wrong": True, "score": 4, "reason": "The dog on the shore has five legs.",
        "fix": "Remove any figure; keep the shore empty."}


def png(width: int = W, height: int = H, colour=BACKGROUND, blocks=()) -> bytes:
    image = Image.new("RGB", (width, height), colour)
    draw = ImageDraw.Draw(image)
    for (x, y, w, h), fill in blocks:
        draw.rectangle([x, y, x + w - 1, y + h - 1], fill=fill)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


PAINTING = png(blocks=[(STONE_BOX, (200, 190, 170)), (BRANCH_BOX, (40, 90, 50))])
EDIT = png(colour=(80, 118, 155))  # a repaint: no subjects, slightly different colours everywhere


def sam_record(index: int, box: tuple[int, int, int, int], conf: float = 0.9) -> str:
    x, y, w, h = box
    payload = _encode_raster([1] * (w * h), w, h)
    return (f"<0f>{index}<|box;x1={x};y1={y};x2={x + w - 1};y2={y + h - 1};w={W};h={H};c={conf}|>"
            f"<|mask;x=0;y=0;data={h},{w},{payload};c={conf}|>")


async def _stream(text: str):
    lane = {"item_id": "msg_1", "output_index": 0, "content_index": 0}
    for event in ({"type": "response.output_text.delta", "delta": text, **lane},
                  {"type": "response.output_text.done", "text": text, **lane},
                  {"type": "response.completed"}):
        yield event


USAGE = SimpleNamespace(prompt_tokens=1000, completion_tokens=200,
                        prompt_tokens_details=SimpleNamespace(cached_tokens=100))


class FakeClient:
    """Answers like Meta Model API (or, as `open_client`, like Muse Spark Open on Baseten).
    `sam` maps a phrase to boxes found on the painting; `sam_on_plate` maps a phrase to boxes
    found on any other picture (the plate). A verdict given as a string is answered verbatim."""

    def __init__(self, scenes=None, verdicts=None, sam=None, sam_on_plate=None, painting=PAINTING, edit=EDIT,
                 fail_edit=False, fail_sam=False, fail_chat=False, chat_delay=0.0, picks=None, missing=(),
                 sam_timeouts=0):
        self.calls: list[str] = []
        self.requests: list[dict] = []
        self.sam_requests: list[dict] = []
        self._scenes = [json.loads(json.dumps(s)) for s in (scenes or [SCENE])]
        self._verdicts = list(verdicts or [PASS])
        self._sam = sam if sam is not None else {"smooth stone": [STONE_BOX], "pine branches": [BRANCH_BOX],
                                                  "water": [WATER_BOX], "sky": [SKY_BOX]}
        self._sam_on_plate = sam_on_plate or {}
        self._painting, self._edit = painting, edit
        self._painting_url = sam_image_url(painting)
        self._fail_edit, self._fail_sam = fail_edit, fail_sam
        self._fail_chat, self._chat_delay = fail_chat, chat_delay
        self._picks = list(picks or [{"choices": []}])
        self._missing = set(missing)  # models answered with 404, as Meta Model API does for one it does not serve
        self._sam_timeouts = sam_timeouts  # the first SAM requests time out, as on a slow hour
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.images = SimpleNamespace(generate=self._generate, edit=self._edit_image)
        self.responses = SimpleNamespace(create=self._responses)

    @staticmethod
    def _pop(queue: list):
        return queue.pop(0) if len(queue) > 1 else queue[0]

    async def _chat(self, **kwargs):
        name = kwargs["response_format"]["json_schema"]["name"]
        if kwargs.get("model") in self._missing:
            request = httpx.Request("POST", "https://api.meta.ai/v1/chat/completions")
            raise NotFoundError("The requested model was not found.", response=httpx.Response(404, request=request),
                                body=None)
        self.calls.append(name)
        self.requests.append(kwargs)
        if self._chat_delay:
            await asyncio.sleep(self._chat_delay)
        if self._fail_chat:
            raise RuntimeError("chat failed")
        queue = {"scene": self._scenes, "pick": self._picks}.get(name, self._verdicts)
        payload = self._pop(queue)
        message = SimpleNamespace(content=payload if isinstance(payload, str) else json.dumps(payload))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=USAGE)

    async def _generate(self, **kwargs):
        self.calls.append("paint")
        self.requests.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(self._painting).decode())])

    async def _edit_image(self, **kwargs):
        self.calls.append("edit")
        self.requests.append(kwargs)
        if self._fail_edit:
            raise RuntimeError("edit failed")
        return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(self._edit).decode())])

    async def _responses(self, **kwargs):
        self.sam_requests.append(kwargs)
        content = kwargs["input"][0]["content"]
        phrase = next(c["text"] for c in content if c["type"] == "input_text")
        url = next(c["image_url"] for c in content if c["type"] == "input_image")
        on_plate = url != self._painting_url
        self.calls.append(f"sam{'@plate' if on_plate else ''}:{phrase}")
        if self._fail_sam:
            raise RuntimeError("sam failed")
        if self._sam_timeouts:
            self._sam_timeouts -= 1
            raise APITimeoutError(request=httpx.Request("POST", "https://api.meta.ai/v1/responses"))
        boxes = (self._sam_on_plate if on_plate else self._sam).get(phrase, [])
        text = "\n".join(sam_record(i, box) for i, box in enumerate(boxes))
        return _stream(text)
