"""A fake Meta Model API for the tests: no network, no cost.

The painting is a plain background with one coloured block per subject; SAM
answers with the blocks it is told about; the critic answers from a queue.
"""
from __future__ import annotations

import base64
import io
import json
from types import SimpleNamespace

from meta_sam_parser._mask_codec import _encode_raster
from PIL import Image, ImageDraw

from elynd_worlds.cutter import sam_image_url

W, H = 120, 200
BACKGROUND = (70, 110, 150)
STONE_BOX = (20, 150, 50, 30)     # x, y, w, h
BRANCH_BOX = (0, 20, 40, 16)

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
PASS = {"belongs": True, "calm": True, "people_or_faces": False, "text_or_symbols": False,
        "subjects_seen": ["smooth stone", "pine branch"], "score": 9, "reason": "Calm and on brief.", "fix": ""}
FAIL = {**PASS, "people_or_faces": True, "score": 4, "reason": "A figure stands on the shore.",
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
    """Answers like Meta Model API. `sam` maps a phrase to boxes found on the painting;
    `sam_on_plate` maps a phrase to boxes found on any other picture (the plate)."""

    def __init__(self, scenes=None, verdicts=None, sam=None, sam_on_plate=None, painting=PAINTING, edit=EDIT,
                 fail_edit=False, fail_sam=False):
        self.calls: list[str] = []
        self.requests: list[dict] = []
        self.sam_requests: list[dict] = []
        self._scenes = [json.loads(json.dumps(s)) for s in (scenes or [SCENE])]
        self._verdicts = list(verdicts or [PASS])
        self._sam = sam if sam is not None else {"smooth stone": [STONE_BOX], "pine branches": [BRANCH_BOX]}
        self._sam_on_plate = sam_on_plate or {}
        self._painting, self._edit = painting, edit
        self._painting_url = sam_image_url(painting)
        self._fail_edit, self._fail_sam = fail_edit, fail_sam
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.images = SimpleNamespace(generate=self._generate, edit=self._edit_image)
        self.responses = SimpleNamespace(create=self._responses)

    @staticmethod
    def _pop(queue: list):
        return queue.pop(0) if len(queue) > 1 else queue[0]

    async def _chat(self, **kwargs):
        name = kwargs["response_format"]["json_schema"]["name"]
        self.calls.append(name)
        self.requests.append(kwargs)
        payload = self._pop(self._scenes) if name == "scene" else self._pop(self._verdicts)
        message = SimpleNamespace(content=json.dumps(payload))
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
        boxes = (self._sam_on_plate if on_plate else self._sam).get(phrase, [])
        text = "\n".join(sam_record(i, box) for i, box in enumerate(boxes))
        return _stream(text)
