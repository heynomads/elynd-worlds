"""Talking to Meta Model API, and counting what it costs.

Meta Model API speaks the OpenAI wire format, so the official OpenAI SDK is the
client. This module adds three things around it:

* `Meter`: every step reports what it used (tokens, images, SAM images, audio
  seconds) and the meter prices it with the published rates. A world knows its
  own cost, apart for each bill (Meta Model API, and the open model's host), and
  a deployment can stop at a budget.
* `_ResponsesOverHTTP`: SAM 3.1 is served through the Responses API, which
  OpenAI SDKs older than 1.66 do not have. The same streaming request is then
  sent with httpx, so the package also runs inside apps that pin an older SDK.
* Small helpers for images (data URLs, decoding, Muse Image's built-in tools,
  and the machine-readable "made by AI" mark every picture keeps).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
from openai import AsyncOpenAI, NotFoundError
from PIL import Image, PngImagePlugin

from .settings import API_BASE, OPEN_API_BASE, Prices

# Muse Image can search the web and run a shell while it paints. A meditation
# world needs neither: turning them off keeps every picture self-contained,
# and nothing from the session ever leaves for a search engine.
NO_IMAGE_TOOLS: dict[str, Any] = {
    "tool_enablement": {"enable_image_search": False, "enable_web_search": False, "enable_shell": False}
}


# The listener behind the current world, as a stable anonymous id (never a name
# or an email). Meta asks for one so that any enforcement targets that one
# listener instead of the whole app. Set by the director for each world.
listener: ContextVar[str | None] = ContextVar("listener", default=None)


def privacy_fields() -> dict[str, Any]:
    """Sent with every Muse Spark and SAM request: keep no copy (store=false), plus the listener's id."""
    fields: dict[str, Any] = {"store": False}
    if listener.get():
        fields["safety_identifier"] = listener.get()
    return fields


def cache_fields(use: str) -> dict[str, Any]:
    """Requests that start the same way (the same instructions, the same library) share a key, so Meta routes
    them together and serves that start from its cache: an eighth of the input price (meta.Prices). One key
    per use, never per listener (dev.meta.ai/docs/prompt-caching)."""
    return {"prompt_cache_key": f"elynd-worlds-{use}"}


# Muse Image marks each picture it returns as made by AI: IPTC's digital source
# type "trainedAlgorithmicMedia", in XMP. Every picture made from one (plate,
# layers, share card) carries the same mark, so cutting and compositing never
# remove it (EU AI Act, Art. 50(2): AI-generated images are marked in a
# machine-readable way).
AI_MADE_XMP: bytes = (
    '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>'
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    '<rdf:Description rdf:about="" xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/" '
    'Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"/>'
    '</rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
).encode()


def encode(image: Image.Image, format: str, **options: Any) -> bytes:
    """`image` as PNG, JPEG or WebP bytes, marked as made by AI (`AI_MADE_XMP`)."""
    if format.upper() == "PNG":
        # Pillow's PNG writer ignores `xmp=`; XMP in PNG is an iTXt chunk.
        info = PngImagePlugin.PngInfo()
        info.add_itxt("XML:com.adobe.xmp", AI_MADE_XMP.decode())
        options["pnginfo"] = info
    else:
        options["xmp"] = AI_MADE_XMP
    out = io.BytesIO()
    image.save(out, format=format, **options)
    return out.getvalue()


def is_marked(picture: bytes) -> bool:
    """Whether a PNG, JPEG or WebP carries the "made by AI" mark."""
    return b"digitalsourcetype/trainedAlgorithmicMedia" in picture


class MetaAPIError(RuntimeError):
    """A request Meta Model API refused or could not complete."""


@dataclass
class Meter:
    """What a world (or a run) used from Meta Model API and from the open model, and what that costs."""

    prices: Prices = field(default_factory=Prices)
    spark_input_tokens: int = 0
    spark_cached_tokens: int = 0
    spark_output_tokens: int = 0
    open_input_tokens: int = 0
    open_cached_tokens: int = 0
    open_output_tokens: int = 0
    images: int = 0
    sam_images: int = 0
    voice_seconds: int = 0
    calls: Counter = field(default_factory=Counter)

    def spark(self, usage: Any, step: str) -> None:
        """Record one Muse Spark answer from its `usage` block."""
        self.calls[step] += 1
        fresh, cached, output = _tokens(usage)
        self.spark_input_tokens += fresh
        self.spark_cached_tokens += cached
        self.spark_output_tokens += output

    def spark_open(self, usage: Any, step: str) -> None:
        """Record one Muse Spark Open answer (billed by its host, not by Meta)."""
        self.calls[f"{step}@open"] += 1
        fresh, cached, output = _tokens(usage)
        self.open_input_tokens += fresh
        self.open_cached_tokens += cached
        self.open_output_tokens += output

    def image(self, step: str, count: int = 1) -> None:
        self.calls[step] += 1
        self.images += count

    def sam(self, step: str, count: int = 1) -> None:
        self.calls[step] += 1
        self.sam_images += count

    def voice(self, seconds: float, step: str = "transcribe") -> None:
        self.calls[step] += 1
        self.voice_seconds += int(seconds)  # billed in whole seconds, rounded down

    @property
    def meta_usd(self) -> float:
        """Meta Model API's bill."""
        p = self.prices
        return round(
            self.spark_input_tokens * p.spark_input_per_million / 1e6
            + self.spark_cached_tokens * p.spark_cached_input_per_million / 1e6
            + self.spark_output_tokens * p.spark_output_per_million / 1e6
            + self.images * p.image
            + self.sam_images * p.sam_image
            + self.voice_seconds * p.voice_hour / 3600,
            5,
        )

    @property
    def open_usd(self) -> float:
        """The open model's host's bill (Baseten)."""
        p = self.prices
        return round(
            self.open_input_tokens * p.open_input_per_million / 1e6
            + self.open_cached_tokens * p.open_cached_input_per_million / 1e6
            + self.open_output_tokens * p.open_output_per_million / 1e6,
            5,
        )

    @property
    def usd(self) -> float:
        """Both bills together."""
        return round(self.meta_usd + self.open_usd, 5)

    def as_dict(self) -> dict[str, Any]:
        return {
            "usd": self.usd,
            "meta_usd": self.meta_usd,
            "open_usd": self.open_usd,
            "spark_tokens": {"input": self.spark_input_tokens, "cached": self.spark_cached_tokens,
                             "output": self.spark_output_tokens},
            "open_tokens": {"input": self.open_input_tokens, "cached": self.open_cached_tokens,
                            "output": self.open_output_tokens},
            "images": self.images,
            "sam_images": self.sam_images,
            "voice_seconds": self.voice_seconds,
            "calls": dict(self.calls),
        }


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _tokens(usage: Any) -> tuple[int, int, int]:
    """(fresh input, cached input, output) tokens of a `usage` block; output includes any reasoning."""
    if usage is None:
        return 0, 0, 0
    prompt = int(_get(usage, "prompt_tokens") or 0)
    cached = int(_get(_get(usage, "prompt_tokens_details"), "cached_tokens") or 0)
    return max(prompt - cached, 0), cached, int(_get(usage, "completion_tokens") or 0)


class _ResponsesOverHTTP:
    """`client.responses.create(..., stream=True)` for OpenAI SDKs that lack it.

    Yields the server-sent events as dicts, which is all SAM's stream parser
    needs. `transport` lets tests answer without a network.
    """

    def __init__(self, key: str, base_url: str = API_BASE, transport: httpx.AsyncBaseTransport | None = None):
        self._key = key
        self._url = base_url.rstrip("/") + "/responses"
        self._transport = transport

    async def create(self, *, timeout: float = 30.0, **body: Any) -> AsyncIterator[dict]:
        body["stream"] = True
        key, url, transport = self._key, self._url, self._transport

        async def events() -> AsyncIterator[dict]:
            async with httpx.AsyncClient(timeout=timeout, transport=transport) as http:
                async with http.stream("POST", url, json=body, headers={
                    "Authorization": f"Bearer {key}", "Accept": "text/event-stream",
                }) as response:
                    if response.status_code >= 400:
                        detail = (await response.aread()).decode(errors="replace")[:300]
                        raise MetaAPIError(f"Responses API {response.status_code}: {detail}")
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            data = line[5:].strip()
                            if data and data != "[DONE]":
                                yield json.loads(data)

        return events()


def connect(api_key: str | None = None, *, base_url: str = API_BASE, timeout: float = 60.0,
            max_retries: int = 2) -> AsyncOpenAI:
    """An async client for Meta Model API. The key defaults to $MODEL_API_KEY."""
    key = (api_key or os.environ.get("MODEL_API_KEY", "")).strip()
    if not key:
        raise MetaAPIError("No Meta Model API key: pass one or set MODEL_API_KEY")
    client = AsyncOpenAI(base_url=base_url, api_key=key, timeout=timeout, max_retries=max_retries)
    if not hasattr(client, "responses"):
        client.responses = _ResponsesOverHTTP(key, base_url)
    return client


def reasoning(effort: str | None) -> dict[str, Any]:
    """How long Muse Spark (or Muse Spark Open) thinks before it answers: minimal, low, medium, high or xhigh.
    None leaves it to the model. Reasoning is billed as output tokens."""
    return {"reasoning_effort": effort} if effort else {}


def connect_open(api_key: str | None = None, *, base_url: str = OPEN_API_BASE, timeout: float = 60.0) -> AsyncOpenAI:
    """An async client for Muse Spark Open (served by Baseten). The key defaults to $OPEN_MODEL_API_KEY.

    It never retries: when the open model fails, Muse Spark on Meta Model API answers instead.
    """
    key = (api_key or os.environ.get("OPEN_MODEL_API_KEY", "")).strip()
    if not key:
        raise MetaAPIError("No open-model API key: pass one or set OPEN_MODEL_API_KEY")
    return AsyncOpenAI(base_url=base_url, api_key=key, timeout=timeout, max_retries=0)


# Models Meta Model API answered 404 for, and until when (time.monotonic()) they are skipped.
_missing: dict[str, float] = {}
MISSING_FOR = 600.0


async def spark_chat(client, models: tuple[str, ...], **request: Any) -> tuple[Any, str]:
    """(answer, model): `client.chat.completions.create` with the first of `models` Meta Model API serves.

    A model it answers 404 for (not found: retired, renamed, or for a while not offered to this account)
    is skipped for ten minutes, so the next requests go straight to the next model instead of each
    paying for the refusal (it took 8 s on 27 Oct 2026).
    """
    error: Exception | None = None
    for model in dict.fromkeys(models):
        if _missing.get(model, 0.0) > time.monotonic():
            continue
        try:
            return await client.chat.completions.create(model=model, **request), model
        except NotFoundError as refused:
            _missing[model] = time.monotonic() + MISSING_FOR
            logging.getLogger("elynd_worlds").warning("%s not found; the next Muse Spark answers", model)
            error = refused
    if error is None:  # every model is being skipped: ask the first again rather than none
        return await client.chat.completions.create(model=models[0], **request), models[0]
    raise error


def data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"


def first_image(response: Any) -> bytes:
    """The first picture of an Images API answer (b64_json)."""
    data = _get(response, "data") or []
    b64 = _get(data[0], "b64_json") if data else None
    if not b64:
        raise MetaAPIError("Muse Image returned no picture")
    return base64.b64decode(b64)


def preview_jpeg(png: bytes, width: int) -> bytes:
    """A smaller JPEG copy of a picture, for a model that only needs to look at it."""
    image = Image.open(io.BytesIO(png)).convert("RGB")
    if image.width > width:
        image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=85)
    return out.getvalue()


def png_size(png: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(png)).size
