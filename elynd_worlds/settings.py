"""What ELYND Worlds runs on: models, limits and prices.

Everything a deployment may want to tune lives here, so the other modules read
like the steps they perform.
"""
from __future__ import annotations

from dataclasses import dataclass, field

API_BASE = "https://api.meta.ai/v1"
# Muse Spark Open, Meta's open-weights Muse Spark, as Baseten serves it (the same OpenAI wire format).
OPEN_API_BASE = "https://inference.baseten.co/v1"


@dataclass(frozen=True)
class Prices:
    """USD, Standard mode, from dev.meta.ai/docs/pricing-rate-limits (26 Oct 2026)."""

    spark_input_per_million: float = 1.25
    spark_cached_input_per_million: float = 0.15
    spark_output_per_million: float = 4.25
    image: float = 0.01              # per generated or edited image returned
    sam_image: float = 2.50 / 1000   # per image segmented
    voice_hour: float = 0.18         # per hour of audio transcribed
    # Muse Spark Open on Baseten (27 Oct 2026): the same rates. It always reasons, and the
    # reasoning is billed as output.
    open_input_per_million: float = 1.25
    open_cached_input_per_million: float = 0.15
    open_output_per_million: float = 4.25


@dataclass(frozen=True)
class Settings:
    # Standard mode only. The "-contributor" variants cost less because Meta may
    # train on what they receive; a listener's meditation must never be used so.
    director_model: str = "muse-spark-1.3"
    critic_model: str = "muse-spark-1.3"
    # When Meta Model API answers 404 for a Muse Spark model (27 Oct 2026: muse-spark-1.3 vanished from this
    # account's catalog for a while), this one answers instead (same prices, Standard mode).
    spark_fallback_model: str = "muse-spark-1.2"
    # Chooses the library world a session opens on (library.py).
    library_model: str = "muse-spark-1.3"
    # The critic runs on Muse Spark Open when the deployment passes an open-model client
    # (critic.py); `critic_model` above takes over whenever it fails.
    open_critic_model: str = "meta-models/muse-spark-open-1026"
    # Little reasoning: 4-6 s a painting instead of 12-18 s, at half the cost, with the same
    # verdicts on our test pictures (all four that must fail still fail). None: the model decides.
    open_critic_reasoning: str | None = "low"
    painter_model: str = "muse-image-1.0"
    cutter_model: str = "sam-3.1"
    voice_model: str = "muse-voice-transcribe-1.0"

    # Portrait and phone-shaped. Muse Image reads the size as an aspect ratio.
    image_size: str = "1080x1920"
    # The critic sees a smaller copy: fewer image tokens, same judgement.
    critic_image_width: int = 576
    # Tagging a library world with its feelings: a path, a summit or a beach shows at this size too.
    moods_image_width: int = 384

    paintings: int = 2               # the first painting plus at most one revision
    pass_score: int = 7              # the critic's score a painting needs, 0-10

    # Layers: a subject becomes a moving layer only if SAM is this sure of it,
    # and it covers a sensible share of the picture (a "subject" covering most
    # of the picture is the background, and moving it would move the world).
    min_confidence: float = 0.35
    min_layer_share: float = 0.0008  # a lantern 1/20 of the width is about 0.15%
    max_layer_share: float = 0.6
    # Of several matches for one subject, only the near ones move: an instance
    # whose area is at least this share of the largest one's.
    near_share: float = 0.4
    # A region (water, sky, grass, stream) moves only if it covers at least this
    # share of the painting: a stray scrap of "water" would ripple on its own.
    min_region_share: float = 0.01

    # Seconds. The meditation never waits for any of them.
    design_timeout: float = 25
    paint_timeout: float = 45
    critic_timeout: float = 20
    open_critic_timeout: float = 18  # then Muse Spark judges instead
    pick_timeout: float = 12

    # Choosing a library world is a short task: the least reasoning, and only the start of a meditation,
    # where its place is set (27 Oct 2026: 0.7 instead of 0.9 cents a choice, the same choices; with the
    # library served from Meta's cache, cache_fields, about 0.3).
    pick_reasoning: str | None = "minimal"
    pick_words: int = 600
    # A library world that fits less than this (0-10) is not the session's world when the meditation
    # describes a place: one is painted for it. A meditation without a place keeps its library world.
    library_min_fit: int = 5
    cut_timeout: float = 20
    # A SAM request that fails on the way is asked again (cutter.segment), after 2 s, then 4 s.
    sam_attempts: int = 3
    sam_retry_seconds: float = 2.0
    world_timeout: float = 120

    # USD. A deployment stops giving the critic the open model once its own ledger of open-model
    # spend reaches this (the Baseten credits are $100; the rest covers worlds in flight).
    open_budget_usd: float = 80.0

    prices: Prices = field(default_factory=Prices)

    def __post_init__(self) -> None:
        for name in ("director_model", "critic_model", "library_model", "spark_fallback_model"):
            if getattr(self, name).endswith("-contributor"):
                raise ValueError(f"{name}: contributor-mode models may train on listeners' meditations")


DEFAULT = Settings()
