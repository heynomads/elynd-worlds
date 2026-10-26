"""What ELYND Worlds runs on: models, limits and prices.

Everything a deployment may want to tune lives here, so the other modules read
like the steps they perform.
"""
from __future__ import annotations

from dataclasses import dataclass, field

API_BASE = "https://api.meta.ai/v1"


@dataclass(frozen=True)
class Prices:
    """USD, Standard mode, from dev.meta.ai/docs/pricing-rate-limits (26 Oct 2026)."""

    spark_input_per_million: float = 1.25
    spark_cached_input_per_million: float = 0.15
    spark_output_per_million: float = 4.25
    image: float = 0.01              # per generated or edited image returned
    sam_image: float = 2.50 / 1000   # per image segmented
    voice_hour: float = 0.18         # per hour of audio transcribed


@dataclass(frozen=True)
class Settings:
    # Standard mode only. The "-contributor" variants cost less because Meta may
    # train on what they receive; a listener's meditation must never be used so.
    director_model: str = "muse-spark-1.3"
    critic_model: str = "muse-spark-1.3"
    painter_model: str = "muse-image-1.0"
    cutter_model: str = "sam-3.1"
    voice_model: str = "muse-voice-transcribe-1.0"

    # Portrait and phone-shaped. Muse Image reads the size as an aspect ratio.
    image_size: str = "1080x1920"
    # The critic sees a smaller copy: fewer image tokens, same judgement.
    critic_image_width: int = 576

    paintings: int = 2               # the first painting plus at most one revision
    pass_score: int = 7              # the critic's score a painting needs, 0-10

    # Layers: a subject becomes a moving layer only if SAM is this sure of it,
    # and it covers a sensible share of the picture (a "subject" covering most
    # of the picture is the background, and moving it would move the world).
    min_confidence: float = 0.35
    min_layer_share: float = 0.002
    max_layer_share: float = 0.6
    # Of several matches for one subject, only the near ones move: an instance
    # whose area is at least this share of the largest one's.
    near_share: float = 0.4

    # Seconds. The meditation never waits for any of them.
    design_timeout: float = 25
    paint_timeout: float = 45
    critic_timeout: float = 20
    cut_timeout: float = 20
    world_timeout: float = 120

    prices: Prices = field(default_factory=Prices)

    def __post_init__(self) -> None:
        for name in ("director_model", "critic_model"):
            if getattr(self, name).endswith("-contributor"):
                raise ValueError(f"{name}: contributor-mode models may train on listeners' meditations")


DEFAULT = Settings()
