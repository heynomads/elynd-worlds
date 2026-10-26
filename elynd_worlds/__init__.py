"""ELYND Worlds: a painted world for every meditation, designed, painted, checked and
brought to life with Meta's models (Muse Spark, Muse Image, SAM 3.1)."""
from .art_director import Meditation
from .director import NoWorld, make_world, make_world_or_none
from .meta import Meter, connect
from .settings import DEFAULT, Settings
from .world import World

__all__ = ["DEFAULT", "Meditation", "Meter", "NoWorld", "Settings", "World", "connect", "make_world",
           "make_world_or_none"]
__version__ = "0.1.0"
