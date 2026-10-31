"""climb_stage.py KEY IN OUT: one painting of a journey's climb, a Muse Image edit of the painting before it (IN).

A journey climbs one mountain, one painting a week: A the forest at the foot of the path (ours is the first stage of
a journey's world, `elynd-worlds stages`), then B the meadow, C the lower slope, D above the trees, E the ridge, F the
top, and G2, from the top, the view back down the way the path came. Each is painted from the one before it, much
further along and higher up the same path, so it stays the same place; climb_weeks.py shows A, B, C, E and F, one a
week, and G2 on day 30 (D is painted on the way to E and not shown). The light rises from the blue hour (A) to sunrise
at the top (F). Every stage is the place alone (painter.PLACE_ALONE_STYLE), from these neutral texts: nothing of a
listener goes into it.

    python scripts/climb/climb_stage.py B climb/A.png climb/B.png      # then C from B, D from C, ... G2 from F

The key comes from $MODEL_API_KEY, or from the file $MODEL_API_KEY_FILE names, as for `elynd-worlds make`; the cost
goes to the same spend ledger ($ELYND_WORLDS_LEDGER, default out/spend.csv)."""
import asyncio, io, os, sys
from pathlib import Path
from PIL import Image
from elynd_worlds import painter
from elynd_worlds.cli import read_key, record
from elynd_worlds.meta import Meter, connect
from elynd_worlds.settings import DEFAULT

KEEP = ("Keep this picture's painterly style, brushwork, palette and the same mountain, so it is clearly the same "
        "place. The path starts at the bottom centre of the picture and leads ahead. " + painter.PLACE_ALONE_STYLE)
ON = "Paint the view from much further along this same path"
STAGES = {
    "B": (f"{ON}, about an hour's walk on: the path has left the forest and crosses an open meadow of long grass, "
          "wildflowers and a few grey rocks toward the foot of this same mountain, which now rises much closer and "
          "larger in the upper half of the picture. Only a few pines remain at the edges. First light before dawn: "
          "the sky behind the mountain turns pale gold at the horizon, the meadow still in soft blue shadow. " + KEEP),
    "C": (f"{ON}: the path now climbs the lower slope of this same mountain in gentle zigzags between low shrubs, "
          "heather and boulders; the summit rises above, much closer than before; down to one side the meadow and "
          "the forest lie in the valley below. The dawn glow brightens: soft pink and gold clouds, cool light on the "
          "slope. " + KEEP),
    "D": (f"{ON}: high on the mountain above the last trees, the path climbs over open rocky grassland toward the "
          "summit ridge, which is near now; the valley far below fills with soft morning mist. The sky is bright and "
          "golden near the horizon; the sun is about to rise. " + KEEP),
    "E": (f"{ON}: the path follows the gentle grassy ridge toward the rounded summit, a short walk ahead now; a sea "
          "of soft clouds lies below on both sides. The sun touches the horizon and its first warm rays catch the "
          "summit. " + KEEP),
    "F": (f"{ON}: the last few steps of the path up to the rounded grassy top of this mountain, marked by a small "
          "cairn of stacked stones; beyond it only open sky above a sea of clouds lit gold. The sun has just risen: "
          "warm golden light on the grass and stones. " + KEEP),
    # A first view back ("looking back down the way the path came") read as a path climbing a ridge; G2 says that the
    # ground falls away and the path runs downhill.
    "G2": ("Paint the view from the very top of this same mountain, turned around and looking down the way the path "
           "came, seen from high above: the ground falls away in front of the viewer. From the bottom centre of the "
           "picture the path runs downhill along the grassy ridge, then down the rocky slope in zigzags, then across "
           "a meadow far below, and disappears into a dark pine forest at the bottom of the valley, smaller and "
           "smaller with distance; the ridge and the path lead down toward the middle of the picture. Wisps of mist "
           "in the valley; distant hills fade into a pale morning sky. The sun has just risen behind the viewer: "
           "warm golden light, long soft shadows pointing down the slope. Keep this picture's painterly style, "
           "brushwork and palette, so it is clearly the same place. " + painter.PLACE_ALONE_STYLE),
    # The first painting without its nearest objects (climb_days.py walks week 1 with it behind: what the first steps
    # reveal behind a rock or a post is painted, not smeared). Save it as climb/A_plate.png.
    "PLATE": ("Remove the nearest objects in the foreground completely (a rock, a post or a stump at the front of the "
              "picture) and paint what would be behind them instead, continuing the ground and the path naturally. "
              "Change nothing else: the same place, light, palette, brushwork and framing. " + painter.PLACE_ALONE_STYLE),
}


async def main():
    key, src, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    client = connect(read_key(os.environ.get("MODEL_API_KEY_FILE")))
    buf = io.BytesIO(); Image.open(src).convert("RGB").resize((1080, 1920), Image.LANCZOS).save(buf, "PNG")
    meter = Meter()
    try:
        picture = await painter._edit(client, buf.getvalue(), STAGES[key], meter, "climb_edit", DEFAULT)
        Image.open(io.BytesIO(picture)).convert("RGB").save(out)
    finally:
        record(Path(os.environ.get("ELYND_WORLDS_LEDGER", "out/spend.csv")), "climb_stage", meter.usd,
               f"the climb's stage {key} -> {out}")
        print("usd", meter.usd)
asyncio.run(main())
