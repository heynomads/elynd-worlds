# ELYND Worlds

**Every meditation gets its own painted world, designed, painted, checked and brought to life with Meta's models.**

[ELYND](https://elynd.app) writes and voices a new guided meditation for each listener, every time. ELYND Worlds,
built during the Meta Global AI Developer Hackathon (Startup track, *Build for Your World*), gives each of those
meditations a place you can see: a calm painting made from the meditation itself that lives as such a place
would. Its water ripples, its clouds drift, its grass waves, a lantern flickers, and the objects in it move as such
things move. It greets you as you arrive, fades while your eyes are closed, returns in long silences and at the
end, and stays with you afterwards as a card you can share.

> **What existed before, what is new.** ELYND, the meditation app (writing, voices, player), existed before
> the hackathon. Earlier on the kickoff day we tried the Meta Model API in a separate private
> prototype. This repository was started from scratch after the kickoff (26 Oct 2026, 12:00 ET) and
> contains none of that prototype's code: everything here was written during the hackathon.

## How a world is made

One agent run per meditation, four Meta models, one of them open weights:

```mermaid
flowchart TD
    M[Meditation script] --> D[Muse Spark · art director<br/>designs the world as strict JSON]
    D --> P[Muse Image · painter<br/>paints it, 9:16]
    P --> C[Muse Spark Open vision · critic<br/>open weights, on Baseten; Muse Spark if it fails<br/>safe? calm? on brief? score]
    P --> S[SAM 3.1 · cutter<br/>one layer per subject]
    P --> E[Muse Image edit<br/>the scene without its moving subjects]
    P --> N[SAM 3.1 · what moves in nature<br/>water, sky, grass, a stream]
    C -- rejected --> R[Muse Spark · revises its design<br/>from the critic's note] --> P
    C -- accepted --> Show[the still painting is shown at once]
    S --> Plate
    E --> Plate[plate: the edit kept only where a moving layer is,<br/>tone-matched, checked clean by SAM 3.1]
    Plate --> World[living world: plate + layers + region map<br/>+ anchors + focus + timeline]
    N --> World
```

| Step | Model | What it does |
|---|---|---|
| Art director | `muse-spark-1.3` | Reads the meditation and designs one world in strict JSON: place, light, palette, air, 1–3 solid foreground subjects (how each is attached, how it moves, whether it glows), what moves in nature (water, sky, grass, a stream, trees), what drifts through the air (leaves, snow, pollen, fireflies) and the painting prompt. It is told never to copy the listener's words, and no picture carries text. |
| Painter | `muse-image-1.0` | Paints the world, portrait for a phone. Built-in web search and shell are off. |
| Critic | `meta-models/muse-spark-open-1026` (Muse Spark Open, open weights, served by Baseten; vision) | Looks at every painting before a listener does: people or faces, text, anything unsettling, on brief, a 0–10 score, and a one-sentence fix. It sees only the painting and the scenery design, never the meditation or the listener. If it fails, takes longer than 18 s or answers out of shape, `muse-spark-1.3` on Meta Model API judges instead. On our six example worlds both agreed on every safety check, and both rejected all four test pictures that must fail (a face, a sign with letters, a storm, a tiny figure). |
| Revision | `muse-spark-1.3` | A rejected painting goes back to the art director with the critic's note, and the agent paints once more (two paintings at most). |
| Cutter | `sam-3.1` | Cuts each designed subject out as its own layer: asks name and plural at once, keeps only the near instances, fills holes, and records the outline box. A layer is opaque two pixels past the outline, so it carries the subject's own soft edge; a floating boat's outline is closed over its oars, which then ride with it. |
| What moves | `sam-3.1` | Finds the water, the sky, the grass and a stream the design names, one request each, and packs them into one region map the phone's shader reads; still subjects are taken out of it, so a stone in a lake never ripples. A glowing subject becomes a light that flickers. |
| Plate | `muse-image-1.0` edit + `sam-3.1` | Repaints the scene without the moving subjects; the repaint is used only where a moving layer is opaque, colour-matched to its surroundings, and SAM checks that no moving subject is left. Otherwise a local fill there. A still subject (a stone, a post) stays exactly as painted, and with nothing moving no repaint is asked for. At rest, plate and layers equal the painting. |

The critique, the cutting, the repaint and the search for what moves only need the painting, so they run at the
same time; a rejection cancels the others. The still painting is shown as soon as the critic accepts it (21–70 s
after the script is written in our runs; the slower ones were made six at a time), and the living world follows
(59–104 s).

## A world from the first second: the library

A world made for a meditation takes about a minute, and listeners close their eyes early. So a session opens on
a **library world**: one of 120 worlds the same agent painted in advance from short themes (30 places guided
meditations visit most, each at four times of day, seasons or weathers; 18 of them dark worlds for falling
asleep; `scripts/library_briefs.py`), each checked by the critic like any other. While the session's own world
is being painted, Muse Spark reads the meditation (the same text the art director reads, at most its first
1,500 words) and names the three library worlds that fit best; the first one the listener has not seen lately
is shown (`library.py`, low reasoning effort: 2–7 s, about half a cent). A sleep meditation chooses only among
the sleep worlds. The phone shows the library world at once, moving, and the session's own world takes its
place in one slow cross-fade when it is complete; if it fails, the library world stays.

```bash
.venv/bin/python scripts/library_briefs.py                       # library/briefs/*.txt
.venv/bin/elynd-worlds make library/briefs/*.txt --out library/worlds --parallel 4
```

## On the phone

`flutter/elynd_worlds_scene` draws a world in a Flutter app:

* **A living painting.** A fragment shader (`shaders/living.frag`) reads the region map and moves each part as it
  moves in nature: ripples travel over the water towards the viewer and its reflections wobble, now and then the
  sun glints on it; the clouds drift; waves of wind roll through the grass; a stream flows; the crowns sway; a
  lantern flickers and its light with it. Leaves, snow, pollen or fireflies drift through the air.
* **Objects move as such things move.** A boat sinks a little on the swell, rises back to where it is painted and
  rolls; something in the air floats on a slow loop and a butterfly folds its wings now and then; plants sway
  in gusts about where they grow; mist drifts. Stones, posts and cabins stay still. Nothing grows or shrinks.
* **Eyes closed.** A per-second timeline follows the session: the whole world while the listener arrives
  (45 s), a glow while their eyes are closed, more of it in long silences, all of it at the end. Sleep worlds
  stay dim and end dark. The phone's (Dart) and the browser's (JavaScript) timelines are tested frame by frame
  against the Python one.
* **Phones crop.** Each world has a focus point (where its subjects are); the painting is cropped around it,
  and subjects painted too low are lifted above the player's controls with a small zoom.
* Reduce-motion is respected: the world then only fades.

## Privacy and safety

* Opt-in. Nothing reaches Meta until a listener turns Painted worlds on, on a screen that says plainly
  what is sent to whom (a one-time card on the home screen, or the switch in Profile, which also turns it
  off again). The server records that consent before it paints anything.
* In ELYND, only the written meditation goes to Meta, with its technique, language and length: no name, no
  email, no mood, no account id.
* Standard mode only. The package refuses `-contributor` models, which may train on what they receive.
* `store: false` on every Muse Spark and SAM request (the Responses API stores requests by default), and a
  pseudonymous listener id (a keyed hash Meta cannot reverse) so that any abuse enforcement targets one
  listener, not the whole app.
* No picture contains people, faces or text; the critic checks every painting.
* The critic on Baseten receives the painting and its scenery design only: no meditation, no listener id.
* Every picture a listener sees or shares is marked as made by AI in its metadata (the IPTC digital source
  type `trainedAlgorithmicMedia` that Muse Image puts on its paintings). The plate, the layers and the share
  card keep the mark (EU AI Act, Art. 50).

## Cost

Every step reports its usage to a meter priced with the published rates. A world costs about **$0.06**
($0.050–0.063 in our last six): two Muse Spark calls, two Muse Image pictures (one when nothing in it moves)
and a handful of SAM images. When Muse Spark Open judges, its share (half a cent to a cent) is Baseten's bill:
the meter keeps the two bills apart. The command line refuses to run past a budget and stops asking the open
model at a budget of its own; the ELYND backend stops making worlds at a total and a daily cap.

## See it in a browser

`web/` is a gallery of worlds the agent made for the example meditations. They live as they do on the phone:
the page compiles the phone's own shader for WebGL 2 and uses the same motion. **Play as a meditation** runs a
session sped up (arriving, eyes closed, a long silence, the end), and each world shows how the agent made it: the
design, the critic's score, the layers, what moves, the time and the cost. Serve the repository and open `/web/`:

```bash
python -m http.server 8000          # then http://localhost:8000/web/  (web/parity.html: the timeline check)
.venv/bin/python scripts/web_gallery.py out/<time>/<name> ...   # rebuild web/worlds/ from saved runs
```

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
export MODEL_API_KEY=...            # a Meta Model API key
export OPEN_MODEL_API_KEY=...       # optional: Muse Spark Open on Baseten as the critic
.venv/bin/elynd-worlds make examples/lake_visualization.txt   # writes out/<time>/<name>/world.json + pictures
.venv/bin/elynd-worlds sheet out/<time>                         # a contact sheet
.venv/bin/pytest                                                # tests, no network
```

To watch a world move without a phone:

```bash
cd flutter/elynd_worlds_scene
flutter test test/render_preview_test.dart --dart-define=WORLD_DIR=/abs/out/<time>/<name> \
  --dart-define=FRAMES_DIR=/abs/frames --dart-define=SILENT=true
ffmpeg -framerate 30 -i /abs/frames/%04d.png -pix_fmt yuv420p preview.mp4
```

## Layout

```
elynd_worlds/          the engine (Python)
  art_director.py      Muse Spark designs and revises
  painter.py           Muse Image paints and repaints without the subjects
  critic.py            Muse Spark Open (or Muse Spark) judges each painting
  library.py           Muse Spark chooses the library world a session opens on
  cutter.py            SAM 3.1 cuts the layers
  regions.py           SAM 3.1 finds the water, sky, grass and a stream
  plate.py             the clean plate behind the moving layers
  director.py          the agent loop
  timeline.py          how much of the world shows over the session
  card.py              the share card
  meta.py              the clients, privacy fields, cost meter
flutter/elynd_worlds_scene/   the scene on the phone (Dart, and the living painting's shader)
web/                   the scene in a browser (JavaScript, WebGL 2) and the gallery
scripts/               the gallery builder; replate.py rebuilds saved worlds' plates offline;
                       library_briefs.py writes the library's themes
library/briefs/        the 120 themes (the library's pictures are kept out of git)
examples/              example meditations written for this repository
tests/                 against a fake Meta Model API
```

© 2026 Comads OÜ.
