# How ELYND Worlds works

For judges: which model sees what, the parts that were hard, and one world made on ELYND's production server, step
by step. What we measured, failures included, is in [`evals/`](evals/).

## Which model sees what

| Step | Model | Runs on | Receives | Returns |
|---|---|---|---|---|
| The place | Llama 3.3 70B, open weights (`place.py`) | Amazon Bedrock, in ELYND's own AWS account | the meditation, which can hold what a listener shared | a few sentences about the place it needs, checked in code (below) |
| Library pick | Muse Spark (`library.pick`, reasoning minimal) | Meta Model API | that description; technique, language, length; the library's short summaries | the three best library worlds, and whether the meditation needs its own world |
| Art director | Muse Spark (`art_director.py`, reasoning low) | Meta Model API | that description (on a revision: the critic's one-sentence fix) | the world's design, strict JSON in a fixed schema |
| Painter | Muse Image (`painter.py`) | Meta Model API | the painting prompt from the design | a 9:16 painting, marked as AI-made in its metadata |
| Critic | Muse Spark Open, open weights (`critic.py`, reasoning low) | Baseten | the painting and its scenery design only | safe, calm, on brief, figures, text; a 0–10 score; a one-sentence fix |
| Cutter, what moves, plate check | SAM 3.1 (`cutter.py`, `regions.py`, `plate.py`) | Meta Model API | the painting (or the plate) and the design's words for its subjects and nature | masks |
| Plate | Muse Image edit (`plate.py`) | Meta Model API | the painting and the subjects that move | the scene without them |
| Library feelings | Muse Spark vision (`library.tag_moods`) | Meta Model API, once per library world | the library painting at 384 px | one to three feelings it helps with |
| Journeys guide and plan | Llama 3.3 70B (`journeys/guide.py`, `planner.py`) | Amazon Bedrock, in ELYND's own AWS account | the listener's answers | the questions, a title, the thirty days |
| A journey's world and climb | Muse Image edits (`stages.py`, `scripts/climb/`) | Meta Model API | a description written by hand, never a listener's words | each stage, judged next to the first for showing the same place (`same_place`) |

Not part of this entry: ELYND's meditation writer and voices, which existed before the hackathon and are not Meta
models. Run once, offline, not at runtime: Depth Anything V2 Small, for the climb's depth maps.

## The parts that were hard

1. **A painting that moves without being redrawn.** A world is a plate, cut-out layers and a region map. The plate is
   the scene without its moving subjects: Muse Image repaints it, the repaint is kept only where a moving layer is
   opaque and is colour-matched to its surroundings, and SAM 3.1 checks that no moving subject is left; otherwise a
   local fill. At rest, plate and layers equal the painting
   (`tests/test_engine.py::test_plate_plus_layers_at_rest_is_the_painting`), so the listener first sees exactly what
   the critic approved.
2. **SAM's masks drive the motion.** `regions.py` asks SAM 3.1 for the water, the sky, the grass and a stream the design
   names, takes still subjects out of them (a stone in a lake never ripples) and packs them into one region map. The
   fragment shader reads it: ripples, reflections, drifting clouds, wind in the grass. The phone (Flutter) and the
   browser (WebGL 2) run the same shader file (`flutter/elynd_worlds_scene/shaders/living.frag` and
   `web/living.frag` are identical).
3. **A boundary between the listener and Meta's API.** Meta's processor terms exclude health data, and a meditation
   can repeat what a listener shared. So the meditation goes only to Llama 3.3 in ELYND's own AWS account, which
   describes the place it needs. `place.py` refuses a description that speaks of a person, names health, feelings or
   the body, takes a name from the meditation or copies a run of its words; it asks once more, and otherwise the
   library world stays (`tests/test_place.py`). This is a filter designed so that nothing of the listener reaches
   Meta, not a proof; it is why, of the meditation, only a short description of a place (with its technique,
   language and length) ever leaves ELYND.
4. **Parallel steps that fail gracefully.** The critique, the cutting, the plate and the search for what moves only
   need the painting, so they run at the same time (`director.py`). A rejection cancels the others and sends the
   critic's fix to the art director for one more painting (`test_a_rejected_painting_is_revised_and_painted_again`,
   `test_two_rejections_mean_no_world`); a failed step only makes the world stiller
   (`test_a_failed_repaint_falls_back_to_the_local_fill`). A painting that no allowed judge could check is never
   shown: the listener keeps a library world.
5. **One timeline in three languages.** How much of the world shows, second by second over a session (arriving, eyes
   closed, long silences, the end), is computed in Python (`timeline.py`), Dart and JavaScript, and the three are
   checked against each other frame by frame (`flutter/elynd_worlds_scene/test/scene_test.dart`, `web/parity.html`).
6. **A journey's place that grows.** A journey's world is painted once, then repainted four times from the same
   viewpoint in another light and season; the critic judges each stage next to the first for showing the same place
   (`stages.py`, `tests/test_stages.py`).

## One world on production, step by step

A personal world made on ELYND's production server on 2 Nov 2026 (13:03 CET), through the app's own API, with the
team's test account and our own words: *"Help me picture my grandmother's courtyard … her red watering can on the
step, the old olive tree, and my old dog Bruno asleep in its shade in the late afternoon."*

| Step | Took | What happened |
|---|---|---|
| Llama 3.3: the place | 1.3 s | "A courtyard in late afternoon, warm golden light, with a red watering can, an old olive tree, and a dog asleep in its shade." No name and no door: what Llama leaves out, Meta never sees. (Production does not store this description; this is the same step re-run on the same text, $0.0006.) |
| Muse Spark: library pick | | A ready-made golden courtyard is on screen 10.8 s after the request (fit 9 of 10), and Muse Spark still asks for a world of its own: the meditation is built around a particular person's place and its own things. |
| Muse Spark: art director | 8.6 s | "Golden Courtyard Rest": a sleeping dog (still, left), a red watering can (still, right), sky and trees that move, golden hour. |
| Muse Image: painter | 17.8 s | One painting. |
| Muse Spark Open: critic | 14.6 s | 9 of 10: "Painting matches the designed golden-hour courtyard with olive tree, warm light, sleeping dog on the left and red watering can on the right clearly visible and calm." |
| **Shown at 44.4 s** | | The painting replaces the library world. These times are on the world agent's own clock, from its start once the meditation was written (the server recorded the painting 54.8 s and the finished world 111.4 s after the request); the steps below started with the critique. |
| SAM 3.1: what moves | 59.2 s | The sky and the olive's leaves. |
| SAM 3.1: cutter | 77.7 s | Tried the dog and the can; both are still by design, so nothing is cut and no plate is needed. |
| **Alive at 109.6 s** | | The world moves through its region map: the sky drifts, the leaves sway, a flock passes. |

Cost: 3.8¢ in all: 3.5¢ at Meta and 0.3¢ at Baseten. A painted world costs about 4 cents (the median over the first 445
library worlds is 4.2¢, over all 1,101 3.8¢); the first 445 library worlds cost about $20.

## Production settings

The ELYND server runs the engine with these settings (this repository's defaults differ where noted):

- Only Muse Spark Open may judge a new painting (`open_critic_only`; off by default here, so the examples run with one
  key). It gets 90 s; a painting it cannot judge in time is never shown.
- A world gets 170 s (the engine's default is 120 s), a revision included, within the 3 minutes the app waits for it.
- At most two paintings (the first and one revision); a painting needs at least 7 of 10 from the critic.
- Muse Spark designs and Muse Spark Open judges with reasoning set to low; the library pick uses minimal reasoning.
- Personal worlds are off until a listener turns them on, and each listener's own worlds have a monthly limit; the
  server stops making worlds at a total and a daily cap.
