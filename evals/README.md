# Evaluations

How well the parts of ELYND Worlds work, measured on the records their runs left behind: every library world's own
record (the critic's verdict, each step's time, the cost), the designer test of 28 Oct 2026, and the team's demo
runs on ELYND's live server. Nothing here calls a model or the network: every number on this page is computed from
`evals/data/` by the scripts in this folder, and the tests check that the page and `evals/results/` match them.

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/python evals/run_evals.py           # every report (or one: evals/eval_critic.py, eval_designer.py, eval_pipeline.py)
.venv/bin/python evals/run_evals.py --check   # and compare with evals/results/*.json
.venv/bin/pytest evals                        # the same, and the tables on this page
```

| Eval | What was measured | n | Result | Limits |
|---|---|---|---|---|
| Critic: the library | final verdict on every library world | 1,101 | all judged by Muse Spark Open (984 at creation, 117 again on 2 Nov); scores 7/8/9/10: 1/10/884/206 (80.3 % at 9) | only passes are in the library; the scores bunch at 9 and say little above the pass mark (7) |
| Critic: turned down in the jobs | first paintings sent back to the art director | 1,082 | 28 turned down, painted again, then passed; first try 95.3 % (26-27 Oct), 98.9 % (2 Nov) | 9 of 28 carry score 0, which is how a missing score reads |
| Critic: against the first critic | the same paintings, muse-spark-1.3 then Muse Spark Open | 120 | 3 turned down (2.5 %, 95 % CI 0.9-7.1 %); same score 85, within one point 117 | all had passed the first critic; 2 of the 3 turn-downs follow a rule change (a designed butterfly) |
| Critic: known bad paintings | a seated person the design did not bring | 14 | 14 of 14 turned down (3-4/10; 95 % CI 78.5-100 %) | one kind of fault, 14 paintings; muse-spark-1.3 had passed all 14 under the rule of 27 Oct |
| Critic: against a look and the code | the 2 Nov run: every picture looked at; looks_dead | 705 | the look turned down 9 the critic passed (1.3 %); the critic passed all 56 the code found dead | the look was by AI reviewer agents, not a person; the critic does not judge motion |
| Designer | Muse Spark vs Muse Spark Open designing; both critics; failures counted | 20 | designed 20/20 vs 18/20; paired score +0.11 (95 % CI -0.17 to +0.44); design 3.0x as fast as Muse Spark at default reasoning, 1.35x as fast as Muse Spark at reasoning low | 20 meditations; the low-reasoning run kept only its summary, so no interval there |
| Pipeline: the live server | the team's demo runs, single worlds | 2 painted, 2 library | painting at 38 / 44 s, living world at 78 / 110 s (91% of the engine's 120 s timeout; the server allows 170 s); 3.7 ¢ / 3.8 ¢ | two runs |
| Pipeline: the library jobs | step times, plates, SAM, cost | 1,082 | plate: no moving subject 851, Muse Image edit 210, local fill 21; SAM left a request unanswered in 88 of 639 (2 Nov); cost median 4.2 ¢ (first 445, 20.12 USD), 3.5 ¢ (2 Nov) | many worlds at once per machine, so step times are a busy batch's; SAM failures recorded from 2 Nov only |
| Place check | Llama 3.3's place description, refused by code | - | no records | the place step does not log the descriptions it turns down |

95 % intervals: Wilson for shares, a bootstrap (10,000 resamples, fixed seed) for differences and ratios.

## The data

* `data/library.json`: every world in the library on 3 Nov 2026, 1,101 of them, 159 for sleep: 443 of the first
  library (26-27 Oct, 89 places), 639 of the second (2 Nov, 119 places) and 19 worlds for a push painted again on
  29-30 Oct (each its own place). For each: the critic's final verdict (and the first one, when it was judged
  again), each step's time from the run's trace, how the plate was made, what SAM found or could not answer, the
  cost from the run's meter, and the scene fields the code check reads. No pictures. The 19 push worlds kept only
  summary fields.
* `data/not_in_library.json`: paintings the jobs made that are not in the library, without ids: 66 of the 2 Nov run,
  6 of the push run, the 2 turned down when judged again, and 14 with a seated person (below).
* `data/designer.json`: the 20 test meditations (written for the test; no listener's words), both designers'
  results, the reasoning-low run's summary, and the six example meditations of `examples/`.
* `data/production.json`: the team's own demo runs through the app on the live server, 1-2 Nov 2026 (timings,
  costs, scores, model names and Llama's place sentence only), and the settings the server runs the engine with.

The files were exported on 3 Nov 2026 from the runs' own records and the jobs' lists of what was kept; the export
read the team's private storage and is not part of this repository.

## a. The critic

Muse Spark Open (open weights, on Baseten) sees the painting and its scenery design, never the meditation, and
answers: belongs, calm, a figure that should not be there, text or symbols, a score of 0-10. A painting passes with
all four right and a score of at least 7 (`critic.passes`). `eval_critic.py`.

* **Every library world has its verdict.** 984 were judged by Muse Spark Open when they were made. The other 117
  had been judged by muse-spark-1.3, the engine's fallback when the open critic does not answer in time (45 s in
  the batch jobs, with many worlds painting at once) or in shape; on 2 Nov Muse Spark Open judged every one of them
  again (34 of the first library, 83 of the second), and its verdict is the one kept.
* **The scores bunch at 9.** 884 of 1,101 (80.3 %), mean 9.18, the lowest 7. The library holds only paintings that
  passed, so these scores cannot show how well the critic tells good from bad, and above the pass mark the score
  says little. What it turns down shows more:
* **Turned down in the jobs: 28 first paintings** (21 of 443, then 7 of 639), each sent back to the art director with
  the critic's note, painted again and passed. Reasons: a barn mirrored upside down in the snow, sailboats and a
  stream the design did not have, harsh moonbeams, a broken jetty that felt sad, fireflies in a world that shows the
  place alone, a tiny person on a road, a cup and a basket the size of the hills, a band of water and grass mirrored
  at the bottom. 9 of the 28 carry a score of 0, which is how the engine reads a missing score; 8 of those 9
  reasons describe a painting that matches its design (read by hand), so they look like answers without a score
  rather than judgements. All 28 reasons are in `results/critic.json`.
* **Against the first critic: 120 paintings** muse-spark-1.3 had passed and Muse Spark Open judged again. It turned
  down 3 (2.5 %, 95 % CI 0.9-7.1 %): two whose design brought a white butterfly (4/10; the place-alone rule of
  29 Oct turns down an animal the brief does not bring, so this is the rule, not the picture) and one broken
  composition, a strip of meadow floating above the sea (8, then 4). The same score on 85, within one point on 117;
  a mean 0.23 lower (95 % CI 0.09-0.40). None of the three is in the library.
* **Paintings known to be wrong: 14** from the first library run (27 Oct) in which Muse Image had added a seated
  person the design did not bring. muse-spark-1.3 had passed all 14 at 9/10 under that day's rule, which accepted a
  small figure at rest, and its notes never mention the person. On 30 Oct Muse Spark Open, with the place-alone
  rule, turned down all 14 (3 or 4 of 10, a wrong figure each time). One kind of fault, 14 paintings. None of them
  is in the library.
* **Against a look at every picture and the code check (2 Nov run, 705 paintings).** After the critic, AI reviewer
  agents (a general-purpose vision model, not one of Meta's and not part of the engine) looked at every picture:
  whole paintings, and the middle band at actual size, where small figures hide; doubtful ones were looked at
  again at full size. They flagged 10. Of the 704 the critic passed, they turned down 9 (1.3 %): 4 posts or
  signposts that read as a cross, 3 broken pictures (a hard seam over a flat strip), 2 tiny marks that might be a
  figure or a perched bird. `library.looks_dead` found 56 that would not move; the critic had passed all 56 (9 or 10
  of 10), and the code passed the one painting the critic turned down. They check different things, so all three
  run; none stands in for another. In the push run (25 worlds made for 20 themes) the critic passed all 25, a look
  turned down 2 (a rock face that read as a face, a signpost that read as a cross) and the code 4.
* **What was checked by whom.** `scripts/library_audit.py` runs `library.looks_dead` only: code, no look at a
  picture. The recorded looks at the pictures were made by the AI reviewer agents above, for the 2 Nov run on all
  705. For the first library there are no per-picture records of a look in this data; its 14 seated people were
  found by looking at the pictures on 29 Oct, after those worlds had gone in. There is no record of a person looking
  at every library painting.
* **The rule each verdict used.** 20 first-library worlds in the library have a designed white butterfly. They were
  judged on 26-27 Oct under that day's rule, which let a design bring an animal; the place-alone rule would turn
  them down, as it did the two such worlds that were judged again.
* **The code check today.** `library.looks_dead`, run again on the committed records of 1,082 library worlds: none
  looks dead (the 19 push worlds kept no region coverage; their run turned down 4 with the same check).

## b. The designer

28 Oct 2026: 20 meditations written for the test (8 English, 4 German, 4 Spanish, 4 French; 8 moods, 5 techniques),
each designed once by Muse Spark (reasoning: the model decides, the engine's setting that day) and once by Muse
Spark Open (reasoning low), painted by Muse Image, and every painting judged by both critics. A design that failed
counts as a failure: no world, not passed. `eval_designer.py`.

| Designer | Designed | Passed of 20 (open / Muse Spark critic) | Mean score of those designed | Mean, a failure counted as 0 | Design time (median) | Cost per world (median) |
|---|---|---|---|---|---|---|
| Muse Spark, the model decides how long it thinks | 20 of 20 (84-100 %) | 20 / 19 | 8.60 / 8.85 | 8.60 / 8.85 | 14.5 s | 3.6 ¢ |
| Muse Spark Open, reasoning low | 18 of 20 (70-97 %) | 18 / 18 | 8.78 / 8.94 | 7.90 / 8.05 | 4.75 s | 2.7 ¢ |
| Muse Spark, reasoning low (the default since; summary only) | 20 of 20 | 20 / 20 | 8.75 / 8.95 | 8.75 / 8.95 | 6.4 s | - |

* **Failures.** Muse Spark Open designed 18 of 20: its host's content filter refused one request (a desert under
  the stars) and one design came back without a title. Means over the 18 it designed hide the two; counted as
  failures, it passed 18 of 20 with either critic.
* **Scores.** On the 18 meditations both designed, the open designer's worlds scored 0.11 higher with either critic
  (95 % CI −0.17 to +0.44 with the open critic, −0.17 to +0.50 with Muse Spark), the same score on 13 and 16 of 18.
  No difference shows; with 18 pairs, one under about half a point would not show either.
* **Speed.** Muse Spark Open designed in a median 4.75 s. Muse Spark took 3.0 times as long when it decided how long
  to think (95 % CI 2.6-4.0, slower on 18 of 18), but 1.35 times as long at reasoning low (6.4 s), the engine's
  default since that day. The reasoning-low run kept only its summary (means, pass counts, the median time), not a
  row per meditation, so it has no interval, and it ran an hour after the other two.
* **Who judged.** 3 of the 38 "open critic" verdicts came from muse-spark-1.3, the engine's fallback when the open
  critic does not answer in time (1 for Muse Spark's designs, 2 for the open model's).
* **The six example meditations** (`examples/`, one world each per designer): mean score 8.67 (Muse Spark) and 8.83
  (Muse Spark Open), mean cost 5.5 ¢ and 4.0 ¢, painted twice 2 and 1 times. Six is too few to rank them.

## c. The pipeline

`eval_pipeline.py`. **On the live server** (the team's demo runs, 1-2 Nov 2026), single worlds, with the settings
ELYND's server uses: only Muse Spark Open may judge (`open_critic_only`), its time limit 90 s instead of the engine's
18 s, the world's 170 s instead of 120 s; reasoning low for design and critic, at most two paintings, pass score 7.

| Demo run | To the painting | To the living world | Of the 120 s / 170 s world limit | Steps (s) | Critic | Cost |
|---|---|---|---|---|---|---|
| observatory | 37.73 s | 77.91 s | 65% / 46% | design 4.8, paint 22.0, critique 8.3, cut 34.0, what moves 46.7, plate 1.4 | 9/10, first painting | 3.7 ¢ |
| courtyard | 44.43 s | 109.55 s | 91% / 64% | design 8.6, paint 17.8, critique 14.6, what moves 59.2, cut 77.7, plate 1.6 | 9/10, first painting | 3.8 ¢ |

Both passed at the first painting. After the painting, SAM is the slow part: cutting the subjects and finding what
moves took 34-78 s, the critique 8-15 s. The courtyard's living world came at 109.55 s, 91 % of the engine's
default `world_timeout` (120 s), 10 s from timing out with that default; the server allows 170 s.
Two more demo runs ended on a library world (a beach that fits, 10 of 10) in 6.5 and 6.9 s, for 0.7 and 0.4 ¢,
with nothing painted. For the courtyard, Llama 3.3's place sentence (the place step run again on the same text; the
server does not store the one it used) was: "A courtyard in late afternoon, warm golden light, with a red watering
can, an old olive tree, and a dog asleep in its shade." (1.3 s, $0.0006).

**The library jobs** made many worlds at once on each machine, with no time limit per world; the 2 Nov run was
CPU-bound (6 workers with up to 3 worlds each on 4 CPUs), so steps waited for one another. These are a busy batch's
times, not one world's.

|  | First library (26-27 Oct) | Second library (2 Nov) |
|---|---|---|
| worlds in the library | 443 | 639 |
| whole world, median / p90 / slowest | 157 / 219 / 318 s | 256 / 357 / 648 s |
| to the painting, median | 57 s | 66 s |
| design / paint / critique, median | 11 / 19 / 18 s | 8 / 16 / 26 s |
| cut / what moves, median (slowest) | 93 / 89 s (190 / 201 s) | 172 / 197 s (324 / 571 s) |
| first painting passed | 422 (95.3 %) | 632 (98.9 %) |
| plate: nothing moves / Muse Image edit / local fill | 321 / 108 / 14 | 530 / 102 / 7 |
| SAM request unanswered after 3 tries | not recorded then | 88 worlds (13.8 %): sky 64, grass 41, water 20, stream 8, grassy ground 5, heather 1; 27 asked again later |
| Muse Image edit (the scene without its subjects) failed | 10 (timeouts; the plate was filled locally) | 0 |
| cost per world, median (p10-p90) | 4.24 ¢ (3.54-5.69) | 3.55 ¢ (3.12-4.72) |
| cost in all | 20.12 USD for the first 445 | 23.84 USD (26.08 with the 66 not kept) |

* **First try.** The first painting passed for 95.3 % (95 % CI 92.9-96.9 %) and 98.9 % (97.8-99.5 %) of the
  worlds in the library; on 2 Nov no world failed both paintings (all 705 have a record). The push run passed 24
  of 25 at the first painting.
* **Plates.** Most worlds have no moving subject, so the plate is the painting itself. Where a subject moves, the
  Muse Image edit was used 210 times and a local fill 21 times (10 of those after the edit timed out).
* **SAM.** On 2 Nov, 88 worlds (13.8 %) had a request for what moves that SAM did not answer after three tries (sky
  64, grass 41, water 20); that part stays still, and 27 were asked again later. The first library's runs did not
  yet record such failures.
* **Cost.** A painted world costs about 4 cents: a median 4.2 ¢ in the first library (the first 445 worlds:
  20.12 USD), 3.5 ¢ in the second, 3.7 and 3.8 ¢ for the demo runs on the server (each cost is both bills: Meta's and Muse Spark Open's host's). Muse Spark Open's share is a
  median 0.3 ¢ a world.

## d. The place check

No records: the place step (`place.py`, Llama 3.3 describing the place a meditation needs) does not log the
descriptions its code check refuses, so its reject rate cannot be computed from existing data. Skipped.
