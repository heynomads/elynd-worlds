"""All evals at once, from the committed data in evals/data/ (no model call, no network).

    python evals/run_evals.py            # every report
    python evals/run_evals.py --check    # and compare each with evals/results/*.json (exit 1 if one differs)
    python evals/run_evals.py --write    # write evals/results/*.json again
    python evals/run_evals.py --table    # the summary table at the top of evals/README.md
"""
from __future__ import annotations

import json
import sys
from typing import Any

import eval_critic
import eval_designer
import eval_pipeline
from eval_common import committed, pct, write

EVALS = {"critic": eval_critic, "designer": eval_designer, "pipeline": eval_pipeline}


def cents(x: float) -> str:
    return f"{x:.1f} ¢"


def table(results: dict[str, dict[str, Any]] | None = None) -> str:
    """The README's summary table, written from the results so the two cannot drift apart."""
    r = results or {name: m.compute() for name, m in EVALS.items()}
    c, d, p = r["critic"], r["designer"], r["pipeline"]
    lib, loop, rj, kb = c["library"], c["in_loop"], c["rejudge"], c["known_bad"]
    s2 = c["against_look_and_code"]["second_run"]
    sc = lib["scores"]
    ft = loop["first_try_pass"]
    a_s, a_o, pair = d["arms"]["spark_default"], d["arms"]["open_low"], d["paired"]
    prod = {x["run"]: x for x in p["production"]["painted"]}
    obs, yard = prod["observatory"], prod["courtyard"]
    j1, j2 = p["jobs"]["first"], p["jobs"]["second"]
    cost = p["cost"]
    diff = pair["open_critic_diff_open_minus_spark"]
    rows = [
        ("Critic: the library", "final verdict on every library world", f"{lib['worlds']:,}",
         f"all judged by Muse Spark Open ({lib['judged_by_open_at_creation']} at creation, "
         f"{lib['judged_again_by_open']} again on 2 Nov); scores 7/8/9/10: {sc.get('7', 0)}/{sc.get('8', 0)}/"
         f"{sc.get('9', 0)}/{sc.get('10', 0)} ({pct(lib['share_scored_9']['share'])} at 9)",
         "only passes are in the library; the scores bunch at 9 and say little above the pass mark (7)"),
        ("Critic: turned down in the jobs", "first paintings sent back to the art director",
         f"{ft['first']['n'] + ft['second']['n']:,}",
         f"{loop['first_painting_turned_down']} turned down, painted again, then passed; first try "
         f"{pct(ft['first']['share'])} (26-27 Oct), {pct(ft['second']['share'])} (2 Nov)",
         f"{loop['score_0']} of {loop['first_painting_turned_down']} carry score 0, which is how a missing score reads"),
        ("Critic: against the first critic", "the same paintings, muse-spark-1.3 then Muse Spark Open",
         str(rj["paintings"]),
         f"{rj['turned_down_by_open']['k']} turned down ({pct(rj['turned_down_by_open']['share'])}, 95 % CI "
         f"{rj['turned_down_by_open']['ci95'][0] * 100:.1f}-{rj['turned_down_by_open']['ci95'][1] * 100:.1f} %); "
         f"same score {rj['same_score']['k']}, within one point {rj['within_one']['k']}",
         "all had passed the first critic; 2 of the 3 turn-downs follow a rule change (a designed butterfly)"),
        ("Critic: known bad paintings", "a seated person the design did not bring", str(kb["paintings"]),
         f"{kb['open_turned_down']['k']} of {kb['paintings']} turned down (3-4/10; 95 % CI "
         f"{kb['open_turned_down']['ci95'][0] * 100:.1f}-100 %)",
         "one kind of fault, 14 paintings; muse-spark-1.3 had passed all 14 under the rule of 27 Oct"),
        ("Critic: against a look and the code", "the 2 Nov run: every picture looked at; looks_dead",
         str(s2["painted"]),
         f"the look turned down {s2['passed_by_critic_turned_down_by_look']['k']} the critic passed "
         f"({pct(s2['passed_by_critic_turned_down_by_look']['share'])}); the critic passed all "
         f"{s2['dead_by_code']} the code found dead",
         "the look was by AI reviewer agents, not a person; the critic does not judge motion"),
        ("Designer", "Muse Spark vs Muse Spark Open designing; both critics; failures counted",
         str(d["meditations"]["n"]),
         f"designed {a_s['designed']['k']}/20 vs {a_o['designed']['k']}/20; paired score {diff['mean']:+.2f} "
         f"(95 % CI {diff['ci95'][0]:+.2f} to {diff['ci95'][1]:+.2f}); design "
         f"{d['speed']['open_vs_spark_default']:.1f}x as fast as Muse Spark at default reasoning, "
         f"{d['speed']['open_vs_spark_low']:.2f}x as fast as Muse Spark at reasoning low",
         "20 meditations; the low-reasoning run kept only its summary, so no interval there"),
        ("Pipeline: the live server", "the team's demo runs, single worlds", "2 painted, 2 library",
         f"painting at {obs['seconds_to_painting']:.0f} / {yard['seconds_to_painting']:.0f} s, living world at "
         f"{obs['seconds_to_ready']:.0f} / {yard['seconds_to_ready']:.0f} s "
         f"({yard['share_of_engine_world_timeout']:.0%} of the engine's 120 s timeout; the server allows 170 s); "
         f"{cents(obs['cost_cents'])} / {cents(yard['cost_cents'])}",
         "two runs"),
        ("Pipeline: the library jobs", "step times, plates, SAM, cost", f"{j1['worlds'] + j2['worlds']:,}",
         f"plate: no moving subject {j1['plates'].get('painting', 0) + j1['plates'].get('none', 0) + j2['plates'].get('painting', 0) + j2['plates'].get('none', 0)}, "
         f"Muse Image edit {j1['plates'].get('muse_image', 0) + j2['plates'].get('muse_image', 0)}, local fill "
         f"{j1['plates'].get('filled', 0) + j2['plates'].get('filled', 0)}; SAM left a request unanswered in "
         f"{j2['sam']['regions_with_a_failed_request']['k']} of {j2['worlds']} (2 Nov); cost median "
         f"{cents(cost['first_library_445']['median_cents'])} (first 445, {cost['first_library_445']['total_usd']:.2f} USD), "
         f"{cents(cost['second_library_639']['median_cents'])} (2 Nov)",
         "many worlds at once per machine, so step times are a busy batch's; SAM failures recorded from 2 Nov only"),
        ("Place check", "Llama 3.3's place description, refused by code", "-",
         "no records", "the place step does not log the descriptions it turns down"),
    ]
    head = "| Eval | What was measured | n | Result | Limits |\n|---|---|---|---|---|\n"
    return head + "\n".join("| " + " | ".join(row) + " |" for row in rows)


def md(head: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(row) + " |" for row in rows])


def designer_table(r: dict[str, Any]) -> str:
    d = r["designer"]
    rows = []
    for key, name in (("spark_default", "Muse Spark, the model decides how long it thinks"),
                      ("open_low", "Muse Spark Open, reasoning low")):
        a = d["arms"][key]
        oc, sc, dz = a["open_critic"], a["spark_critic"], a["designed"]
        rows.append([name, f"{dz['k']} of 20 ({dz['ci95'][0] * 100:.0f}-{dz['ci95'][1] * 100:.0f} %)",
                     f"{oc['passed_of_all']['k']} / {sc['passed_of_all']['k']}",
                     f"{oc['mean_of_designed']:.2f} / {sc['mean_of_designed']:.2f}",
                     f"{oc['mean_failures_as_0']:.2f} / {sc['mean_failures_as_0']:.2f}",
                     f"{a['design_s']['median']} s", f"{a['cost_usd']['median_per_world'] * 100:.1f} ¢"])
    lo = d["spark_low"]
    rows.append(["Muse Spark, reasoning low (the default since; summary only)", f"{lo['designed']['k']} of 20",
                 f"{lo['open_critic_passed_of_all']['k']} / {lo['spark_critic_passed_of_all']['k']}",
                 f"{lo['open_critic_mean']:.2f} / {lo['spark_critic_mean']:.2f}",
                 f"{lo['open_critic_mean']:.2f} / {lo['spark_critic_mean']:.2f}", f"{lo['design_s_median']} s", "-"])
    return md(["Designer", "Designed", "Passed of 20 (open / Muse Spark critic)", "Mean score of those designed",
               "Mean, a failure counted as 0", "Design time (median)", "Cost per world (median)"], rows)


def production_table(r: dict[str, Any]) -> str:
    names = {"design": "design", "paint 1": "paint", "critique 1": "critique", "cut 1": "cut",
             "regions 1": "what moves", "plate": "plate"}
    rows = []
    for x in r["pipeline"]["production"]["painted"]:
        steps = ", ".join(f"{names.get(k, k)} {v:.1f}" for k, v in x["steps"].items())
        rows.append([x["run"], f"{x['seconds_to_painting']} s", f"{x['seconds_to_ready']} s",
                     f"{x['share_of_engine_world_timeout']:.0%} / {x['share_of_server_world_timeout']:.0%}", steps,
                     f"{x['critic_score']}/10, first painting", f"{x['cost_cents']:.1f} ¢"])
    return md(["Demo run", "To the painting", "To the living world", "Of the 120 s / 170 s world limit",
               "Steps (s)", "Critic", "Cost"], rows)


def jobs_table(r: dict[str, Any]) -> str:
    j1, j2, c = r["pipeline"]["jobs"]["first"], r["pipeline"]["jobs"]["second"], r["pipeline"]["cost"]

    def s(j: dict[str, Any], *keys: str) -> str:
        return " / ".join(f"{j['step_seconds'][k]['median']:.0f}" for k in keys) + " s"

    def plates(j: dict[str, Any]) -> str:
        p = j["plates"]
        return f"{p.get('painting', 0) + p.get('none', 0)} / {p.get('muse_image', 0)} / {p.get('filled', 0)}"

    def sam(j: dict[str, Any]) -> str:
        f = j["sam"]["regions_with_a_failed_request"]
        if not f["k"]:
            return "not recorded then"
        top = ", ".join(f"{k} {v}" for k, v in sorted(j["sam"]["failed_requests_by_phrase"].items(), key=lambda kv: -kv[1]))
        return f"{f['k']} worlds ({pct(f['share'])}): {top}; {j['sam']['asked_again_later']} asked again later"

    def edit(j: dict[str, Any]) -> str:
        n = sum(j["muse_image_edit_failed"].values())
        return f"{n} (timeouts; the plate was filled locally)" if n else "0"

    rows = [
        ["worlds in the library", f"{j1['worlds']}", f"{j2['worlds']}"],
        ["whole world, median / p90 / slowest",
         *(f"{j['whole_world_s']['median']:.0f} / {j['whole_world_s']['p90']:.0f} / {j['whole_world_s']['max']:.0f} s"
           for j in (j1, j2))],
        ["to the painting, median", *(f"{j['to_painting_s']['median']:.0f} s" for j in (j1, j2))],
        ["design / paint / critique, median", s(j1, "design", "paint", "critique"), s(j2, "design", "paint", "critique")],
        ["cut / what moves, median (slowest)",
         *(f"{s(j, 'cut', 'regions')} ({j['step_seconds']['cut']['max']:.0f} / {j['step_seconds']['regions']['max']:.0f} s)"
           for j in (j1, j2))],
        ["first painting passed", *(f"{j['first_try_pass']['k']} ({pct(j['first_try_pass']['share'])})" for j in (j1, j2))],
        ["plate: nothing moves / Muse Image edit / local fill", plates(j1), plates(j2)],
        ["SAM request unanswered after 3 tries", sam(j1), sam(j2)],
        ["Muse Image edit (the scene without its subjects) failed", edit(j1), edit(j2)],
        ["cost per world, median (p10-p90)",
         *(f"{v['median_cents']:.2f} ¢ ({v['p10_cents']:.2f}-{v['p90_cents']:.2f})"
           for v in (c["first_library_445"], c["second_library_639"]))],
        ["cost in all", f"{c['first_library_445']['total_usd']:.2f} USD for the first 445",
         f"{c['second_library_639']['total_usd']:.2f} USD ({c['second_run_all_705']['total_usd']:.2f} with the 66 not kept)"],
    ]
    return md(["", "First library (26-27 Oct)", "Second library (2 Nov)"], rows)


def tables(results: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {"summary": table(results), "designer": designer_table(results), "production": production_table(results),
            "jobs": jobs_table(results)}


def main(argv: list[str]) -> int:
    results = {name: m.compute() for name, m in EVALS.items()}
    if "--table" in argv:
        for name, text in tables(results).items():
            print(f"<!-- {name} -->\n{text}\n")
        return 0
    status = 0
    for name, m in EVALS.items():
        print(m.report(results[name]), end="\n\n")
        if "--write" in argv:
            write(name, results[name])
        if "--check" in argv:
            same = json.loads(json.dumps(results[name])) == committed(name)
            print(f"results/{name}.json: {'the same' if same else 'DIFFERENT'}\n")
            status |= 0 if same else 1
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
