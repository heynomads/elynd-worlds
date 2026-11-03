"""The pipeline: how long each step takes, how often the first painting passes, how plates are made, what SAM could
not answer, and what a world costs. From the team's demo runs on the live server (evals/data/production.json) and
the library jobs (evals/data/library.json, evals/data/not_in_library.json).

The library jobs ran many worlds at once on each machine (the 2 Nov run: 6 workers with up to 3 worlds each on
4 CPUs), so their step times measure a busy batch, not one world; the demo runs are single worlds on the server.

    python evals/eval_pipeline.py             # the report
    python evals/eval_pipeline.py --check     # and compare with evals/results/pipeline.json
"""
from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from eval_common import load, main_guard, pct, quantile, rate, spread

SEQUENTIAL = ("design", "paint", "critique", "revise")   # the steps before the still painting, one after another
PARALLEL = ("cut", "regions", "remove subjects")          # after a painting, beside the critique


def kind(step: str) -> str:
    return step.rstrip("0123456789").strip()


def steps_of(w: dict[str, Any]) -> list[tuple[str, float, str]]:
    return [(kind(s), sec, note) for s, sec, note in w["trace"]]


def to_painting(w: dict[str, Any]) -> float:
    """Seconds from the start to the accepted painting: the sequential steps added up."""
    return sum(sec for k, sec, _ in steps_of(w) if k in SEQUENTIAL)


def job_summary(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"worlds": len(worlds)}
    per_step: dict[str, list[float]] = {}
    for w in worlds:
        for k, sec, note in steps_of(w):
            if note.startswith(("cancelled", "failed")) or k == "regions again":
                continue
            per_step.setdefault(k, []).append(sec)
    out["step_seconds"] = {k: spread(v) for k, v in sorted(per_step.items())}
    out["to_painting_s"] = spread([to_painting(w) for w in worlds])
    out["whole_world_s"] = spread([w["seconds"] for w in worlds])
    out["over_120_s"] = rate(sum(1 for w in worlds if w["seconds"] > 120), len(worlds))
    out["over_170_s"] = rate(sum(1 for w in worlds if w["seconds"] > 170), len(worlds))
    out["first_try_pass"] = rate(sum(1 for w in worlds if w["paintings"] == 1), len(worlds))
    out["painted_twice"] = sum(1 for w in worlds if w["paintings"] == 2)
    out["plates"] = dict(Counter(next(n for k, _, n in steps_of(w) if k == "plate") for w in worlds))
    # SAM: what moves (regions) and the cut-outs; failures are recorded in the trace from the 2 Nov run on.
    regions = [n for w in worlds for k, _, n in steps_of(w) if k == "regions" and not n.startswith("cancelled")]
    failed_phrases = Counter(p.strip() for n in regions if "failed:" in n for p in n.split("failed:")[1].split(","))
    out["sam"] = {
        "regions_steps": len(regions),
        "regions_with_a_failed_request": rate(sum(1 for n in regions if "failed:" in n), len(regions)),
        "failed_requests_by_phrase": dict(sorted(failed_phrases.items())),
        "nothing_found": sum(1 for n in regions if n.split(";")[0].strip() == "none"),
        "asked_again_later": sum(1 for w in worlds if any(k == "regions again" for k, _, _ in steps_of(w))),
        "cut_failed": sum(1 for w in worlds for k, _, n in steps_of(w) if k == "cut" and n.startswith("failed")),
        "cut_found_no_layer": sum(1 for w in worlds for k, _, n in steps_of(w) if k == "cut" and n == "no layers"),
    }
    out["muse_image_edit_failed"] = dict(Counter(
        n for w in worlds for k, _, n in steps_of(w) if k == "remove subjects" and n.startswith("failed")))
    out["sam_images_per_world"] = spread([w["cost"]["sam_images"] for w in worlds], 0)
    return out


def costs(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "median_cents": round(statistics.median(values) * 100, 2),
            "p10_cents": round(quantile(values, 0.1) * 100, 2), "p90_cents": round(quantile(values, 0.9) * 100, 2),
            "mean_cents": round(statistics.mean(values) * 100, 2), "total_usd": round(sum(values), 2)}


def compute() -> dict[str, Any]:
    prod = load("production.json")
    lib = load("library.json")["worlds"]
    out = load("not_in_library.json")
    server, engine = prod["server_settings"], prod["engine_defaults"]

    painted, chosen = [], []
    for r in prod["runs"]:
        if r["status"] == "painted":
            steps = {s: sec for s, sec, _ in r["trace"]}
            painted.append({
                "run": r["run"], "paintings": r["paintings"], "critic_score": r["critic"]["score"],
                "seconds_to_painting": r["seconds_to_painting"], "seconds_to_ready": r["seconds_to_ready"],
                "steps": steps, "slowest_after_painting": max((s for s in steps if kind(s) in PARALLEL), key=steps.get),
                "plate": r["plate"], "cost_cents": round(r["cost_usd"] * 100, 2),  # both bills: open_cost_usd is part of it
                "open_share_cents": round(r["open_cost_usd"] * 100, 2),
                "share_of_engine_world_timeout": round(r["seconds_to_ready"] / engine["world_timeout_s"], 2),
                "share_of_server_world_timeout": round(r["seconds_to_ready"] / server["world_timeout_s"], 2)})
        else:
            chosen.append({"run": r["run"], "library_fit": r["library_fit"],
                           "seconds_to_library_world": r["seconds_to_library_world"],
                           "cost_cents": round(r["cost_usd"] * 100, 2)})
    production = {"painted": painted, "library_chosen": chosen, "server_settings": server, "engine_defaults": engine,
                  "place": next(r["place"] for r in prod["runs"] if "place" in r)}

    jobs = {run: job_summary([w for w in lib if w["run"] == run]) for run in ("first", "second")}
    push = [w for w in lib if w["run"] == "push"]
    push_all = push + out["push_run"]["not_kept"]
    jobs["push"] = {
        "paintings_recorded": len(push_all), "kept": len(push),
        "whole_world_s": spread([w["seconds"] for w in push_all]),
        "design_s": spread([w["design_seconds"] for w in push_all]),
        "first_try_pass": rate(sum(1 for w in push_all if w["paintings"] == 1), len(push_all)),
        "plates": dict(Counter(w["plate"] for w in push_all)),
    }

    first_445 = [w["cost"]["usd"] for w in lib if w["run"] == "first"] + [d["cost_usd"] for d in out["rejudge_turned_down"]]
    cost = {
        "first_library_445": costs(first_445),
        "second_library_639": costs([w["cost"]["usd"] for w in lib if w["run"] == "second"]),
        "second_run_all_705": costs([w["cost"]["usd"] for w in lib if w["run"] == "second"]
                                    + [d["cost_usd"] for d in out["second_run"]["not_registered"]]),
        "push_19": costs([w["cost"]["usd"] for w in push]),
        "library_1101": costs([w["cost"]["usd"] for w in lib]),
        "open_critic_share_cents": spread([w["cost"]["open_usd"] * 100 for w in lib], 2),
        "demo_runs_painted_cents": [p["cost_cents"] for p in painted],
    }
    return {"production": production, "jobs": jobs, "cost": cost}


def report(r: dict[str, Any]) -> str:
    p, jobs, c = r["production"], r["jobs"], r["cost"]
    lines = ["THE PIPELINE",
             f"server settings: {p['server_settings']}; engine defaults: {p['engine_defaults']}"]
    for x in p["painted"]:
        lines.append(f"demo run {x['run']}: painting at {x['seconds_to_painting']} s, living world at "
                     f"{x['seconds_to_ready']} s ({x['share_of_engine_world_timeout']:.0%} of the engine's 120 s, "
                     f"{x['share_of_server_world_timeout']:.0%} of the server's 170 s); steps {x['steps']}; slowest after "
                     f"the painting: {x['slowest_after_painting']}; plate {x['plate']}; {x['cost_cents']} cents")
    for x in p["library_chosen"]:
        lines.append(f"demo run {x['run']}: a library world (fit {x['library_fit']}) at {x['seconds_to_library_world']} s, "
                     f"{x['cost_cents']} cents")
    for run in ("first", "second"):
        j = jobs[run]
        lines.append(f"library job, {run} run ({j['worlds']} worlds in the library): whole world median "
                     f"{j['whole_world_s']['median']} s, p90 {j['whole_world_s']['p90']} s, max {j['whole_world_s']['max']} s; "
                     f"to the painting median {j['to_painting_s']['median']} s")
        lines.append("  steps (median / p90 / max s): " + "; ".join(
            f"{k} {v['median']}/{v['p90']}/{v['max']}" for k, v in j["step_seconds"].items()))
        lines.append(f"  first-try pass {j['first_try_pass']['k']}/{j['first_try_pass']['n']} "
                     f"({pct(j['first_try_pass']['share'])}); plates {j['plates']}; SAM: regions with a failed request "
                     f"{j['sam']['regions_with_a_failed_request']['k']}/{j['sam']['regions_steps']} "
                     f"{j['sam']['failed_requests_by_phrase']}, asked again later {j['sam']['asked_again_later']}; "
                     f"Muse Image edit failed {j['muse_image_edit_failed']}")
    pj = jobs["push"]
    lines.append(f"push run: {pj['paintings_recorded']} recorded, {pj['kept']} kept; whole world median "
                 f"{pj['whole_world_s']['median']} s, max {pj['whole_world_s']['max']} s; first-try pass "
                 f"{pj['first_try_pass']['k']}/{pj['first_try_pass']['n']}")
    for k in ("first_library_445", "second_library_639", "second_run_all_705", "push_19", "library_1101"):
        v = c[k]
        lines.append(f"cost {k}: median {v['median_cents']} cents (p10 {v['p10_cents']}, p90 {v['p90_cents']}), "
                     f"total {v['total_usd']} USD")
    lines.append(f"cost of the demo runs that painted: {c['demo_runs_painted_cents']} cents; open critic's share per "
                 f"library world: median {c['open_critic_share_cents']['median']} cents")
    return "\n".join(lines)


if __name__ == "__main__":
    main_guard("pipeline", compute, report)
