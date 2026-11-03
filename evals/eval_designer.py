"""The designer: Muse Spark against Muse Spark Open as the art director, on 20 test meditations (28 Oct 2026).

Per meditation, both designs were painted by Muse Image and each painting was judged by both critics. A design that
failed counts as a failure here (no world, not passed), never as a missing row. Design speed is compared with
Muse Spark both when it decides how long to think (the setting of the test) and at reasoning low (the engine's
default since that day). From evals/data/designer.json.

    python evals/eval_designer.py             # the report
    python evals/eval_designer.py --check     # and compare with evals/results/designer.json
"""
from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from eval_common import OPEN, bootstrap, ci, load, main_guard, pct, quantile, rate

ARMS = ("spark_default", "open_low")
NAMES = {"spark_default": "Muse Spark, the model decides how long it thinks",
         "open_low": "Muse Spark Open, reasoning low",
         "spark_low": "Muse Spark, reasoning low (the engine's default since)"}


def arm_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    cells = [r[arm] for r in rows]
    made = [c for c in cells if c["designed"]]
    n = len(cells)
    out = {
        "meditations": n,
        "designed": rate(len(made), n),
        "failures": [f"{c['error']}: {c['detail']}" for c in cells if not c["designed"]],
    }
    for critic in ("open_critic", "spark_critic"):
        passed = sum(1 for c in made if c[critic]["pass"])
        scores = [c[critic]["score"] for c in made]
        out[critic] = {
            "passed_of_all": rate(passed, n),             # a failed design counts as not passed
            "mean_of_designed": round(statistics.mean(scores), 2),
            "mean_failures_as_0": round(sum(scores) / n, 2),
            "median_failures_lowest": statistics.median(scores + [0] * (n - len(made))),
        }
    out["open_critic"]["fell_back_to_spark"] = sum(1 for c in made if c["open_critic"]["judge"] != OPEN)
    times = [c["design_s"] for c in made]
    out["design_s"] = {"n": len(times), "median": round(statistics.median(times), 2),
                       "p90": round(quantile(times, 0.9), 2), "min": min(times), "max": max(times)}
    costs = [c["cost_usd"] for c in made]
    out["cost_usd"] = {"median_per_world": round(statistics.median(costs), 4),
                       "spent_incl_failures": round(sum(c["cost_usd"] for c in cells), 4)}
    return out


def compute() -> dict[str, Any]:
    data = load("designer.json")
    rows = data["twenty"]
    arms = {arm: arm_summary(rows, arm) for arm in ARMS}

    # Paired, on the meditations both designers designed.
    both = [r for r in rows if r["spark_default"]["designed"] and r["open_low"]["designed"]]
    paired: dict[str, Any] = {"meditations": len(both)}
    for critic in ("open_critic", "spark_critic"):
        diffs = [r["open_low"][critic]["score"] - r["spark_default"][critic]["score"] for r in both]
        paired[f"{critic}_diff_open_minus_spark"] = {
            "mean": round(statistics.mean(diffs), 2), "ci95": bootstrap(statistics.mean, diffs),
            "open_higher": sum(1 for d in diffs if d > 0), "same": sum(1 for d in diffs if d == 0),
            "open_lower": sum(1 for d in diffs if d < 0)}
    pairs = [(r["spark_default"]["design_s"], r["open_low"]["design_s"]) for r in both]

    def ratio_of_medians(ps: list[tuple[float, float]]) -> float:
        return statistics.median(a for a, _ in ps) / statistics.median(b for _, b in ps)

    paired["design_s_ratio_spark_default_over_open"] = {
        "ratio_of_medians": round(ratio_of_medians(pairs), 2), "ci95": bootstrap(ratio_of_medians, pairs),
        "open_faster_on": sum(1 for a, b in pairs if b < a)}

    low = data["spark_low_twenty"]
    open_median = arms["open_low"]["design_s"]["median"]
    spark_low = {
        "meditations": len(rows),
        "designed": rate(low["designed"], len(rows)),
        "open_critic_passed_of_all": rate(low["open_critic_pass"], len(rows)),
        "spark_critic_passed_of_all": rate(low["spark_critic_pass"], len(rows)),
        "open_critic_mean": round(low["open_critic_mean"], 2),
        "spark_critic_mean": round(low["spark_critic_mean"], 2),
        "design_s_median": low["design_s_median"],
        "speed_ratio_spark_low_over_open_low": round(low["design_s_median"] / open_median, 2),
        "limits": "only the run's summary was kept (no row per meditation): no paired difference and no interval",
    }
    speed = {
        "open_low_median_s": open_median,
        "spark_default_median_s": arms["spark_default"]["design_s"]["median"],
        "spark_low_median_s": low["design_s_median"],
        "open_vs_spark_default": round(arms["spark_default"]["design_s"]["median"] / open_median, 2),
        "open_vs_spark_low": spark_low["speed_ratio_spark_low_over_open_low"],
    }

    six = data["six_examples"]["rows"]
    six_out = {}
    for arm in ARMS:
        rs = [x for x in six if x["designer"] == arm]
        six_out[arm] = {"meditations": len(rs), "made": len(rs),
                        "mean_score_open_critic": round(statistics.mean(x["score"] for x in rs), 2),
                        "median_design_s": round(statistics.median(x["design_s"] for x in rs), 2),
                        "painted_twice": sum(1 for x in rs if x["paintings"] == 2),
                        "mean_cost_usd": round(statistics.mean(x["cost_usd"] for x in rs), 4),
                        "median_cost_usd": round(statistics.median(x["cost_usd"] for x in rs), 4)}
    languages = dict(Counter(r["language"] for r in rows))
    return {"meditations": {"n": len(rows), "languages": languages,
                            "moods": len({r["mood"] for r in rows}), "techniques": len({r["technique"] for r in rows})},
            "arms": arms, "paired": paired, "spark_low": spark_low, "speed": speed, "six_examples": six_out}


def report(r: dict[str, Any]) -> str:
    lines = ["THE DESIGNER (20 test meditations, 28 Oct 2026)",
             f"meditations: {r['meditations']['n']} in {r['meditations']['languages']}"]
    for arm in ARMS:
        a = r["arms"][arm]
        oc, sc = a["open_critic"], a["spark_critic"]
        lines.append(
            f"{NAMES[arm]}: designed {a['designed']['k']}/{a['designed']['n']} (95 % CI {ci(a['designed']['ci95'])})"
            + (f"; failed: {'; '.join(a['failures'])}" if a["failures"] else ""))
        lines.append(
            f"  passed, failures counted: open critic {oc['passed_of_all']['k']}/20, Muse Spark critic "
            f"{sc['passed_of_all']['k']}/20; mean score of the designed: {oc['mean_of_designed']} / "
            f"{sc['mean_of_designed']}; with a failure as 0: {oc['mean_failures_as_0']} / {sc['mean_failures_as_0']}")
        lines.append(f"  design time: median {a['design_s']['median']} s, 90th percentile {a['design_s']['p90']} s "
                     f"(n {a['design_s']['n']}); cost per world {a['cost_usd']['median_per_world']} USD; "
                     f"open critic fell back to Muse Spark on {oc['fell_back_to_spark']}")
    p = r["paired"]
    for critic in ("open_critic", "spark_critic"):
        d = p[f"{critic}_diff_open_minus_spark"]
        lines.append(f"paired on {p['meditations']}: {critic} score, open minus spark {d['mean']:+.2f} "
                     f"(95 % CI {ci(d['ci95'], False)}); higher/same/lower {d['open_higher']}/{d['same']}/{d['open_lower']}")
    sp = p["design_s_ratio_spark_default_over_open"]
    lines.append(f"design time, Muse Spark (default) over Muse Spark Open: {sp['ratio_of_medians']}x "
                 f"(95 % CI {sp['ci95'][0]}-{sp['ci95'][1]}x), open faster on {sp['open_faster_on']}/{p['meditations']}")
    lo = r["spark_low"]
    lines.append(f"{NAMES['spark_low']}: designed {lo['designed']['k']}/20, passed {lo['open_critic_passed_of_all']['k']}"
                 f"/{lo['spark_critic_passed_of_all']['k']}, means {lo['open_critic_mean']}/{lo['spark_critic_mean']}, "
                 f"median {lo['design_s_median']} s: Muse Spark Open is {lo['speed_ratio_spark_low_over_open_low']}x "
                 f"as fast ({lo['limits']})")
    s6 = r["six_examples"]
    lines.append("six example meditations: " + "; ".join(
        f"{arm} mean score {v['mean_score_open_critic']}, design {v['median_design_s']} s, mean cost "
        f"{v['mean_cost_usd']} USD, painted twice {v['painted_twice']}" for arm, v in s6.items()))
    return "\n".join(lines)


if __name__ == "__main__":
    main_guard("designer", compute, report)
