"""The critic: Muse Spark Open (open weights, on Baseten) judging every library painting.

What it scored over the whole library, what it turned down and why, how it compares with the first critic
(muse-spark-1.3) on the paintings both judged, how it does on paintings known to be wrong, and how it lines up with
the code check (`library.looks_dead`) and with a later look at the pictures. From evals/data/library.json and
evals/data/not_in_library.json; the code check is the engine's own function, run again on the committed records.

    python evals/eval_critic.py             # the report
    python evals/eval_critic.py --check     # and compare with evals/results/critic.json
"""
from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Any

from eval_common import OPEN, PASS_SCORE, SPARK, bootstrap, ci, histogram, load, main_guard, pct, rate

CRITIQUE = re.compile(r"^score (\d+) by ([^:]+?)(?:: (.*))?$")


def passed(verdict: dict[str, Any]) -> bool:
    """critic.passes for a full verdict; a record that kept only the score passes at the pass score."""
    return (verdict.get("belongs", True) and verdict.get("calm", True) and not verdict.get("figures_wrong")
            and not verdict.get("text_or_symbols") and int(verdict["score"]) >= PASS_SCORE)


def look_kind(detail: str) -> str:
    """What a look at the picture found, in one word."""
    text = detail.lower()
    if re.search(r"\b(seam|broken)\b", text):
        return "a broken picture (a seam or a floating strip)"
    if re.search(r"\bcross\b", text):
        return "a cross (a signpost or a post)"
    if re.search(r"\b(figure|bird|face)\b", text):
        return "a possible figure, animal or face"
    return "other"


def code_check(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    """library.looks_dead, the engine's function, on every library world whose full record is committed."""
    from elynd_worlds.library import looks_dead
    full = [w for w in worlds if "living" in w]
    dead = []
    for w in full:
        manifest = {"living": w["living"], "layers": w["layers"]}
        if looks_dead(manifest, w["scene"]):
            dead.append(w["id"])
    return {"checked": len(full), "look_dead": len(dead), "dead_ids": dead,
            "not_checked": len(worlds) - len(full),
            "not_checked_why": "the 19 push worlds of 29-30 Oct kept only summary fields (no region coverage)"}


def compute() -> dict[str, Any]:
    lib = load("library.json")["worlds"]
    out = load("not_in_library.json")
    runs = ("first", "push", "second")

    # 1. Coverage and scores over the whole library.
    final = [w["verdict"] for w in lib]
    scores = [v["score"] for v in final]
    library = {
        "worlds": len(lib),
        "by_run": {r: sum(1 for w in lib if w["run"] == r) for r in runs},
        "places": {r: len({w["id"].split("-")[0] for w in lib if w["run"] == r}) for r in runs},
        "sleep_worlds": None,
        "final_judge": dict(Counter(v["judge"] for v in final)),
        "judged_by_open_at_creation": sum(1 for w in lib if w["verdict"]["judge"] == OPEN and not w["earlier"]),
        "judged_again_by_open": sum(1 for w in lib if w["earlier"]),
        "all_pass": all(passed(v) for v in final),
        "scores": histogram(scores),
        "scores_by_run": {r: histogram([w["verdict"]["score"] for w in lib if w["run"] == r]) for r in runs},
        "mean_score": round(statistics.mean(scores), 2),
        "share_scored_9": rate(sum(1 for s in scores if s == 9), len(scores)),
    }
    library["sleep_worlds"] = sum(1 for w in lib if w["sleep"])

    # 2. Paintings the critic turned down during the library jobs (the first painting; the agent painted again).
    turned_down = []
    for w in lib:
        for step, _seconds, note in w.get("trace", []):
            if step.startswith("critique") and int(step.split()[-1]) < w["paintings"]:
                m = CRITIQUE.match(note)
                turned_down.append({"run": w["run"], "score": int(m.group(1)), "judge": m.group(2),
                                    "reason": m.group(3) or ""})
    first_try = {r: rate(sum(1 for w in lib if w["run"] == r and w["paintings"] == 1),
                         sum(1 for w in lib if w["run"] == r)) for r in runs}
    in_loop = {
        "first_painting_turned_down": len(turned_down),
        "by_run": {r: sum(1 for t in turned_down if t["run"] == r) for r in runs},
        "first_try_pass": first_try,
        "scores_turned_down": histogram([t["score"] for t in turned_down]),
        "scores_turned_down_by_run": {r: histogram([t["score"] for t in turned_down if t["run"] == r]) for r in runs},
        "score_0": sum(1 for t in turned_down if t["score"] == 0),
        "score_0_note": "critic.py reads a missing score as 0; the reasons are kept below",
        "judges": dict(Counter(t["judge"] for t in turned_down)),
        "reasons": [f"{t['score']}/10 by {t['judge']}: {t['reason']}" for t in turned_down],
    }

    # 3. The first critic and Muse Spark Open on the same paintings (muse-spark-1.3 judged at creation when the open
    #    critic did not answer in time; Muse Spark Open judged each again on 2 Nov 2026).
    pairs = [(w["earlier"]["score"], w["verdict"]["score"], True) for w in lib if w["earlier"]]
    pairs += [(d["earlier"]["score"], d["verdict"]["score"], False) for d in out["rejudge_turned_down"]]
    pairs += [(d["earlier"]["score"], d["score"], False) for d in out["second_run"]["not_registered"] if d["why"] == "critic"]
    diffs = [b - a for a, b, _ in pairs]
    rejudge = {
        "paintings": len(pairs),
        "first_critic": SPARK,
        "passed_by_first_critic": len(pairs),
        "turned_down_by_open": rate(sum(1 for _, _, kept in pairs if not kept), len(pairs)),
        "turned_down_why": ["a designed white butterfly, under the place-alone rule (4/10)"] * len(out["rejudge_turned_down"])
                           + [d["detail"].split("review: ")[-1].split(" (")[0] + " (4/10)"
                              for d in out["second_run"]["not_registered"] if d["why"] == "critic"],
        "same_score": rate(sum(1 for d in diffs if d == 0), len(diffs)),
        "within_one": rate(sum(1 for d in diffs if abs(d) <= 1), len(diffs)),
        "mean_first": round(statistics.mean(a for a, _, _ in pairs), 2),
        "mean_open": round(statistics.mean(b for _, b, _ in pairs), 2),
        "mean_difference_open_minus_first": round(statistics.mean(diffs), 2),
        "mean_difference_ci95": bootstrap(statistics.mean, diffs),
        "by_run": {r: sum(1 for w in lib if w["earlier"] and w["run"] == r) for r in runs},
    }

    # 4. Paintings known to be wrong: a seated person the design did not bring.
    bad = out["known_bad_figures"]["paintings"]
    known_bad = {
        "paintings": len(bad),
        "at_creation_passed": rate(sum(1 for b in bad if passed(b["at_creation"])), len(bad)),
        "at_creation_judge": dict(Counter(b["at_creation"]["judge"] for b in bad)),
        "open_turned_down": rate(sum(1 for b in bad if not passed(b["checked_again"])), len(bad)),
        "open_scores": histogram([b["checked_again"]["score"] for b in bad]),
        "open_figures_wrong": sum(1 for b in bad if b["checked_again"]["figures_wrong"]),
    }

    # 5. The critic against the later look and the code check, on the runs where both are recorded.
    second = out["second_run"]
    not_reg = second["not_registered"]
    painted = second["painted"]
    critic_down = [d for d in not_reg if d["why"] == "critic"]
    critic_passed = painted - len(critic_down)
    look_down = [d for d in not_reg if d["why"] == "look"]
    dead = [d for d in not_reg if d["why"] == "dead"]
    push = out["push_run"]
    against = {
        "second_run": {
            "painted": painted,
            "critic_passed": critic_passed,
            "critic_turned_down": len(critic_down),
            "looked_at": second["looked_at"],
            "look_flagged": second["flagged_by_look"],
            "passed_by_critic_turned_down_by_look": rate(len(look_down), critic_passed),
            "look_kinds": dict(Counter(look_kind(d["detail"]) for d in look_down)),
            "look_scores_from_critic": histogram([d["score"] for d in look_down]),
            "dead_by_code": len(dead),
            "dead_passed_by_critic": rate(sum(1 for d in dead if d["score"] >= PASS_SCORE), len(dead)),
            "dead_scores_from_critic": histogram([d["score"] for d in dead]),
            "critic_turned_down_alive_by_code": sum(1 for d in critic_down if d["why"] != "dead"),
            "in_library": painted - len(not_reg),
        },
        "push_run": {
            "paintings": push["paintings_made"],
            "critic_passed": sum(1 for w in lib if w["run"] == "push" and passed(w["verdict"]))
                             + sum(1 for d in push["not_kept"] if d["score"] >= PASS_SCORE),
            "turned_down_by_look": sum(1 for d in push["not_kept"] if d["why"] == "look"),
            "look_kinds": dict(Counter(look_kind(d["detail"]) for d in push["not_kept"] if d["why"] == "look")),
            "dead_by_code": sum(1 for d in push["not_kept"] if d["why"] == "dead"),
            "scores_of_not_kept": histogram([d["score"] for d in push["not_kept"]]),
        },
        "code_check_on_library": code_check(lib),
    }

    # 6. What the verdicts in the library were judged against: the first library's designs could bring a fluttering
    #    white butterfly; the place-alone rule (29 Oct 2026) turns that down, and the two judged again under it were.
    butterflies = [w for w in lib if any("butterfly" in (s.get("name") or "") for s in (w.get("scene") or {}).get("subjects", []))]
    rules = {
        "designed_butterfly_in_library": len(butterflies),
        "by_run": dict(Counter(w["run"] for w in butterflies)),
        "their_judges": dict(Counter(w["verdict"]["judge"] for w in butterflies)),
        "judged_again_under_place_alone_rule": sum(1 for w in butterflies if w["earlier"]),
        "note": "judged on 26-27 Oct 2026 under that day's rule, which allowed an animal the design brings; "
                "the two such paintings judged again under the place-alone rule were turned down (4/10)",
    }
    return {"library": library, "in_loop": in_loop, "rejudge": rejudge, "known_bad": known_bad,
            "against_look_and_code": against, "rules": rules}


def report(r: dict[str, Any]) -> str:
    lib, loop, rj, kb, ag, ru = (r["library"], r["in_loop"], r["rejudge"], r["known_bad"],
                                 r["against_look_and_code"], r["rules"])
    s2, code = ag["second_run"], ag["code_check_on_library"]
    lines = [
        "THE CRITIC (Muse Spark Open)",
        f"library: {lib['worlds']} worlds ({', '.join(f'{k} {v}' for k, v in lib['by_run'].items())}); final judge "
        f"{lib['final_judge']}; {lib['judged_by_open_at_creation']} judged by it at creation, "
        f"{lib['judged_again_by_open']} judged again on 2 Nov after a {SPARK} verdict; all pass: {lib['all_pass']}",
        f"scores (final verdicts): {lib['scores']}; mean {lib['mean_score']}; scored 9: "
        f"{lib['share_scored_9']['k']} ({pct(lib['share_scored_9']['share'])})",
        f"  by run: {lib['scores_by_run']}",
        f"first painting turned down in the jobs: {loop['first_painting_turned_down']} {loop['by_run']}; "
        f"scores {loop['scores_turned_down']}; first-try pass: "
        + ", ".join(f"{k} {v['k']}/{v['n']} ({pct(v['share'])}, 95 % CI {ci(v['ci95'])})"
                    for k, v in loop["first_try_pass"].items()),
        f"{SPARK} then Muse Spark Open on the same {rj['paintings']} paintings: turned down "
        f"{rj['turned_down_by_open']['k']} ({pct(rj['turned_down_by_open']['share'])}, 95 % CI "
        f"{ci(rj['turned_down_by_open']['ci95'])}): {'; '.join(rj['turned_down_why'])}",
        f"  same score {rj['same_score']['k']}/{rj['paintings']}, within one point {rj['within_one']['k']}/"
        f"{rj['paintings']}; mean {rj['mean_first']} -> {rj['mean_open']} "
        f"(difference {rj['mean_difference_open_minus_first']:+.2f}, 95 % CI {ci(rj['mean_difference_ci95'], False)})",
        f"known bad (a seated person the design did not bring): {kb['paintings']}; passed at creation by "
        f"{kb['at_creation_judge']}: {kb['at_creation_passed']['k']}; turned down by Muse Spark Open: "
        f"{kb['open_turned_down']['k']}/{kb['paintings']} (95 % CI {ci(kb['open_turned_down']['ci95'])}), "
        f"scores {kb['open_scores']}",
        f"2 Nov run: {s2['painted']} painted, critic passed {s2['critic_passed']}; a look at every picture flagged "
        f"{s2['look_flagged']}; passed by the critic but turned down by the look: "
        f"{s2['passed_by_critic_turned_down_by_look']['k']} ({pct(s2['passed_by_critic_turned_down_by_look']['share'])}): "
        f"{s2['look_kinds']}",
        f"  dead by the code check: {s2['dead_by_code']}, all passed by the critic "
        f"({s2['dead_passed_by_critic']['k']}/{s2['dead_by_code']}, scores {s2['dead_scores_from_critic']}); "
        f"the critic's one turn-down was alive by the code check",
        f"code check (looks_dead) again on the library: {code['checked']} checked, {code['look_dead']} look dead, "
        f"{code['not_checked']} not checked ({code['not_checked_why']})",
        f"designed butterflies in the library: {ru['designed_butterfly_in_library']} {ru['by_run']} ({ru['note']})",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main_guard("critic", compute, report)
