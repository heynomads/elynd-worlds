"""The evals recompute exactly the committed results and the README's summary table from evals/data/ (no network)."""
from __future__ import annotations

import json

import pytest

import eval_critic
import eval_designer
import eval_pipeline
import run_evals
from eval_common import HERE, OPEN, committed


@pytest.fixture(scope="module")
def results():
    return {name: json.loads(json.dumps(m.compute())) for name, m in run_evals.EVALS.items()}


@pytest.mark.parametrize("name", ["critic", "designer", "pipeline"])
def test_results_are_reproduced(results, name):
    assert results[name] == committed(name), f"run: python evals/run_evals.py --write, then look at the diff of {name}"


def test_readme_table_is_current(results):
    assert run_evals.table(results) in (HERE / "README.md").read_text(encoding="utf-8")


def test_every_library_world_judged_by_the_open_model(results):
    lib = results["critic"]["library"]
    assert lib["worlds"] == 1101 and lib["final_judge"] == {OPEN: 1101} and lib["all_pass"]
    assert lib["sleep_worlds"] == 159
    assert lib["places"]["first"] + lib["places"]["second"] == 208
    assert lib["judged_by_open_at_creation"] + lib["judged_again_by_open"] == 1101


def test_rejudge_counts(results):
    rj = results["critic"]["rejudge"]
    first_run = rj["by_run"]["first"] + 2          # the two turned down are not in the library
    assert first_run == 36 and rj["turned_down_by_open"]["k"] == 3 and rj["paintings"] == 120


def test_designer_counts_failures(results):
    open_low = results["designer"]["arms"]["open_low"]
    assert open_low["designed"]["k"] == 18 and open_low["open_critic"]["passed_of_all"]["n"] == 20
    assert open_low["open_critic"]["mean_failures_as_0"] < open_low["open_critic"]["mean_of_designed"]
    assert results["designer"]["speed"]["open_vs_spark_low"] < 1.5 < results["designer"]["speed"]["open_vs_spark_default"]


def test_cost_and_timeouts(results):
    cost = results["pipeline"]["cost"]["first_library_445"]
    assert cost["n"] == 445 and cost["median_cents"] == pytest.approx(4.2, abs=0.05) and 19.5 < cost["total_usd"] < 20.5
    courtyard = next(p for p in results["pipeline"]["production"]["painted"] if p["run"] == "courtyard")
    assert courtyard["seconds_to_ready"] < 120 < 170


def test_the_code_check_still_passes_the_library():
    code = eval_critic.code_check(json.loads((HERE / "data" / "library.json").read_text())["worlds"])
    assert code["checked"] == 1082 and code["look_dead"] == 0


def test_modules_print_a_report(results):
    for m, name in ((eval_critic, "critic"), (eval_designer, "designer"), (eval_pipeline, "pipeline")):
        assert m.report(results[name]).strip()
