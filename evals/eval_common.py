"""Shared helpers for the evals: the committed data, 95 % intervals and short summaries. Standard library only.

Every number in evals/results/ and in evals/README.md is computed from the files in evals/data/ by the eval_*.py
scripts; nothing calls a model or the network.
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

OPEN = "meta-models/muse-spark-open-1026"   # Muse Spark Open on Baseten (open weights)
SPARK = "muse-spark-1.3"                     # Muse Spark on Meta Model API
PASS_SCORE = 7                               # Settings.pass_score
Z95 = 1.959963984540054


def load(name: str) -> Any:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def wilson(k: int, n: int) -> list[float] | None:
    """95 % Wilson score interval for k successes out of n."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + Z95 ** 2 / n
    centre = (p + Z95 ** 2 / (2 * n)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def rate(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "share": round(k / n, 3) if n else None, "ci95": wilson(k, n)}


def quantile(values: Sequence[float], q: float) -> float:
    """The q-quantile with linear interpolation between the closest ranks."""
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def spread(values: Sequence[float], digits: int = 1) -> dict[str, Any]:
    """n, median, 90th percentile and the slowest (or largest) value."""
    xs = [float(v) for v in values]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "median": round(statistics.median(xs), digits), "p90": round(quantile(xs, 0.9), digits),
            "max": round(max(xs), digits)}


def bootstrap(stat: Callable[[list[Any]], float], items: Sequence[Any], reps: int = 10000,
              seed: int = 2026) -> list[float]:
    """95 % percentile interval of stat over resamples of items (with replacement; a fixed seed, so it repeats)."""
    rng = random.Random(seed)
    n = len(items)
    values = sorted(stat([items[int(rng.random() * n)] for _ in range(n)]) for _ in range(reps))
    return [round(quantile(values, 0.025), 3), round(quantile(values, 0.975), 3)]


def histogram(scores: Sequence[int]) -> dict[str, int]:
    return {str(s): sum(1 for x in scores if x == s) for s in sorted(set(scores))}


def pct(share: float | None) -> str:
    return "n/a" if share is None else f"{share * 100:.1f} %"


def ci(interval: Sequence[float] | None, as_pct: bool = True) -> str:
    if interval is None:
        return "n/a"
    lo, hi = interval
    return f"{lo * 100:.1f}-{hi * 100:.1f} %" if as_pct else f"{lo:+.2f} to {hi:+.2f}"


def write(name: str, result: dict[str, Any]) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def committed(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def run(name: str, compute: Callable[[], dict[str, Any]], report: Callable[[dict[str, Any]], str],
        argv: list[str]) -> int:
    """`python evals/eval_<name>.py` prints the report; --write saves results/<name>.json; --check compares with it."""
    result = compute()
    print(report(result))
    if "--write" in argv:
        print(f"\nwrote {write(name, result).relative_to(HERE.parent)}")
    if "--check" in argv:
        same = json.loads(json.dumps(result)) == committed(name)
        print(f"\nresults/{name}.json: {'the same' if same else 'DIFFERENT'}")
        return 0 if same else 1
    return 0


def main_guard(name: str, compute, report) -> None:
    sys.exit(run(name, compute, report, sys.argv[1:]))
