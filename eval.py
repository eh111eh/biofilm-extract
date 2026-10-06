"""Compare extracted records with ground_truth.json (Table 3).

Per (strain, temperature, metric) cell the prediction is one of:
  correct   value within tolerance of the ground truth
  wrong     value given, but outside tolerance          -> counts as a hallucination
  abstain   null where the ground truth has a value     -> missing, but not a false claim
For ground-truth cells that are 'not reported' (no G'=G'' crossover):
  correct   not_detected=true
  abstain   null without the flag
  wrong     a numeric value was given                   -> hallucination

Two criteria are reported:
  exact       |pred - gt| <= 0.5% (formatting only, e.g. '10,997' vs 10997.0)
  consistent  gt lies inside the rounding interval implied by how many digits the prediction states
              ('~8,000' -> [7500, 8500] accepts 8047; '4.2' -> [4.15, 4.25] rejects 4.3).
              A value stated to one digit is not penalised for being a rounded value.
"""
from decimal import Decimal
import argparse
import json
from collections import defaultdict

from schema import METRICS

EXACT_TOL = 0.005


def load(path: str) -> dict:
    return {(r["strain"], r["temperature_C"]): r for r in json.load(open(path))["records"]}


def exact(p, g):
    return p is not None and g is not None and abs(p - g) <= EXACT_TOL * max(abs(g), 1e-12)


def consistent(p, g):
    if p is None or g is None:
        return False
    if exact(p, g):
        return True
    exp = Decimal(repr(float(p))).normalize().as_tuple().exponent  # position of last non-zero digit
    return abs(p - g) <= 0.5 * 10 ** exp + 1e-12


def classify(g: dict, p: dict | None, ok) -> str:
    if g["not_detected"]:
        if p and p["mean"] is not None:
            return "wrong"
        return "correct" if p and p["not_detected"] else "abstain"
    if p is None or p["mean"] is None:
        return "abstain"
    return "correct" if ok(p["mean"], g["mean"]) else "wrong"


def score(gt: dict, pred: dict) -> dict:
    cells = defaultdict(lambda: defaultdict(int))  # key -> counters
    for (strain, T), g in gt.items():
        pr = pred.get((strain, T))
        for m in METRICS:
            gm, pm = g[m], (pr[m] if pr else None)
            keys = (m, f"{m}@{T}", "ALL", f"ALL@{T}")
            for key in keys:
                cells[key]["n"] += 1
            for tag, ok in (("exact", exact), ("cons", consistent)):
                c = classify(gm, pm, ok)
                for key in keys:
                    cells[key][f"{tag}_{c}"] += 1
            if not gm["not_detected"] and gm["sd"] is not None:  # SD scored separately
                hit = pm is not None and consistent(pm["sd"], gm["sd"])
                for key in (m, "ALL", "SD"):
                    cells[key]["sd_n"] += 1
                    cells[key]["sd_ok"] += int(hit)
    return cells


def pct(a, b):
    return "  -  " if b == 0 else f"{100 * a / b:5.1f}"


def report(name: str, cells: dict) -> str:
    out = [f"\n=== {name} ===",
           f"{'field':14} {'n':>3} | {'exact':>6} {'consist.':>8} | {'answered':>8} {'precision':>9} {'halluc.':>7} | {'SD acc':>6}"]
    for k in METRICS + ["ALL", "ALL@30", "ALL@50"]:
        c = cells[k]
        n = c["n"]
        answered = c["cons_correct"] + c["cons_wrong"]
        out.append(
            f"{k:14} {n:>3} | {pct(c['exact_correct'], n):>6} {pct(c['cons_correct'], n):>8} | "
            f"{pct(answered, n):>8} {pct(c['cons_correct'], answered):>9} {pct(c['cons_wrong'], n):>7} | "
            f"{pct(c['sd_ok'], c['sd_n']):>6}"
        )
    out.append("acc = correct / all cells (abstain counts as a miss) | answered = non-null | precision = correct / answered | halluc. = wrong / all")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+", help="prediction JSON files ({'records': [...]})")
    ap.add_argument("--gt", default="ground_truth.json")
    a = ap.parse_args()
    gt = load(a.gt)
    for path in a.preds:
        print(report(path, score(gt, load(path))))
