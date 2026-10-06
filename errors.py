"""Error analysis: why is each non-correct cell not correct?

  python3 errors.py out/<model>/masked/pred.json

'text' = what a careful human can read from the report prose with Table 3 hidden
(tests/oracle_masked.json). Verdicts:
  TEXT != TABLE   agent copied the prose correctly, but the report's prose and Table 3 disagree
  HALLUCINATION   agent gave a number that the prose does not contain
  WRONG           the prose had the right value, the agent gave a different one
  MISSED          the prose had the right value, the agent answered null
"""
import sys

from eval import consistent, load
from schema import METRICS

gt = load("ground_truth.json")
oracle = load("tests/oracle_masked.json")
pred = load(sys.argv[1])


def given(x):
    return x is not None and (x["mean"] is not None or x["not_detected"])


def show(x):
    if not given(x):
        return "-"
    return "not detected" if x["not_detected"] else f"{x['mean']:g}"


def same(g, x):
    """x agrees with ground-truth cell g (value within rounding, or both 'not detected')."""
    if g["not_detected"] or x["not_detected"]:
        return g["not_detected"] and x["not_detected"]
    return consistent(x["mean"], g["mean"])


counts = {"OK": 0, "TEXT != TABLE": 0, "HALLUCINATION": 0, "WRONG": 0, "MISSED": 0}
rows = []
for (s, T), g in sorted(gt.items()):
    for m in METRICS:
        gm = g[m]
        om = oracle.get((s, T), {}).get(m)
        pm = pred.get((s, T), {}).get(m)
        in_text = given(om)
        text_ok = in_text and same(gm, om)
        if given(pm):
            if same(gm, pm):
                counts["OK"] += 1
                continue
            if in_text and not text_ok and not pm["not_detected"] and not om["not_detected"] and pm["mean"] == om["mean"]:
                verdict = "TEXT != TABLE"
            elif not in_text:
                verdict = "HALLUCINATION"
            else:
                verdict = "WRONG"
        elif text_ok:
            verdict = "MISSED"
        else:
            continue  # null, and nothing recoverable from the text: correct abstention
        counts[verdict] += 1
        rows.append((s, T, m, show(gm), show(om), show(pm), verdict))

print(f"{'strain':6} {'T':>3} {'metric':12} {'Table 3':>12} {'in text':>12} {'agent':>12}  verdict")
for r in rows:
    print(f"{r[0]:6} {r[1]:>3} {r[2]:12} {r[3]:>12} {r[4]:>12} {r[5]:>12}  {r[6]}")
print("\n" + "  ".join(f"{k}: {v}" for k, v in counts.items()))
print("(all other cells: agent answered null and the prose does not contain the value = correct abstention)")