"""Parse Table 3 (appendix) from the *unmasked* PDF text layer into ground_truth.json.

The result is printed as a table: check it against the PDF page by eye before trusting it.
"""
import json
import re
import sys

from ingest import TABLE3_RE, load_pages
from schema import METRICS

CELL = re.compile(r"^([\d,]+(?:\.\d+)?) ± ([\d,]+(?:\.\d+)?)([a-z])?$")
NOT_REPORTED = re.compile(r"^—([a-z])$")


def num(s: str) -> float:
    return float(s.replace(",", ""))


def parse(pdf_path: str) -> list[dict]:
    pages = load_pages(pdf_path, mask_table3=False)
    block = next(m.group(0) for t in pages if (m := TABLE3_RE.search(t)))
    lines = [l.strip() for l in block.split("\n")]
    start = lines.index("WSO [Pa]") + 1
    rows = []
    i = start
    while i + 6 < len(lines) and re.fullmatch(r"\d{4}", lines[i]):
        strain, temp, cells = lines[i], int(lines[i + 1].replace("◦C", "")), lines[i + 2 : i + 7]
        rec = {"strain": strain, "temperature_C": temp}
        for name, c in zip(METRICS, cells):
            m, nr = CELL.match(c), NOT_REPORTED.match(c)
            if m:
                rec[name] = {"mean": num(m.group(1)), "sd": num(m.group(2)),
                             "approximate": m.group(3) == "b", "not_detected": False, "evidence": None}
            elif nr:
                rec[name] = {"mean": None, "sd": None, "approximate": False, "not_detected": True, "evidence": None}
            else:
                raise ValueError(f"unparsed cell {c!r} in row {strain} {temp}")
        rows.append(rec)
        i += 7
    return rows


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/SHP_Report.pdf"
    rows = parse(pdf)
    assert len(rows) == 14, f"expected 14 rows, got {len(rows)}"
    json.dump({"records": rows}, open("ground_truth.json", "w"), indent=1, ensure_ascii=False)
    print(f"{'strain':6} {'T':>3} " + " ".join(f"{m:>20}" for m in METRICS))
    for r in rows:
        cells = []
        for m in METRICS:
            x = r[m]
            cells.append("—" if x["not_detected"] else f"{x['mean']} ± {x['sd']}" + ("*" if x["approximate"] else ""))
        print(f"{r['strain']:6} {r['temperature_C']:>3} " + " ".join(f"{c:>20}" for c in cells))
    print("\n* = flagged approximate in the report; — = not reported (no G'=G'' crossover)")
