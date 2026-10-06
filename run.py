"""Run the extraction agent.

  python run.py --mode masked            # Table 3 hidden: text-only recovery (RAG), honest about gaps
  python run.py --mode full              # Table 3 visible: table-reading upper bound
  python run.py --mode masked --strains 2106 3610 --force

Per-strain results are cached in out/<model>/<mode>/<strain>.json (with full tool-call trace);
out/<model>/<mode>/pred.json is the merged file that eval.py consumes.
"""
import argparse
import json
import os
import time

from agent import MODEL, make_client, run_strain
from google.genai import errors
from ingest import chunk_pages, load_pages
from schema import STRAINS
from tools import Toolbox

ap = argparse.ArgumentParser()
ap.add_argument("--pdf", default="data/SHP_Report.pdf")
ap.add_argument("--mode", choices=["masked", "full"], required=True)
ap.add_argument("--strains", nargs="*", default=STRAINS)
ap.add_argument("--force", action="store_true", help="ignore cached per-strain results")
ap.add_argument("--sleep", type=float, default=4.0, help="seconds between strains (free-tier rate limits)")
a = ap.parse_args()

pages = load_pages(a.pdf, mask_table3=(a.mode == "masked"))
chunks = chunk_pages(pages)
out_dir = os.path.join("out", MODEL, a.mode)  # results are kept per model
os.makedirs(out_dir, exist_ok=True)
client = make_client()
print(f"model={MODEL} mode={a.mode} pages={len(pages)} chunks={len(chunks)}")

merged, total_tokens, skipped = [], 0, []
for strain in a.strains:
    path = os.path.join(out_dir, f"{strain}.json")
    cached = json.load(open(path)) if os.path.exists(path) else None
    if cached and cached["status"] == "ok" and not a.force:
        res = cached
        print(f"{strain}: cached (ok)")
    else:
        t0 = time.time()
        try:
            res = run_strain(client, Toolbox(pages, chunks, expected_strain=strain), strain)
        except errors.APIError as e:
            if getattr(e, "code", 0) >= 500:  # Google's servers are busy: skip, retry on the next run
                print(f"{strain}: skipped, server busy ({e.code}). Run the same command again later.")
                skipped.append(strain)
                continue
            raise
        res["model"], res["mode"], res["seconds"] = MODEL, a.mode, round(time.time() - t0, 1)
        json.dump(res, open(path, "w"), indent=1, ensure_ascii=False)
        print(f"{strain}: {res['status']} in {res['steps']} tool calls, {res['tokens']} tokens, {res['seconds']}s")
        time.sleep(a.sleep)
    total_tokens += res.get("tokens", 0)
    merged += res["records"]

json.dump({"records": merged, "model": MODEL, "mode": a.mode},
          open(os.path.join(out_dir, "pred.json"), "w"), indent=1, ensure_ascii=False)
print(f"wrote {out_dir}/pred.json  ({len(merged)} records, {total_tokens} tokens total)")
if skipped:
    print(f"SKIPPED (server busy): {' '.join(skipped)} -> run the same command again to do just these")
print(f"next: python eval.py {out_dir}/pred.json")