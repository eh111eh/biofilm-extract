"""Tools the agent can call + the validator behind `submit_extraction`.

Pipeline the model is told to follow:  find table -> read values -> validate (submit).
"""
import json
import re

from pydantic import ValidationError

from ingest import Chunk, sentences
from retrieve import BM25
from schema import METRICS, RANGES, STRAINS, TEMPS, Extraction

TABLE_CAPTION = re.compile(r"^Table (\d+): (.+)$", re.M)
FOOTNOTE = re.compile(r"^([a-d] |ns:|Note:)")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def norm(s: str) -> str:
    """Aggressive normalisation so a quoted snippet can be matched against PDF text
    regardless of line breaks, hyphenation, degree/prime glyph variants."""
    s = s.lower()
    for a, b in [("°", "◦"), ("∼", "~"), ("′′", "''"), ("″", "''"), ("′", "'"), ("’", "'")]:
        s = s.replace(a, b)
    return re.sub(r"[\s\-−–—]+", "", s)


def numbers_in(s: str) -> list[float]:
    out = []
    for tok in _NUM.findall(s):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            pass
    return out


class Toolbox:
    def __init__(self, pages: list[str], chunks: list[Chunk], expected_strain: str | None = None):
        self.pages, self.chunks = pages, chunks
        self.bm25 = BM25(chunks)
        self.corpus_norm = norm(" ".join(pages))
        self.expected_strain = expected_strain
        self.sents = sentences(pages)

    # ---- retrieval ------------------------------------------------------------------
    def search_report(self, query: str, section: str = "", k: int = 5) -> dict:
        k = max(1, min(int(k), 8))
        hits = self.bm25.search(query, k=k, section=section or "")
        return {
            "results": [
                {"chunk_id": c.id, "page": c.page, "section": c.section, "text": c.text}
                for _, c in hits
            ]
        }

    def find_mentions(self, strain: str, max_results: int = 20) -> dict:
        """Prose sentences that mention a strain AND contain a number other than the strain id / 30 / 50."""
        sid = re.sub(r"\D", "", str(strain))
        if sid not in STRAINS:
            return {"error": f"unknown strain '{strain}', use one of {STRAINS}"}
        pat = re.compile(rf"(?<![\d.]){sid}(?![\d])")
        out = []
        for pno, sec, sent in self.sents:
            if not pat.search(sent):
                continue
            extra = [n for n in numbers_in(pat.sub(" ", sent)) if n not in (30.0, 50.0)]
            if extra:
                out.append({"page": pno, "section": sec, "sentence": sent[:450]})
        return {"strain": sid, "n_found": len(out), "sentences": out[:max_results],
                "note": "Sentences may mention several strains; only use numbers that belong to this strain."}

    # ---- tables ---------------------------------------------------------------------
    def list_tables(self) -> dict:
        out = []
        for pno, t in enumerate(self.pages, start=1):
            for m in TABLE_CAPTION.finditer(t):
                out.append({"table_id": int(m.group(1)), "caption": m.group(2).strip(), "page": pno})
        return {"tables": out}

    def read_table(self, table_id: int) -> dict:
        table_id = int(table_id)
        for pno, t in enumerate(self.pages, start=1):
            m = re.search(rf"^Table {table_id}: .+$", t, re.M)
            if not m:
                continue
            nxt = self.pages[pno] if pno < len(self.pages) else ""
            text = t[m.start():] + "\n" + nxt  # tables can run onto the next page
            lines, body = text.split("\n"), 0
            kept = []
            for ln in lines:
                ln = ln.strip()
                if kept and re.match(r"^Table \d+: ", ln):
                    break
                if body >= 6 and len(ln) > 70 and not FOOTNOTE.match(ln):
                    break  # first prose sentence after the table body
                kept.append(ln)
                body += 1
            return {"table_id": table_id, "page": pno,
                    "note": "Text layer, one cell per line, row-major. Column headers come first.",
                    "text": "\n".join(kept)[:4000]}
        return {"error": f"Table {table_id} not found",
                "available": [t["table_id"] for t in self.list_tables()["tables"]]}

    # ---- validation (the 'verify' step) ------------------------------------------------
    def validate(self, ext: Extraction) -> list[str]:
        issues: list[str] = []
        seen = set()
        for i, r in enumerate(ext.records):
            p = f"records[{i}]"
            if r.strain not in STRAINS:
                issues.append(f"{p}.strain: '{r.strain}' not one of {STRAINS}")
            if self.expected_strain and r.strain != self.expected_strain:
                issues.append(f"{p}.strain: expected only '{self.expected_strain}'")
            if r.temperature_C not in TEMPS:
                issues.append(f"{p}.temperature_C: {r.temperature_C} not in {TEMPS}")
            key = (r.strain, r.temperature_C)
            if key in seen:
                issues.append(f"{p}: duplicate record for {key}")
            seen.add(key)
            for name in METRICS:
                m = getattr(r, name)
                q = f"{p}.{name}"
                if m.not_detected and name != "gamma_f_pct":
                    issues.append(f"{q}.not_detected: only valid for gamma_f_pct")
                if m.not_detected and m.mean is not None:
                    issues.append(f"{q}: not_detected=true requires mean=null")
                if m.sd is not None and m.mean is None:
                    issues.append(f"{q}.sd: sd given without mean")
                if m.evidence is not None:
                    if len(m.evidence.strip()) < 8:
                        issues.append(f"{q}.evidence: too short (>= 8 chars of verbatim text)")
                    elif norm(m.evidence) not in self.corpus_norm:
                        issues.append(f"{q}.evidence: not found verbatim in the report; copy it exactly")
                if m.mean is not None or m.not_detected:
                    if m.evidence is None:
                        issues.append(f"{q}: needs a verbatim evidence quote")
                if m.mean is not None:
                    lo, hi = RANGES[name]
                    if not (lo < m.mean <= hi) and not (name in ("WSO_Pa", "tan_delta0") and m.mean == 0):
                        issues.append(f"{q}.mean: {m.mean} outside plausible range ({lo}, {hi}]")
                    if m.sd is not None and m.sd < 0:
                        issues.append(f"{q}.sd: negative")
                    if m.evidence:
                        nums = numbers_in(m.evidence)
                        if not any(abs(n - m.mean) <= 1e-6 * max(1.0, abs(m.mean)) for n in nums):
                            issues.append(f"{q}.mean: {m.mean} does not appear in its evidence quote")
                        if m.sd is not None and not any(abs(n - m.sd) <= 1e-6 * max(1.0, abs(m.sd)) for n in nums):
                            issues.append(f"{q}.sd: {m.sd} does not appear in its evidence quote")
        if self.expected_strain:
            for t in TEMPS:
                if (self.expected_strain, t) not in seen:
                    issues.append(f"missing record for strain {self.expected_strain} at {t} C")
        return issues

    def submit_extraction(self, records_json: str) -> dict:
        try:
            ext = Extraction.model_validate_json(records_json)
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            return {"status": "errors", "issues": [f"schema error: {str(e)[:600]}"]}
        issues = self.validate(ext)
        if issues:
            return {"status": "errors", "issues": issues[:25]}
        return {"status": "ok", "records": ext.model_dump()["records"]}

    # ---- dispatch ---------------------------------------------------------------------
    def dispatch(self, name: str, args: dict) -> dict:
        try:
            if name == "search_report":
                return self.search_report(args.get("query", ""), args.get("section", ""), args.get("k", 5))
            if name == "find_mentions":
                return self.find_mentions(args.get("strain", ""))
            if name == "list_tables":
                return self.list_tables()
            if name == "read_table":
                return self.read_table(args["table_id"])
            if name == "submit_extraction":
                return self.submit_extraction(args["records_json"])
            return {"error": f"unknown tool {name}"}
        except Exception as e:  # tool errors go back to the model, not up the stack
            return {"error": f"{type(e).__name__}: {e}"}