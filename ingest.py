"""PDF -> page texts -> section-tagged chunks.

mask_table3=True removes the Table 3 block from the text the agent can see,
so evaluation against Table 3 isn't just copy-paste from the table itself.
"""
import re
from dataclasses import dataclass

import pymupdf

TABLE3_RE = re.compile(
    r"Table 3: Summary of average rheological parameters.*?should be treated as approximate\.",
    re.S,
)
MASK_PLACEHOLDER = "[Table 3 removed]"

# In this PDF's text layer a heading is two lines: "5.1.2" then "50◦C: Thermal softening ..."
NUM_ONLY = re.compile(r"^([1-7](?:\.[1-9]){0,2})$")
APPX_ONLY = re.compile(r"^([A-D])$")


@dataclass
class Chunk:
    id: str
    page: int        # 1-indexed PDF page
    section: str
    text: str


def load_pages(pdf_path: str, mask_table3: bool = True) -> list[str]:
    doc = pymupdf.open(pdf_path)
    pages = [p.get_text() for p in doc]
    if mask_table3:
        pages = [TABLE3_RE.sub(MASK_PLACEHOLDER, t) for t in pages]
    return pages


def _walk_headings(pages: list[str]):
    """Yield (page_no, line, current_section) for every line, tracking headings."""
    section = "Front matter"
    last_top = 0  # top-level section numbers must appear in order (rejects table cells like "1")
    for pno, text in enumerate(pages, start=1):
        lines = [l.strip() for l in text.split("\n")]
        if pno <= 3:  # title page, abstract, table of contents
            for ln in lines:
                yield pno, ln, section
            continue
        i = 0
        while i < len(lines):
            ln = lines[i]
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if NUM_ONLY.match(ln) and nxt and len(nxt) < 90 and not nxt.endswith(".") and (
                nxt[:1].isupper() or ("." in ln and nxt[:1].isdigit())
            ):
                top = int(ln.split(".")[0])
                if "." in ln or top == last_top + 1:
                    if "." not in ln:
                        last_top = top
                    section = f"{ln} {nxt}"
                    i += 1
            elif APPX_ONLY.match(ln) and i + 1 < len(lines) and lines[i + 1][:1].isupper() and len(lines[i + 1]) < 70:
                section = f"Appendix {ln} {lines[i + 1]}"
                i += 1
            yield pno, ln, section
            i += 1


def _groups(pages: list[str]) -> list[tuple[int, str, list[str]]]:
    groups: list[tuple[int, str, list[str]]] = []
    for pno, ln, sec in _walk_headings(pages):
        if not ln:
            continue
        if groups and groups[-1][0] == pno and groups[-1][1] == sec:
            groups[-1][2].append(ln)
        else:
            groups.append((pno, sec, [ln]))
    return groups


def chunk_pages(pages: list[str], size: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pno, sec, lines in _groups(pages):
        text = " ".join(lines)
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            if end < len(text):  # prefer to cut at a sentence boundary
                cut = text.rfind(". ", start + size // 2, end)
                end = cut + 1 if cut != -1 else end
            piece = text[start:end].strip()
            if len(piece) > 40:
                chunks.append(Chunk(f"p{pno}-{len(chunks)}", pno, sec, piece))
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(∼~≈])")
_SKIP_SECTIONS = ("Front matter", "7 References", "Appendix B", "Appendix C", "Appendix D")
_NUMERICISH = re.compile(r"^[\d.,±◦C%()~∼≈<>/-]+$")


def sentences(pages: list[str]) -> list[tuple[int, str, str]]:
    """(page, section, sentence) for prose sentences. Table-body blobs (mostly numeric tokens)
    and non-result sections (references, statistics tables) are dropped."""
    out = []
    for pno, sec, lines in _groups(pages):
        if sec.startswith(_SKIP_SECTIONS):
            continue
        for sent in _SENT_SPLIT.split(" ".join(lines)):
            toks = sent.split()
            if len(toks) < 6 or sum(bool(_NUMERICISH.match(t)) for t in toks) / len(toks) > 0.5:
                continue
            out.append((pno, sec, sent.strip()))
    return out


if __name__ == "__main__":
    import sys

    pages = load_pages(sys.argv[1], mask_table3="--full" not in sys.argv)
    cs = chunk_pages(pages)
    print(len(pages), "pages ->", len(cs), "chunks")
    seen = []
    for c in cs:
        if c.section not in seen:
            seen.append(c.section)
    print("\n".join(seen))