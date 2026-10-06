# biofilm-extract

A tool-using LLM agent that reads my Senior Honours Project report (a 29-page PDF on thermal remodelling of *Bacillus subtilis* biofilm matrices) and extracts its rheology results (G′₀, tan δ₀, yield strain, crossover strain, weak strain overshoot) per strain and temperature as validated JSON, plus an evaluation harness that measures whether the agent **invents values when the document does not contain them**.

> Status: working prototype on **one document**. All results below are single runs per model. See [Limitations](#limitations).

## Why this exists

Pulling numbers out of a document is easy to demo and hard to trust. The question here is not "can an LLM copy a table?" but "when the answer is not in the text, does the agent say so?" The agent is built so that an unknown value is a valid answer (`null`) and an unsupported value is rejected.

## How it works

```
PDF ──► ingest.py ──► section-tagged chunks + sentence index
                          │
        ┌─────────────────┴────────────────────────────────┐
        │  Gemini function-calling loop (agent.py)           │
        │  FIND    find_mentions · list_tables               │
        │  READ    read_table · search_report (BM25)         │
        │  VERIFY  submit_extraction ──► validator ──┐       │
        │             ▲                              │issues │
        │             └───────── fix and resubmit ◄──┘       │
        └──────────────────────────────────────────────────┘
                          │ validated JSON: 14 records × 5 metrics
                          ▼
              eval.py / errors.py  vs  ground_truth.json (Table 3)
```

The validator (`tools.py`) rejects a submission unless:

1. it matches the Pydantic schema (`schema.py`);
2. every non-null value carries an `evidence` quote that appears **verbatim** in the report (normalised for line breaks, hyphenation and degree/prime glyphs);
3. the value itself appears in its own quote;
4. the value is in a physically plausible range.

Rejections go back to the model as a list of issues to fix. If the model stalls, a step budget forces a final `submit_extraction` call (exercised with simulated failures). The loop is written by hand rather than with automatic function calling, so every tool call is logged in `out/<model>/<mode>/<strain>.json`.

## Evaluation design

- **Ground truth.** Table 3 of the report: 14 strain × temperature rows × 5 metrics = 70 cells, plus standard deviations. `make_ground_truth.py` parses it from the PDF text layer; I checked all 14 rows by eye against the printed table.
- **Masked mode.** If the agent can read Table 3, the task is transcription. In masked mode the Table 3 text is removed before the agent sees the report. Values that the prose itself states can still be recovered.
- **Recoverable set.** I read the prose and listed every value it states for a given strain and temperature (`tests/oracle_masked.json`): 20 cells. For 17 of them the prose value agrees with Table 3 (to its stated rounding); for 3 it does not (for example the prose says WSO 4.2 Pa for strain 2106 at 50 °C, Table 3 says 4.3). The 17-cell ceiling is therefore a human judgement, not an automatic measurement.
- **Scoring.** Per cell: *correct* (within the rounding implied by the digits given, so "∼8,000" accepts 8047), *wrong*, or *abstain* (`null`). A crossover strain reported as "not detected" counts as its own value. `errors.py` classifies every non-correct cell as `TEXT != TABLE` (the agent copied the prose faithfully, the report disagrees with itself), `HALLUCINATION` (a number the prose does not contain), `WRONG`, or `MISSED` (the prose had it, the agent answered `null`).

## Results

All 7 strains, one run per model, same prompt and tools.

**Masked mode (Table 3 hidden)**

| | `gemini-3.5-flash-lite` | `gemini-3.8-flash` |
|---|---|---|
| Recoverable cells found (of 17) | 11 | **16** |
| Cells answered (of 70) | 14 | 19 |
| Correct vs Table 3 (of 70) | 11 (15.7%) | 16 (22.9%) |
| Answered but ≠ Table 3 | 3 | 3 |
| of which `HALLUCINATION` / `WRONG` | 0 / 0 | 0 / 0 |
| of which `TEXT != TABLE` | 3 | 3 |
| `MISSED` (value was in the prose) | 6 | 1 |

Consistent-accuracy by metric (share of 14 cells correct):

| | G′₀ | tan δ₀ | γ_y | γ_f | WSO |
|---|---|---|---|---|---|
| `flash-lite` | 21.4% | 21.4% | 0% | 35.7% | 0% |
| `3.8-flash` | 35.7% | 35.7% | 0% | 42.9% | 0% |

**Full mode (Table 3 visible), `gemini-3.5-flash-lite` only:** 70/70 cells and all SDs correct. This is a transcription task and shares its source with the ground truth, so it shows that the pipeline and the scorer work, not that extraction is solved.

**What the numbers say**

- Every value either model returned appears in the report's prose. The three "wrong" cells in each run are places where the prose and Table 3 disagree: 2103 at 30 °C (tan δ₀ ≈0.23 vs 0.24) and the WSO values of 2106 and 2107 at 50 °C. The agent copied the prose correctly.
- γ_y has no recoverable cell: the prose gives only ranges shared by several strains, and both models abstained rather than assigning one.
- The larger model recovers more but is not a strict superset: `gemini-3.8-flash` missed one value that `flash-lite` found (strain 2108, G′₀ at 30 °C).
- SD accuracy is near zero (3.0%) because the prose rarely states standard deviations.

## Limitations

- **One document and a hard-coded schema.** Strains, temperatures and the five metrics live in `schema.py`, and the table parser and heading detection are written for this PDF's text layer. This is a domain-specific pipeline, not a general extractor, and I have not tested it on a second document.
- **Single runs.** Outputs are not deterministic; a rerun can change a cell or two. The 16 vs 11 gap is large enough to be informative, but I would not claim a fixed difference.
- **Small sample.** 70 cells, 17 of them recoverable from prose.
- **Text only.** Figures are not read, so values shown only in plots are out of scope.
- **Ground truth and recoverable set are mine.** Table 3 is parsed from the same PDF text layer the agent reads (verified by eye), and the 17/20-cell recoverable set is a manual judgement.
- **BM25 retrieval, not embeddings.** Adequate for a 29-page report with distinctive tokens; not evaluated against dense retrieval.
- **Provider quirks.** Developed on Google AI Studio's free tier: 20 requests/day on the Flash models, frequent 503/504 errors (a single strain took up to ~9 minutes of retries). `run.py` caches per strain and skips strains that fail on server errors so a rerun resumes where it stopped.

## Run it

Python 3.10+ (developed on 3.12). Put the report at `data/SHP_Report.pdf` (not included in this repo).

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="..."                 # Google AI Studio key
export GEMINI_MODEL="gemini-3.5-flash-lite" # or gemini-3.8-flash

python make_ground_truth.py                 # Table 3 -> ground_truth.json (check it by eye)
python run.py --mode masked                 # or --mode full; add --strains 2106 3610 for a subset
python eval.py   out/$GEMINI_MODEL/masked/pred.json
python errors.py out/$GEMINI_MODEL/masked/pred.json
```

Results are stored per model under `out/<model>/<mode>/`. Roughly 30,000 tokens per strain.

## Repository layout

| File | Role |
|---|---|
| `ingest.py` | PDF → pages → section-tagged chunks and sentences; optional Table 3 masking |
| `schema.py` | Pydantic schema and plausibility ranges |
| `retrieve.py` | BM25 retriever with domain-aware tokenisation (`G′₀`, `tan δ₀`, `30◦C`, …) |
| `tools.py` | Agent tools and the validator behind `submit_extraction` |
| `agent.py` | Gemini function-calling loop with step budget, forced submit, retries |
| `run.py` | Per-strain runner with caching |
| `make_ground_truth.py` | Table 3 → `ground_truth.json` |
| `eval.py` | Cell-level scoring (correct / wrong / abstain, rounding-aware) |
| `errors.py` | Classifies every non-correct cell |
| `tests/oracle_masked.json` | Values stated in the prose (hand-built recoverable set) |

## Next

- A Claude backend behind the same tool interface, scored on the same cells.
- Move the schema into a config file and run on a second document with its own ground truth.
- Repeat runs to estimate variance; compare BM25 with embedding retrieval.