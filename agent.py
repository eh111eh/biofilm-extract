"""Tool-using extraction agent on the Gemini API (google-genai SDK).

The loop is written by hand (automatic function calling disabled) so every step is
logged to a trace and the validate->fix cycle is explicit.

Gemini 3 note: the model's own `content` object is appended to the history untouched.
It carries thought signatures that the API expects to see again on the next turn;
rebuilding the Parts by hand can trigger 400 errors.

Budget: after MAX_STEPS tool-using turns (or earlier if the model stalls) the agent is
forced to call submit_extraction, so a run can never end empty-handed.
"""
import os
import time

import httpx

from google import genai
from google.genai import errors, types

from schema import EXAMPLE
from tools import Toolbox

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")  # check the exact id in AI Studio
MAX_STEPS = 8
TIMEOUT_S = 90  # seconds to wait for one API answer before retrying
WARN_AT = 5  # from this step on, tell the model to wrap up

S = types.Schema


def _decl(name, desc, props=None, required=None):
    params = None
    if props:
        params = S(type="OBJECT", properties={k: S(type=t, description=d) for k, (t, d) in props.items()},
                   required=required or list(props))
    return types.FunctionDeclaration(name=name, description=desc, parameters=params)


DECLS = [
    _decl("find_mentions",
          "Return the prose sentences of the report that mention one strain and contain numbers. "
          "Best first step for finding values stated in the text. Sentences can mention several strains.",
          {"strain": ("STRING", "strain id, e.g. '2106' or '3610'")}),
    _decl("list_tables", "List the tables that are available in the report (id, caption, page)."),
    _decl("read_table", "Return the raw text of one listed table, one cell per line, row-major, header first.",
          {"table_id": ("INTEGER", "Table number from list_tables")}),
    _decl("search_report",
          "BM25 keyword search over report passages, for a specific metric or topic. "
          "Use descriptive words ('yield strain gamma_y 50C increase'), not bare numbers.",
          {"query": ("STRING", "descriptive keywords"),
           "section": ("STRING", "optional substring of a section title, e.g. '5.1.3'"),
           "k": ("INTEGER", "number of passages, 1-8")},
          required=["query"]),
    _decl("submit_extraction",
          "Validate and submit the records as a JSON string. Returns status 'ok' or a list of issues to fix. "
          "You are finished only when it returns status 'ok'.",
          {"records_json": ("STRING", "JSON object {\"records\": [...]} following the schema")}),
]

SYSTEM = f"""You extract structured rheology results from a Senior Honours Project report on Bacillus subtilis biofilms.
You have text only (figures are not readable). Work in three stages:
1. FIND   - in your FIRST turn call find_mentions (your strain) and list_tables together (parallel calls).
            If a listed table holds per-strain, per-temperature rheology values, read_table it.
2. READ   - use search_report only for a specific metric you still miss.
3. VERIFY - call submit_extraction. If it returns issues, fix exactly those and resubmit.

Extract, for ONE strain at BOTH 30 C and 50 C, these metrics (field names in parentheses):
  plateau storage modulus G'_0 in Pa (G0_prime_Pa), baseline loss tangent tan(delta)_0 (tan_delta0),
  yield strain in % (gamma_y_pct), crossover strain in % (gamma_f_pct), weak strain overshoot in Pa (WSO_Pa).

IMPORTANT - how to work:
- The report text sometimes refers to tables or appendices you cannot see. If list_tables does not list a table,
  it is NOT available: do not search for it. Many values will simply not be in the text you have.
- A submission with mostly null fields is a valid, good answer. Submit by your 4th turn at the latest.
  Do not keep searching for numbers that are not there.
- Only report what the report states for that exact strain AND temperature. Never infer, interpolate or compute.
  A range shared by several strains (e.g. '~160-185 Pa' for two strains) is NOT a value for either: use null.
- 'mean' is the headline value, 'sd' the +/- value if given. Numbers like 10,997 mean 10997.
- approximate=true when the text says ~, ≈ or 'approximate' for that value.
- not_detected=true (gamma_f_pct only) when the report says no G'=G'' crossover was detected for that condition; then mean=null.
- evidence: a verbatim quote (copy exact characters from tool output, 8-200 chars) that contains the number. Required
  for every non-null value and every not_detected=true. The submit step rejects quotes that are not in the report.
- If a table and prose disagree, prefer the table; do not average.
- Strain names: 'NCIB 3610' -> '3610', 'NRS 2106' -> '2106', etc. temperature_C is 30 or 50.
- Return exactly two records (30 and 50) for your strain.
Output format example (placeholder values only; strain XXXX does not exist):
{EXAMPLE}
Report sections you can filter on: {{sections}}
"""


def _call(client, contents, cfg):
    for attempt in range(8):
        try:
            return client.models.generate_content(model=MODEL, contents=contents, config=cfg)
        except httpx.TimeoutException:
            if attempt < 2:
                print(f"    ... no answer within {TIMEOUT_S}s, retrying ({attempt + 1}/2)", flush=True)
                continue
            raise SystemExit(f"\nThe API did not answer 3 times in a row (timeout {TIMEOUT_S}s). "
                             "Try again later or switch model. Finished strains are cached in out/.")
        except errors.APIError as e:
            msg = str(e)
            if getattr(e, "code", None) == 429 and ("PerDay" in msg or "per day" in msg.lower()):
                raise SystemExit(
                    f"\nDaily free-tier quota used up for model '{MODEL}'. Retrying will not help today.\n"
                    "Options: set another model (export GEMINI_MODEL=...), check limits at "
                    "https://aistudio.google.com/rate-limit, or enable billing. Finished strains are cached in out/.")
            if getattr(e, "code", None) in (429, 500, 503, 504) and attempt < 7:
                wait = min(60, 3 * 2 ** attempt)
                print(f"    ... API said {e.code}, waiting {wait}s (retry {attempt + 1}/7)", flush=True)
                time.sleep(wait)
                continue
            raise


def _say(contents, text):
    """Add a user text without creating two consecutive user turns."""
    if contents and contents[-1].role == "user":
        contents[-1].parts.append(types.Part(text=text))
    else:
        contents.append(types.Content(role="user", parts=[types.Part(text=text)]))


def _turn(client, tb, contents, cfg, trace, step):
    """One model call + execution of its tool calls. Returns (records_or_None, tokens, made_calls)."""
    resp = _call(client, contents, cfg)
    tokens = getattr(resp.usage_metadata, "total_token_count", 0) or 0
    cand = resp.candidates[0] if resp.candidates else None
    if cand is None or cand.content is None or not cand.content.parts:
        trace.append({"step": step, "event": "empty_response",
                      "finish_reason": str(getattr(cand, "finish_reason", None))})
        return None, tokens, False
    contents.append(cand.content)  # keep thought signatures intact
    calls = [p.function_call for p in cand.content.parts if p.function_call]
    if not calls:
        return None, tokens, False
    replies, result = [], None
    for c in calls:
        args = dict(c.args or {})
        out = tb.dispatch(c.name, args)
        print(f"    step {step}: {c.name}" + (f" -> {out.get('status')}" if c.name == "submit_extraction" else ""), flush=True)
        trace.append({"step": step, "tool": c.name, "args": args,
                      "result": str(out)[:400] + ("…" if len(str(out)) > 400 else "")})
        replies.append(types.Part.from_function_response(name=c.name, response={"result": out}))
        if c.name == "submit_extraction" and out.get("status") == "ok":
            result = out["records"]
    contents.append(types.Content(role="user", parts=replies))
    return result, tokens, True


def run_strain(client, tb: Toolbox, strain: str, max_steps: int = MAX_STEPS) -> dict:
    sections = sorted({c.section for c in tb.chunks})
    base = dict(
        system_instruction=SYSTEM.replace("{sections}", "; ".join(sections)),
        tools=[types.Tool(function_declarations=DECLS)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )  # temperature left at the API default on purpose (Gemini 3 guidance)
    cfg = types.GenerateContentConfig(**base)
    force_cfg = types.GenerateContentConfig(
        **base,
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
            mode="ANY", allowed_function_names=["submit_extraction"])),
    )
    contents = [types.Content(role="user", parts=[types.Part(
        text=f"Extract all five metrics for strain {strain} at 30 C and at 50 C (two records). Strain {strain} only.")])]
    trace, result, tokens, nudged = [], None, 0, False
    print(f"  [{strain}] started", flush=True)

    for step in range(1, max_steps + 1):
        if step == WARN_AT:
            _say(contents, f"Only {max_steps - step + 1} steps left. Call submit_extraction now with what you have; "
                           "use null for anything you did not find.")
        result, t, made_calls = _turn(client, tb, contents, cfg, trace, step)
        tokens += t
        if result is not None:
            break
        if not made_calls:
            if nudged:
                break
            nudged = True
            _say(contents, "Call submit_extraction with the JSON records. Do not answer in prose.")
            trace.append({"step": step, "event": "nudge"})

    forced = 0
    if result is None:  # budget used up / stalled: force a submission (up to 3 validated attempts)
        _say(contents, "Step budget used up. Submit your best extraction NOW via submit_extraction. "
                       "Use null for anything not found; do not guess.")
        for attempt in range(3):
            forced += 1
            result, t, made_calls = _turn(client, tb, contents, force_cfg, trace, f"forced{attempt + 1}")
            tokens += t
            if result is not None or not made_calls:
                break

    return {"strain": strain, "status": "ok" if result else "failed", "forced_submit": forced,
            "records": result or [], "steps": len(trace), "tokens": tokens, "trace": trace}


def make_client() -> genai.Client:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY (Google AI Studio key).")
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_S * 1000))