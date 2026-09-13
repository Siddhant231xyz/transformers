"""
CI Space client — reusable query functions for Claude Code.

Usage (from Claude Code):
    import sys; sys.path.insert(0, "C:/Users/yih-d/Desktop/Yih-Dar/Project/transformers/ci_db")
    from ci_client import last_pass, transitions, failures, error_history, investigate
    from ci_client import error_timeline   # → print-ready Unicode table

All functions return plain Python dicts/lists — no printing, no side effects.
Use pp() to pretty-print any result.
Use print(error_timeline(...)) to render a period/error history table.

Space base URL: https://ydshieh-transformers-daily-ci-db.hf.space
"""

import json
import pprint
import re
import textwrap
import time
from pathlib import Path

import requests

BASE = "https://ydshieh-transformers-daily-ci-db.hf.space"
_TIMEOUT = 30

# ── file-based cache for state_changes (mode-3) ────────────────────────────────
# One JSON file per (test, gpu), stored in ci_db/cache/.
# Default TTL: 24 hours. Pass cache_ttl=0 to force refresh, cache_ttl=-1 to disable.

_CACHE_DIR = Path(__file__).parent / "cache"
_DEFAULT_TTL = 86400  # seconds


def _cache_key(test: str, gpu: str) -> str:
    """Sanitize (test, gpu) into a safe filename."""
    safe = re.sub(r"[^\w\-.]", "_", test)
    return f"{safe}__{gpu}.json"


def _cache_load(test: str, gpu: str, ttl: int) -> dict | None:
    """Return cached mode-3 state_changes data if fresh, else None.

    ttl semantics: >0 = max age in seconds; 0 = force refresh; -1 = no cache; -2 = permanent (use if exists).
    """
    if ttl == -1:
        return None
    path = _CACHE_DIR / _cache_key(test, gpu)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        age = time.time() - data.get("_cached_at", 0)
        if ttl > 0 and age > ttl:
            return None
        # ttl == -2: always valid if file exists (permanent)
        return data["payload"]
    except Exception:
        return None


def _cache_save(test: str, gpu: str, payload: dict) -> None:
    """Persist mode-3 state_changes response to disk."""
    _CACHE_DIR.mkdir(exist_ok=True)
    path = _CACHE_DIR / _cache_key(test, gpu)
    path.write_text(
        json.dumps({"_cached_at": time.time(), "payload": payload}, indent=2),
        encoding="utf-8",
    )


def cache_clear(test: str | None = None, gpu: str | None = None) -> int:
    """
    Delete cache files.
      cache_clear()           → delete all cache files
      cache_clear(test, gpu)  → delete the specific file for (test, gpu)
    Returns the number of files deleted.
    """
    if not _CACHE_DIR.exists():
        return 0
    if test and gpu:
        path = _CACHE_DIR / _cache_key(test, gpu)
        if path.exists():
            path.unlink()
            return 1
        return 0
    files = list(_CACHE_DIR.glob("*.json"))
    for f in files:
        f.unlink()
    return len(files)


# ── low-level GET ──────────────────────────────────────────────────────────────

def _get(endpoint: str, **params) -> dict:
    """Raw GET to the Space. Returns parsed JSON dict. Raises on HTTP error."""
    # Remove None params so they don't appear in the URL
    params = {k: v for k, v in params.items() if v is not None}
    r = requests.get(f"{BASE}/{endpoint.lstrip('/')}", params=params, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


# ── pretty printer ─────────────────────────────────────────────────────────────

def pp(data) -> None:
    """Pretty-print any result from this module."""
    pprint.pprint(data, width=120, sort_dicts=False)


# ── health ─────────────────────────────────────────────────────────────────────

def health() -> dict:
    """
    Check Space health and get row counts.

    Returns:
        {"status": "ok", "db": "ci.db", "counts": {"ci_runs": N, "failures": N}}
    """
    return _get("health")


# ── failures ───────────────────────────────────────────────────────────────────

def failures(
    test:      str | None = None,
    gpu:       str | None = None,   # "single" | "multi"
    trace:     str | None = None,
    model:     str | None = None,   # entity_key e.g. "models_t5"
    from_date: str | None = None,   # "YYYY-MM-DD"
    to_date:   str | None = None,   # "YYYY-MM-DD"
    mode:      str = "contains",    # "contains" | "exact"
    sort:      str = "latest",      # "latest" | "oldest"
    limit:     int | None = None,   # None = no limit (all rows)
    job_type:  str = "models",
) -> list[dict]:
    """
    List individual test failure rows matching the given filters.

    Returns list of dicts with keys:
        date, entity_key, gpu_type, test_line, trace, job_link
    """
    data = _get("failures",
                test=test, gpu=gpu, trace=trace, model=model,
                **{"from": from_date, "to": to_date},
                mode=mode, sort=sort, limit=limit, job_type=job_type)
    return data.get("results", [])


# ── runs ───────────────────────────────────────────────────────────────────────

def runs(
    model:        str | None = None,
    gpu:          str | None = None,
    from_date:    str | None = None,
    to_date:      str | None = None,
    sort:         str = "latest",
    limit:        int | None = None,  # None = no limit (all rows)
    job_type:     str = "models",
    skip_invalid: bool = False,
) -> list[dict]:
    """
    List model-level CI run summaries.

    Returns list of dicts with keys:
        date, entity_key, job_type, success, errors, skipped, error,
        job_link_single, job_link_multi, commit_sha
    """
    data = _get("runs",
                model=model, gpu=gpu,
                **{"from": from_date, "to": to_date},
                sort=sort, limit=limit, job_type=job_type,
                skip_invalid=str(skip_invalid).lower())
    return data.get("results", [])


# ── last_pass ──────────────────────────────────────────────────────────────────

def last_pass(
    test:      str,
    gpu:       str,             # "single" | "multi"
    model:     str | None = None,
    mode:      str = "contains",
    from_date: str | None = None,
    to_date:   str | None = None,
    job_type:  str = "models",
) -> dict:
    """
    Find the last date this test passed for the given GPU.

    Returns dict with keys:
        test, model, gpu,
        last_pass   – last passing date, or None if never passed
        first_fail  – first failing date after last_pass, or None
        fail_count  – number of valid failing runs since last_pass
    """
    return _get("last-pass",
                test=test, gpu=gpu, model=model, mode=mode,
                **{"from": from_date, "to": to_date},
                job_type=job_type)


# ── transitions ────────────────────────────────────────────────────────────────

def transitions(
    test:         str,
    gpu:          str,              # "single" | "multi"
    model:        str | None = None,
    mode:         str = "contains",
    trace:        str | None = None,   # if set → trace-appearance mode
    error_change: bool = False,        # if True → any-error-change mode
    from_date:    str | None = None,
    to_date:      str | None = None,
    past_days:    int | None = None,
    sort:         str = "oldest",      # default oldest so history reads top-to-bottom
    limit:        int = 500,
    job_type:     str = "models",
) -> list[dict]:
    """
    Find state-change pairs for a test.

    Mode depends on arguments:
      - error_change=False, trace=None  → pass→fail pairs
            result keys: pass_date, fail_date
      - error_change=False, trace=TEXT  → (pass|fail-without-trace)→(fail-with-trace) pairs
            result keys: prev_date, prev_state, trace_date
      - error_change=True               → every state change (pass→fail + error changes)
            result keys: prev_date, prev_state, change_date, change_type, prev_trace, new_trace

    Returns list of result dicts (empty list if none found).
    """
    data = _get("transitions",
                test=test, gpu=gpu, model=model, mode=mode,
                trace=trace,
                error_change=str(error_change).lower() if error_change else None,
                **{"from": from_date, "to": to_date},
                past_days=past_days, sort=sort, limit=limit, job_type=job_type)
    return data.get("results", [])


# ── error_history ──────────────────────────────────────────────────────────────

def error_history(
    test:      str,
    gpu:       str,
    model:     str | None = None,
    mode:      str = "contains",
    from_date: str | None = None,
    to_date:   str | None = None,
    past_days: int | None = None,
    limit:     int = 500,
) -> list[dict]:
    """
    Convenience wrapper: transitions(error_change=True), sorted oldest-first.

    Shows every time the failure error changed (or the test newly broke).
    Result keys: prev_date, prev_state, change_date, change_type, prev_trace, new_trace
    """
    return transitions(test=test, gpu=gpu, model=model, mode=mode,
                       error_change=True, from_date=from_date, to_date=to_date,
                       past_days=past_days, sort="oldest", limit=limit)


# ── investigate ────────────────────────────────────────────────────────────────

def investigate(
    test:      str,
    gpu:       str,
    model:     str | None = None,
    mode:      str = "contains",
    from_date: str | None = None,
    to_date:   str | None = None,
) -> dict:
    """
    One-shot investigation: last_pass + pass→fail transitions for both GPUs
    (or the specified GPU only).

    Returns dict:
        {
          "single": { "last_pass": {...}, "transitions": [...] },
          "multi":  { "last_pass": {...}, "transitions": [...] },
        }
    If gpu is specified, only that GPU key is populated.
    """
    gpus = [gpu] if gpu else ["single", "multi"]
    result = {}
    for g in gpus:
        lp = last_pass(test=test, gpu=g, model=model, mode=mode,
                       from_date=from_date, to_date=to_date)
        tr = transitions(test=test, gpu=g, model=model, mode=mode,
                         from_date=from_date, to_date=to_date,
                         sort="oldest", limit=500)
        result[g] = {"last_pass": lp, "transitions": tr}
    return result


# ── daily_changes ──────────────────────────────────────────────────────────────

def daily_changes(
    date:      str | None = None,   # defaults to today
    gpu:       str = "single",
    job_type:  str = "models",
    limit:     int = 500,
    cache_ttl: int = _DEFAULT_TTL,
) -> list[dict]:
    """
    Return all tests that had a state change on the given date.

    change_type values:
      'new_failure'   – was passing on the previous valid run, now failing
      'error_changed' – was failing before with a different error trace

    Result keys: test_line, entity_key, gpu_type, change_type,
                 new_trace, prev_trace, prev_date

    date defaults to today (YYYY-MM-DD). Past dates are cached forever
    (data never changes); today's date uses the standard TTL.
    """
    import datetime
    if date is None:
        date = datetime.date.today().isoformat()

    today = datetime.date.today().isoformat()
    ttl = cache_ttl if date == today else -2  # sentinel: past date → cache forever

    # Load from cache
    cache_key_name = f"daily_changes__{date}__{gpu}.json"
    cache_path = _CACHE_DIR / cache_key_name
    if ttl != -1:  # caching enabled
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                age = time.time() - data.get("_cached_at", 0)
                # past dates: always valid; today: check TTL
                if ttl == -2 or (cache_ttl > 0 and age <= cache_ttl):
                    return data["payload"]
            except Exception:
                pass

    result = _get("daily-changes", date=date, gpu=gpu, job_type=job_type, limit=limit)
    payload = result.get("results", [])

    # Save to cache
    if ttl != -1:
        _CACHE_DIR.mkdir(exist_ok=True)
        cache_path.write_text(
            json.dumps({"_cached_at": time.time(), "payload": payload}, indent=2),
            encoding="utf-8",
        )

    return payload


def integration_test_timelines(
    date:        str | None = None,   # defaults to today
    gpu:         str = "single",
    filter_str:  str = "IntegrationTest",  # substring to filter test class names
    job_type:    str = "models",
    show_commit: bool = True,
    show_trace:  bool = False,
    cache_ttl:   int = _DEFAULT_TTL,
) -> dict[str, str]:
    """
    For a given date, fetch all state changes via daily_changes(), filter to
    tests whose class name contains `filter_str`, then return a dict mapping
    each test path → rendered mode-3 error_timeline table string.

    Usage:
        tables = integration_test_timelines('2026-08-22')
        for test, table in tables.items():
            print(table)
    """
    import datetime as _dt
    if date is None:
        date = _dt.date.today().isoformat()
    today = _dt.date.today().isoformat()

    changes = daily_changes(date=date, gpu=gpu, job_type=job_type, cache_ttl=cache_ttl)
    tests = [c["test_line"] for c in changes if filter_str in c["test_line"]]
    # deduplicate while preserving order
    seen = set()
    unique_tests = [t for t in tests if not (t in seen or seen.add(t))]

    # For past dates, treat test mode-3 caches as permanent (no re-fetch needed)
    timeline_ttl = cache_ttl if date == today else -2

    result = {}
    for test in unique_tests:
        result[test] = error_timeline(
            test, gpu=gpu,
            error_change=True,
            show_commit=show_commit,
            show_trace=show_trace,
            cache_ttl=timeline_ttl,
        )
    return result


# ── state_changes ──────────────────────────────────────────────────────────────

def state_changes(
    test:         str,
    gpu:          str,
    model:        str | None = None,
    mode:         str = "contains",
    trace:        str | None = None,   # if set → 3-state mode (pass|fail|fail-with-trace)
    error_change: bool = False,        # if True → also report trace changes within failures
    from_date:    str | None = None,
    to_date:      str | None = None,
    job_type:     str = "models",
    cache_ttl:    int = _DEFAULT_TTL,  # seconds; 0=force refresh, -1=disable cache
) -> dict:
    """
    Return every date where the test state changed, plus the initial state.

    Three modes:
      trace=None,  error_change=False → 2 states: pass | fail
      trace=TEXT,  error_change=False → 3 states: pass | fail | fail-with-trace
      error_change=True               → pass | fail (with trace field per period)

    Returns:
        {
          "initial": {"state": ..., "trace": ..., "commit_sha": ...} | None,
          "changes": [{"date": ..., "commit_sha": ..., "state": ..., "trace": ...,
                       "prev_date": ..., "prev_commit_sha": ...}, ...]
        }
    `prev_date` = last day of the previous state (the previous period spans up to this date).

    Cache:
      Mode-3 (error_change=True) responses are cached to disk in ci_db/cache/ so that
      subsequent calls (even in a new process) skip the Space request.
      cache_ttl=86400  use cache if < 24 h old  (default)
      cache_ttl=0      force a fresh fetch and update the cache
      cache_ttl=-1     disable cache entirely
      Modes 1 and 2 are NOT cached directly — use local_derive=True on periods() /
      error_timeline() to derive them from the cached mode-3 data.
    """
    # Cache applies only to mode-3 (error_change=True, no trace filter, no date range)
    # Date-filtered queries are not cached (results would change as new data arrives).
    use_cache = (error_change and trace is None
                 and from_date is None and to_date is None
                 and cache_ttl != -1)

    if use_cache and cache_ttl != 0:
        cached = _cache_load(test, gpu, cache_ttl)
        if cached is not None:
            return cached

    result = _get("state-changes",
                  test=test, gpu=gpu, model=model, mode=mode,
                  trace=trace,
                  error_change=str(error_change).lower() if error_change else None,
                  **{"from": from_date, "to": to_date},
                  job_type=job_type)

    if use_cache:
        _cache_save(test, gpu, result)

    return result


# ── error_timeline ─────────────────────────────────────────────────────────────
#
# Builds a ready-to-print Unicode box table from state_changes().
# All three modes (pass/fail, trace-appearance, error-change) are supported.
# Pass periods and recovery periods are included.
#
# Table structure:
#   ┌─────────────────────────┬───────────────────────────────────────────────┐
#   │ Period                  │ Error / Status                                │
#   ├─────────────────────────┼───────────────────────────────────────────────┤
#   │ ? → 2025-11-10          │ Passing                                       │
#   ├─────────────────────────┼───────────────────────────────────────────────┤
#   │ 2025-11-13 → 2026-01-05 │ torch.OutOfMemoryError: CUDA out of memory... │
#   ├─────────────────────────┼───────────────────────────────────────────────┤
#   │ 2026-01-06 → present    │ RuntimeError: ignore_mismatched_sizes         │
#   └─────────────────────────┴───────────────────────────────────────────────┘

_RE_LINE_PFX = re.compile(r"^\(line \d+\)\s*")


def _norm_trace(trace: str | None) -> str:
    """
    Normalize a trace for period-merging comparison.

    Steps:
      1. Strip '(line N)' prefix (line numbers change as code shifts).
      2. Take only the first line of the trace (skip stack frames).
      3. Truncate to 80 chars — the stable part of an error message.
         This collapses runtime-variable suffixes like process IDs or
         free-memory amounts that change day to day for the same error.

    Two traces that differ only in line numbers, stack frames, or runtime
    values (PIDs, memory amounts) will compare equal after normalization.
    """
    if not trace:
        return ""
    t = _RE_LINE_PFX.sub("", trace.strip())
    return t.split("\n")[0].strip()[:80]


def _trace_label(trace: str | None, state: str) -> str:
    """
    Human-readable error label for the Error column (display only, not comparison).
    Returns the full first line after stripping the line prefix.
    """
    if state == "pass":
        return "Passing"
    if state == "fail-with-trace":
        prefix = "[TARGET TRACE] "
    elif state == "fail":
        prefix = ""
    else:
        prefix = f"[{state}] "
    if not trace:
        return f"{prefix}(failing — no trace)"
    t = _RE_LINE_PFX.sub("", trace.strip())
    return prefix + t.split("\n")[0].strip()


def _fmt_period(start: str | None, end: str | None) -> str:
    if start is None and end is None:
        return "?"
    if start is None:
        return f"? → {end}"
    if end is None:
        return f"{start} → present"
    if start == end:
        return start
    return f"{start} → {end}"


def _build_periods(sc: dict) -> list[dict]:
    """
    Build contiguous periods from state_changes() output.

    sc = {"initial": {state, trace}, "changes": [{date, state, trace, prev_date}, ...]}

    The initial period starts at an unknown date (shown as None → "?").
    Each change gives us:
      - The new period's start date (change["date"])
      - The previous period's end date (change["prev_date"])

    Consecutive periods with the same normalized state+trace are merged.

    Returns list of dicts: {start, end, state, trace}
    """
    initial = sc.get("initial")
    changes = sc.get("changes", [])

    if initial is None:
        return []

    raw: list[dict] = [
        {
            "start":           None,
            "end":             changes[0]["prev_date"] if changes else None,
            "state":           initial["state"],
            "prev_state":      None,
            "trace":           initial["trace"],
            "commit_sha":      initial.get("commit_sha"),
            "end_commit_sha":  changes[0].get("prev_commit_sha") if changes else None,
        }
    ]

    for i, ch in enumerate(changes):
        next_ch = changes[i + 1] if i + 1 < len(changes) else None
        raw.append({
            "start":           ch["date"],
            "end":             next_ch["prev_date"] if next_ch else None,
            "state":           ch["state"],
            "prev_state":      raw[-1]["state"],
            "trace":           ch["trace"],
            "commit_sha":      ch.get("commit_sha"),
            "end_commit_sha":  next_ch.get("prev_commit_sha") if next_ch else None,
        })

    # Merge consecutive periods with the same normalized state+trace
    merged: list[dict] = []
    for p in raw:
        if (merged
                and merged[-1]["state"] == p["state"]
                and _norm_trace(merged[-1]["trace"]) == _norm_trace(p["trace"])):
            merged[-1]["end"]            = p["end"]
            merged[-1]["end_commit_sha"] = p["end_commit_sha"]
        else:
            merged.append(dict(p))

    return merged


def _derive_mode1(p3: list[dict]) -> list[dict]:
    """
    Derive mode-1 periods (pass | fail) from mode-3 periods.

    Mode 3 splits failing runs when the trace changes; mode 1 treats all consecutive
    fail periods as one.  We merge any run of fail periods into a single entry,
    keeping the start commit of the first and the end commit of the last.
    """
    merged: list[dict] = []
    for p in p3:
        if merged and merged[-1]["state"] == "fail" and p["state"] == "fail":
            merged[-1]["end"]            = p["end"]
            merged[-1]["end_commit_sha"] = p["end_commit_sha"]
        else:
            entry = dict(p)
            entry["prev_state"] = merged[-1]["state"] if merged else p.get("prev_state")
            merged.append(entry)
    return merged


def _derive_mode2(p3: list[dict], trace: str, mode: str) -> list[dict]:
    """
    Derive mode-2 periods (pass | fail | fail-with-trace) from mode-3 periods.

    Each fail period is re-labelled by checking whether its representative trace
    contains the target substring (or matches exactly if mode='exact').
    Consecutive same-state / same-trace periods are then merged.
    """
    relabeled: list[dict] = []
    for p in p3:
        p2 = dict(p)
        if p["state"] == "fail":
            t = p.get("trace") or ""
            has = (t == trace) if mode == "exact" else (trace in t)
            p2["state"] = "fail-with-trace" if has else "fail"
        relabeled.append(p2)

    merged: list[dict] = []
    for p in relabeled:
        if (merged
                and merged[-1]["state"] == p["state"]
                and _norm_trace(merged[-1]["trace"]) == _norm_trace(p["trace"])):
            merged[-1]["end"]            = p["end"]
            merged[-1]["end_commit_sha"] = p["end_commit_sha"]
        else:
            entry = dict(p)
            entry["prev_state"] = merged[-1]["state"] if merged else p.get("prev_state")
            merged.append(entry)
    return merged


def periods(
    test:         str,
    gpu:          str,
    model:        str | None = None,
    mode:         str = "contains",
    trace:        str | None = None,
    error_change: bool = False,
    from_date:    str | None = None,
    to_date:      str | None = None,
    local_derive: bool = False,   # fetch mode-3 once and derive mode 1/2 locally
    cache_ttl:    int = _DEFAULT_TTL,
) -> list[dict]:
    """
    Return the period list for a test — same as error_timeline() but as data, not a table.

    Each dict: {start, end, state, prev_state, trace, commit_sha, end_commit_sha}
      start          – first date of the period (None = unknown / before DB coverage)
      end            – last date of the period  (None = ongoing / present)
      state          – "pass" | "fail" | "fail-with-trace"
      prev_state     – state of the preceding period (None for first period)
      trace          – representative trace for the period (None for pass periods)
      commit_sha     – commit on the first day of this period
      end_commit_sha – commit on the last day of this period

    Same three modes — controlled by trace / error_change.

    local_derive=True:
      Fetches mode-3 (error_change=True) from the Space once, then derives
      mode-1 or mode-2 locally — no extra request.  Not applicable when
      error_change=True (mode 3 is already the native mode).
    """
    if local_derive and not error_change:
        sc3 = state_changes(test=test, gpu=gpu, model=model, mode=mode,
                            error_change=True,
                            from_date=from_date, to_date=to_date,
                            cache_ttl=cache_ttl)
        if sc3.get("initial") is None:
            return []
        p3 = _build_periods(sc3)
        return _derive_mode1(p3) if trace is None else _derive_mode2(p3, trace, mode)

    sc = state_changes(
        test=test, gpu=gpu, model=model, mode=mode,
        trace=trace, error_change=error_change,
        from_date=from_date, to_date=to_date,
        cache_ttl=cache_ttl,
    )
    if sc.get("initial") is None:
        return []
    return _build_periods(sc)


def error_timeline(
    test:             str,
    gpu:              str,
    model:            str | None = None,
    mode:             str = "contains",
    trace:            str | None = None,   # if set → trace-appearance mode (3 states)
    error_change:     bool = False,        # if True → error-change mode (track trace shifts)
    from_date:        str | None = None,
    to_date:          str | None = None,
    period_col_width: int = 25,
    error_col_width:  int = 60,
    max_label_len:    int = 140,
    show_commit:      bool = False,        # add a Commit column (first 8 chars of SHA)
    show_trace:       bool = False,        # add a Trace column (full raw trace text)
    trace_col_width:  int = 80,
    local_derive:     bool = False,        # fetch mode-3 once and derive mode 1/2 locally
    cache_ttl:        int = _DEFAULT_TTL,  # passed through to state_changes / periods
) -> str:
    """
    Build and return a Unicode box table showing the full state timeline.

    Three modes (mirrors the three transition query modes):
      trace=None,  error_change=False → pass / fail periods  (mode 1)
      trace=TEXT,  error_change=False → pass / fail / fail-with-trace  (mode 2)
      error_change=True               → pass / fail, split when error trace changes  (mode 3)

    Pass periods AND recovery periods (fail → pass) are always included.
    Consecutive periods with the same normalized error are merged into one row.

    Usage:
        # Mode 1
        print(error_timeline("tests/.../test_foo.py::Class::test_name", gpu="single"))
        # Mode 2
        print(error_timeline(..., trace="CUDAGraphs"))
        # Mode 3
        print(error_timeline(..., error_change=True))
    """
    _periods = periods(
        test=test, gpu=gpu, model=model, mode=mode,
        trace=trace, error_change=error_change,
        from_date=from_date, to_date=to_date,
        local_derive=local_derive, cache_ttl=cache_ttl,
    )

    if not _periods:
        return f"No data found.\n  test={test!r}\n  gpu={gpu}"

    # Build row data — "Error / Status" is a pure state label, never contains trace text.
    # Trace text lives only in the Trace column (when show_trace=True).
    def _label(p: dict) -> str:
        state      = p["state"]
        prev_state = p.get("prev_state")
        if state == "pass":
            return "Passing"
        if state == "fail-with-trace":          # mode 2
            return "Failing [FOUND TARGET TRACE]"
        # state == "fail"
        if trace is not None:                   # mode 2: "fail" means no target trace
            return "Failing [NO TARGET TRACE]"
        if prev_state == "fail":                # mode 3: fail→fail = trace changed
            return "Failing [TRACE CHANGED]"
        return "Failing"

    period_strs = [_fmt_period(p["start"], p["end"]) for p in _periods]
    labels      = [_label(p) for p in _periods]

    # Column widths
    w0 = max(period_col_width, max(len(s) for s in period_strs), len("Period"))
    w1 = min(error_col_width,  max(len(s) for s in labels))
    w1 = max(w1, len("Error / Status"))
    w2 = 8   # commit: first 8 chars of SHA ("Commit" = 6 chars, fits)
    w3 = trace_col_width

    # show_commit adds two columns: start-commit and end-commit
    col_widths  = [w0, w1] + ([w2, w2] if show_commit else []) + ([w3] if show_trace else [])
    col_headers = ["Period", "Error / Status"] + (["Commit", "End commit"] if show_commit else []) + (["Trace"] if show_trace else [])

    def hline(L, M, R):
        return L + M.join("─" * (w + 2) for w in col_widths) + R

    def row_line(cells):
        return "│ " + " │ ".join(c.ljust(w) for c, w in zip(cells, col_widths)) + " │"

    def _sha8(sha):
        return sha[:8] if sha else "(none)"

    def period_rows(p: dict, period_str: str, label: str) -> list[list[str]]:
        cols: list[list[str]] = []
        cols.append([period_str])
        cols.append(textwrap.wrap(label, w1) or [""])
        if show_commit:
            cols.append([_sha8(p.get("commit_sha"))])
            cols.append([_sha8(p.get("end_commit_sha"))])
        if show_trace:
            if p["state"] != "pass" and p.get("trace"):
                stripped = _RE_LINE_PFX.sub("", p["trace"].strip())
                cols.append(textwrap.wrap(stripped, w3) or [""])
            else:
                cols.append([""])
        n = max(len(c) for c in cols)
        return [[c[i] if i < len(c) else "" for c in cols] for i in range(n)]

    lines = [
        f"Test : {test}",
        f"GPU  : {gpu}",
        "",
        hline("┌", "┬", "┐"),
        row_line(col_headers),
        hline("├", "┼", "┤"),
    ]

    for i, p in enumerate(_periods):
        for row in period_rows(p, period_strs[i], labels[i]):
            lines.append(row_line(row))
        if i < len(_periods) - 1:
            lines.append(hline("├", "┼", "┤"))

    lines.append(hline("└", "┴", "┘"))
    return "\n".join(lines)


# ── get_changes_for_json ───────────────────────────────────────────────────────

_HF_DATASET_REPO  = "hf-internal-testing/transformers_daily_ci"
_HF_DATASET_TYPE  = "dataset"


def _parse_hf_url(path_or_url: str) -> str | None:
    """
    If path_or_url is a full HF blob URL, extract and return the filename portion.
    e.g. https://huggingface.co/datasets/<repo>/blob/main/2026-08-24/runs/.../f.json
      → '2026-08-24/runs/.../f.json'
    Returns None if not a HF URL.
    """
    m = re.search(r"huggingface\.co/datasets/[^/]+/[^/]+/blob/[^/]+/(.+)", path_or_url)
    return m.group(1) if m else None


def _load_json(path_or_url: str) -> dict:
    """
    Load a model_results.json from:
      - Full HF blob URL
      - HF relative path  (e.g. '2026-08-24/runs/.../model_results.json')
      - Local file path
    """
    hf_filename = _parse_hf_url(path_or_url)
    if hf_filename is None and not Path(path_or_url).exists():
        hf_filename = path_or_url  # treat as HF relative path

    if hf_filename is not None:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=_HF_DATASET_REPO,
            filename=hf_filename,
            repo_type=_HF_DATASET_TYPE,
            token=None,
        )
        with open(local, encoding="utf-8") as f:
            return json.load(f)

    with open(path_or_url, encoding="utf-8") as f:
        return json.load(f)


def _json_to_failures(data: dict, gpu: str) -> dict[str, tuple[str, str | None]]:
    """
    Parse model_results.json → {test_line: (entity_key, trace)} for the given gpu.
    Only includes tests that appear in the failure list (passing tests are absent).
    """
    result: dict[str, tuple[str, str | None]] = {}
    for entity_key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        fails_by_gpu = entry.get("failures") or {}
        if not isinstance(fails_by_gpu, dict):
            continue
        for f in (fails_by_gpu.get(gpu) or []):
            if not isinstance(f, dict):
                continue
            test_line = f.get("line", "")
            result[test_line] = (entity_key, f.get("trace") or None)
    return result


def get_changes_for_json(
    path_or_url: str,
    ref_date:    str | None = None,  # DB date to compare against; auto-inferred from path
    gpu:         str = "single",
    job_type:    str = "models",
) -> list[dict]:
    """
    Compare a model_results.json (manual / ad-hoc run) against the scheduled run
    already in the DB for ref_date, and return a daily_changes()-style list.

    path_or_url accepts:
      - Full HF blob URL:  https://huggingface.co/datasets/.../blob/main/2026-08-24/runs/.../model_results.json
      - HF relative path:  2026-08-24/runs/.../model_results.json
      - Local file path

    ref_date is inferred from the first YYYY-MM-DD in path_or_url when not given.

    Returns list of dicts with the same keys as daily_changes():
        test_line, entity_key, gpu_type, change_type, new_trace, prev_trace, prev_date

    change_type values:
        'new_failure'   – failing in manual run, passing in DB ref
        'new_pass'      – passing in manual run, failing in DB ref
        'error_changed' – failing in both but with a different trace
    Tests unchanged between the two runs (same pass/fail + same trace) are excluded.

    The JSON is never written to the DB.
    """
    # Infer ref_date from path if not provided
    if ref_date is None:
        m = re.search(r"\d{4}-\d{2}-\d{2}", path_or_url)
        if not m:
            raise ValueError(
                f"Cannot infer ref_date from {path_or_url!r}; pass ref_date= explicitly"
            )
        ref_date = m.group(0)

    # Load and parse the manual run
    data = _load_json(path_or_url)
    manual = _json_to_failures(data, gpu)          # {test_line: (entity_key, trace)}

    # Load reference JSON from HF (same source as the DB) — avoids Space API limits
    _JOB_PATHS = {"models": "ci_results_run_models_gpu/model_results.json"}
    ref_hf_path = f"{ref_date}/{_JOB_PATHS.get(job_type, _JOB_PATHS['models'])}"
    ref_data = _load_json(ref_hf_path)
    ref_fails = _json_to_failures(ref_data, gpu)   # {test_line: (entity_key, trace)}
    db_trace  = {t: tr for t, (_, tr) in ref_fails.items()}
    db_entity = {t: ek for t, (ek, _) in ref_fails.items()}

    # Diff
    results: list[dict] = []
    for test_line in sorted(set(manual) | set(db_trace)):
        in_manual = test_line in manual
        in_db     = test_line in db_trace

        if in_manual and not in_db:
            entity_key, new_trace = manual[test_line]
            results.append(dict(
                test_line=test_line, entity_key=entity_key, gpu_type=gpu,
                change_type="new_failure",
                new_trace=new_trace, prev_trace=None, prev_date=ref_date,
            ))
        elif in_db and not in_manual:
            results.append(dict(
                test_line=test_line,
                entity_key=db_entity.get(test_line, ""),
                gpu_type=gpu,
                change_type="new_pass",
                new_trace=None, prev_trace=db_trace[test_line], prev_date=ref_date,
            ))
        elif in_manual and in_db:
            entity_key, new_trace = manual[test_line]
            prev_trace = db_trace[test_line]
            if _norm_trace(new_trace) != _norm_trace(prev_trace):
                results.append(dict(
                    test_line=test_line, entity_key=entity_key, gpu_type=gpu,
                    change_type="error_changed",
                    new_trace=new_trace, prev_trace=prev_trace, prev_date=ref_date,
                ))

    return results


# ── filter_changes ─────────────────────────────────────────────────────────────

def filter_changes(
    changes:        list[dict],
    filter_str:     str | None       = None,   # include only if test_line contains this
    exclude_traces: list[str] | None = None,   # exclude if new_trace contains any of these
    change_types:   list[str] | None = None,   # allowlist of change_type values
) -> list[dict]:
    """
    Filter a list of change dicts (from daily_changes() or get_changes_for_json()).

    Args:
        filter_str:     Substring required in test_line  (e.g. "IntegrationTest")
        exclude_traces: Substrings — drop the row if new_trace contains any
                        (e.g. ["OutOfMemoryError", "CUDA out of memory"])
        change_types:   Allowlist of change_type values
                        (e.g. ["new_failure", "new_pass"])

    Returns a filtered list (original dicts are not modified).

    Works on the output of both daily_changes() and get_changes_for_json().
    """
    out = []
    for c in changes:
        if filter_str and filter_str not in (c.get("test_line") or ""):
            continue
        if change_types and c.get("change_type") not in change_types:
            continue
        if exclude_traces:
            trace = c.get("new_trace") or ""
            if any(ex in trace for ex in exclude_traces):
                continue
        out.append(c)
    return out
