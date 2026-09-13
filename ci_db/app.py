#!/usr/bin/env python3
"""
Transformers Daily CI DB — FastAPI query API.

Endpoints
---------
GET /health         Health check
GET /failures       List individual test failures
GET /runs           List model-level CI run summaries
GET /last-pass      Find the last date a test passed
GET /transitions    Find pass→fail (or trace-appearance) transition dates
"""

import os
import re
import sqlite3
import threading
import time
from datetime import date, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

DB_PATH = os.environ.get("DB_PATH", "ci.db")
INGEST_INTERVAL = int(os.environ.get("INGEST_INTERVAL", 600))  # seconds (default 10 min)

# Registered as a SQLite scalar function so SQL can call strip_line_prefix(col).
# Strips the leading "(line N)" prefix that the CI system prepends to every trace,
# so that error-change comparisons are not confused by line-number drift.
_re_line_prefix = re.compile(r"^\(line \d+\)\s*")

def _strip_line_prefix(trace: str | None) -> str | None:
    if trace is None:
        return None
    return _re_line_prefix.sub("", trace)

app = FastAPI(
    title="Transformers Daily CI DB",
    description="Query daily CI results for huggingface/transformers",
    version="1.0.0",
)


# ── background ingest ─────────────────────────────────────────────────────────

def _run_ingest() -> None:
    """Fetch any dates not yet in the DB, from the day after the latest present date up to today.
    For each newly ingested date, also fetch its commit SHA from GitHub (no token needed for 1/day)."""
    import ci_ingest
    import ci_backfill_sha
    import requests as _requests

    conn = sqlite3.connect(DB_PATH)
    ci_ingest.create_schema(conn)
    # Ensure commit_sha column exists (added by backfill script; safe to re-run)
    try:
        conn.execute("ALTER TABLE ci_runs ADD COLUMN commit_sha TEXT")
        conn.commit()
    except Exception:
        pass

    row = conn.execute("SELECT MAX(date) FROM ci_runs").fetchone()
    latest = row[0] if row and row[0] else None
    from_date = "2024-08-01" if latest is None else (date.fromisoformat(latest) + timedelta(days=1)).isoformat()
    today = date.today().isoformat()
    if from_date > today:
        conn.close()
        return

    print(f"[ingest] {from_date} → {today}", flush=True)
    new_dates = []
    for date_str in ci_ingest.iter_dates(from_date, today):
        for job_type, rel_path in ci_ingest.JOB_FILES.items():
            if ci_ingest.already_ingested(conn, date_str, job_type):
                continue
            data = ci_ingest.fetch_json(date_str, rel_path, token=None)
            if data is None:
                print(f"[ingest] {date_str} [{job_type}] not found", flush=True)
                continue
            n = ci_ingest.ingest_date(conn, date_str, job_type, data)
            print(f"[ingest] {date_str} [{job_type}] {n} entities", flush=True)
            new_dates.append(date_str)
            time.sleep(1)

    # Fetch commit SHA for each newly ingested date (1 GitHub API request per date)
    if new_dates:
        session = _requests.Session()
        session.headers["Accept"] = "application/vnd.github.v3+json"
        for date_str in sorted(set(new_dates)):
            link_row = conn.execute("""
                SELECT COALESCE(MIN(job_link_single), MIN(job_link_multi)) AS job_link
                FROM ci_runs WHERE date = ? AND commit_sha IS NULL
            """, (date_str,)).fetchone()
            if not link_row or not link_row[0]:
                continue
            run_id = ci_backfill_sha.extract_run_id(link_row[0])
            if not run_id:
                continue
            sha = ci_backfill_sha.fetch_sha(run_id, session)
            if sha and sha is not ci_backfill_sha.NOT_FOUND:
                conn.execute("UPDATE ci_runs SET commit_sha = ? WHERE date = ?", (sha, date_str))
                conn.commit()
                print(f"[sha] {date_str} sha={sha[:12]}", flush=True)
            else:
                print(f"[sha] {date_str} run={run_id} not found or error", flush=True)

    conn.close()

def _ingest_loop() -> None:
    while True:
        try:
            _run_ingest()
        except Exception as e:
            print(f"[ingest] error: {e}", flush=True)
        time.sleep(INGEST_INTERVAL)

threading.Thread(target=_ingest_loop, daemon=True, name="ci-ingest").start()


# ── DB ────────────────────────────────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.create_function("strip_line_prefix", 1, _strip_line_prefix)
    return conn


# ── helpers ───────────────────────────────────────────────────────────────────

def valid_run_sql(gpu: Optional[str], alias: str = "r") -> str:
    """
    Returns SQL fragment that is TRUE when a CI run is valid for the given GPU.
    The model-level `error` flag is set if ANY GPU had issues, so per-GPU validity
    is determined solely by whether job_link_{gpu} is present.
    """
    if gpu == "single":
        return f"{alias}.job_link_single IS NOT NULL AND {alias}.job_link_single != ''"
    if gpu == "multi":
        return f"{alias}.job_link_multi  IS NOT NULL AND {alias}.job_link_multi  != ''"
    return (
        f"{alias}.success > 0 AND {alias}.error = 0 AND ("
        f"{alias}.job_link_single IS NOT NULL OR "
        f"{alias}.job_link_multi  IS NOT NULL)"
    )


def make_match(col: str, value: str, mode: str) -> tuple[str, str]:
    if mode == "exact":
        return f"{col} = ?", value
    return f"{col} LIKE ?", f"%{value}%"


def entity_key_from_test(test_line: str) -> Optional[str]:
    """
    Infer entity_key from test path.
    e.g. "tests/models/t5/test_modeling_t5.py::..." → "models_t5"
    """
    m = re.match(r"tests/(\w+)/(\w+)/", test_line)
    return f"{m.group(1)}_{m.group(2)}" if m else None


def rows_to_dicts(rows) -> list[dict]:
    return [dict(r) for r in rows]


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    return """
    <html><body>
    <h2>Transformers Daily CI DB</h2>
    <p>See <a href="/docs">/docs</a> for the interactive API explorer.</p>
    <ul>
      <li><a href="/health">/health</a></li>
      <li><a href="/failures?limit=5">/failures</a></li>
      <li><a href="/runs?limit=5">/runs</a></li>
      <li>/last-pass?test=...&amp;gpu=single</li>
      <li>/transitions?test=...&amp;gpu=single</li>
    </ul>
    </body></html>
    """


@app.get("/health")
def health():
    try:
        conn = get_conn()
        counts = {
            "ci_runs":  conn.execute("SELECT COUNT(*) FROM ci_runs").fetchone()[0],
            "failures": conn.execute("SELECT COUNT(*) FROM failures").fetchone()[0],
        }
        conn.close()
        return {"status": "ok", "db": DB_PATH, "counts": counts}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/failures")
def failures(
    test:      Optional[str] = Query(None, description="Filter by test line"),
    trace:     Optional[str] = Query(None, description="Filter by trace text"),
    model:     Optional[str] = Query(None, description="entity_key, e.g. models_t5"),
    gpu:       Optional[str] = Query(None, enum=["single", "multi"]),
    from_date: Optional[str] = Query(None, alias="from", description="YYYY-MM-DD"),
    to_date:   Optional[str] = Query(None, alias="to",   description="YYYY-MM-DD"),
    mode:      str           = Query("contains", enum=["contains", "exact"]),
    sort:      str           = Query("latest",   enum=["latest", "oldest"]),
    limit:     Optional[int] = Query(None, ge=1, description="Max rows to return; omit for all rows"),
    job_type:  str           = Query("models"),
):
    clauses: list[str] = ["f.job_type = ?"]
    params:  list      = [job_type]

    if test:
        c, p = make_match("f.test_line", test, mode)
        clauses.append(c); params.append(p)
    if trace:
        c, p = make_match("f.trace", trace, mode)
        clauses.append(c); params.append(p)
    if model:
        clauses.append("f.entity_key = ?"); params.append(model)
    if gpu:
        clauses.append("f.gpu_type = ?"); params.append(gpu)
    if from_date:
        clauses.append("f.date >= ?"); params.append(from_date)
    if to_date:
        clauses.append("f.date <= ?"); params.append(to_date)

    order = "ASC" if sort == "oldest" else "DESC"
    limit_clause = f"LIMIT {limit}" if limit is not None else ""
    sql = f"""
        SELECT f.date, f.entity_key, f.gpu_type, f.test_line, f.trace, f.job_link
        FROM failures f
        WHERE {" AND ".join(clauses)}
        ORDER BY f.date {order}, f.entity_key, f.gpu_type
        {limit_clause}
    """
    conn = get_conn()
    try:
        rows = conn.execute(sql, params).fetchall()
        return {"count": len(rows), "results": rows_to_dicts(rows)}
    finally:
        conn.close()


@app.get("/runs")
def runs(
    model:        Optional[str] = Query(None, description="entity_key, e.g. models_t5"),
    gpu:          Optional[str] = Query(None, enum=["single", "multi"]),
    from_date:    Optional[str] = Query(None, alias="from", description="YYYY-MM-DD"),
    to_date:      Optional[str] = Query(None, alias="to",   description="YYYY-MM-DD"),
    sort:         str           = Query("latest", enum=["latest", "oldest"]),
    limit:        Optional[int] = Query(None, ge=1, description="Max rows to return; omit for all rows"),
    job_type:     str           = Query("models"),
    skip_invalid: bool          = Query(False, description="Exclude runs with missing job_link for the queried GPU"),
):
    clauses: list[str] = ["r.job_type = ?"]
    params:  list      = [job_type]

    if model:
        clauses.append("r.entity_key = ?"); params.append(model)
    if from_date:
        clauses.append("r.date >= ?"); params.append(from_date)
    if to_date:
        clauses.append("r.date <= ?"); params.append(to_date)
    if skip_invalid:
        clauses.append(valid_run_sql(gpu))

    order = "ASC" if sort == "oldest" else "DESC"
    limit_clause = f"LIMIT {limit}" if limit is not None else ""
    sql = f"""
        SELECT r.date, r.entity_key, r.job_type,
               r.success, r.errors, r.skipped, r.error,
               r.job_link_single, r.job_link_multi, r.commit_sha
        FROM ci_runs r
        WHERE {" AND ".join(clauses)}
        ORDER BY r.date {order}, r.entity_key
        {limit_clause}
    """
    conn = get_conn()
    try:
        rows = conn.execute(sql, params).fetchall()
        return {"count": len(rows), "results": rows_to_dicts(rows)}
    finally:
        conn.close()


@app.get("/last-pass")
def last_pass(
    test:      str           = Query(..., description="Test line (required)"),
    gpu:       str           = Query(..., enum=["single", "multi"]),
    model:     Optional[str] = Query(None, description="entity_key (inferred from test path if omitted)"),
    mode:      str           = Query("contains", enum=["contains", "exact"]),
    from_date: Optional[str] = Query(None, alias="from"),
    to_date:   Optional[str] = Query(None, alias="to"),
    job_type:  str           = Query("models"),
):
    entity_key = model or entity_key_from_test(test)
    if not entity_key:
        raise HTTPException(422, "Cannot infer entity_key from test path; supply ?model=")

    valid_sql  = valid_run_sql(gpu)
    m_sql, m_p = make_match("f.test_line", test, mode)

    sql = f"""
        SELECT MAX(r.date) AS last_pass
        FROM ci_runs r
        WHERE r.entity_key = ? AND r.job_type = ?
          AND {valid_sql}
          AND NOT EXISTS (
              SELECT 1 FROM failures f
              WHERE f.date       = r.date
                AND f.job_type   = r.job_type
                AND f.entity_key = r.entity_key
                AND f.gpu_type   = ?
                AND {m_sql}
          )
    """
    params = [entity_key, job_type, gpu, m_p]
    if from_date:
        sql += " AND r.date >= ?"; params.append(from_date)
    if to_date:
        sql += " AND r.date <= ?"; params.append(to_date)

    conn = get_conn()
    try:
        row  = conn.execute(sql, params).fetchone()
        last = row["last_pass"] if row else None

        result: dict = {
            "test": test, "model": entity_key, "gpu": gpu, "last_pass": last,
            "first_fail": None, "fail_count": 0,
        }

        if last:
            fail_sql = f"""
                SELECT MIN(r.date) AS first_fail, COUNT(*) AS fail_count
                FROM ci_runs r
                WHERE r.entity_key = ? AND r.job_type = ?
                  AND {valid_sql} AND r.date > ?
                  AND EXISTS (
                      SELECT 1 FROM failures f
                      WHERE f.date       = r.date
                        AND f.job_type   = r.job_type
                        AND f.entity_key = r.entity_key
                        AND f.gpu_type   = ?
                        AND {m_sql}
                  )
            """
            fr = conn.execute(
                fail_sql, [entity_key, job_type, last, gpu, m_p]
            ).fetchone()
            if fr:
                result["first_fail"] = fr["first_fail"]
                result["fail_count"] = fr["fail_count"]

        return result
    finally:
        conn.close()


@app.get("/state-changes")
def state_changes(
    test:         str           = Query(..., description="Test line (required)"),
    gpu:          str           = Query(..., enum=["single", "multi"]),
    model:        Optional[str] = Query(None),
    mode:         str           = Query("contains", enum=["contains", "exact"]),
    trace:        Optional[str] = Query(None,
                                        description="If set, use 3 states: pass | fail | fail-with-trace"),
    error_change: bool          = Query(False,
                                        description="If true, also report fail→fail transitions "
                                                    "when the error trace changes"),
    from_date:    Optional[str] = Query(None, alias="from"),
    to_date:      Optional[str] = Query(None, alias="to"),
    job_type:     str           = Query("models"),
):
    """
    Return every date where the test state changed, plus the initial state.

    Three modes (determined by parameters):

    1. pass/fail only  (trace=None, error_change=False)
       States: 'pass' | 'fail'
       Reports both pass→fail AND fail→pass (recovery) transitions.

    2. trace-appearance  (trace=TEXT, error_change=False)
       States: 'pass' | 'fail' | 'fail-with-trace'
       'fail' = test failing but target trace string absent.

    3. error-change  (error_change=True)
       States: 'pass' | 'fail'
       Also reports fail→fail when the normalized error trace changes.
       The `trace` field in each row carries the representative trace for that period.

    Response:
        {
          "initial": {"state": ..., "trace": ...},
          "changes": [{"date": ..., "state": ..., "trace": ..., "prev_date": ...}, ...]
        }
    `initial` is the state before the first recorded change.
    `prev_date` = last day of the previous state (so previous period spans up to prev_date).
    """
    entity_key = model or entity_key_from_test(test)
    if not entity_key:
        raise HTTPException(422, "Cannot infer entity_key from test path; supply ?model=")

    valid_sql  = valid_run_sql(gpu)
    m_sql, m_p = make_match("f.test_line", test, mode)

    date_filter, date_params = "", []
    if from_date:
        date_filter += " AND r.date >= ?"; date_params.append(from_date)
    if to_date:
        date_filter += " AND r.date <= ?"; date_params.append(to_date)

    conn = get_conn()
    try:
        if trace and not error_change:
            # 3-state mode: pass | fail | fail-with-trace
            t_sql, t_p = make_match("f.trace", trace, mode)
            # ? order: gpu, m_p, t_p  (trace EXISTS),  gpu, m_p  (fail EXISTS),
            #          gpu, m_p  (trace subquery),  entity_key, job_type,  [date params]
            params = [gpu, m_p, t_p, gpu, m_p, gpu, m_p, entity_key, job_type] + date_params
            state_expr = f"""
                CASE
                    WHEN EXISTS (
                        SELECT 1 FROM failures f
                        WHERE f.date=r.date AND f.job_type=r.job_type
                          AND f.entity_key=r.entity_key AND f.gpu_type=?
                          AND {m_sql} AND {t_sql}
                    ) THEN 'fail-with-trace'
                    WHEN EXISTS (
                        SELECT 1 FROM failures f
                        WHERE f.date=r.date AND f.job_type=r.job_type
                          AND f.entity_key=r.entity_key AND f.gpu_type=?
                          AND {m_sql}
                    ) THEN 'fail'
                    ELSE 'pass'
                END
            """
            change_cond = "prev_state IS NULL OR state != prev_state"
            trace_select = f"""
                (SELECT f.trace FROM failures f
                 WHERE f.date=r.date AND f.job_type=r.job_type
                   AND f.entity_key=r.entity_key AND f.gpu_type=?
                   AND {m_sql}
                 ORDER BY f.rowid LIMIT 1) AS trace
            """

        elif error_change:
            # error-change mode: report pass↔fail AND fail→fail when trace changes
            # ? order: gpu, m_p (EXISTS),  gpu, m_p (trace subquery),
            #          entity_key, job_type,  [date params]
            params = [gpu, m_p, gpu, m_p, entity_key, job_type] + date_params
            state_expr = f"""
                CASE WHEN EXISTS (
                    SELECT 1 FROM failures f
                    WHERE f.date=r.date AND f.job_type=r.job_type
                      AND f.entity_key=r.entity_key AND f.gpu_type=?
                      AND {m_sql}
                ) THEN 'fail' ELSE 'pass' END
            """
            trace_select = f"""
                (SELECT f.trace FROM failures f
                 WHERE f.date=r.date AND f.job_type=r.job_type
                   AND f.entity_key=r.entity_key AND f.gpu_type=?
                   AND {m_sql}
                 ORDER BY f.rowid LIMIT 1) AS trace
            """
            change_cond = """
                prev_state IS NULL
                OR state != prev_state
                OR (state='fail'
                    AND SUBSTR(COALESCE(strip_line_prefix(trace),''), 1, 80)
                     != SUBSTR(COALESCE(strip_line_prefix(prev_trace),''), 1, 80))
            """

        else:
            # Simple pass/fail mode — also fetch a representative trace per day
            # so the table can display the error text.
            # ? order: gpu, m_p (EXISTS),  gpu, m_p (trace subquery),
            #          entity_key, job_type,  [date params]
            params = [gpu, m_p, gpu, m_p, entity_key, job_type] + date_params
            state_expr = f"""
                CASE WHEN EXISTS (
                    SELECT 1 FROM failures f
                    WHERE f.date=r.date AND f.job_type=r.job_type
                      AND f.entity_key=r.entity_key AND f.gpu_type=?
                      AND {m_sql}
                ) THEN 'fail' ELSE 'pass' END
            """
            trace_select = f"""
                (SELECT f.trace FROM failures f
                 WHERE f.date=r.date AND f.job_type=r.job_type
                   AND f.entity_key=r.entity_key AND f.gpu_type=?
                   AND {m_sql}
                 ORDER BY f.rowid LIMIT 1) AS trace
            """
            change_cond = "prev_state IS NULL OR state != prev_state"

        sql = f"""
            WITH daily_state AS (
                SELECT r.date,
                    r.commit_sha,
                    {state_expr} AS state,
                    {trace_select}
                FROM ci_runs r
                WHERE r.entity_key=? AND r.job_type=?
                  AND {valid_sql} {date_filter}
            ),
            with_prev AS (
                SELECT date, commit_sha, state, trace,
                       LAG(date)       OVER (ORDER BY date) AS prev_date,
                       LAG(commit_sha) OVER (ORDER BY date) AS prev_commit_sha,
                       LAG(state)      OVER (ORDER BY date) AS prev_state,
                       LAG(trace)      OVER (ORDER BY date) AS prev_trace
                FROM daily_state
            )
            SELECT date, commit_sha, state, trace, prev_date, prev_commit_sha
            FROM with_prev
            WHERE {change_cond}
            ORDER BY date
        """

        rows = conn.execute(sql, params).fetchall()

        if not rows:
            return {"initial": None, "changes": []}

        # First row (prev_date IS NULL) = initial state
        first = rows[0]
        initial = {"state": first["state"], "trace": first["trace"], "commit_sha": first["commit_sha"]}
        changes = [
            {
                "date":            r["date"],
                "commit_sha":      r["commit_sha"],
                "state":           r["state"],
                "trace":           r["trace"],
                "prev_date":       r["prev_date"],
                "prev_commit_sha": r["prev_commit_sha"],
            }
            for r in rows[1:]
        ]
        return {"initial": initial, "changes": changes}

    finally:
        conn.close()


@app.get("/daily-changes")
def daily_changes(
    date:     str           = Query(..., description="Date to check (YYYY-MM-DD)"),
    gpu:      str           = Query("single", enum=["single", "multi"]),
    job_type: str           = Query("models"),
    limit:    int           = Query(500, ge=1, le=2000),
):
    """
    Return all tests that had a state change on the given date:
      - 'new_failure'   : was passing on the previous valid run, failing today
      - 'error_changed' : was failing on the previous valid run with a different error trace

    Result fields: test_line, entity_key, gpu_type, change_type,
                   new_trace, prev_trace, prev_date
    """
    valid_sql = valid_run_sql(gpu)

    sql = f"""
        WITH
        today_fails AS (
            SELECT f.test_line, f.entity_key, f.gpu_type,
                   MIN(f.rowid) AS min_rowid
            FROM failures f
            WHERE f.date = ? AND f.gpu_type = ? AND f.job_type = ?
            GROUP BY f.test_line, f.entity_key, f.gpu_type
        ),
        today_traces AS (
            SELECT tf.test_line, tf.entity_key, tf.gpu_type,
                   f.trace AS new_trace
            FROM today_fails tf
            JOIN failures f ON f.rowid = tf.min_rowid
        ),
        prev_runs AS (
            SELECT r.entity_key, MAX(r.date) AS prev_date
            FROM ci_runs r
            WHERE r.date < ? AND r.job_type = ?
              AND {valid_sql}
            GROUP BY r.entity_key
        ),
        prev_fails AS (
            SELECT f.test_line, f.entity_key, f.gpu_type,
                   (SELECT f2.trace FROM failures f2
                    WHERE f2.date = f.date AND f2.entity_key = f.entity_key
                      AND f2.gpu_type = f.gpu_type AND f2.test_line = f.test_line
                      AND f2.job_type = ?
                    ORDER BY f2.rowid LIMIT 1) AS prev_trace
            FROM failures f
            JOIN prev_runs pr ON f.entity_key = pr.entity_key AND f.date = pr.prev_date
            WHERE f.gpu_type = ? AND f.job_type = ?
        )
        SELECT tt.test_line, tt.entity_key, tt.gpu_type,
               tt.new_trace,
               CASE WHEN pf.test_line IS NULL THEN 'new_failure'
                    ELSE 'error_changed' END AS change_type,
               pf.prev_trace,
               pr.prev_date
        FROM today_traces tt
        LEFT JOIN prev_runs pr ON tt.entity_key = pr.entity_key
        LEFT JOIN prev_fails pf ON tt.test_line  = pf.test_line
                                AND tt.entity_key = pf.entity_key
                                AND tt.gpu_type   = pf.gpu_type
        WHERE pf.test_line IS NULL
           OR SUBSTR(COALESCE(strip_line_prefix(tt.new_trace), ''), 1, 80)
           != SUBSTR(COALESCE(strip_line_prefix(pf.prev_trace), ''), 1, 80)
        ORDER BY change_type, tt.entity_key, tt.test_line
        LIMIT ?
    """
    # param order: date, gpu, job_type | date, job_type | job_type | gpu, job_type | limit
    params = [date, gpu, job_type, date, job_type, job_type, gpu, job_type, limit]

    conn = get_conn()
    try:
        rows = conn.execute(sql, params).fetchall()
        return {"date": date, "gpu": gpu, "count": len(rows), "results": rows_to_dicts(rows)}
    finally:
        conn.close()


@app.get("/transitions")
def transitions(
    test:         str           = Query(..., description="Test line (required)"),
    gpu:          str           = Query(..., enum=["single", "multi"]),
    model:        Optional[str] = Query(None, description="entity_key (inferred from test path if omitted)"),
    mode:         str           = Query("contains", enum=["contains", "exact"]),
    trace:        Optional[str] = Query(None,
                                        description="If set, find (pass|fail-without-trace)→(fail-with-trace) "
                                                    "transitions instead of pass→fail"),
    error_change: bool          = Query(False,
                                        description="If true, report every state change: "
                                                    "pass→fail AND fail→fail-with-different-error. "
                                                    "Results include prev_trace and new_trace."),
    from_date:    Optional[str] = Query(None, alias="from"),
    to_date:      Optional[str] = Query(None, alias="to"),
    past_days:    Optional[int] = Query(None, description="Shorthand for from=(today - N days)"),
    sort:         str           = Query("latest", enum=["latest", "oldest"]),
    limit:        int           = Query(50, ge=1, le=500),
    job_type:     str           = Query("models"),
):
    entity_key = model or entity_key_from_test(test)
    if not entity_key:
        raise HTTPException(422, "Cannot infer entity_key from test path; supply ?model=")

    effective_from = from_date
    if past_days and not effective_from:
        effective_from = (date.today() - timedelta(days=past_days)).isoformat()

    valid_sql  = valid_run_sql(gpu)
    m_sql, m_p = make_match("f.test_line", test, mode)
    order      = "ASC" if sort == "oldest" else "DESC"

    date_filter, date_params = "", []
    if effective_from:
        date_filter += " AND r.date >= ?"; date_params.append(effective_from)
    if to_date:
        date_filter += " AND r.date <= ?"; date_params.append(to_date)

    conn = get_conn()
    try:
        if error_change:
            # Report every state change:
            #   pass → fail (any error)
            #   fail → fail with a different error trace
            #
            # One representative trace per day is chosen (first inserted row).
            # ? order: gpu, m_p (EXISTS), gpu, m_p (trace subquery),
            #          entity_key, job_type, [date params]
            ec_m_sql, ec_m_p = make_match("f.test_line", test, mode)
            params = [gpu, ec_m_p, gpu, ec_m_p, entity_key, job_type] + date_params
            sql = f"""
                WITH daily_state AS (
                    SELECT
                        r.date,
                        CASE WHEN EXISTS (
                            SELECT 1 FROM failures f
                            WHERE f.date=r.date AND f.job_type=r.job_type
                              AND f.entity_key=r.entity_key AND f.gpu_type=?
                              AND {ec_m_sql}
                        ) THEN 1 ELSE 0 END AS is_failing,
                        (SELECT f.trace FROM failures f
                         WHERE f.date=r.date AND f.job_type=r.job_type
                           AND f.entity_key=r.entity_key AND f.gpu_type=?
                           AND {ec_m_sql}
                         ORDER BY f.rowid LIMIT 1) AS trace
                    FROM ci_runs r
                    WHERE r.entity_key=? AND r.job_type=?
                      AND {valid_sql} {date_filter}
                ),
                with_prev AS (
                    SELECT date, is_failing, trace,
                           LAG(date)       OVER (ORDER BY date) AS prev_date,
                           LAG(is_failing) OVER (ORDER BY date) AS prev_is_failing,
                           LAG(trace)      OVER (ORDER BY date) AS prev_trace
                    FROM daily_state
                )
                SELECT
                    prev_date,
                    CASE WHEN prev_is_failing=0 THEN 'pass' ELSE 'fail' END AS prev_state,
                    date AS change_date,
                    CASE WHEN prev_is_failing=0 THEN 'new-failure' ELSE 'error-changed' END AS change_type,
                    prev_trace,
                    trace AS new_trace
                FROM with_prev
                WHERE
                    (prev_is_failing=0 AND is_failing=1)
                    OR (prev_is_failing=1 AND is_failing=1
                        AND COALESCE(strip_line_prefix(trace),'') != COALESCE(strip_line_prefix(prev_trace),''))
                ORDER BY prev_date {order}
                LIMIT {limit}
            """
            rows = conn.execute(sql, params).fetchall()
            return {
                "test": test, "model": entity_key, "gpu": gpu,
                "mode": "error-change",
                "count": len(rows), "results": rows_to_dicts(rows),
            }

        if trace:
            t_sql, t_p = make_match("f.trace", trace, mode)
            # ? order: gpu, m_p, t_p  (trace EXISTS), gpu, m_p  (fail EXISTS),
            #          entity_key, job_type, [date params]
            params = [gpu, m_p, t_p, gpu, m_p, entity_key, job_type] + date_params
            sql = f"""
                WITH valid_runs AS (
                    SELECT r.date,
                        CASE
                            WHEN EXISTS (
                                SELECT 1 FROM failures f
                                WHERE f.date=r.date AND f.job_type=r.job_type
                                  AND f.entity_key=r.entity_key AND f.gpu_type=?
                                  AND {m_sql} AND {t_sql}
                            ) THEN 'trace'
                            WHEN EXISTS (
                                SELECT 1 FROM failures f
                                WHERE f.date=r.date AND f.job_type=r.job_type
                                  AND f.entity_key=r.entity_key AND f.gpu_type=?
                                  AND {m_sql}
                            ) THEN 'fail'
                            ELSE 'pass'
                        END AS state
                    FROM ci_runs r
                    WHERE r.entity_key=? AND r.job_type=?
                      AND {valid_sql} {date_filter}
                ),
                with_prev AS (
                    SELECT date, state,
                           LAG(date)  OVER (ORDER BY date) AS prev_date,
                           LAG(state) OVER (ORDER BY date) AS prev_state
                    FROM valid_runs
                )
                SELECT prev_date, prev_state, date AS trace_date
                FROM with_prev
                WHERE state='trace' AND prev_state != 'trace'
                ORDER BY prev_date {order}
                LIMIT {limit}
            """
            rows = conn.execute(sql, params).fetchall()
            return {
                "test": test, "model": entity_key, "gpu": gpu,
                "trace": trace, "mode": "trace-appearance",
                "count": len(rows), "results": rows_to_dicts(rows),
            }
        else:
            # ? order: gpu, m_p  (EXISTS), entity_key, job_type, [date params]
            params = [gpu, m_p, entity_key, job_type] + date_params
            sql = f"""
                WITH valid_runs AS (
                    SELECT r.date,
                        CASE WHEN EXISTS (
                            SELECT 1 FROM failures f
                            WHERE f.date=r.date AND f.job_type=r.job_type
                              AND f.entity_key=r.entity_key AND f.gpu_type=?
                              AND {m_sql}
                        ) THEN 'fail' ELSE 'pass' END AS state
                    FROM ci_runs r
                    WHERE r.entity_key=? AND r.job_type=?
                      AND {valid_sql} {date_filter}
                ),
                with_prev AS (
                    SELECT date, state,
                           LAG(date)  OVER (ORDER BY date) AS prev_date,
                           LAG(state) OVER (ORDER BY date) AS prev_state
                    FROM valid_runs
                )
                SELECT prev_date AS pass_date, date AS fail_date
                FROM with_prev
                WHERE state='fail' AND prev_state='pass'
                ORDER BY pass_date {order}
                LIMIT {limit}
            """
            rows = conn.execute(sql, params).fetchall()
            return {
                "test": test, "model": entity_key, "gpu": gpu,
                "mode": "pass-to-fail",
                "count": len(rows), "results": rows_to_dicts(rows),
            }
    finally:
        conn.close()
