"""SQLite log store — ARCH-02 §6.3, §6.4.

Three tables: documents (ingestion inventory), questions_log (M15),
test_runs (evidence trail, kept indefinitely). The question log is
pruned at 90 days; the file lives inside the Render free service whose
filesystem is EPHEMERAL — it is never the only copy of anything that
cannot be regenerated (ARCH-02 §6.4).
"""

import csv
import json
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  source_path TEXT,
  file_hash TEXT NOT NULL,
  lang TEXT,
  pages INTEGER,
  chunks INTEGER,
  status TEXT NOT NULL,
  refuse_note TEXT,
  indexed_at TEXT
);
CREATE TABLE IF NOT EXISTS questions_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  asked_at TEXT NOT NULL,
  channel TEXT NOT NULL,
  session_id TEXT,
  question TEXT NOT NULL,
  lang TEXT,
  refused INTEGER NOT NULL,
  answer TEXT,
  sources TEXT,
  latency_ms INTEGER,
  helpful INTEGER
);
CREATE TABLE IF NOT EXISTS test_runs (
  run_id TEXT PRIMARY KEY,
  model TEXT NOT NULL,
  embed_model TEXT NOT NULL,
  threshold REAL,
  recall_at3 REAL,
  correct INTEGER,
  refused_ok INTEGER,
  citations_ok INTEGER,
  report_path TEXT
);
"""


class LogDB:
    def __init__(self, path):
        self.path = path
        with sqlite3.connect(path) as conn:
            conn.executescript(SCHEMA)

    def log(self, question, lang, refused, answer, sources, latency_ms,
            channel="widget", session_id=None):
        import datetime
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO questions_log (asked_at, channel, session_id,"
                " question, lang, refused, answer, sources, latency_ms)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 channel, session_id, question, lang, int(bool(refused)),
                 answer, json.dumps(sources, ensure_ascii=False), latency_ms))

    def unanswered(self, limit=100):
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT asked_at, channel, question FROM questions_log"
                " WHERE refused = 1 ORDER BY asked_at DESC LIMIT ?",
                (limit,)).fetchall()
        return [{"asked_at": r[0], "channel": r[1], "question": r[2]}
                for r in rows]

    def stats(self, chunk_count):
        with sqlite3.connect(self.path) as conn:
            total, refused = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(refused), 0) FROM questions_log"
            ).fetchone()
        return {"chunks_indexed": chunk_count, "questions_total": total,
                "refusals": refused,
                "refusal_rate": (refused / total) if total else 0.0}

    def export_csv(self, fileobj):
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT asked_at, channel, question, refused, latency_ms"
                " FROM questions_log ORDER BY asked_at").fetchall()
        writer = csv.writer(fileobj)
        writer.writerow(["asked_at", "channel", "question", "refused",
                         "latency_ms"])
        writer.writerows(rows)

    def record_test_run(self, run_id, model, embed_model, threshold,
                        recall_at3, correct, refused_ok, citations_ok,
                        report_path):
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO test_runs VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, model, embed_model, threshold, recall_at3, correct,
                 refused_ok, citations_ok, report_path))
