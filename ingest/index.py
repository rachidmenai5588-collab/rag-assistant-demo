"""Ingestion CLI — ARCH-02 §4, §6.5.

    python -m ingest.index --config config.yaml --corpus demo/corpus

parse -> clean -> normalise -> chunk -> embed -> upsert to Qdrant.
Enforces the tier document limit (M9): the 51st document on Basic is
refused with an upgrade quote, not silently truncated. Records every
document in the SQLite inventory (documents table) with its file hash,
so reindex.py can detect changes.
"""

import argparse
import datetime
import hashlib
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import store as store_mod
from app.chunk import chunk_text
from app.embed import embed, prepare_for_embedding
from app.logdb import LogDB
from ingest.parse import extract_text, file_hash, ParseRefusal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from server import _load_config_yaml  # noqa: E402  (shared minimal YAML reader)


def detect_lang(text):
    arabic = sum(1 for ch in text if "\u0600" <= ch <= "\u06ff")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if arabic and latin:
        return "mixed"
    if arabic:
        return "ar"
    return "en"


def ingest_corpus(config, corpus_dir, client, collection, db, limit):
    """Ingest every supported file in `corpus_dir`. Returns a report dict."""
    report = {"indexed": [], "refused": [], "skipped_unchanged": []}
    files = sorted(f for f in os.listdir(corpus_dir)
                   if os.path.isfile(os.path.join(corpus_dir, f)))
    for fname in files:
        path = os.path.join(corpus_dir, fname)
        doc_id = fname
        fhash = file_hash(path)
        with sqlite3.connect(db.path) as conn:
            row = conn.execute(
                "SELECT file_hash FROM documents WHERE doc_id = ?",
                (doc_id,)).fetchone()
        if row and row[0] == fhash:
            report["skipped_unchanged"].append(doc_id)
            continue
        if len(report["indexed"]) + len(report["skipped_unchanged"]) >= limit:
            report["refused"].append(
                {"doc_id": doc_id, "reason": "tier_limit",
                 "note": f"Tier limit is {limit} documents. This document "
                         f"was refused, not silently truncated — upgrade "
                         f"to the +$100 tier (500 docs) to include it."})
            continue
        try:
            text, pages = extract_text(path)
        except ParseRefusal as exc:
            report["refused"].append({"doc_id": doc_id, "reason": exc.reason,
                                      "note": exc.note})
            with sqlite3.connect(db.path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO documents (doc_id, title,"
                    " source_path, file_hash, lang, pages, chunks, status,"
                    " refuse_note, indexed_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (doc_id, fname, path, fhash, None, None, 0,
                     exc.reason, exc.note,
                     datetime.datetime.now(datetime.timezone.utc).isoformat()))
            continue
        lang = detect_lang(text)
        chunks = chunk_text(
            text,
            target_chars=int(config.get("chunking", {}).get("target_chars", 1800)),
            overlap_chars=int(config.get("chunking", {}).get("overlap_chars", 400)),
            hard_cap=int(config.get("chunking", {}).get("max_chars_hard_cap", 3000)))
        # Embed the chunk text in the form the embedding model expects
        # (prepare_for_embedding: raw by default — see embed.py).
        # BM25 uses the normalised form separately (text_norm payload).
        vectors = embed([prepare_for_embedding(c, config) for c in chunks], config)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        store_mod.upsert_chunks(client, collection, doc_id, fname, chunks,
                                vectors, page=None, lang=lang, updated_at=now)
        with sqlite3.connect(db.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents (doc_id, title, source_path,"
                " file_hash, lang, pages, chunks, status, refuse_note,"
                " indexed_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (doc_id, fname, path, fhash, lang, pages, len(chunks),
                 "indexed", None, now))
        report["indexed"].append({"doc_id": doc_id, "chunks": len(chunks),
                                  "lang": lang, "pages": pages})
    return report


def main():
    ap = argparse.ArgumentParser(description="Ingest a document corpus into Qdrant.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--db", default="data/questions.db")
    args = ap.parse_args()
    config = _load_config_yaml(args.config)
    limit = int(config.get("tier_doc_limit", 50))
    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    db = LogDB(args.db)
    client = store_mod.get_client(config)
    collection = store_mod.ensure_collection(client, config)
    report = ingest_corpus(config, args.corpus, client, collection, db, limit)
    total_chunks = sum(d["chunks"] for d in report["indexed"])
    print(f"Indexed: {len(report['indexed'])} documents, {total_chunks} chunks")
    print(f"Skipped unchanged: {len(report['skipped_unchanged'])}")
    for r in report["refused"]:
        print(f"REFUSED {r['doc_id']}: {r['reason']} — {r['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
