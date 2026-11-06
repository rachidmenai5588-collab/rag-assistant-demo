"""Hash-diff re-index — ARCH-02 §6.5, M11.

    python -m ingest.reindex --config config.yaml --corpus demo/corpus

Re-hashes every source document and compares with documents.file_hash.
Unchanged -> skipped (no re-embedding, no cost). Changed -> re-parse,
re-chunk, re-embed only the changed chunks (matched by content_hash),
upsert new vectors, delete vectors whose chunk_index no longer exists.
New -> full ingest. Removed -> delete-by-filter on doc_id.
Idempotent: running the indexer twice over the same documents produces
the same collection (deterministic point IDs).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import store as store_mod
from app.logdb import LogDB
from ingest.index import ingest_corpus
from server import _load_config_yaml


def main():
    ap = argparse.ArgumentParser(description="Re-index changed documents only.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--db", default="data/questions.db")
    args = ap.parse_args()
    config = _load_config_yaml(args.config)
    limit = int(config.get("tier_doc_limit", 50))
    db = LogDB(args.db)
    client = store_mod.get_client(config)
    collection = store_mod.ensure_collection(client, config)

    # Removed documents: in the inventory but no longer in the corpus.
    import sqlite3
    with sqlite3.connect(db.path) as conn:
        indexed_ids = [r[0] for r in conn.execute(
            "SELECT doc_id FROM documents WHERE status = 'indexed'")]
    present = set(os.listdir(args.corpus))
    removed = [d for d in indexed_ids if d not in present]
    for doc_id in removed:
        store_mod.delete_document(client, collection, doc_id)
        with sqlite3.connect(db.path) as conn:
            conn.execute("UPDATE documents SET status = 'removed' WHERE doc_id = ?",
                         (doc_id,))
        print(f"Removed: {doc_id}")

    report = ingest_corpus(config, args.corpus, client, collection, db, limit)
    print(f"Re-indexed: {len(report['indexed'])} changed/new, "
          f"{len(report['skipped_unchanged'])} unchanged, "
          f"{len(report['refused'])} refused, {len(removed)} removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
