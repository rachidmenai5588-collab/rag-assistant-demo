"""Qdrant client wrapper — ARCH-02 §2.6, §6.1.

One collection per buyer: `kb_<buyer-slug>`. The same client serves
production (Qdrant Cloud, via QDRANT_URL + QDRANT_API_KEY) and tests
(local in-memory mode, no server). Payload per point (ARCH-02 §6.1):
doc_id, doc_title, chunk_index, page, text_raw, text_norm,
content_hash, lang, updated_at.
"""

import os

from qdrant_client import QdrantClient, models

from app.chunk import chunk_id


class StoreError(RuntimeError):
    pass


def get_client(config):
    """Qdrant client: cloud when QDRANT_URL is set, else local in-memory."""
    url = os.environ.get("QDRANT_URL") or (config.get("qdrant") or {}).get("url")
    if url:
        api_key = os.environ.get("QDRANT_API_KEY")
        return QdrantClient(url=url, api_key=api_key, timeout=30)
    return QdrantClient(location=":memory:")


def ensure_collection(client, config):
    """Create the collection if it does not exist (idempotent)."""
    from app.embed import embedding_dimensions
    name = (config.get("qdrant") or {}).get("collection", "kb_default")
    dims = embedding_dimensions(config)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dims, distance=models.Distance.COSINE),
        )
    return name


def upsert_chunks(client, collection, doc_id, doc_title, chunks, vectors,
                  page=None, lang="ar", updated_at=None):
    """Upsert chunk vectors + payloads. Deterministic point IDs (§6.5)."""
    from app.normalize import normalize_arabic
    from app.chunk import content_hash
    points = []
    for i, (text, vector) in enumerate(zip(chunks, vectors)):
        points.append(models.PointStruct(
            id=chunk_id(doc_id, i),
            vector=vector,
            payload={
                "doc_id": doc_id,
                "doc_title": doc_title,
                "chunk_index": i,
                "page": page,
                "text_raw": text,
                "text_norm": normalize_arabic(text),
                "content_hash": content_hash(text),
                "lang": lang,
                "updated_at": updated_at,
            },
        ))
    client.upsert(collection_name=collection, points=points, wait=True)
    return len(points)


def delete_document(client, collection, doc_id):
    """Delete all points of one document (re-index / M11)."""
    client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(filter=models.Filter(
            must=[models.FieldCondition(key="doc_id",
                                        match=models.MatchValue(value=doc_id))])),
        wait=True,
    )


def dense_search(client, collection, vector, top_k=20):
    """Cosine similarity search. Returns [{score, payload}] (score = cosine)."""
    results = client.query_points(
        collection_name=collection, query=vector, limit=top_k,
        with_payload=True).points
    return [{"score": float(r.score), "payload": r.payload} for r in results]


def fetch_all_chunks(client, collection):
    """All stored chunks (payloads) — the BM25 corpus. Small corpora only
    (<=500 docs); this is the in-process BM25 index of ARCH-02 §3.2."""
    points, _ = client.scroll(collection_name=collection, limit=100000,
                              with_payload=True, with_vectors=False)
    return [p.payload for p in points]


def count_points(client, collection):
    return client.count(collection_name=collection, exact=True).count


def health_check(client, collection):
    """True if the collection is reachable."""
    try:
        return client.collection_exists(collection)
    except Exception:
        return False
