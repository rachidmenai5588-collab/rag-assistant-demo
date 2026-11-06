"""Hybrid retrieval — ARCH-02 §3.2.

Dense vectors (Qdrant cosine, top-20) + BM25 over Arabic-normalised
text (top-20), fused by Reciprocal Rank Fusion (k=60), top-8 into the
prompt. BM25 is a SPARSE COMPLEMENT fused by RRF — never a replacement
for dense retrieval. There is no keyword search anywhere in this
retrieval path: a question is always embedded, and dense similarity
always participates.

Refusal (ARCH-02 §8, M5): two layers — (1) if the best dense cosine
score is below `refusal_threshold`, refuse without calling the LLM;
(2) the grounded prompt + refusal wording in the question's language.

Reranking: a config flag, default OFF (ARCH-02 §3.2). If enabled but no
reranker is available in this environment, retrieve() raises loudly
rather than silently returning un-reranked results.
"""

import math
from collections import Counter

from app.embed import prepare_for_embedding
from app.normalize import normalize_arabic, tokenize_normalized
from app import store as store_mod


class RetrievalError(RuntimeError):
    pass


# ---------------------------------------------------------------- BM25
def bm25_scores(query_tokens, documents):
    """Classic BM25 (k1=1.5, b=0.75) over normalised-token documents.

    `documents` is a list of normalised chunk texts. Returns a list of
    (doc_index, score) sorted by score descending.
    """
    n = len(documents)
    if n == 0 or not query_tokens:
        return []
    tokenized = [doc.split() for doc in documents]
    doc_lens = [len(t) for t in tokenized]
    avg_len = sum(doc_lens) / n if n else 0
    df = Counter()
    for tokens in tokenized:
        for tok in set(tokens):
            df[tok] += 1
    scores = []
    for i, tokens in enumerate(tokenized):
        tf = Counter(tokens)
        score = 0.0
        dl = doc_lens[i]
        for tok in set(query_tokens):
            if tok not in tf:
                continue
            idf = math.log(1 + (n - df[tok] + 0.5) / (df[tok] + 0.5))
            f = tf[tok]
            denom = f + 1.5 * (1 - 0.75 + 0.75 * (dl / avg_len if avg_len else 1))
            score += idf * (f * (1.5 + 1)) / denom
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ---------------------------------------------------------------- RRF
def reciprocal_rank_fusion(rankings, k=60):
    """Fuse ranked lists by RRF. `rankings`: list of lists of chunk keys
    (doc_id#chunk_index), best first. Returns [(key, rrf_score)] sorted
    by fused score descending. No score calibration between dense and
    sparse scales is needed — that is why RRF is chosen (ARCH-02 §3.2).
    """
    fused = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------- retrieve
def retrieve(question, client, collection, config, embed_fn):
    """Run the full hybrid retrieval for one question.

    Returns a dict:
      { "chunks": [payload, ...] (top-8, fused order),
        "best_dense_score": float (top dense cosine),
        "refused": bool,            # True when best dense score < threshold
        "dense_ranking": [...], "bm25_ranking": [...], "fused": [...] }
    `embed_fn` is embed(texts, config) — injected so the harness can
    substitute the local backend without touching this module.
    """
    retr = config.get("retrieval", {})
    dense_k = int(retr.get("dense_top_k", 20))
    bm25_k = int(retr.get("bm25_top_k", 20))
    rrf_k = int(retr.get("rrf_k", 60))
    final_k = int(retr.get("final_top_k", 8))
    threshold = float(retr.get("refusal_threshold", 0.30))

    if config.get("retrieval", {}).get("rerank_enabled"):
        raise RetrievalError(
            "Reranking is enabled in config but no cross-encoder reranker "
            "is available in this environment. Disable rerank_enabled or "
            "provide a buyer-funded host >= 1 GB RAM / verified API "
            "reranker (ARCH-02 §3.2). Refusing to return un-reranked "
            "results as if they were reranked.")

    # Dense: embed the question in the form the embedding model expects
    # (prepare_for_embedding: raw by default — see embed.py). BM25 below
    # uses the NORMALISED question tokens.
    q_norm = normalize_arabic(question)
    q_vec = embed_fn([prepare_for_embedding(question, config)], config)[0]
    dense = store_mod.dense_search(client, collection, q_vec, top_k=dense_k)
    best_dense = dense[0]["score"] if dense else 0.0

    # Sparse: BM25 over the normalised chunk corpus.
    all_chunks = store_mod.fetch_all_chunks(client, collection)
    by_key = {}
    for payload in all_chunks:
        key = f"{payload['doc_id']}#{payload['chunk_index']}"
        by_key[key] = payload
    bm25_ranked = bm25_scores(tokenize_normalized(q_norm),
                              [p.get("text_norm", "") for p in all_chunks])
    bm25_keys = []
    for idx, score in bm25_ranked[:bm25_k]:
        if score <= 0:
            continue
        payload = all_chunks[idx]
        bm25_keys.append(f"{payload['doc_id']}#{payload['chunk_index']}")

    dense_keys = [f"{d['payload']['doc_id']}#{d['payload']['chunk_index']}"
                  for d in dense]
    fused = reciprocal_rank_fusion([dense_keys, bm25_keys], k=rrf_k)

    chunks = [by_key[key] for key, _ in fused[:final_k] if key in by_key]
    refused = best_dense < threshold
    return {
        "chunks": chunks,
        "best_dense_score": best_dense,
        "refused": refused,
        "dense_ranking": dense_keys,
        "bm25_ranking": bm25_keys,
        "fused": fused,
    }
