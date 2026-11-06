"""Embedding provider behind one interface — ARCH-02 §2.3.

    embed(texts, config) -> list[list[float]]

Primary path: OpenAI `text-embedding-3-small` (buyer's funded key,
`OPENAI_API_KEY`). Local fallback (testing only, never the production
default): `sentence-transformers` multilingual model, used only when
no OpenAI key is configured AND `embedding.provider` is "local".

The provider is behind this one interface so a future swap is a
one-module change. There is NO other embedding path in the template:
no TF-IDF, no keyword counting, no hash vectors masquerading as
embeddings. If no backend is available, embed() raises loudly —
the service never silently falls back to keyword search.
"""

import os

import requests

from app.normalize import normalize_arabic

OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"


class EmbeddingError(RuntimeError):
    pass


def prepare_for_embedding(text, config):
    """Text form sent to the embedding model.

    Default (normalize_input: false): RAW text. Neural embedding models
    are trained on raw text; normalisation (ة→ه, ى→ي) produces
    out-of-distribution forms that measurably degrade dense retrieval
    (diagnostic on the Saffron & Sage corpus with MiniLM: Recall@3 0.95
    raw vs 0.85 normalised; top-1 18/20 vs 14/20). BM25 ALWAYS uses the
    normalised form (text_norm payload) — normalisation is a sparse-
    retrieval technique, not an embedding technique. Set
    embedding.normalize_input: true only with measured evidence for the
    embedding model in use; re-verify with OpenAI text-embedding-3-small
    at first delivery.
    """
    if (config.get("embedding") or {}).get("normalize_input", False):
        return normalize_arabic(text)
    return text


def embed(texts, config, api_key=None):
    """Embed a list of texts. Returns a list of float vectors."""
    if not texts:
        return []
    provider = (config.get("embedding") or {}).get("provider", "openai")
    if provider == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EmbeddingError(
                "OPENAI_API_KEY is not set. The OpenAI embedding path "
                "needs the buyer's funded key (or set embedding.provider "
                "to 'local' for test-only local embeddings).")
        return _embed_openai(texts, config, key)
    if provider == "local":
        return _embed_local(texts, config)
    raise EmbeddingError(f"Unknown embedding provider: {provider!r}")


def _embed_openai(texts, config, api_key):
    model = (config.get("embedding") or {}).get("openai_model",
                                               "text-embedding-3-small")
    resp = requests.post(
        OPENAI_EMBED_URL,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={"model": model, "input": texts},
        timeout=60,
    )
    if resp.status_code != 200:
        raise EmbeddingError(
            f"OpenAI embeddings failed: HTTP {resp.status_code}: "
            f"{resp.text[:300]}")
    data = resp.json()["data"]
    data.sort(key=lambda d: d["index"])
    return [d["embedding"] for d in data]


_local_model = None


def _embed_local(texts, config):
    """Test-only local multilingual embeddings via sentence-transformers.

    Model: paraphrase-multilingual-MiniLM-L12-v2 (ARCH-02 §2.3 fallback).
    Requires the `sentence-transformers` package + a model download;
    NOT a production dependency and NOT installed by requirements.txt.
    """
    global _local_model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingError(
            "Local embedding backend requested but sentence-transformers "
            "is not installed. Install it (test environment only) or use "
            "the OpenAI provider with a funded key.") from exc
    model_name = (config.get("embedding") or {}).get(
        "local_model", "paraphrase-multilingual-MiniLM-L12-v2")
    if _local_model is None:
        _local_model = SentenceTransformer(model_name)
    vectors = _local_model.encode(texts, normalize_embeddings=True,
                                  show_progress_bar=False)
    return [list(map(float, v)) for v in vectors]


def embedding_dimensions(config) -> int:
    """Vector size the Qdrant collection must be created with."""
    provider = (config.get("embedding") or {}).get("provider", "openai")
    if provider == "local":
        # paraphrase-multilingual-MiniLM-L12-v2 outputs 384 dims.
        return 384
    return int((config.get("embedding") or {}).get("dimensions", 1536))
