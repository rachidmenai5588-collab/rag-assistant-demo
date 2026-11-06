"""Sentence-aware chunking — ARCH-02 §6.2.

Target 500–800 tokens per chunk (~350–600 Arabic words). The template
measures by characters as a proxy (Arabic tokens are denser than
English) and records the actual chunk-size distribution in the accuracy
report. Boundaries: paragraph breaks first, then sentence boundaries;
never split mid-sentence except for pathological documents. Overlap:
~100–150 tokens (one sentence) with the previous chunk. Tables/lists:
kept as one chunk where they fit; otherwise split by row groups with
the header row carried as context into each chunk.
"""

import hashlib
import re

_SENTENCE_END = re.compile(r"(?<=[.!?؟۔])\s+")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def split_sentences(text: str) -> list:
    """Split text into sentences, keeping the terminator attached."""
    parts = _SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, target_chars: int = 1800, overlap_chars: int = 400,
               hard_cap: int = 3000) -> list:
    """Chunk `text` into overlapping, sentence-aware chunks.

    Returns a list of chunk strings. A chunk is never empty. Overlap is
    the tail of the previous chunk (last ~`overlap_chars` characters,
    cut at a sentence boundary where possible).
    """
    text = text.strip()
    if not text:
        return []
    # Paragraphs first: a table/list paragraph is kept whole when it fits.
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    chunks: list = []
    current = ""
    for para in paragraphs:
        # A paragraph longer than the target is split by sentences.
        pieces = [para] if len(para) <= target_chars else _split_long_paragraph(
            para, target_chars, hard_cap)
        for piece in pieces:
            if not current:
                current = piece
            elif len(current) + 1 + len(piece) <= target_chars:
                current = current + "\n\n" + piece
            else:
                chunks.append(current)
                # Overlap: carry the tail of the finished chunk forward.
                overlap = _overlap_tail(current, overlap_chars)
                current = (overlap + "\n\n" + piece).strip() if overlap else piece
                # If even with overlap the piece alone exceeds the cap, split it.
                while len(current) > hard_cap:
                    chunks.append(current[:hard_cap])
                    current = current[hard_cap:]
    if current:
        chunks.append(current)
    return chunks


def _split_long_paragraph(para: str, target_chars: int, hard_cap: int) -> list:
    sentences = split_sentences(para)
    pieces: list = []
    current = ""
    for sent in sentences:
        if len(sent) > hard_cap:
            # Pathological single sentence: hard-split at the cap.
            if current:
                pieces.append(current)
                current = ""
            for i in range(0, len(sent), hard_cap):
                pieces.append(sent[i:i + hard_cap])
            continue
        if not current:
            current = sent
        elif len(current) + 1 + len(sent) <= target_chars:
            current = current + " " + sent
        else:
            pieces.append(current)
            current = sent
    if current:
        pieces.append(current)
    return pieces


def _overlap_tail(chunk: str, overlap_chars: int) -> str:
    """Last ~overlap_chars of a chunk, cut at a sentence boundary if possible."""
    if len(chunk) <= overlap_chars:
        return chunk
    tail = chunk[-overlap_chars:]
    # Prefer to start at a sentence boundary inside the tail.
    m = _SENTENCE_END.search(tail)
    if m:
        return tail[m.end():].strip()
    return tail.strip()


def chunk_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic point ID: UUID derived from sha256(doc_id + chunk_index)
    — ARCH-02 §6.5. Qdrant requires UUID (or int) point IDs; the first 32
    hex chars of the SHA-256 form a valid UUID. Upserts by this ID make
    indexing idempotent: re-running the indexer over unchanged documents
    produces the same collection.
    """
    import uuid
    digest = hashlib.sha256(f"{doc_id}#{chunk_index}".encode("utf-8")).hexdigest()
    return str(uuid.UUID(hex=digest[:32]))


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
