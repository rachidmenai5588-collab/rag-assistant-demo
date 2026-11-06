"""Answer pipeline orchestration — ARCH-02 §4.

chat.py wires retrieve -> generate -> cite -> log. It contains no HTTP
code (server.py owns that) and no retrieval/generation logic of its
own. The widget and Telegram both call answer_question().
"""

import time

from app import store as store_mod
from app.embed import embed
from app.retrieve import retrieve
from app.generate import generate, REFUSAL_AR, REFUSAL_EN


def answer_question(question, client, collection, config, lang="ar",
                    log_fn=None):
    """Answer one question end-to-end.

    Returns dict(answer, sources, refused, latency_ms, best_dense_score).
    Failure modes (ARCH-02 §7.5): OpenAI/Qdrant unreachable -> caller
    returns the maintenance message; here they raise, and server.py
    converts them. No stack trace, no empty reply, no hallucinated
    apology is ever returned as an answer.
    """
    started = time.monotonic()
    retrieval = retrieve(question, client, collection, config, embed)
    if retrieval["refused"]:
        result = {"answer": REFUSAL_AR if lang == "ar" else REFUSAL_EN,
                  "sources": [], "refused": True,
                  "best_dense_score": retrieval["best_dense_score"]}
    else:
        gen = generate(question, retrieval["chunks"], config, lang=lang)
        result = {"answer": gen["answer"], "sources": gen["sources"],
                  "refused": gen["refused"],
                  "best_dense_score": retrieval["best_dense_score"]}
        if "refusal_reason" in gen:
            result["refusal_reason"] = gen["refusal_reason"]
    result["latency_ms"] = int((time.monotonic() - started) * 1000)
    if log_fn:
        log_fn(question=question, lang=lang, refused=result["refused"],
               answer=result["answer"], sources=result["sources"],
               latency_ms=result["latency_ms"])
    return result
