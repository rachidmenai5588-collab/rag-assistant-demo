"""Grounded generation — ARCH-02 §8, M7, §8.1.

The system prompt is built from a versioned template file
(prompts/grounded.ar.txt, grounded.en.txt), not from inline strings —
reviewable, versioned, identical across deliveries. The prompt instructs
answering ONLY from the provided context; if the answer is not present,
say you don't know.

Citation enforcement (§8.1): after generation, verify (a) the answer
references at least one retrieved chunk (by doc_id); (b) if the answer
contains a number, price, or date token, that token appears in one of
the cited chunks. If either check fails, return a REFUSAL instead of
the answer — an uncited answer is treated as an ungrounded answer.
This deliberately trades occasional false refusals (annoying) for
invented facts (a failed M4 and a buyer dispute).
"""

import json
import os
import re

import requests

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

NUMBER_TOKEN = re.compile(r"\d+(?:[.,]\d+)?\s*(?:%|€|\$|د\.ج|دج|يوم|أيام|ساعة|شهر)?")

REFUSAL_AR = "عذراً، لا توجد هذه المعلومة في مستندات المتجر."
REFUSAL_EN = "Sorry, this information is not in the store's documents."


class GenerationError(RuntimeError):
    pass


def load_prompt(lang="ar"):
    fname = "grounded.ar.txt" if lang == "ar" else "grounded.en.txt"
    path = os.path.join(PROMPTS_DIR, fname)
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def build_messages(question, chunks, lang="ar"):
    """Assemble the grounded prompt: system template + retrieved context
    (raw text, with doc_id citations) + the question as a clearly
    delimited user block (prompt-injection boundary, ARCH-02 §9)."""
    system = load_prompt(lang)
    context_blocks = []
    for c in chunks:
        context_blocks.append(
            f"[SOURCE doc_id={c['doc_id']} title={c.get('doc_title','')} "
            f"page={c.get('page')}] {c.get('text_raw','')}")
    context = "\n\n".join(context_blocks)
    user = (f"السؤال:\n{question}\n\nأجب بالاعتماد على المقاطع أعلاه فقط."
            if lang == "ar" else
            f"Question:\n{question}\n\nAnswer using ONLY the passages above.")
    return [
        {"role": "system", "content": system + "\n\nالمقاطع المرجعية:\n" + context
         if lang == "ar" else system + "\n\nReference passages:\n" + context},
        {"role": "user", "content": user},
    ]


def _extract_number_tokens(text):
    return set(m.group(0).strip() for m in NUMBER_TOKEN.finditer(text or ""))


def citation_check(answer, chunks):
    """Return (ok, reason). (a) >=1 citation to a retrieved chunk;
    (b) every number/price/date token in the answer appears in a cited chunk.
    The check is deliberately conservative (ARCH-02 §8.1)."""
    if not chunks:
        return False, "no retrieved chunks to cite"
    cited_ids = set()
    for c in chunks:
        if c["doc_id"] in (answer or ""):
            cited_ids.add(c["doc_id"])
    if not cited_ids:
        return False, "answer cites no retrieved document"
    cited_text = " ".join(c.get("text_raw", "") for c in chunks
                          if c["doc_id"] in cited_ids)
    for token in _extract_number_tokens(answer):
        digits = re.sub(r"\D", "", token)
        if digits and digits not in re.sub(r"\D", " ", cited_text).split() \
                and digits not in cited_text:
            # Number tokens are compared digit-by-digit against cited text.
            if digits not in cited_text.replace(" ", ""):
                return False, f"number token {token!r} not in any cited chunk"
    return True, "ok"


def generate(question, chunks, config, lang="ar", api_key=None):
    """Generate a grounded answer. Returns dict(answer, sources, refused).

    If retrieval already refused, no LLM call is made. If the citation
    check fails, a refusal is returned instead of the answer.
    """
    if not chunks:
        return {"answer": REFUSAL_AR if lang == "ar" else REFUSAL_EN,
                "sources": [], "refused": True}
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise GenerationError(
            "OPENAI_API_KEY is not set. Generation needs the buyer's funded "
            "key. (Tests use the extractive stand-in in tests/harness.py, "
            "never this module.)")
    model = config.get("llm_model", "gpt-4o-mini")
    messages = build_messages(question, chunks, lang)
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        json={"model": model, "temperature": config.get("llm_temperature", 0.0),
              "messages": messages},
        timeout=120,
    )
    if resp.status_code != 200:
        raise GenerationError(
            f"OpenAI chat failed: HTTP {resp.status_code}: {resp.text[:300]}")
    answer = resp.json()["choices"][0]["message"]["content"].strip()
    ok, reason = citation_check(answer, chunks)
    if not ok:
        return {"answer": REFUSAL_AR if lang == "ar" else REFUSAL_EN,
                "sources": [], "refused": True,
                "refusal_reason": f"citation check failed: {reason}"}
    sources = [{"doc_id": c["doc_id"], "doc_title": c.get("doc_title", ""),
                "page": c.get("page"),
                "excerpt": c.get("text_raw", "")[:300]} for c in chunks[:3]]
    return {"answer": answer, "sources": sources, "refused": False}
