# Product 2 — AI Knowledge Assistant (RAG): reusable template

**ASREO Phase 3 template build. Status: built and measured on the
Saffron & Sage demo corpus — see `demo/accuracy-report.md` for the
real numbers. Not a finished commercial product: M4/M5/M6 fail under
the test-only measurement conditions (read the report before quoting
any number).**

## What this is

A single-tenant, per-buyer RAG chatbot: one small Python web service
per buyer, deployed on the **buyer's** free-tier accounts (Render +
Qdrant Cloud), calling the **buyer's** OpenAI API key for embeddings
and generation. Hand-rolled pipeline, no LangChain. Arabic-first:
one shared normalisation function, hybrid dense + BM25 retrieval
fused by Reciprocal Rank Fusion, grounded generation with citation
enforcement and a number/price/date token check, refusal below a
similarity threshold.

## Layout

```
config.yaml            per-buyer configuration (model names live here, never in code)
requirements.txt       6 pinned production dependencies
render.yaml            Render service definition (buyer deploys from this)
app/
  server.py            Flask app: routes only (POST /api/chat, /healthz, widget, Telegram webhook, admin)
  chat.py              orchestrate retrieve -> generate -> cite -> log
  retrieve.py          dense + BM25 + RRF fusion + refusal threshold
  generate.py          grounded prompt (from prompts/) + OpenAI chat + citation check
  normalize.py         normalize_arabic() — THE shared function
  embed.py             embed(texts) interface: OpenAI primary, local test fallback
  chunk.py             sentence-aware chunking + deterministic point IDs
  store.py             Qdrant wrapper (cloud + local in-memory mode)
  logdb.py             SQLite log: documents / questions_log / test_runs
  telegram.py          Bot API calls (webhook + M16 escalation)
  rate_limit.py        in-memory token bucket (30 req/min/IP)
  widget/              RTL-correct embeddable chat widget
ingest/
  parse.py             PDF/DOCX/TXT extraction + scanned-PDF refusal
  index.py             CLI: parse -> chunk -> embed -> upsert (+ tier limit M9)
  reindex.py           hash-diff re-index (M11)
prompts/               grounded.ar.txt, grounded.en.txt (versioned, not inline)
tests/
  test_normalize.py    M2: 10 pairs, 10/10 (pytest)
  test_retrieval.py    M3: Recall@3 measurement
  test_answers.py      M4/M5/M6 judgement helpers (auto-suggest + human review)
  testset_gen.py       test-set generator scaffold + checklist + paraphrase prompt
  harness.py           runs the full M3–M6 measurement, writes the accuracy report
demo/
  corpus/              Saffron & Sage sample corpus (5 docs: returns, delivery,
                       hours, warranty, payments; hours/phone from the real
                       restaurant template)
  testset/             30-question test set + human judgements
  accuracy-report.md   MEASURED M3–M6 report (read it — M4/M5/M6 fail, honestly)
  results.json         per-question machine-readable results
docs/
  deployment-runbook.md  exact public deployment steps (needs a browser)
  handover-pack.md       per-buyer handover checklist (M12)
  update-guide-ar.md     Arabic update guide template (M11)
```

## Quick start (local test)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pip install sentence-transformers   # test-only local embeddings
.venv/bin/python -m pytest tests/test_normalize.py -q
.venv/bin/python -m tests.harness --config config.yaml --corpus demo/corpus \
    --testset demo/testset/testset-30.json --out demo/accuracy-report.md
.venv/bin/gunicorn -w 1 -b 127.0.0.1:5000 "app.server:create_app()"
```

Without an OpenAI key the service runs in local-embedding mode for
retrieval tests, and `/api/chat` returns the maintenance message for
answerable questions (generation needs the buyer's funded key — by
design, never a silent fallback).

## Measured results (demo corpus, local MiniLM embeddings, extractive stand-in)

| Criterion | Target | Measured |
|---|---|---|
| M3 Recall@3 | ≥ 0.80 | **0.950 — PASS** (factual 1.00, paraphrase 1.00, dialect 0.80) |
| M4 correctness (human-reviewed) | ≥ 0.85 | **0.80 (16/20) — FAIL** |
| M5 refusals | ≥ 9/10, 0 invented | **4/10 — FAIL** (0 invented facts — see report) |
| M6 citations | 100% | **16/20 — FAIL** (16/16 of answered questions cite the expected doc) |
| Retrieval latency p50/p95 | ≤ 30 s (M10) | 84 ms / 109 ms (warm, local) |

Full analysis, including why M4/M5 fail (threshold false-refusals on
dialect questions; layer-1 threshold cannot separate on this corpus +
model; the LLM layer-2 refusal is unmeasured without a key) is in
`demo/accuracy-report.md` and `BUILD-REPORT.md`. Do not quote the M3
number without the M4/M5 numbers beside it.

## Hard rules (ARCH-02)

- Secrets in env vars only. Never in the repo, never in `config.yaml`.
- The chat model name lives in `config.yaml` only — never hard-code a
  model. `gpt-4o-mini` is deprecated 14 Jan 2027: re-verify at delivery.
- No keyword search in the retrieval path. BM25 is a sparse complement
  fused by RRF, never a replacement for dense retrieval.
- Reranking is a config flag, default OFF — not in Basic.
- Scanned/image-only PDFs are refused with a written explanation.
- No order is accepted without the VM/agent build environment.
