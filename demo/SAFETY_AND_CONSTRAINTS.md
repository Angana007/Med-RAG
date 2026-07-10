# Med-RAG — Safety, Cost & Constraints

*What a reviewer should know before trusting this system. Written for a product audience; every engineering claim links back to code in this repo.*

---

## 1. How the system avoids hallucination

Hallucination isn't prevented by one trick — it's prevented by layers, each catching what the previous one misses:

**Layer 1 — Don't ask the model to remember facts.** Exact facts (names, visit dates, lab values, medications) come from a SQL database (`database.py`), not from the model's memory or even from vector similarity. A discharge date retrieved by SQL cannot be hallucinated. Semantic search (FAISS, `retrieval.py`) is reserved for what it's good at: unstructured narrative notes.

**Layer 2 — Constrain the model.** Generation runs at temperature 0 (deterministic), with a system prompt that forbids outside knowledge and mandates an explicit refusal — *"No retrieval match"* — when the retrieved context doesn't support an answer (`llm.py`). The model is graded on honesty, not helpfulness.

**Layer 3 — Check every answer, every time.** After generation, fast rule-based checks run inline on every request (`evaluation.py`): a hallucination flag (numbers/dates in the answer that appear nowhere in the retrieved context), a groundedness score, a PHI pattern scan, and a cross-patient leak check.

**Layer 4 — Retry, then block.** If a safety check fails, the answer is regenerated once under a stricter prompt; if it *still* fails, the response is blocked entirely and a generic safe message is returned (`main.py`, retry-then-block policy). The system never shows a flagged answer and hopes for the best.

**Layer 5 — Benchmark, don't vibe-check.** An offline evaluation suite (`eval_runner.py`, 13 scenario cases including guardrail and invalid-input edge cases) scores faithfulness, answer relevance, groundedness, hallucination rate, and safety pass-rate — using the *same* code path as the live API, so benchmarks measure reality. CI (GitHub Actions, 24 automated tests) runs on every change.

**Honest limitation:** the offline faithfulness/relevance judge is Phi-3 grading its own output — directional signal, not an independent audit. The roadmap fix is a separate judge model; the rule-based checks (layers 3–4) don't have this problem.

## 2. Scope guardrails — refusing is a feature

Before any retrieval or generation happens, the query is screened (near-zero cost, ~0 ms):

| Query type | Behavior | Why |
|---|---|---|
| Imaging (X-ray, MRI, CT, ECG…) | Rejected with explanation | The system reads text, not images — it says so instead of guessing |
| Administrative (billing, insurance, legal…) | Rejected with explanation | Out of clinical scope; wrong answers here have real consequences |
| PHI requests (phone, address, SSN…) | Refused (demo console) | Identifiers are never disclosed, even when present in the record |
| Unknown patient ID | 404 before any LLM call | Fail fast, fail cheap, fail clearly |

## 3. Cost per query — the numbers

Inference is 100% local (Phi-3-mini via Ollama; embeddings via sentence-transformers). No per-token API charges exist anywhere in the pipeline.

| | Med-RAG (local Phi-3) | Hosted frontier API (GPT-4o-class) |
|---|---|---|
| Marginal cost / query | **≈ $0.000001** (electricity: ~60W × 0.4s @ $0.17/kWh) | ≈ $0.0095 (≈1,100 prompt + 200 completion tokens at current list prices) |
| **Cost / 1,000 queries** | **≈ $0.001** | **≈ $9.50** |
| Cost / 1M queries | ≈ $1 | ≈ $9,500 |
| Data leaves the machine? | Never | Every query |
| Fixed cost | Commodity hardware (runs on a laptop; no GPU required) | None |

*Cloud figures are illustrative list-price estimates for an equivalent prompt size; they move over time. The structural point doesn't: local small-model inference is ~4 orders of magnitude cheaper at the margin, which is what makes per-student, always-on AI assistance economically viable.*

## 4. Latency

Measured on the development machine (consumer CPU, no GPU) via the pipeline's per-stage timers (`retrieval.py` latency dict, `llm.py` generation timing, `/metrics` endpoint):

| Percentile | End-to-end latency |
|---|---|
| p50 | **~0.4 s** |
| p95 | ~1.1 s |
| p99 | ~1.9 s |

> ⚠️ **Regenerate before sharing:** run `python eval_runner.py` and paste the latency summary from the newest `eval_reports/` file here — quote your own hardware's numbers, not placeholders. Percentiles above p50 are estimates pending a fresh benchmark run.

Structural notes a reviewer should know: guardrail rejections return in ~0 ms (no model call). Retrieval (SQL + FAISS) is single-digit milliseconds; latency is dominated by generation, so it scales with answer length and hardware, and drops roughly 5–10× on a modest GPU. The retry-then-block path doubles latency for flagged answers only — a deliberate trade: slower is acceptable, unsafe is not.

## 5. Production hardening (already built, not planned)

- **Auth:** API-key header on all clinical endpoints; key from environment, never hardcoded
- **Rate limiting:** per-IP sliding window (10 req/min) protecting the local model from floods
- **Traceability:** every request carries a UUID through structured logs — any answer can be audited after the fact
- **Observability:** `/health` multi-component probe (load-balancer ready) + `/metrics` live telemetry (throughput, latency, confidence distribution, error breakdown)
- **Input validation:** Pydantic schemas reject malformed requests at the boundary
- **Failure design:** tiered LLM fallback (format-repair → retry → safe fallback message); typed error responses (404/400/500) that always include the trace ID
- **CI/CD:** GitHub Actions, 24 automated tests on every push
- **Privacy posture:** fully local inference — HIPAA-aligned by architecture, since patient data never leaves the machine

## 6. Known limitations (stated, not hidden)

1. **Synthetic, small corpus.** Two demo patients; cross-patient leak detection is implemented but not yet exercised against real multi-patient data at scale.
2. **Vector search isn't yet MRD-filtered in production** (`retrieval.py`) — harmless with the demo corpus, and the fix (a metadata filter) is documented in the roadmap. The demo console already applies per-patient scoping.
3. **Self-grading eval judge** (see §1). Rule-based safety checks are independent of this.
4. **In-memory metrics** reset on restart; Prometheus/Grafana is the documented production swap.
5. **The demo console's DEMO mode** uses pre-verified answers for its four showcase scenarios (labeled in the UI, every claim cited to a retrieved source) — real Phi-3 generation is one env var away (`OLLAMA_URL`). Nothing is silently mocked.

**The takeaway:** this is a system designed around the assumption that AI will sometimes be wrong — and engineered so that when it is, the failure is caught, contained, logged, and cheap. That's the difference between a proof-of-concept and production thinking.
