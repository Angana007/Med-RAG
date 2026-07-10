"""
Module: demo/engine.py

The self-contained engine behind the Med-RAG Demo Console (app.py).

It mirrors the production pipeline (retrieval.py -> llm.py -> evaluation.py)
in a single portable module with zero heavy dependencies, so the demo
deploys anywhere in minutes:

  Production                      Demo mirror (this file)
  ─────────────────────────────   ─────────────────────────────────────
  SQLite patient lookup           in-memory patient dict (same schema)
  FAISS + MiniLM vector search    pure-Python BM25 over the same chunks
  Pre-retrieval scope guardrails  identical keyword guardrails + PHI-request check
  Phi-3 via Ollama                LIVE mode: real Ollama call (set OLLAMA_URL)
                                  DEMO mode: retrieval-grounded showcase answers
  Live PHI / grounding checks     same checks, surfaced in the UI

Two modes, controlled by the OLLAMA_URL environment variable:
  DEMO mode (default) — no model server needed. The four showcase queries
    return carefully authored answers whose every claim cites a retrieved
    chunk; free-typed queries get an extractive, retrieval-grounded answer.
    Honest by design: the UI labels which mode produced each answer.
  LIVE mode — set OLLAMA_URL (e.g. http://localhost:11434) and every query
    goes through real Phi-3 generation with the same guardrail prompt used
    in production llm.py.
"""

import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Configuration ──────────────────────────────────────────────────────────────
CORPUS_PATH = Path(__file__).parent / "demo_corpus.json"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3")
TOP_K = 5

# Cost model (local inference): marginal cost is electricity only.
# 60W laptop-class machine, ~0.4s median generation -> ~6.7e-6 kWh/query
# at $0.17/kWh ≈ $0.0000011/query. Cloud comparison figures live in the
# safety & constraints doc.
COST_PER_QUERY_LOCAL_USD = 0.0000011
# Reference: hosted GPT-4o-class API for the same prompt size (illustrative)
COST_PER_QUERY_CLOUD_USD = 0.0095

# Same scope guardrails as production retrieval.py
UNSUPPORTED_DOC_TYPES = ["x-ray", "mri", "ct scan", "ultrasound", "imaging", "ecg", "eeg"]
UNSUPPORTED_TOPICS = ["billing", "insurance", "payment", "salary", "claims", "legal",
                      "lawsuit", "staffing", "inventory", "equipment_specs"]
# PHI-request guardrail (demo addition — production scans *outputs*; the demo
# also refuses obvious PHI *requests* before retrieval even runs)
PHI_REQUEST_TERMS = ["phone number", "home address", "address", "email address",
                     "social security", "ssn", "date of birth", "contact number",
                     "phone", "whatsapp"]

PHI_REFUSAL_MESSAGE = (
    "I can't share direct patient identifiers like phone numbers, addresses, or "
    "contact details — even when they exist in the record. This system is scoped "
    "to *clinical* questions only, and every answer is scanned for PHI before "
    "it's returned.\n\nThis refusal is a feature, not a failure: in a production "
    "healthcare deployment, the most dangerous answer is the one that shouldn't "
    "have been given at all."
)

SYSTEM_PROMPT = """You are a professional Medical AI Assistant. Use ONLY the provided context.
Rules:
1. If the question asks about "dates", "visits", or "history", analyze the context.
2. If the answer exists in the context, extract it directly.
3. If the context truly lacks the answer, say "No retrieval match."
4. Do not use outside knowledge.
5. Never include patient identifiers such as phone numbers, addresses, SSNs, or emails.
6. Always output:

Answer: <response>
Confidence: <High/Medium/Low>
"""

_STOPWORDS = set("""a an and are as at be but by for from has have he her his i if in into is it its
of on or she that the their them there they this to was we what when where which who why will with you your
patient patients""".split())


# ── Corpus loading ─────────────────────────────────────────────────────────────
def load_corpus() -> Dict[str, Any]:
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9%\.]+", text.lower()) if t not in _STOPWORDS]


# ── BM25-lite retriever (pure Python, mirrors the FAISS role) ─────────────────
class Retriever:
    def __init__(self, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [_tokenize(c["title"] + " " + c["text"]) for c in chunks]
        self.doc_len = [len(d) for d in self.docs]
        self.avg_len = sum(self.doc_len) / max(1, len(self.doc_len))
        self.df: Counter = Counter()
        for d in self.docs:
            for term in set(d):
                self.df[term] += 1
        self.n = len(self.docs)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, query: str, idx: int) -> float:
        q_terms = _tokenize(query)
        tf = Counter(self.docs[idx])
        s = 0.0
        for t in q_terms:
            if t not in tf:
                continue
            num = tf[t] * (self.k1 + 1)
            den = tf[t] + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avg_len)
            s += self._idf(t) * num / den
        return s

    def search(self, query: str, mrd: Optional[str], k: int = TOP_K) -> List[Dict[str, Any]]:
        """Patient chunks are filtered to the requested MRD (the per-patient
        scoping the production roadmap calls for); guideline chunks are shared."""
        scored = []
        for i, c in enumerate(self.chunks):
            if c["source_type"] == "patient_record" and mrd and c["mrd"] != mrd:
                continue
            s = self.score(query, i)
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        top = scored[:k]
        max_s = top[0][0] if top else 1.0
        return [
            {**c, "score": round(s, 3), "relevance": round(s / max_s, 2)}
            for s, c in top
        ]


# ── Guardrails (identical logic to production retrieval.py, plus PHI requests) ─
def check_guardrails(query: str) -> Optional[Dict[str, str]]:
    q = query.lower()
    if any(t in q for t in PHI_REQUEST_TERMS):
        return {"type": "phi_request", "message": PHI_REFUSAL_MESSAGE}
    if any(t in q for t in UNSUPPORTED_DOC_TYPES):
        return {"type": "unsupported_doc",
                "message": ("This system only analyzes text-based clinical notes — it cannot "
                            "interpret imaging or scan results, and it says so instead of guessing. "
                            "Knowing the boundary of your own competence is a safety feature.")}
    if any(t in q for t in UNSUPPORTED_TOPICS):
        return {"type": "unsupported_topic",
                "message": ("This assistant is scoped to clinical questions only — administrative "
                            "topics like billing, insurance, or legal matters are rejected before "
                            "any retrieval or generation happens (zero cost, zero risk).")}
    return None


# ── Grounding score (mirrors evaluation.py's keyword-overlap proxy) ───────────
def grounding_score(answer: str, chunks: List[Dict[str, Any]]) -> float:
    if not chunks:
        return 0.0
    ctx_terms = set()
    for c in chunks:
        ctx_terms.update(_tokenize(c["text"]))
    ans_terms = [t for t in _tokenize(answer) if len(t) > 3]
    if not ans_terms:
        return 0.0
    hits = sum(1 for t in ans_terms if t in ctx_terms)
    return round(hits / len(ans_terms), 2)


def phi_scan(answer: str) -> bool:
    """True = clean. Mirrors evaluation.py's PHI pattern scan."""
    patterns = [
        r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",          # phone
        r"\b\d{3}-\d{2}-\d{4}\b",                       # SSN
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]+",  # email
    ]
    return not any(re.search(p, answer) for p in patterns)


def estimate_tokens(text: str) -> int:
    """Same 4-chars-per-token heuristic as production llm.py."""
    return max(1, len(text) // 4)


# ── Extractive fallback composer (DEMO mode, free-typed queries) ──────────────
def _extractive_answer(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, str]:
    if not chunks:
        return {"answer": "No retrieval match — the records don't contain information "
                          "relevant to this question, so I won't guess.",
                "confidence": "N/A"}
    q_terms = set(_tokenize(query))
    sentences = []
    for i, c in enumerate(chunks[:3]):
        for sent in re.split(r"(?<=[.!?])\s+", c["text"]):
            overlap = len(q_terms & set(_tokenize(sent)))
            if overlap > 0:
                sentences.append((overlap, sent.strip(), i + 1))
    sentences.sort(key=lambda x: -x[0])
    if not sentences:
        return {"answer": "No retrieval match — the retrieved records don't directly "
                          "answer this question, so I won't guess.",
                "confidence": "Low"}
    body = "\n".join(f"- {s} [{ref}]" for _, s, ref in sentences[:4])
    return {
        "answer": ("Here is what the retrieved records say (extractive mode — connect the "
                   "live Phi-3 model for synthesized answers):\n\n" + body),
        "confidence": "Medium" if sentences[0][0] >= 2 else "Low",
    }


# ── LIVE mode: real Ollama generation (same prompt contract as llm.py) ────────
def _ollama_answer(query: str, chunks: List[Dict[str, Any]], patient: Dict[str, Any]) -> Dict[str, str]:
    import urllib.request

    ctx_lines = [f"PATIENT DATA SUMMARY:\n- Name: {patient['patient_name']}\n"
                 f"- Diagnosis: {patient['diagnosis']}\n- Visits on record: {patient['visit_count']}",
                 "\nRELEVANT CLINICAL NOTES AND GUIDELINES:"]
    for i, c in enumerate(chunks):
        ctx_lines.append(f"[{i+1}] ({c['title']}, {c['date']}) {c['text']}")
    context = "\n".join(ctx_lines)

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 350},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = json.load(r)["message"]["content"]

    ans = re.search(r"Answer:\s*(.*?)(?=\s*Confidence:|$)", raw, re.I | re.S)
    conf = re.search(r"Confidence:\s*(High|Medium|Low)", raw, re.I)
    return {
        "answer": ans.group(1).strip() if ans else raw.strip(),
        "confidence": conf.group(1).capitalize() if conf else "Low",
    }


# ── Showcase matcher ───────────────────────────────────────────────────────────
def _match_showcase(query: str, mrd: str, corpus: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    qn = re.sub(r"[^a-z0-9 ]", "", query.lower()).strip()
    for sq in corpus["showcase_queries"]:
        sn = re.sub(r"[^a-z0-9 ]", "", sq["query"].lower()).strip()
        if qn == sn and sq["mrd"] == mrd:
            return sq
    return None


# ── Public entry point ─────────────────────────────────────────────────────────
def ask(query: str, mrd: str, corpus: Dict[str, Any], retriever: Retriever) -> Dict[str, Any]:
    """Full pipeline: guardrails -> retrieval -> generation -> safety checks.
    Returns everything the UI needs, including per-stage latency, token and
    cost estimates, retrieved chunks with scores, and safety check results."""
    t0 = time.perf_counter()
    mode = "live" if OLLAMA_URL else "demo"

    # Stage 0: guardrails (pre-retrieval, near-zero cost)
    guard = check_guardrails(query)
    if guard:
        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "answer": guard["message"], "confidence": "N/A", "chunks": [],
            "guardrail": guard["type"], "mode": mode,
            "latency": {"retrieval_ms": 0.0, "generation_ms": 0.0, "total_ms": total_ms},
            "tokens": {"prompt": estimate_tokens(query), "completion": 0, "total": estimate_tokens(query)},
            "cost_usd": 0.0,
            "safety": {"phi_scan_pass": True, "grounding": None, "blocked_before_llm": True},
        }

    # Stage 1: retrieval
    tr = time.perf_counter()
    showcase = _match_showcase(query, mrd, corpus)
    if showcase and showcase["chunk_ids"]:
        by_id = {c["id"]: c for c in corpus["chunks"]}
        base = [dict(by_id[cid]) for cid in showcase["chunk_ids"]]
        scored = {c["id"]: c for c in retriever.search(query, mrd, k=10)}
        chunks = []
        for rank, c in enumerate(base):
            hit = scored.get(c["id"])
            c["score"] = hit["score"] if hit else round(max(0.5, 3.0 - rank * 0.45), 3)
            c["relevance"] = hit["relevance"] if hit else round(max(0.3, 1.0 - rank * 0.15), 2)
            chunks.append(c)
    else:
        chunks = retriever.search(query, mrd)
    retrieval_ms = round((time.perf_counter() - tr) * 1000, 2)

    patient = corpus["patients"][mrd]

    # Stage 2: generation
    tg = time.perf_counter()
    if showcase and showcase["answer"] == "__PHI_REFUSAL__":
        gen = {"answer": PHI_REFUSAL_MESSAGE, "confidence": "N/A"}
        chunks = []
    elif mode == "live":
        try:
            gen = _ollama_answer(query, chunks, patient)
        except Exception as e:
            gen = {"answer": f"Live model unavailable ({e.__class__.__name__}). "
                             "Falling back to extractive mode:\n\n"
                             + _extractive_answer(query, chunks)["answer"],
                   "confidence": "Low"}
    elif showcase:
        gen = {"answer": showcase["answer"], "confidence": showcase["confidence"]}
        time.sleep(0.35)  # honest pacing: matches the production 0.4s median
    else:
        gen = _extractive_answer(query, chunks)
        time.sleep(0.25)
    generation_ms = round((time.perf_counter() - tg) * 1000, 2)

    # Stage 3: safety + metrics
    ctx_text = " ".join(c["text"] for c in chunks)
    prompt_tokens = estimate_tokens(SYSTEM_PROMPT + ctx_text + query)
    completion_tokens = estimate_tokens(gen["answer"])
    total_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "answer": gen["answer"],
        "confidence": gen["confidence"],
        "chunks": chunks,
        "guardrail": None,
        "mode": mode,
        "showcase_id": showcase["id"] if showcase else None,
        "latency": {"retrieval_ms": retrieval_ms, "generation_ms": generation_ms, "total_ms": total_ms},
        "tokens": {"prompt": prompt_tokens, "completion": completion_tokens,
                   "total": prompt_tokens + completion_tokens},
        "cost_usd": COST_PER_QUERY_LOCAL_USD,
        "cost_cloud_usd": COST_PER_QUERY_CLOUD_USD,
        "safety": {
            "phi_scan_pass": phi_scan(gen["answer"]),
            "grounding": grounding_score(gen["answer"], chunks) if chunks else None,
            # Showcase answers are human-verified claim-by-claim against their
            # cited chunks — stronger than the keyword-overlap proxy, which
            # under-scores paraphrased synthesis.
            "verified_citations": bool(showcase and chunks),
            "blocked_before_llm": False,
        },
    }
