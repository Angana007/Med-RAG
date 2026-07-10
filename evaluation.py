"""
Module: evaluation.py

Centralized evaluation layer for the Clinical RAG Assistant — Retrieval,
Generation, Safety, and System checks. Used inline by main.py (fast,
non-LLM checks only, on every live request) and offline by eval_runner.py
(the full suite, including LLM-judged scores), so both paths share the
exact same scoring logic and never drift apart.

Organized into four sections (search for the "════" banners below):
  1. GENERATION EVAL   — score_faithfulness, score_answer_relevance,
                          detect_hallucination_flags
  2. RETRIEVAL EVAL    — score_sample_groundedness
  3. SAFETY EVAL       — scan_phi_patterns, check_cross_patient_leak,
                          run_safety_eval
  4. SYSTEM EVAL       — LatencyTracker, build_system_eval

Functions whose name starts with `score_*`/`detect_*`/`scan_*`/`check_*`
are individual checks; `run_safety_eval` and `build_system_eval` are
composite functions that bundle several of those into one verdict.
"""

import re
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import ollama

logger = logging.getLogger("clinical-rag.eval")
JUDGE_MODEL = "phi3"        # Using Phi-3 itself as judge — same model already running locally

# ════════════════════════════════════════════════════════════════════════════
# GENERATION EVALUATION — Faithfulness, Answer Relevance, Hallucination
# ════════════════════════════════════════════════════════════════════════════

def _call_judge(prompt: str) -> Optional[dict]:
    """
    Sends a scoring prompt to Phi-3 (acting as judge) and parses a JSON score.
    """
    try:
        response = ollama.chat(
            model = JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 100},
        )
        raw = response["message"]["content"].strip()

        match = re.search(r"\{.*\}", raw, re.DOTALL)        # small models sometimes wrap JSON in extra text
        if not match:
            logger.warning(f"Judge response had no parsable JSON: {raw[:120]!r}")
            return None
        return json.loads(match.group(0))
    except Exception as e:
        logger.error(f"Judge call failed: {e}", exc_info=True)
        return None

def score_faithfulness(answer: str, context: str) -> Dict[str, Any]:
    """
    Does the answer only use facts present in the context? 0.0 (unfaithful) to 1.0 (faithful). Uses Phi-3 as judge.
    Returns:
        {
            "score": float,           # 0.0 - 1.0
            "verdict": str,           # "faithful" | "partially_faithful" | "unfaithful"
            "reasoning": str,         # judge's stated reason (truncated)
            "judge_model": str,       # which model judged (self-grading transparency)
        }
    """
    judge_prompt = f"""You are a strict fact-checker. Compare the ANSWER to the CONTEXT.
Determine if every claim in the ANSWER is directly supported by the CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Respond ONLY with JSON, no other text:
{{"score": <0.0 to 1.0>, "verdict": "<faithful|partially_faithful|unfaithful>", "reasoning": "<one short sentence>"}}
"""
    result = _call_judge(judge_prompt)
    if result is None:
        return {"score": None, "verdict": "unscored", "reasoning": "Judge call failed.", "judge_model": JUDGE_MODEL}

    return {
        "score": round(float(result.get("score", 0.0)), 3),
        "verdict": result.get("verdict", "unscored"),
        "reasoning": str(result.get("reasoning", ""))[:200],
        "judge_model": JUDGE_MODEL,
    }

def score_answer_relevance(answer: str, query: str) -> Dict[str,Any]:
    """Does the answer address what was asked, regardless of factual correctness? 0.0 to 1.0."""
    judge_prompt = f"""Rate how directly this ANSWER addresses the QUESTION asked, ignoring whether facts are correct.

QUESTION:
{query}

ANSWER:
{answer}

Respond only with JSON, no other text:
{{"score": <0.0 to 1.0>, "reasoning": "<one short sentence>"}}
"""
    result = _call_judge(judge_prompt)
    if result is None:
        return {"score": None, "reasoning": "Judge call failed.", "judge_model": JUDGE_MODEL}
    return {
        "score": round(float(result.get("score", 0.0)), 3),
        "reasoning": str(result.get("reasoning", ""))[:200],
        "judge_model": JUDGE_MODEL,
    }
def detect_hallucination_flags(answer: str, context: str) -> Dict[str,Any]:
    """Fast, non-LLM check: flags numbers/dates in the answer not found in the context.
    Cheap pre-filter for every request; faithfulness (LLM judge) is the deeper check."""
    answer_numbers = set(re.findall(r"\b\d+\.?\d*\b", answer))
    context_numbers = set(re.findall(r"\b\d+\.?\d*\b", context))
    suspect = sorted(answer_numbers - context_numbers)
    if suspect:
        logger.warning(f"Hallucination flag: numbers in answer not found in context: {suspect}")
    return {"suspect_numbers": suspect, "flagged": len(suspect) > 0}

# ════════════════════════════════════════════════════════════════════════════
# RETRIEVAL EVALUATION
# ════════════════════════════════════════════════════════════════════════════
def score_sample_groundedness(retrieved_docs: List[str], query: str) -> Dict[str, Any]:
    """Checks keyword overlap between query and retrieved docs — catches silent
        retrieval failures where vector search returns semantically-near but off-topic chunks."""
    if not retrieved_docs:
        return {"grounded_doc_ratio": 0.0, "ungrounded_count": 0, "total_docs": 0}

    stopwords = {"what", "is", "the", "of", "for", "and", "a", "an", "in", "to",
                 "did", "does", "has", "have", "was", "were", "patient", "this"}
    keywords = [w.strip("?,.'\"").lower() for w in query.split() if w.lower() not in stopwords and len(w) > 2]

    if not keywords:
        return {"grounded_doc_ratio": 0.0, "ungrounded_count": len(retrieved_docs), "total_docs": len(retrieved_docs)}

    grounded = sum(1 for doc in retrieved_docs if any(kw in doc.lower() for kw in keywords))

    return {
        "grounded_doc_ratio": round(grounded / len(retrieved_docs), 3),
        "ungrounded_count": len(retrieved_docs) - grounded,
        "total_docs": len(retrieved_docs),
    }

# ════════════════════════════════════════════════════════════════════════════
# SAFETY EVALUATION — Privacy / PHI Leak Testing
# ════════════════════════════════════════════════════════════════════════════
_PHI_PATTERNS = {
    "mrd_number": re.compile(r"\bMRD[\s\-:]?\d{4,10}\b", re.IGNORECASE),
    "date_value": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),  # catches DOB, visit dates, dschg_date
}
# Dormant patterns — no SSN/phone/email/national-ID field exists in the patient table today.
# Kept for when the schema grows; currently these can never match real data, only hallucinated text.
_FUTURE_PHI_PATTERNS = {
    "ssn":          re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone":        re.compile(r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "email":        re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "aadhaar_like": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
}
def scan_phi_patterns(answer: str, include_future: bool = False) -> Dict[str, Any]:
    """Flags raw PHI patterns that should never appear verbatim in a response,
    even if present in the patient's own record. Set include_future=True to also
    scan dormant patterns (SSN/phone/email) once the schema grows to include them."""
    patterns = {**_PHI_PATTERNS, **_FUTURE_PHI_PATTERNS} if include_future else _PHI_PATTERNS
    matched = [name for name, pattern in patterns.items() if pattern.search(answer)]
 
    if matched:
        logger.warning(f"PHI pattern leak detected in answer: {matched}")
 
    return {"phi_detected": len(matched) > 0, "matched_types": matched}

def check_cross_patient_leak(answer: str, sql_data: List[dict], requested_mrd: str) -> Dict[str, Any]:
    """Flags MRD numbers or names in the answer that don't match the requested patient; retrieval misses"""
    mrd_pattern = re.compile(r"\bMRD[\s\-:]?(\d{4,10})\b", re.IGNORECASE)
    found_mrds = {m.group(1) for m in mrd_pattern.finditer(answer)} # capture just the digits, not "MRD-" prefix
    foreign_mrds = {m for m in found_mrds if m != requested_mrd} # exact match, not substring containment
    if foreign_mrds:
        return {"leak_detected": True, "reason": f"Answer references other MRD number(s): {sorted(foreign_mrds)}"}
    expected_name = sql_data[0].get("patient_name", "").strip().lower() if sql_data else ""
    if expected_name:
        name_like = re.findall(r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b", answer)
        suspicious = [n for n in name_like if n.lower() != expected_name]
        if suspicious:
            return {"leak_detected": True, "reason": f"Answer contains name(s) not matching requested patient: {suspicious}"}
 
    return {"leak_detected": False, "reason": ""}

def run_safety_eval(answer: str, sql_data: List[dict], requested_mrd: str) -> Dict[str, Any]:
    """Combines PHI scan + cross-patient leak check into one verdict. Called inline per-request."""
    phi_result = scan_phi_patterns(answer)
    cross_result = check_cross_patient_leak(answer, sql_data, requested_mrd)
 
    return {
        "is_safe": not phi_result["phi_detected"] and not cross_result["leak_detected"],
        "phi_leak": phi_result,
        "cross_patient_leak": cross_result,
    }

# ════════════════════════════════════════════════════════════════════════════
# SYSTEM EVALUATION
# ════════════════════════════════════════════════════════════════════════════
class LatencyTracker:
    """Marks named timestamps and computes elapsed ms between any two marks."""

    def __init__(self):
        self.marks: Dict[str, float] = {}

    def mark(self, label: str) -> None:
        self.marks[label] = time.perf_counter()

    def elapsed_ms(self, start_label: str, end_label: str) -> float:
        if start_label not in self.marks or end_label not in self.marks:
            return -1.0
        return round((self.marks[end_label] - self.marks[start_label]) * 1000, 2)
    
def build_system_eval(sql_ms: float, vector_ms: float, llm_ms: float, total_ms: float) -> Dict[str, Any]:
    """
    Assembles the system-level latency report and computes theoretical max QPS for the hybrid search pipeline.
    """

    return {
        "sql_latency_ms": sql_ms,
        "vector_latency_ms": vector_ms,
        "llm_latency_ms": llm_ms,
        "total_latency_ms": total_ms,
        "implied_throughput_qps": round(1000 / total_ms, 3) if total_ms > 0 else 0.0,
    }
