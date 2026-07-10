"""
Module: eval_runner.py

Offline evaluation runner — the CLI entry point for benchmarking the
Clinical RAG Assistant end to end.

Executes a list of (mrd_number, query) test cases through the FULL
pipeline (retrieval -> generation -> evaluation, i.e. retrieval.py ->
llm.py -> evaluation.py, the same modules main.py uses live) and saves a
timestamped JSON report to eval_reports/ for regression tracking across
code changes.

Usage:
    python eval_runner.py                              # uses eval_test_cases.json
    python eval_runner.py --test-file my_cases.json     # or a custom file

Requires Ollama running locally with the `phi3` model pulled (both answer
generation and the LLM-judge scoring in evaluation.py call out to it) —
see the README's "Local LLM Setup" section.

Report shape: {"generated_at", "test_file", "summary": {...aggregate
stats...}, "cases": [...one detailed result per test case...]}. See
summarize_report() below for exactly which stats land in "summary".
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from retrieval import get_hybrid_context
from llm import generate_answer
from evaluation import (
    score_faithfulness,
    score_answer_relevance,
    detect_hallucination_flags,
    score_sample_groundedness,
    run_safety_eval,
    build_system_eval,
)

logger = logging.getLogger("clinical-rag.eval_runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DEFAULT_TEST_FILE = "eval_test_cases.json"
REPORT_DIR        = Path("eval_reports")

def load_test_cases(path: str) -> List[Dict[str, str]]:
    """
    Loads test cases from a JSON file.
    Expected format:
        [
            {"mrd_number": "MRD1001", "query": "What medications is the patient on?"},
            {"mrd_number": "MRD1002", "query": "When was the last visit?"}
        ]

    If the file doesn't exist, writes a starter template so the user has
    something to edit instead of hitting a confusing FileNotFoundError.
    """
    file_path = Path(path)
    if not file_path.exists():
        starter = [
            {"mrd_number": "MRD1001", "query": "What medications is the patient currently taking?"},
            {"mrd_number": "MRD1001", "query": "When was the patient's last visit?"},
        ]
        file_path.write_text(json.dumps(starter, indent=2))
        logger.warning(
            f"Test file '{path}' not found. Created a starter template with "
            f"{len(starter)} sample case(s) — edit it with your real MRD numbers and re-run."
        )
        return starter

    return json.loads(file_path.read_text())

def run_single_case(case: Dict[str, str]) -> Dict[str, Any]:
    """
    Runs ONE test case through the full pipeline and scores it across
    every category in the architecture diagram: retrieval, generation,
    safety, and system latency.
    """
    mrd_number = case["mrd_number"]
    query      = case["query"]

    result: Dict[str, Any] = {"mrd_number": mrd_number, "query": query}
    total_start = time.perf_counter()

    # ── Retrieval ───────────────────────────────────────────────────────────
    context, error_msg = get_hybrid_context(mrd_number, query)
    if error_msg:
        result["error"] = error_msg
        result["status"] = "retrieval_failed"
        return result

    llm_context  = context.get("formatted_context", "")
    sql_data     = context.get("sql_results", [])
    vector_docs  = context.get("vector_results", [])
    retrieval_latency = context.get("latency", {})
    retrieval_eval    = context.get("eval", {})

    # ── Generation ──────────────────────────────────────────────────────────
    llm_start = time.perf_counter()
    raw_answer = generate_answer(query, llm_context)
    llm_ms = round((time.perf_counter() - llm_start) * 1000, 2)

    total_ms = round((time.perf_counter() - total_start) * 1000, 2)

    # ── Generation Evaluation ──────────────────────────────────────────────
    faithfulness   = score_faithfulness(raw_answer, llm_context)
    relevance      = score_answer_relevance(raw_answer, query)
    hallucination  = detect_hallucination_flags(raw_answer, llm_context)

    # ── Retrieval Evaluation: Sample Groundedness ──────────────────────────
    groundedness = score_sample_groundedness(vector_docs, query)

    # ── Safety Evaluation ───────────────────────────────────────────────────
    safety = run_safety_eval(raw_answer, sql_data, mrd_number)

    # ── System Evaluation ───────────────────────────────────────────────────
    system_eval = build_system_eval(
        sql_ms=retrieval_latency.get("sql_ms", -1),
        vector_ms=retrieval_latency.get("vector_ms", -1),
        llm_ms=llm_ms,
        total_ms=total_ms,
    )

    result.update({
        "status": "success",
        "answer": raw_answer,
        "retrieval_eval": retrieval_eval,          # precision@k, mrr (from retrieval.py)
        "groundedness": groundedness,
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "hallucination_check": hallucination,
        "safety_eval": safety,
        "system_eval": system_eval,
    })

    logger.info(
        f"[{mrd_number}] Faithfulness: {faithfulness['score']} | "
        f"Relevance: {relevance['score']} | Safe: {safety['is_safe']} | "
        f"Total: {total_ms}ms"
    )

    return result

def summarize_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates per-case results into summary statistics — the numbers
    you'd actually put in a README benchmark table.
    """
    successful = [r for r in results if r.get("status") == "success"]
    n = len(successful)

    if n == 0:
        return {"total_cases": len(results), "successful": 0, "note": "No successful runs to summarize."}

    avg_faithfulness = sum(r["faithfulness"]["score"] or 0 for r in successful) / n
    avg_relevance    = sum(r["answer_relevance"]["score"] or 0 for r in successful) / n
    avg_total_ms     = sum(r["system_eval"]["total_latency_ms"] for r in successful) / n
    safety_pass_rate = sum(1 for r in successful if r["safety_eval"]["is_safe"]) / n
    hallucination_rate = sum(1 for r in successful if r["hallucination_check"]["flagged"]) / n

    return {
        "total_cases":          len(results),
        "successful":           n,
        "failed":               len(results) - n,
        "avg_faithfulness":     round(avg_faithfulness, 3),
        "avg_answer_relevance": round(avg_relevance, 3),
        "avg_total_latency_ms": round(avg_total_ms, 2),
        "safety_pass_rate":     round(safety_pass_rate, 3),
        "hallucination_flag_rate": round(hallucination_rate, 3),
    }

def main(test_file: str = DEFAULT_TEST_FILE):
    REPORT_DIR.mkdir(exist_ok=True)

    test_cases = load_test_cases(test_file)
    logger.info(f"Loaded {len(test_cases)} test case(s) from '{test_file}'")

    results = [run_single_case(case) for case in test_cases]
    summary = summarize_report(results)

    report = {
        "generated_at": datetime.now().isoformat(),
        "test_file": test_file,
        "summary": summary,
        "cases": results,
    }

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"eval_report_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2))

    logger.info(f"Report written to {report_path}")
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")

    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run offline evaluation suite.")
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE, help="Path to JSON test cases file.")
    args = parser.parse_args()
    main(args.test_file)