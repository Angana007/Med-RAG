"""
Test suite for the Med-RAG retrieval and safety layers.

Covers everything that does NOT require a live Ollama instance, so this
suite runs fast locally and in CI:

  - Pre-retrieval scope guardrails (imaging / billing queries blocked)
  - MRD validation and structured (SQL) retrieval hits
  - Error handling for unknown patients
  - Safety checks: PHI pattern scan, cross-patient leak detection,
    hallucination flagging (evaluation.py's pure functions)
  - Groundedness scoring
  - SQL lookup latency budget

LLM-dependent quality metrics (faithfulness, answer relevance) are
exercised separately via eval_runner.py, which needs Ollama running.

Run from the repo root:  pytest tests/ -v
"""

import os
import sys
import time

import pytest

# Make repo-root modules importable and ensure relative paths
# (clinical_data.db, faiss_index/) resolve regardless of where pytest
# is invoked from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from database import get_patient_metadata  # noqa: E402
from retrieval import get_hybrid_context  # noqa: E402
from evaluation import (  # noqa: E402
    check_cross_patient_leak,
    detect_hallucination_flags,
    scan_phi_patterns,
    score_sample_groundedness,
)

DEMO_MRD = "20001"  # the synthetic demo patient that ships with the repo


# ---------------------------------------------------------------------------
# Scope guardrails — blocked BEFORE any retrieval or LLM call
# ---------------------------------------------------------------------------

def test_guardrail_blocks_imaging_query():
    context, error = get_hybrid_context(DEMO_MRD, "Can you interpret the patient's most recent MRI scan?")
    assert context is None
    assert error is not None
    assert "Unsupported document type" in error


def test_guardrail_blocks_billing_query():
    context, error = get_hybrid_context(DEMO_MRD, "What is the patient's outstanding billing balance?")
    assert context is None
    assert error is not None
    assert "Unsupported question" in error


# ---------------------------------------------------------------------------
# MRD validation & structured retrieval
# ---------------------------------------------------------------------------

def test_invalid_mrd_returns_clear_error():
    context, error = get_hybrid_context("99999", "What is the patient's diagnosis?")
    assert context is None
    assert error is not None
    assert "99999" in error and "not found" in error


def test_sql_retrieval_hit_for_demo_patient():
    records = get_patient_metadata(DEMO_MRD)
    assert len(records) > 0, "Demo patient should exist in clinical_data.db"
    assert all(r["mrd_number"] == DEMO_MRD for r in records)
    assert records[0].get("patient_name"), "Records should carry a patient name"


def test_sql_retrieval_miss_returns_empty_list():
    assert get_patient_metadata("00000") == []


def test_sql_lookup_latency_within_budget():
    start = time.perf_counter()
    get_patient_metadata(DEMO_MRD)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 100, f"Indexed SQL lookup took {elapsed_ms:.1f} ms (budget: 100 ms)"


# ---------------------------------------------------------------------------
# Safety checks (evaluation.py — pure functions, no LLM)
# ---------------------------------------------------------------------------

def test_phi_scan_flags_raw_mrd_in_answer():
    result = scan_phi_patterns("The record for MRD-20001 shows a diabetes diagnosis.")
    assert result["phi_detected"] is True
    assert "mrd_number" in result["matched_types"]


def test_phi_scan_passes_clean_answer():
    result = scan_phi_patterns("The patient's diagnosis is Type 2 Diabetes.")
    assert result["phi_detected"] is False
    assert result["matched_types"] == []


def test_cross_patient_leak_detects_foreign_mrd():
    sql_data = [{"patient_name": "Jennifer Conrad"}]
    result = check_cross_patient_leak(
        "Records for MRD-30055 show a different treatment plan.", sql_data, DEMO_MRD
    )
    assert result["leak_detected"] is True
    assert "30055" in result["reason"]


def test_cross_patient_leak_allows_own_patient():
    sql_data = [{"patient_name": "Jennifer Conrad"}]
    result = check_cross_patient_leak(
        "Jennifer Conrad was counseled on diet and lifestyle modification.", sql_data, DEMO_MRD
    )
    assert result["leak_detected"] is False


def test_hallucination_flag_on_unsupported_numbers():
    context = "Patient advised follow-up in 4 weeks."
    answer = "The patient's HbA1c was 9.2 at the last visit."
    result = detect_hallucination_flags(answer, context)
    assert result["flagged"] is True
    assert "9.2" in result["suspect_numbers"]


def test_hallucination_flag_clean_when_grounded():
    context = "Patient advised follow-up in 4 weeks."
    answer = "Follow-up is recommended in 4 weeks."
    result = detect_hallucination_flags(answer, context)
    assert result["flagged"] is False


# ---------------------------------------------------------------------------
# Groundedness scoring
# ---------------------------------------------------------------------------

def test_groundedness_scores_relevant_docs():
    docs = ["Blood glucose monitored. HbA1c reviewed against prior visit."]
    result = score_sample_groundedness(docs, "What was noted about the patient's HbA1c review?")
    assert result["grounded_doc_ratio"] == 1.0
    assert result["ungrounded_count"] == 0


def test_groundedness_flags_offtopic_docs():
    docs = ["Vital signs stable. No acute complications noted."]
    result = score_sample_groundedness(docs, "What immunizations were administered?")
    assert result["grounded_doc_ratio"] == 0.0
    assert result["ungrounded_count"] == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
