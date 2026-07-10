"""
Module: retrieval.py

Implements the Hybrid Retrieval strategy at the heart of the system: for
a given (mrd_number, query), combines structured SQL lookups (database.py
— patient identity + visit metadata) with semantic vector search
(embeddings.py's FAISS index — clinical note narratives) into one
context object for the LLM.

Single public entry point: get_hybrid_context(mrd_number, query) -> see
its docstring below for the exact return shape. Both main.py (live API)
and eval_runner.py (offline benchmark) call this same function, so
retrieval behavior is identical in both paths.

Two guardrails run BEFORE any retrieval happens (cheap keyword checks on
the query text): UNSUPPORTED_DOC_TYPES rejects questions about imaging
this system can't interpret (x-ray/MRI/etc.), and UNSUPPORTED_TOPICS
rejects out-of-scope administrative questions (billing/insurance/legal/
etc.) so they never reach the LLM.
"""

import os
import time
from typing import Tuple, Optional, Dict, Any
from database import get_patient_metadata
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration Constants
VECTOR_DB_PATH = "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

UNSUPPORTED_DOC_TYPES = ["x-ray", "mri", "ct scan", "ultrasound", "imaging", "ecg", "eeg"]
UNSUPPORTED_TOPICS = ["billing", "insurance", "payment", "salary", "claims", "legal", "lawsuit","staffing", "inventory", "equipment_specs"]

def get_hybrid_context(mrd_number: str, query: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Combines SQL 'Facts' with Vector 'Narratives' to build a holistic view
    of the patient's record for the AI.
    Args:
        mrd_number (str): The unique patient ID to filter results.
        query (str): The user's medical question.

    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[str]]: A tuple containing
        (context, error_message). context is a dict with keys:
        "formatted_context" (str, what gets sent to the LLM), "sql_results"
        (list of patient metadata dicts), "vector_results" (list of matched
        clinical note strings), and "latency" (sql_ms/vector_ms/total_ms) —
        this shape is what main.py and eval_runner.py expect for safety
        checks, groundedness scoring, and latency reporting.
    """
    total_start = time.perf_counter()
    query_lower = query.lower()

    # Check for Unsupported Document Types
    if any(doc in query_lower for doc in UNSUPPORTED_DOC_TYPES):
        return None, "Unsupported document type: This system only analyzes text-based clinical notes and cannot interpret imaging or scan results yet."

    # Check for Unsupported Topics (Billing/Insurance)
    if any(topic in query_lower for topic in UNSUPPORTED_TOPICS):
        return None, "Unsupported question: This assistant is for clinical inquiries only and does not have access to administrative or billing data."

    #Structured Data Retrieval (SQL): We fetch patient demographic and visit history metadata first
    sql_start = time.perf_counter()
    sql_data = get_patient_metadata(mrd_number)
    sql_ms = round((time.perf_counter() - sql_start) * 1000, 2)
    if not sql_data:
        return None, f"MRD {mrd_number} not found in the database. Please verify the ID."

    #Semantic Data Retrieval (Vector Store): We use embeddings to find the most relevant information
    vector_start = time.perf_counter()
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        #Check if the FAISS index exists before loading the model
        if not os.path.exists(VECTOR_DB_PATH):
            return None, f"Vector database (FAISS) not found. Please run the embedding.py"
        vector_db = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization = True
        )

        #We use the metadata filter to restrict search to THIS patient only.
        docs = vector_db.similarity_search(query, k=3)
        vector_results = [doc.page_content for doc in docs]
        vector_context = "\n".join(vector_results)

        if not vector_context:
            vector_context = "No relevant clinical notes found for this query."

    except Exception as e:
        return None, f"Error during vector retrieval: {str(e)}"
    vector_ms = round((time.perf_counter() - vector_start) * 1000, 2)

    #Full Context Synthesis:  We combine the SQL 'Facts' with the Vector 'Narratives' to create a holistic view.
    patient_name = sql_data[0].get('patient_name', 'Unknown')      #Adding 'Unknown' will avoid "KeyError" Crash in case patient_name is not there
    visit_count = len(sql_data)

    formatted_sql = f"PATIENT DATA SUMMARY:\n- Name: {patient_name}\n- Total Records Found: {visit_count}"

    full_context = (
        f"{formatted_sql} \n\n"
        f"RELEVANT CLINICAL NOTES: \n"
        f"{vector_context}"
    )

    total_ms = round((time.perf_counter() - total_start) * 1000, 2)

    context = {
        "formatted_context": full_context,
        "sql_results": sql_data,
        "vector_results": vector_results,
        "semantic_chunks": vector_results,   # alias — main.py's detect_retrieval_source() checks this key too
        "latency": {"sql_ms": sql_ms, "vector_ms": vector_ms, "total_ms": total_ms},
        # No ground-truth relevance labels exist for this dataset, so precision@k / MRR
        # can't be computed — left empty rather than faked. groundedness (keyword-overlap
        # proxy) is still scored downstream in evaluation.py from vector_results.
        "eval": {},
    }

    return context, None