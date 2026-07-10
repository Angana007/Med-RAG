"""
Module: llm.py

Manages the interaction with the local Large Language Model (Phi-3, via
Ollama). Focuses on strict prompt engineering to ensure medical accuracy
and safety by preventing the AI from using outside knowledge.

Two public entry points, both returning a plain "Answer: ...\\nConfidence:
..." formatted string (never raises — always falls back to
FALLBACK_RESPONSE on unrecoverable failure):
  - generate_answer(query, context)        normal path, used by main.py
                                             and eval_runner.py.
  - generate_answer_strict(query, context)  stricter safety prompt, used
                                             ONLY as a one-shot regeneration
                                             after a safety violation is
                                             detected (see main.py's
                                             retry-then-block policy).

Production additions on top of a bare ollama.chat() call:
    1. Response time tracking (ms-level, per generation)
    2. Token counting (prompt tokens, completion tokens, total — a cheap
       4-chars-per-token estimate, see count_tokens())
    3. Tiered fallback handling (format fix -> retry -> safe fallback)
    4. Structured logging (replaces silent except)
"""

import ollama
import time
import logging

logger = logging.getLogger("clinical-rag.llm")

# ============================================================
# Configuration 
# ============================================================
MODEL_NAME = "phi3"
MAX_RETRIES = 2       # How many times to retry before giving up
NUM_PREDICT = 250     # Token limit per response — keeps answers concise
TEMPERATURE = 0.0     # Zero = deterministic, essential for clinical facts
TIMEOUT_ERROR = "The medical assistant is currently unavailable (Timeout). Please try again later."
FALLBACK_RESPONSE = f"Answer: {TIMEOUT_ERROR}\nConfidence Score: N/A"

# ============================================================
# Token Counter
# ============================================================
def count_tokens(text: str) -> int:
    """
    Estimates the number of LLM tokens using the heuristic:
    1 token ≈ 4 characters (for typical English text).
    This lightweight approximation is suitable for monitoring and logging,
    but not for exact token accounting; swap with tiktoken if you need precision.

    Args: 
        text: Any string (prompt, context, or response).
    Returns:
        int: Estimated token count
    """
    return max(1, len(text) // 4)

# ============================================================
# Format Validation
# ============================================================
def _is_valid_format(text: str) -> bool:
    """
    Checks if the LLM response follows the required output structure.
    Both 'Answer:' and 'Confidence:' must be present.
    """
    return "Answer:" in text and "Confidence:" in text

def _fix_format(raw_text: str) -> str:
    """
    Repairs responses missing the 'Confidence:' line.
    Returns the repaired text or the original if no repair is possible.
    """
    if "Answer:" in raw_text and "Confidence:" not in raw_text:
        logger.warning("LLM response missing 'Confidence:' - appending Low as fallback.")
        return raw_text.strip() + "\nConfidence: Low"
    return raw_text

# ============================================================
# Generation Function 
# ============================================================
def generate_answer(query: str, context: str) -> str:
    """
    Sends clinical context to phi3 and enforces a structured and safe response.

    Features:
    - Response time and token logging
    - Format repair, retry, and safe fallback
    - Structured logging for all execution paths

    Handles:
    - Unsupported questions (Non-medical or out of scope questions)
    - No retrieval match (when context does not answer the query)
    - Confidence score generation
    - LLM timeout or crash

    Args:
        query:   The clinical question from the user
        context: The retrieved patient context string (from retrieval.py)
    Returns:
        str: Structured LLM response in "Answer: ...\nConfidence: ..." format.
            Falls back to FALLBACK_RESPONSE on unrecoverable failure.
    """
    # ============================================================
    # Prompt Construction 
    # ============================================================
    # The system prompt is IMPORTANT since it is the GUARDRAIL for clinical safety.
    system_message = """
    You are a professional Medical AI Assistant. Use ONLY the provided context. 
    Rules:
    1. If the question asks about "dates", "visits", or "history", analyze the context.
    2. If the answer exists in the context, extract it directly.
    3. If the context truly lacks the answer, say "No retrieval match."
    4. Do not use outside knowledge.
    5. Always output:

    Answer: <response>
    Confidence: <High/Medium/Low>
    """
    
    #Building the prompt for phi3: We use chat method
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"}
    ]

    # Token counting: Counted before the call so we always have prompt tokens even if LLM fails.
    prompt_text = system_message + context + query
    prompt_tokens = count_tokens(prompt_text)
    logger.info(
        f"LLM request | Model: {MODEL_NAME} | "
        f"Prompt tokens (est.): {prompt_tokens} | "
        f"Max completion tokens: {NUM_PREDICT}"
    )

    # Generation with Retry Loop
    last_exception = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Response Time Tracking
            gen_start = time.perf_counter()
            print("About to call ollama.chat()")
            response =ollama.chat(
                model = MODEL_NAME,
                messages=messages,
                options={
                    "temperature": TEMPERATURE,
                    "num_predict": NUM_PREDICT,
                }
            )
            print("ollama.chat() returned")
            gen_ms = round((time.perf_counter() - gen_start) * 1000, 2)

            # Token counting
            answer_text = response["message"]["content"]
            completion_tokens = count_tokens(answer_text)
            total_tokens = prompt_tokens + completion_tokens

            logger.info(
                f"LLM response | Attempt: {attempt}/{MAX_RETRIES} | "
                f"Generation time: {gen_ms}ms | "
                f"Completion tokens (est.): {completion_tokens} | "
                f"Total tokens (est.): {total_tokens}"
            )

            # Tiered fallback: Level 1 (Format Fix)
            if not _is_valid_format(answer_text):
                repaired = _fix_format(answer_text)
                if _is_valid_format(repaired):
                    logger.info("Format repaired successfully: returning fixed response.")
                    return repaired
                # Format still broken after repair → fall through to retry
                logger.warning(
                    f"Attempt {attempt}: Response format invalid after repair attempt. "
                    f"Retrying..." if attempt < MAX_RETRIES else "No retries left."
                )
                continue    # retry the LLM call
            # Valid format: return immediately
            return answer_text
        
        except Exception as e:
            last_exception = e
            gen_ms_fail = round((time.perf_counter() - gen_start) * 1000, 2) if 'gen_start' in dir() else "n/a"

            # Tiered Fallback: Level 2 — Retry
            if attempt < MAX_RETRIES:
                logger.warning(
                    f"LLM call failed on attempt {attempt}/{MAX_RETRIES} "
                    f"after {gen_ms_fail}ms | Error: {e} | Retrying..."
                )
            else:
                # Tiered Fallback: Level 3 — Safe fallback
                # All retries exhausted. Return the safe clinical fallback string.
                # Ensure the API always returns a structured response.
                logger.error(
                    f"LLM failed after {MAX_RETRIES} attempts | "
                    f"Last error: {last_exception}",
                    exc_info=True
                )
    return FALLBACK_RESPONSE


# ============================================================
# Strict Regeneration (Safety Retry)
# ============================================================
# Invoked ONLY when the initial response fails a live safety check
# (e.g., cross-patient leakage or unintended PHI exposure).
#
# A stricter system prompt is used to regenerate the answer exactly once,
# explicitly prohibiting the detected safety violation.
#
# NOTE:
# This is independent of generate_answer()'s internal MAX_RETRIES loop.
# MAX_RETRIES handles transient generation failures (exceptions, invalid
# output, formatting issues), whereas this retry is exclusively for
# post-generation safety violations.
# =============================================================================

STRICT_SAFETY_SUFFIX = """
ADDITIONAL STRICT SAFETY RULES (this answer was flagged for a safety violation):
- NEVER include any other patient's name, MRD number, or identifying detail.
- NEVER output raw identifiers such as SSN, phone number, email, or date of birth,
  even if present in the context — describe their existence in general terms only
  if relevant, never the literal value.
- If you cannot answer without including such information, respond with:
  "Answer: This information cannot be safely disclosed in this format.\\nConfidence: Low"  
"""

def generate_answer_strict(query: str, context: str) -> str:
    """
    Regenerates an answer with a stricter safety-focused system prompt.
    Used as the SECOND attempt after a safety violation is detected in
    the first answer — see run_safety_eval() in evaluation.py and the
    retry-then-block policy in main.py's /query endpoint.

    This does NOT replace generate_answer()'s own retry loop (which handles
    format errors and exceptions). Instead, it performs a single, safety-specific
    regeneration only if the initial answer fails a live safety check for PHI
    exposure or cross-patient information leakage.

    Args:
        query:   The clinical question.
        context: The retrieved patient context.

    Returns:
        str: A new "Answer: ...\\nConfidence: ..." formatted response,
            generated under stricter safety constraints. 
    """
    system_message = f"""
    You are a professional Medical AI Assistant. Use ONLY the provided context.
    Rules:
    1. If the question asks about "dates", "visits", or "history", analyze the context.
    2. If the answer exists in the context, extract it directly.
    3. If the context truly lacks the answer, say "No retrieval match."
    4. Do not use outside knowledge.
    5. Always output:

    Answer: <response>
    Confidence: <High/Medium/Low>
    {STRICT_SAFETY_SUFFIX}
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"}
    ]
    logger.warning("Running strict regeneration after safety violation on first attempt.")
    try:
        gen_start = time.perf_counter()
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options={"temperature": TEMPERATURE, "num_predict": NUM_PREDICT},
        )
        gen_ms = round((time.perf_counter() - gen_start) * 1000, 2)
        answer_text = response["message"]["content"]

        logger.info(f"Strict regeneration completed in {gen_ms}ms")

        if not _is_valid_format(answer_text):
            answer_text = _fix_format(answer_text)

        return answer_text
    except Exception as e:
        logger.error(f"Strict regeneration failed: {e}", exc_info=True)
        return FALLBACK_RESPONSE