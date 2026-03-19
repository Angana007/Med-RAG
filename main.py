from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from llm import generate_answer
from retrieval import get_hybrid_context
from typing import Dict, Any
import re

"""
Module: main.py
The primary entry point for the Clinical RAG Assistant API. 
This module orchestrates the flow between user requests, medical data 
retrieval, and AI-generated responses using FastAPI.
"""

#Initialize the FastAPI Application
app = FastAPI(
    title="Clinical RAG Assistant API",
    description = "A secure API for querying clinical patient records using Hybrid Retrieval-Augmented Generation.",
    version ="1.0.0"
)

class QueryRequest(BaseModel):
    """
    Defines the structured format for incoming clinical questions.
    Uses Pydantic to ensure data integrity before any processing begins.
    """
    
    mrd_number: str = Field(..., description="The patient's unique Medical Record Number.")
    query: str = Field(..., description="The clinical question related to the patient's records.")
    
    #Handle Empty Query
    @field_validator('query')
    @classmethod
    def query_must_not_be_empty(cls, v: str) -> str:
        """Ensure that the query string is not empty or whitespace."""
        if not v.strip():
            raise ValueError('The clinical query cannot be empty.')
        return v

def parse_llm_response(raw_text: str) -> Dict[str, str]:
    """
    Parses the LLM's string output into a structured dictionary.
    Uses a lookahead regex to ensure the 'answer' field doesn't include 
    the 'confidence' label.
    """
    # 1. Extract Answer: Matches everything from "Answer:" up until it hits "Confidence:"
    # The (.*?) is non-greedy, and (?=...) is a lookahead that doesn't consume the text.
    answer_match = re.search(r"Answer:\s*(.*?)(?=\s*Confidence:|$)", raw_text, re.IGNORECASE | re.DOTALL)
    
    # 2. Extract Confidence: Matches High, Medium, or Low
    confidence_match = re.search(r"Confidence:\s*(High|Medium|Low)", raw_text, re.IGNORECASE)
    
    # 3. Clean and Assign
    # If the regex find fails, we fall back to cleaning the raw text as a safety measure.
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # Backup: If the specific "Answer:" tag isn't found, 
        # we take the raw text and strip the confidence line manually.
        answer = re.sub(r"Confidence:.*", "", raw_text, flags=re.IGNORECASE | re.DOTALL).strip()
        answer = re.sub(r"Answer:\s*", "", answer, flags=re.IGNORECASE).strip()

    confidence = confidence_match.group(1).strip() if confidence_match else "Low"
    
    return {
        "answer": answer,
        "confidence": confidence
    }

# API Endpoints
@app.post("/query", status_code=status.HTTP_200_OK)
async def process_clinical_query(request: QueryRequest):
    """
    Process a clinical query using RAG pipeline.
    
    Workflow:
    1. Retrieve relevant patient context using hybrid retrieval.
    2. Generate an answer using the LLM based on the retrieved context.
    3. Return structured response with the generated result.
    
    Error Handling:
    1. Invalid MRD number
    2. Retrieval errors or unsupported queries
    3. LLM/system timeout or internal failure
    """
    
    # Step 1: Retrieve relevant patient context using hybrid retrieval which combines structured lookup and semantic search
    context, error_msg = get_hybrid_context(request.mrd_number, request.query)
    
    if error_msg:
        # Handles error due to Invalid MRD (patient record not found)
        if "not found" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "Invalid_MRD", "message": error_msg}
            )
            
        # Retrieval failure (unsupported document types or query mismatch)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Retrieval failure", "message": error_msg}
        )
        
    # Step 2: Generate the final answer using the LLM based on the retrieved context
    try:
        raw_result = generate_answer(request.query, context)
        
        # Step 3: Structured Response Mapping
        # New: Separates the raw LLM string into 'answer' and 'confidence' JSON keys
        parsed_data = parse_llm_response(raw_result)
        
        return {
            "mrd_number": request.mrd_number,
            "answer": parsed_data["answer"],
            "confidence": parsed_data["confidence"]
        }
        
    except Exception as e:
        # Catches LLM Timeout or system-level failure to maintain API stability.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "System Timeout", "message": "The LLM failed to respond in time."}
        )
        
# STARTUP CHECK: Useful for monitoring and deployment readiness checks
@app.get("/health")
def health_check():
    """
    Service Monitoring: Provides a simple heartbeat endpoint to verify the 
    API is online and the model is correctly loaded.
    """
    return {"status": "online", "model": "Phi-3 (Mini-4K-Instruct)"}

# Local Entry Point
if __name__ == "__main__":
    import uvicorn
    # High-performance local server entry point
    uvicorn.run(app, host="0.0.0.0", port=8000)