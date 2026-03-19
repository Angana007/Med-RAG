import os
from typing import Tuple, Optional
from database import get_patient_metadata 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

"""
Module: retrieval.py
Implements a Hybrid Retrieval strategy. 
It combines structured SQL lookups (for patient identity) with semantic 
vector search (for medical notes) to provide a complete clinical context.
"""

# Configuration Constants
VECTOR_DB_PATH = "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

UNSUPPORTED_DOC_TYPES = ["x-ray", "mri", "ct scan", "ultrasound", "imaging", "ecg", "eeg"]
UNSUPPORTED_TOPICS = ["billing", "insurance", "payment", "salary", "claims", "legal", "lawsuit","staffing", "inventory", "equipment_specs"]

def get_hybrid_context(mrd_number: str, query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Combines SQL 'Facts' with Vector 'Narratives' to build a holistic view 
    of the patient's record for the AI.
    Args:
        mrd_number (str): The unique patient ID to filter results.
        query (str): The user's medical question.

    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple containing (formatted_context, error_message).
    """
    query_lower = query.lower()
    
    # Check for Unsupported Document Types
    if any(doc in query_lower for doc in UNSUPPORTED_DOC_TYPES):
        return None, "Unsupported document type: This system only analyzes text-based clinical notes and cannot interpret imaging or scan results yet."

    # Check for Unsupported Topics (Billing/Insurance)
    if any(topic in query_lower for topic in UNSUPPORTED_TOPICS):
        return None, "Unsupported question: This assistant is for clinical inquiries only and does not have access to administrative or billing data."
    
    #Structured Data Retrieval (SQL): We fetch patient demographic and visit history metadata first
    sql_data = get_patient_metadata(mrd_number)
    if not sql_data:
        return None, f"MRD {mrd_number} not found in the database. Please verify the ID."
    
    #Semantic Data Retrieval (Vector Store): We use embeddings to find the most relevant information
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
        vector_context = "\n".join([doc.page_content for doc in docs])
        
        if not vector_context:
            vector_context = "No relevant clinical notes found for this query."
            
    except Exception as e:
        return None, f"Error during vector retrieval: {str(e)}"
    
    #Full Context Synthesis:  We combine the SQL 'Facts' with the Vector 'Narratives' to create a holistic view.
    patient_name = sql_data[0].get('patient_name', 'Unknown')      #Adding 'Unknown' will avoid "KeyError" Crash in case patient_name is not there
    visit_count = len(sql_data)
    
    formatted_sql = f"PATIENT DATA SUMMARY:\n- Name: {patient_name}\n- Total Records Found: {visit_count}"
    
    full_context = (
        f"{formatted_sql} \n\n"
        f"RELEVANT CLINICAL NOTES: \n"
        f"{vector_context}"
    )
    
    return full_context, None