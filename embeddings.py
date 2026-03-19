import os
import json
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

"""
Module: embeddings.py
Transforms raw clinical text into searchable mathematical vectors.
This allows the system to find relevant medical notes based on the 
meaning of a user's question, even if the exact words don't match.
"""

VECTOR_DB_PATH = 'faiss_index'

def process_clinical_docs():
    """
    Cleans medical records, splits them into manageable chunks, 
    and builds a searchable FAISS vector index.
    """
    
    # Using a lightweight, high-performance model for local clinical text embeddings.
    embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    documents = []
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    
    for file_name in json_files:
        with open(file_name, 'r') as f:
            data = json.load(f)
            records = data if isinstance(data, list) else [data]
            
            for record in records:
                # Clean HTML: Real-world clinical exports often contain noise;
                # parsing ensures the vector store index contains only searchable medical text.
                raw_html = record.get('description', '')
                clean_text = BeautifulSoup(raw_html, "html.parser").get_text(separator=" ")
                
                # Attach metadata so we can filter by MRD during search
                metadata = {
                    "mrd_number": str(record.get("mrd_number")),
                    "visit_id": record.get("visit_id"),
                    "document_type": record.get('document_type'),
                    "date": record.get('dschg_date'),
                    "patient_name": record.get('patient_name'),
                    "doctor": record.get('doctor_name')
                }
                
                documents.append(Document(page_content=clean_text, metadata=metadata))
                
    # Chunking Logic: 500-character chunks with overlap ensure that clinical context (like symptoms and dates)
    # isn't severed at the boundaries, improving retrieval accuracy for the LLM.
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    chunks = splitter.split_documents(documents)
    
    # Save the index locally so we don't have to re-process the files every time the server starts.
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(VECTOR_DB_PATH)
    print("Vector database created successfully.")
    
if __name__ == "__main__":
    process_clinical_docs()