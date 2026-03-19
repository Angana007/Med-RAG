# 🩺 Med-RAG: Local Hybrid-Retrieval Medical Chatbot

## Overview: 
---

A specialized Medical NLP Chatbot designed for high-accuracy clinical query answering. This system implements a Hybrid Retrieval-Augmented Generation (RAG) architecture, combining the precision of SQL with the semantic depth of Vector Search.

🌟 Key Features
100% Local & Private: No external API calls (OpenAI/Anthropic). All data stays on-premise, ensuring HIPAA-compliant design principles.

Hybrid Retrieval: Merges structured SQLite lookups (for patient metadata) with semantic FAISS vector search (for unstructured clinical notes).

Longitudinal View: Smart SQL logic identifies all historical visit IDs associated with an MRD to analyze patient trends over time.

Deterministic Guardrails: Hard-coded system prompts and temperature: 0 settings to eliminate hallucinations.
---------------------------------------------------------------------------------------------
## System Architecture:
---

                    ┌─────────────────────────────┐
                    │        Client/User          │
                    │           (Curl)            │
                    └─────────────┬───────────────┘
                                  │
                                  │ HTTP Request
                                  ▼
                        ┌──────────────────┐
                        │   FastAPI Server │
                        │  Chatbot Backend │
                        └─────────┬────────┘
                                  │
                        Query Processing Layer
                                  │
             ┌────────────────────┴────────────────────┐
             │                                         │
             ▼                                         ▼
   ┌──────────────────────┐                 ┌────────────────────────┐
   │ Structured Retrieval │                 │ Semantic Retrieval     │
   │ (SQL Database)       │                 │ (Vector Search / RAG)  │
   │                      │                 │                        │
   │ Patient Metadata     │                 │ Embedded Clinical Notes│
   │ MRD Number           │                 │ Progress Notes         │
   │ Visit Information    │                 │ Consultation Reports   │
   │ Doctor Details       │                 │ Medical Descriptions   │
   └──────────┬───────────┘                 └──────────┬─────────────┘
              │                                       │
              ▼                                       ▼
       ┌───────────────┐                      ┌─────────────────┐
       │   SQLite DB   │                      │ Vector Database │
       │               │                      │     (FAISS)     │
       └───────────────┘                      └────────┬────────┘
              │                                        │
              │                                        │
              └─────────────┬──────────────────────────┘
                            │                           
                            ▼
                     Retrieved Context Chunks
                            │
                            ▼
              ┌────────────────────────┐
              │ Prompt Construction    │
              │                        │
              │ - System Prompt        │
              │ - SQL Metadata         │
              │ - Vector Context       │
              │ - User Query           │
              └─────────────┬──────────┘
                            │
                            ▼
              ┌────────────────────────┐
              │ Local LLM Inference    │
              │                        │
              │ Phi-3 Mini via Ollama  │
              └─────────────┬──────────┘
                            │
                            ▼
                     Generated Clinical Answer
                            │
                            ▼
                     JSON API Response
                            │
                            ▼
                     Client/User

---------------------------------------------------------------------------------------------
## 🛠️ Technical Stack
---

LLM Inference:	Ollama (Phi-3 Mini)
Orchestration:	LangChain / LangChain-Community
API Framework:	FastAPI
Vector Store:	FAISS (Facebook AI Similarity Search)
Database:	    SQLite / SQLAlchemy
Embeddings: 	HuggingFace (sentence-transformers)
---------------------------------------------------------------------------------------------
## Retrieval Strategy: 
---

The system employs a Hybrid Retrieval Architecture  combining
- Structured (SQL): A SQLite database stores structured patient metadata directly from the provided JSON records, including the MRD number, patient name, gender, and doctor details. The system also tracks visit IDs, discharge dates, and specific document types to maintain a complete clinical profile. This structured storage enables precise patient lookups and ensures that all information is validated before it reaches the retrieval stage. Crucially, the SQL layer identifies all unique visits associated with an MRD, enabling the system to retrieve information from past-dated events across the patient's entire medical history.

- Semantic Retrieval (Vector RAG): Clinical descriptions and notes are embedded using sentence embeddings and stored in a FAISS vector database. The system retrieves doctor observations, progress notes, treatment summaries, and clinical descriptions based on their semantic similarity to the user's question.

Query Execution Flow:
1. Validation: Check for invalid MRDs or empty queries.
2. MRD Identification: Locate the specific patient record using the mrd_number.
3. Structured Lookup: Retrieve metadata and identify all historical visit IDs via SQL to ensure a longitudinal view.
4. Semantic Search: Perform vector similarity search for clinical context (e.g., discharge medications).
5. Context Synthesis: Merge SQL and Vector results into a single grounded prompt.
6. Local Inference: Generate a clinical answer using Phi-3 (Ollama) with a confidence score

This longitudinal approach improves both accuracy and completeness by allowing the LLM to analyze trends and compare data across multiple dates of service.
---------------------------------------------------------------------------------------------
## Chunking Strategy
---

Clinical notes in JSON records can be long and unstructured. To improve retrieval quality, notes are split into semantic chunks before embedding.

To support semantic retrieval, we will adopt the Chunking Strategy as follows:

- Chunk size: 500 characters
- Overlap: 50 characters

This enhances retrieval precision, semantic search accuracy, and the grounding of LLM responses. Each chunk is embedded and stored independently in the FAISS index.
---------------------------------------------------------------------------------------------
## Prompt Design and Guardrails:
---

Clinical AI prefers safety over creativity. The system prompt enforces strict grounding to prevent hallucinations and ensure answers come only from retrieved clinical data.

Guardrails Implemented:
- Model instructed to use ONLY retrieved context, preventing outside knowledge or invented treatments.
- If the answer is missing from context, the model returns “No retrieval match.”
- Temperature has been set to 0 to produce consistent, factual outputs.

We enforce a structured consistent response format:
Answer: <response>
Confidence: <High/Medium/Low>

Sample System Prompt:

You are a Medical AI Assistant. Use ONLY the provided context.
Rules:
1. Do not use outside knowledge.
2. If the answer is missing, say "No retrieval match."
3. No assumptions or speculation.
4. Output format:
Answer: <text>
Confidence: <score>

The above template will minimize hallucinations, ensure medical compliance and guarantee all answers are directly derived from and traceable to specific patient records.
-----------------------------------------------------------------------------------------------
## Project Setup & Usage:
---

To ensure the Med-RAG Clinical Assistant runs correctly in your local environment, follow these steps:

1. Environment & Dependencies
First, create a virtual environment to keep your dependencies isolated:

python -m venv venv
# On Windows:
.\venv\Scripts\activate

Install the core RAG and API stack:
pip install fastapi uvicorn sqlalchemy pydantic faiss-cpu \
            sentence-transformers langchain langchain-community \
            langchain-huggingface beautifulsoup4 ollama

2. Local LLM Setup (Ollama)
This project leverages Phi-3 for local inference.

Download and Install Ollama

Verify installation: ollama --version

Pull the model: ollama pull phi3

3. Data Initialization
Since medical data is kept private and not tracked in this repo, you must run the ingestion scripts to generate your local SQLite database and FAISS vector index:

# Populate the SQLite Database
python database.py

# Generate embeddings and create the FAISS index
python embeddings.py

4. Start the API Server
Launch the FastAPI backend:
python main.py

The server will be available at http://127.0.0.1:8000. You can access the interactive Swagger documentation at:

👉 http://127.0.0.1:8000/docs

API Interaction Example
Endpoint: POST /query

Request Body:
{
  "mrd_number": "17319",
  "query": "What did the doctor say about the patient condition?"
}

cURL Command:
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"mrd_number": "17319", "query": "What is the patient condition?"}'

Expected Response:
{
  "mrd_number": "17319",
  "answer": "The doctor noted that the patient was stable with no complaints.",
  "confidence": "High"
}