# 🩺 Med-RAG: Local Hybrid-Retrieval Medical Chatbot
---

A specialized Medical NLP Chatbot designed for high-accuracy clinical query answering. This system implements a Hybrid Retrieval-Augmented Generation (RAG) architecture, combining the precision of SQL with the semantic depth of Vector Search.

## 🌟 Key Features


- **100% Local & Private**  
  No external API calls (OpenAI/Anthropic). All data stays on-premise, ensuring HIPAA-compliant design principles.

- **Hybrid Retrieval**  
  Merges structured SQLite lookups (for patient metadata) with semantic FAISS vector search (for unstructured clinical notes).

- **Longitudinal View**  
  Smart SQL logic identifies all historical visit IDs associated with an MRD to analyze patient trends over time.

- **Deterministic Guardrails**  
  Hard-coded system prompts and `temperature = 0` to eliminate hallucinations.


## 🏗️ System Architecture:
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


## 🏗️ System Architecture

```mermaid
flowchart TD

A[Client / User (cURL)] -->|HTTP Request| B[FastAPI Server<br/>Chatbot Backend]

B --> C[Query Processing Layer]

C --> D[Structured Retrieval<br/>(SQL Database)]
C --> E[Semantic Retrieval<br/>(Vector Search / RAG)]

D --> D1[Patient Metadata<br/>MRD Number<br/>Visit Info<br/>Doctor Details]
E --> E1[Embedded Clinical Notes<br/>Progress Notes<br/>Consultation Reports]

D1 --> F[SQLite DB]
E1 --> G[FAISS Vector DB]

F --> H[Retrieved Context]
G --> H

H --> I[Prompt Construction<br/>System Prompt + SQL + Vector + Query]

I --> J[Local LLM Inference<br/>Phi-3 via Ollama]

J --> K[Generated Clinical Answer]

K --> L[JSON API Response]

L --> A
```

## 🛠️ Technical Stack
---

- **LLM Inference:** Ollama (Phi-3 Mini)  
- **Orchestration:** LangChain / LangChain-Community  
- **API Framework:** FastAPI  
- **Vector Store:** FAISS (Facebook AI Similarity Search)  
- **Database:** SQLite / SQLAlchemy  
- **Embeddings:** HuggingFace (sentence-transformers)


## 🔍 Retrieval Strategy: 
---

The system employs a **Hybrid Retrieval Architecture** combining:

### 🧾 Structured (SQL)
- Stores patient metadata from JSON records:
  - MRD number, patient name, gender
  - Doctor details
  - Visit IDs, discharge dates, document types
- Enables precise lookups and validation
- Retrieves **all historical visits for longitudinal analysis**

### 🧠 Semantic Retrieval (Vector RAG)
- Clinical notes embedded using sentence transformers
- Stored in FAISS vector database
- Retrieves:
  - Doctor observations
  - Progress notes
  - Treatment summaries
  - Clinical descriptions

### ⚙️ Query Execution Flow
1. **Validation** → Check for invalid MRDs or empty queries  
2. **MRD Identification** → Locate patient record  
3. **Structured Lookup** → Fetch metadata + visit history  
4. **Semantic Search** → Retrieve relevant clinical context  
5. **Context Synthesis** → Merge SQL + vector results  
6. **Local Inference** → Generate answer + confidence score  

✅ This **longitudinal approach** improves accuracy by analyzing trends across multiple visits.

## ✂️ Chunking Strategy
---

Clinical notes are long and unstructured, so they are split before embedding:

- **Chunk Size:** 500 characters  
- **Overlap:** 50 characters  

🎯 Benefits:
- Better semantic retrieval  
- Higher accuracy  
- Stronger grounding of LLM responses  

Each chunk is embedded and stored independently in FAISS.


## 🛡️ Prompt Design and Guardrails:
---

Clinical AI prioritizes **safety over creativity**.

### 🚧 Guardrails Implemented:
- Use **ONLY retrieved context**
- No external knowledge or hallucinations
- Return `"No retrieval match"` if answer is missing
- `temperature = 0` for deterministic outputs

### 📌 Response Format:

Answer: <response>
Confidence: <High/Medium/Low>

### 🧠 Sample System Prompt:

You are a Medical AI Assistant. Use ONLY the provided context.

Rules:

Do not use outside knowledge.

If the answer is missing, say "No retrieval match."

No assumptions or speculation.

Output format:
Answer: <text>
Confidence: <score>


✅ Ensures:
- Hallucination-free outputs  
- Medical compliance  
- Traceable answers  

---

## 🚀 Project Setup & Usage
---

To ensure the Med-RAG Clinical Assistant runs correctly in your local environment, follow these steps:

### 1️⃣ Environment & Dependencies

First, create a virtual environment to keep your dependencies isolated:

```bash
python -m venv venv
```

On Windows:

```bash
.\venv\Scripts\activate
```

Install the core RAG and API stack:

```bash
pip install fastapi uvicorn sqlalchemy pydantic faiss-cpu \
            sentence-transformers langchain langchain-community \
            langchain-huggingface beautifulsoup4 ollama
```

### 2️⃣ Local LLM Setup (Ollama)

This project leverages Phi-3 for local inference.

Verify installation: ```bash ollama --version ```

Pull the model: ```bash ollama pull phi3 ```

### 3️⃣ Data Initialization

Since medical data is kept private and not tracked in this repo, you must run the ingestion scripts to generate your local SQLite database and FAISS vector index:

Populate the SQLite Database

```bash python database.py ```

Generate embeddings and create the FAISS index

``` bash python embeddings.py ```

### 4️⃣ Start the API Server

Launch the FastAPI backend:

```bash python main.py ```

The server will be available at http://127.0.0.1:8000. You can access the interactive Swagger documentation at:

👉 http://127.0.0.1:8000/docs

## 📡 API Usage Example 

Endpoint: 

```bash POST /query ```

Request Body:

```bash
{
  "mrd_number": "17319",
  "query": "What did the doctor say about the patient condition?"
}
```

cURL Command:

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"mrd_number": "17319", "query": "What is the patient condition?"}'
```

Expected Response:

```bash
{
  "mrd_number": "17319",
  "answer": "The doctor noted that the patient was stable with no complaints.",
  "confidence": "High"
}
```
## ✅ Summary

Med-RAG delivers a **fully local, privacy-first clinical AI system** that combines:

- **Structured SQL precision**
- **Semantic vector intelligence**
- **Deterministic LLM outputs**

➡️ **Result:** Accurate, explainable, and safe medical query answering system
