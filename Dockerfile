# Med-RAG API — FastAPI + hybrid retrieval
# Build:  docker build -t med-rag .
# Run (with a local Ollama):
#   docker run -p 8000:8000 -e OLLAMA_HOST=http://host.docker.internal:11434 med-rag
# Or bring up the full stack (API + Ollama) with: docker compose up
FROM python:3.10-slim

WORKDIR /app

# CPU-only torch keeps the image several GB smaller than the CUDA default
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Application code + the committed demo dataset. clinical_data.db and
# faiss_index/ are NOT tracked in git — entrypoint.sh builds both from
# synthetic_patient_records.json on first startup.
COPY main.py retrieval.py database.py embeddings.py llm.py evaluation.py setup_db.py ./
COPY synthetic_patient_records.json entrypoint.sh ./
RUN chmod +x entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# entrypoint.sh bootstraps demo data if missing, then starts uvicorn on 0.0.0.0:8000
ENTRYPOINT ["./entrypoint.sh"]
