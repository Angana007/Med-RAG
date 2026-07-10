#!/bin/sh
# Med-RAG container entrypoint — bootstraps the demo data stores on first
# startup (both are derived from synthetic_patient_records.json and are
# not tracked in git), then starts the API.
set -e

if [ ! -f clinical_data.db ]; then
    echo "[entrypoint] clinical_data.db not found — building demo database..."
    python setup_db.py
fi

if [ ! -d faiss_index ]; then
    echo "[entrypoint] faiss_index/ not found — building FAISS vector index"
    echo "[entrypoint] (first run downloads the embedding model, ~90 MB)..."
    python embeddings.py
fi

exec python main.py
