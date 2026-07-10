# Med-RAG Demo Console — single-container deployment
# Build:  docker build -t medrag-demo ./demo
# Run:    docker run -p 8501:8501 medrag-demo
# Live mode (optional): docker run -p 8501:8501 -e OLLAMA_URL=http://host.docker.internal:11434 medrag-demo
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py engine.py demo_corpus.json ./

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
