# Med-RAG Demo Console

A polished, deployable front-end for [Med-RAG](../Readme.md) — built so anyone (including non-engineers) can try the system in under five minutes.

**What you'll see:** a clinical chat console where every answer shows its retrieved sources with ranking scores, its confidence, its latency breakdown, and its cost per query. Four one-click scenarios demonstrate *reasoning* — drug-safety analysis, longitudinal trend interpretation, contradiction checking, and a safety refusal.

**Data note:** every patient in this demo is synthetic. No real health information anywhere.

---

## Option A — Run it in your browser, zero install (recommended for reviewers)

The demo is a three-file Streamlit app, so it deploys to **Streamlit Community Cloud** (free, permanent URL) in ~3 minutes:

1. Fork or push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pick the repo, set **Main file path** to `demo/app.py`, click **Deploy**.
4. Share the resulting `https://<app-name>.streamlit.app` URL.

No environment variables needed — the app starts in DEMO mode automatically.

> **Hugging Face Spaces alternative:** create a new *Streamlit* Space, upload the four files in this folder (`app.py`, `engine.py`, `demo_corpus.json`, `requirements.txt`), done. Same result, also free.

## Option B — Run locally (2 commands)

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

Opens at http://localhost:8501.

## Option C — Docker

```bash
docker build -t medrag-demo ./demo
docker run -p 8501:8501 medrag-demo
```

### One-click cloud deploy (Railway / Render)

Both platforms auto-detect the Dockerfile:

- **Railway:** *New Project → Deploy from GitHub repo*, set the root directory to `demo/`. Railway assigns a public URL under *Settings → Networking → Generate Domain*.
- **Render:** *New → Web Service → connect repo*, set **Root Directory** `demo`, environment **Docker**. Free tier note: the service sleeps when idle and takes ~40s to wake on first visit.

---

## DEMO mode vs LIVE mode (honest by design)

| | DEMO mode (default) | LIVE mode |
|---|---|---|
| Retrieval + ranking | ✅ fully live (BM25 hybrid mirror of the FAISS pipeline) | ✅ fully live |
| Guardrails + PHI scan | ✅ fully live | ✅ fully live |
| Answer generation | Showcase queries: pre-verified answers, every claim cited to a retrieved source. Free-typed queries: extractive, grounded answers. | Real Phi-3 via Ollama, temperature 0, same system prompt as production |
| Infrastructure needed | none | an Ollama server |

The mode is displayed in the UI on every single answer — nothing is silently mocked.

**To enable LIVE mode:** copy `.env.example` to `.env` (or set the variable directly) with:

```bash
export OLLAMA_URL=http://localhost:11434   # requires: ollama pull phi3
streamlit run demo/app.py
```

To expose your local Phi-3 to a cloud-hosted demo, use the existing Cloudflare Tunnel setup and set `OLLAMA_URL` to the tunnel URL. Put the tunnel behind Cloudflare Access if it will be public for more than a review window.

## How this relates to the production system

This console is the *presentation layer*. The production system in the repo root is the real thing: FastAPI with API-key auth and rate limiting, SQLite + FAISS hybrid retrieval, live PHI/cross-patient safety checks with a retry-then-block policy, a RAGAS-style eval suite, structured logging with UUID tracing, and CI with 24 automated tests. The demo engine mirrors that pipeline stage-for-stage in one dependency-light module so reviewers don't need a GPU, a model download, or a database to experience it.
