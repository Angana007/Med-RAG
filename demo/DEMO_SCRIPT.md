# Med-RAG Demo — Video Script

A recording script for a portfolio/recruiter-facing walkthrough of Med-RAG (Local Hybrid-Retrieval Medical Chatbot). Two cuts included: a **full ~4-minute walkthrough** for anyone who wants the engineering detail, and a **90-second elevator cut** for recruiters skimming a portfolio.

---

## Before you hit record

**Have these open/ready in advance** (switching windows live wastes seconds and reads as unpolished):

1. Terminal #1 — server running (`python main.py`), font size bumped up (18–20pt), dark theme, window narrow enough to read on a phone screen.
2. Browser tab — Swagger UI at `http://127.0.0.1:8000/docs`, already expanded to the `/query` endpoint.
3. Terminal #2 — ready to run `python eval_runner.py`, with a completed `eval_reports/eval_report_<timestamp>.json` already generated (run it once before recording so you're not waiting on Phi-3 inference on camera).
4. This repo's `README.md` open in an editor/preview showing the architecture Mermaid diagram (or export it as an image beforehand — GitHub renders it natively, editor preview panes are fine too).
5. A code editor with `main.py`'s retry-then-block section and `evaluation.py`'s safety checks pre-scrolled to the right lines.
6. Screen recording tool (Loom, OBS, QuickTime) set to 1080p minimum, system audio off, mic levels checked.

**Data used throughout:** the shipped demo patient, MRD `20001`, five visits, diagnosis Type 2 Diabetes (`synthetic_patient_records.json`). All example answers below reflect what the retrieved context actually supports — say them naturally, don't read them like a teleprompter.

---

## Full Script (~4 minutes)

### 0:00–0:15 — Cold open (hook)

**On screen:** Terminal with the server already running; cursor blinking in Swagger UI.

**Say:**
> "This is Med-RAG — a clinical question-answering system that runs entirely on my laptop. No OpenAI, no Anthropic, no cloud calls. I built it to explore what it actually takes to make a RAG chatbot safe enough for healthcare data — not just accurate, but provably safe and measurably evaluated."

*(Pace: confident, not rushed. This line is your thesis — everything after it supports this claim.)*

### 0:15–0:50 — The problem, in one breath

**On screen:** README's Purpose/Differentiators section, or just talk to camera.

**Say:**
> "Most RAG demos stop at 'embed some documents, retrieve top-k chunks, ask an LLM.' That's fine for a blog search bot. It's not fine for patient records, where a wrong answer or a leaked identifier is a real harm, not a bad UX moment. So I built three things most RAG tutorials skip: hybrid retrieval that doesn't trust an LLM with facts it shouldn't have to remember, live safety checks that can actually block a response, and a benchmarking framework that scores the system instead of just vibes-checking it."

### 0:50–1:35 — Architecture, fast

**On screen:** The Mermaid architecture diagram from `README.md` (screen-record it rendered, or a static export).

**Say, pointing/cursor-tracing the diagram as you go:**
> "Here's the flow. A request comes in with a patient ID and a question. First, a cheap guardrail check — if you ask about billing or an X-ray, it's rejected before anything expensive happens. Then hybrid retrieval: structured facts — name, visit dates, doctors — come straight out of SQLite, because asking an LLM to 'remember' a discharge date via vector similarity is asking for trouble. Unstructured clinical notes go through FAISS vector search instead, which is what embeddings are actually good at. Both get merged into one context object, which goes to Phi-3, running locally through Ollama, temperature zero, for deterministic output. And *then* — this is the part most demos skip — every answer gets scanned for PHI leakage and cross-patient leakage before the client ever sees it."

### 1:35–2:20 — Live query, happy path

**On screen:** Swagger UI, `/query` endpoint expanded. Fill in the request body live.

**Say while typing:**
> "Let's ask it something real."

**Request body (type this on screen):**
```json
{
  "mrd_number": "20001",
  "query": "How has the patient's blood glucose control changed across visits?"
}
```

**Click Execute. While it's loading:**
> "This is running fully local inference — SQL lookup, vector search over five visit notes, then a Phi-3 generation call, all on this machine."

**When the response comes back, read the key fields, not the whole JSON blob:**
> "Answer, confidence score, which retrieval path was used — hybrid, in this case — and the latency breakdown. Notice `safety_flagged: false` and an empty `safety_details` — that's the live safety layer telling us nothing tripped."

*(If you don't have Ollama warm, or want a guaranteed clean take: pre-run this exact call once, screen-record the real response, and splice it in — just don't claim it's live if it isn't.)*

### 2:20–2:55 — Guardrails and the safety retry policy

**On screen:** Swagger UI again, or a code cutaway to `main.py`'s retry-then-block section.

**Say:**
> "Now watch what happens with an out-of-scope question."

**Request body:**
```json
{
  "mrd_number": "20001",
  "query": "What is the patient's outstanding billing balance?"
}
```

**Say, after showing the 400 response:**
> "Rejected before it ever reaches the LLM — billing questions are explicitly out of scope for a clinical assistant. And if a generated answer *does* trip a live safety check — say, it accidentally surfaces another patient's MRD number — the system doesn't just log a warning. It regenerates the answer once with a stricter prompt, re-checks it, and if it's still unsafe, blocks the response entirely and returns a generic safe message instead. That policy lives right here—"

**Cut to code:** `main.py`, the `# Retry-Then-Block Policy` section (around the safety_flagged check).

> "—regenerate once, re-check, block if it's still bad. Every violation is logged internally regardless of what the client sees."

### 2:55–3:35 — Evaluation framework

**On screen:** Terminal #2, pre-generated `eval_reports/eval_report_<timestamp>.json` open, or the `eval_runner.py` output in the terminal scrollback.

**Say:**
> "The part I think matters most: this isn't just tested by hand. There's a benchmark suite — thirteen test cases covering normal clinical questions, guardrail edge cases, and an invalid patient ID — that runs the exact same retrieval, generation, and safety code the live API uses, and scores every answer for faithfulness, relevance, hallucination, retrieval groundedness, and safety, using Phi-3 itself as a judge."

**Show the summary block:**
```json
{
  "total_cases": 13,
  "successful": 13,
  "avg_faithfulness": 0.xx,
  "safety_pass_rate": 0.xx,
  "hallucination_flag_rate": 0.xx
}
```

> "That's the difference between 'it worked when I tried it' and 'here's a number I can track across code changes.'"

*(Fill in your actual numbers from a real `eval_runner.py` run — don't recite placeholders on camera.)*

### 3:35–4:00 — Close

**On screen:** Back to you / README title.

**Say:**
> "So — hybrid retrieval, local-only inference, live safety enforcement with a real escalation policy, and an evaluation framework that scores the system instead of assuming it works. That's Med-RAG. Code's on GitHub, link below — happy to walk through any part of it in more depth."

**On screen (end card):** GitHub URL, your name/contact, maybe the architecture diagram one more time as a static frame.

---

## 90-Second Elevator Cut (for a recruiter skim)

Use only these beats, tightened:

1. **(0:00–0:15)** Cold open line, unchanged.
2. **(0:15–0:35)** One sentence each on hybrid retrieval, local inference, live safety, and eval — no diagram walkthrough, just the four differentiators as a fast list.
3. **(0:35–1:05)** One live query shown end-to-end (the happy-path example above), reading only the answer + confidence + safety_flagged fields.
4. **(1:05–1:25)** Ten seconds on the eval framework — show the summary JSON, say the one line about "scored, not vibes-checked."
5. **(1:25–1:30)** Close line + end card.

---

## Recording tips

- **Read nothing verbatim.** These are talking points, not a teleprompter script — say them in your own words so it sounds like you know the system (you do).
- **Pre-run anything that calls Ollama** before recording. Live LLM inference has unpredictable latency and occasionally messy output; a clean pre-run take splices in fine and nobody will know.
- **Zoom your terminal and browser** — recruiters often watch on a phone or a small laptop preview thumbnail. If text isn't readable at 50% size, it's too small.
- **Keep total runtime under 5 minutes** for a portfolio piece. Most viewers decide whether to keep watching in the first 15 seconds — that cold open line is doing real work, don't rush past it.
- **Caption or subtitle it** if you can (Loom and YouTube both auto-generate these) — a large fraction of recruiters skim videos muted.
