# Med-RAG Demo Video — Script for Lucidus AI

**Target length:** 4:00–4:30 · **Audience:** Rhea (CPO) and Aakaanksh (CEO) — product-minded, not ML engineers. Every technical claim is translated into a product consequence.

---

## Before you record

1. Deploy the console (see `demo/README.md`, Option A) so the URL bar shows a **public link** — it silently proves "this is shipped, not a localhost screencast." Open it fresh (empty chat).
2. Set browser zoom to 110–125% so the metrics chips are legible in 1080p.
3. Recording: OBS/Loom/QuickTime, 1080p, mic checked, notifications off (macOS: Focus mode / Windows: Do Not Disturb).
4. Practice the four clicks once — the whole demo is buttons, no typing needed except one free query.
5. Optional: keep the repo's README architecture diagram open in a second tab for the 30-second architecture beat.

**Delivery note:** you're not presenting a project, you're giving them a tour of a product. Calm, first-person, plain English. The word "basically" is banned.

---

## 0:00–0:35 — What this is (plain English, no jargon)

**On screen:** the console, freshly loaded. Move the cursor slowly across the hero banner badges.

> "Hi Rhea, hi Aakaanksh — this is Med-RAG, a clinical question-answering system I built and shipped end-to-end. A clinician picks a patient, asks a question in plain English, and gets an answer that is grounded *only* in that patient's records — with the sources, the confidence level, the response time, and the cost of every single answer shown right on screen.
>
> I chose healthcare deliberately: it's the hardest place to put an AI assistant, because a confident wrong answer isn't a bad user experience — it's a harm. That's the same bar an AI tutor should be held to, and that's why I want to walk you through it."

**Highlight:** the public URL, then the three hero badges (PHI-scanned · retrieval-grounded · runs fully offline).

## 0:35–1:20 — Why it's not a search engine

**On screen:** stay on the landing view; hover over the four scenario buttons.

> "A search engine finds documents. This system makes *judgments* about them — and, just as importantly, knows when *not* to. These four scenarios are the tour: one needs drug-safety reasoning across a patient's labs and hospital protocols, one needs trend analysis across six months of visits, one needs the system to notice that two notes could contradict each other, and one is a trap — a question the system should refuse.
>
> Under the hood there are two kinds of memory, because facts and stories need different tools: exact facts — names, dates, lab values — come from a database that can't hallucinate, and the narrative clinical notes come from semantic search. The AI is forbidden from using anything outside what was retrieved. If the records don't contain the answer, it says 'no retrieval match' instead of improvising."

## 1:20–3:10 — Live demo: three queries that show judgment

### Query 1 (1:20–2:05) — ⚠️ Drug-safety reasoning

**Click:** the **"⚠️ Drug-safety reasoning"** button.

> "Marcus Okafor: diabetic, kidney function declining, still on a full dose of metformin. I'm asking what the care team should watch for. Notice this isn't a lookup — the answer doesn't exist in any single document."

**When the answer renders, point out in order:**
1. The answer connects *three* sources: his latest lab result, the hospital's dosing protocol, and his reported painkiller use — "it caught that his eGFR is approaching the protocol's cutoff *before* he crossed it, and flagged that his over-the-counter ibuprofen plus his blood-pressure medication is a known high-risk combination."
2. Open **"Why this answer?"** — scroll the source cards. "Every bracketed number in the answer is one of these sources, with its retrieval ranking score. Total transparency: a clinician — or a teacher, or a student — can check the AI's homework."
3. The metrics chips: "Sub-second, a few hundred tokens, and the cost per query is effectively zero — I'll come back to why that matters."

### Query 2 (2:05–2:40) — 📈 Longitudinal trend analysis

**Click:** the **"📈 Longitudinal trend analysis"** button.

> "Second patient, different skill: is her diabetes control improving? The system reads all five visits and builds the trend — and the interesting part is the middle: her numbers *rebounded* at visit four because she'd stopped her medication over side effects. The system doesn't just report the dip; it explains it, and cites the care standard that predicts exactly this pattern. That's the difference between retrieval and reasoning — and it's precisely the skill a learning-analytics system needs when it looks at a student's progress over a semester."

### Query 3 (2:40–3:10) — 🔒 The refusal

**Click:** the **"🔒 Safety refusal (PHI)"** button.

> "Last one: I'm asking for the patient's phone number and home address. Watch the latency — *zero milliseconds of AI time*. The guardrail rejects it before the model ever runs. No tokens spent, no chance of a leak. In production there's a second net behind this one: every generated answer is scanned for identifiers, regenerated once under a stricter prompt if flagged, and blocked outright if it's still unsafe. The safest answer is sometimes no answer — designing for that is a product decision, not just an engineering one."

**Optional (if under time):** type a free query — *"Can you interpret the patient's MRI scan?"* — "It also knows the edge of its own competence: it can't read images, so it says so."

## 3:10–3:45 — Architecture in 30 seconds (can it scale? is it safe?)

**On screen:** either the README architecture diagram tab, or stay on the console and gesture through the footer's three columns.

> "Quick look under the hood, because 'polished demo' and 'production system' are different claims. Behind this UI is a FastAPI service with API-key auth, per-user rate limiting, structured logs where every request carries a trace ID, health and metrics endpoints, and a CI pipeline running twenty-four automated tests on every change. Quality isn't vibes: a benchmark suite scores every release for faithfulness, relevance, and safety pass-rate.
>
> And the economics: the model runs locally, so the marginal cost is about a tenth of a cent per *thousand* queries — versus roughly ten dollars per thousand on a hosted frontier model. At the scale of classrooms, that ratio decides which features are even possible."

## 3:45–4:15 — The bridge to Lucidus

**On screen:** back to the console landing view.

> "Everything you just saw maps one-to-one onto education. Swap patient records for a student's learning history, clinical protocols for curriculum standards, and 'what should the care team watch for' becomes 'what should this student review before Thursday's exam.' The architecture doesn't change — the corpus does.
>
> The demo's live at the link below — click the four buttons, try to break it, ask it something unfair. I built it to be tested, not just watched. Thanks for the time."

**End card (2s):** demo URL + your name + email.

---

## What to highlight, per beat (cheat sheet)

| Time | Click | Metric/element to point at | The point it proves |
|---|---|---|---|
| 0:20 | — | public URL + hero badges | shipped, not a notebook |
| 1:45 | Query 1 → "Why this answer?" | source cards + ranking scores | transparency, multi-source reasoning |
| 2:00 | — | latency + cost chips | production thinking |
| 2:30 | Query 2 | the visit-4 rebound row in the table | reasoning over time, not lookup |
| 2:50 | Query 3 | "blocked before the model ever ran" badge, 0ms | safety by architecture |
| 3:30 | footer | cost comparison + "24 tests in CI" | scale + rigor |

## Recording workflow

1. One full rehearsal without recording (checks pacing; target ≤ 4:30).
2. Record in a single take if possible — small stumbles read as authentic; edits read as produced.
3. If you must edit, cut only at the section boundaries above.
4. Export 1080p MP4. To trim the head/tail with ffmpeg:
   ```bash
   ffmpeg -i raw.mov -ss 00:00:03 -to 00:04:25 -c:v libx264 -crf 20 -c:a aac demo_medrag_lucidus.mp4
   ```
5. Upload to Loom/YouTube-unlisted and send the link *plus* the live demo URL in the same message — the video is the pitch, the URL is the proof.
