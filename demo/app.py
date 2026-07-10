"""
Med-RAG Demo Console — Streamlit UI

A polished, deployable front-end for the Med-RAG clinical RAG system,
built for the Lucidus AI review. Run locally:

    pip install -r demo/requirements.txt
    streamlit run demo/app.py

DEMO mode (default): zero infrastructure, instant answers, retrieval fully live.
LIVE mode: export OLLAMA_URL=http://localhost:11434 for real Phi-3 generation.
"""

import statistics
import streamlit as st

import engine

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Med-RAG Console",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.6rem; max-width: 1200px; }
  header[data-testid="stHeader"] { background: transparent; }

  .mrx-hero {
    background: linear-gradient(120deg, #0f3057 0%, #1a5276 60%, #14746f 100%);
    border-radius: 14px; padding: 22px 28px; color: #fff; margin-bottom: 14px;
  }
  .mrx-hero h1 { margin: 0; font-size: 1.55rem; color: #fff; }
  .mrx-hero p { margin: 6px 0 0 0; opacity: .85; font-size: .95rem; }

  .mrx-badge {
    display: inline-block; padding: 3px 12px; border-radius: 999px;
    font-size: .74rem; font-weight: 600; letter-spacing: .04em;
    margin-right: 6px; margin-top: 10px;
  }
  .badge-green  { background: #d1f2df; color: #14532d; }
  .badge-blue   { background: #dbeafe; color: #1e3a8a; }
  .badge-amber  { background: #fef3c7; color: #92400e; }
  .badge-slate  { background: rgba(255,255,255,.18); color: #fff; }

  .metric-chip {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 8px 14px; text-align: center;
  }
  .metric-chip .v { font-size: 1.05rem; font-weight: 700; color: #0f3057; }
  .metric-chip .l { font-size: .68rem; color: #64748b; text-transform: uppercase; letter-spacing: .05em; }

  .chunk-card {
    border: 1px solid #e2e8f0; border-left: 4px solid #1a5276;
    border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; background: #fbfdff;
    font-size: .87rem;
  }
  .chunk-card.guideline { border-left-color: #14746f; background: #f6fffd; }
  .chunk-card .meta { color: #64748b; font-size: .74rem; margin-bottom: 4px; }
  .score-pill {
    float: right; background: #0f3057; color: #fff; border-radius: 999px;
    padding: 1px 10px; font-size: .72rem; font-weight: 600;
  }
  .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── State ──────────────────────────────────────────────────────────────────────
if "corpus" not in st.session_state:
    st.session_state.corpus = engine.load_corpus()
    st.session_state.retriever = engine.Retriever(st.session_state.corpus["chunks"])
if "messages" not in st.session_state:
    st.session_state.messages = []
if "latencies" not in st.session_state:
    st.session_state.latencies = []
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

corpus = st.session_state.corpus
retriever = st.session_state.retriever
live_mode = bool(engine.OLLAMA_URL)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 Med-RAG Console")
    st.caption("Clinical decision-support RAG · fully local · safety-checked")

    mrd = st.selectbox(
        "Patient record",
        options=list(corpus["patients"].keys()),
        format_func=lambda m: f"{corpus['patients'][m]['patient_name'].title()} · MRD {m}",
    )
    p = corpus["patients"][mrd]
    st.markdown(
        f"**Diagnosis:** {p['diagnosis']}  \n"
        f"**Physician:** {p['doctor_name']}  \n"
        f"**Visits on record:** {p['visit_count']}"
    )

    st.divider()
    st.markdown("**Engine status**")
    if live_mode:
        st.success(f"LIVE · Phi-3 via Ollama\n\n`{engine.OLLAMA_URL}`", icon="🟢")
    else:
        st.info("DEMO · retrieval is live, showcase answers are pre-verified against "
                "the retrieved sources. Set `OLLAMA_URL` for live Phi-3 generation.",
                icon="🔵")

    st.divider()
    st.markdown("**Session metrics**")
    lat = st.session_state.latencies
    if lat:
        s = sorted(lat)
        pct = lambda q: s[min(len(s) - 1, int(q * len(s)))]
        st.markdown(
            f"Queries: **{len(lat)}**  \n"
            f"p50 latency: **{pct(.5):,.0f} ms**  \n"
            f"p95 latency: **{pct(.95):,.0f} ms**  \n"
            f"Session cost: **${st.session_state.total_cost:.6f}**"
        )
    else:
        st.caption("Run a query to populate metrics.")

    st.divider()
    st.caption(
        "All patient data is **synthetic**. Answers are grounded exclusively in "
        "retrieved records — the model is forbidden from using outside knowledge, "
        "and says *\"No retrieval match\"* rather than guessing."
    )

# ── Hero ───────────────────────────────────────────────────────────────────────
mode_badge = ('<span class="mrx-badge badge-slate">🟢 LIVE · Phi-3 local</span>' if live_mode
              else '<span class="mrx-badge badge-slate">🔵 DEMO mode</span>')
st.markdown(f"""
<div class="mrx-hero">
  <h1>Med-RAG · Clinical Reasoning Console</h1>
  <p>Ask questions about a patient's record. Every answer shows its sources, its confidence,
  its latency, and its cost — because a system you can't inspect is a system you can't trust.</p>
  {mode_badge}
  <span class="mrx-badge badge-slate">🔒 PHI-scanned outputs</span>
  <span class="mrx-badge badge-slate">📚 100% retrieval-grounded</span>
  <span class="mrx-badge badge-slate">💻 Runs fully offline</span>
</div>
""", unsafe_allow_html=True)

# ── Showcase query buttons ─────────────────────────────────────────────────────
st.markdown("##### Try a scenario that needs *judgment*, not just lookup:")
sq_for_patient = [q for q in corpus["showcase_queries"]]
cols = st.columns(len(sq_for_patient))
pending_query = None
pending_mrd = None
for col, sq in zip(cols, sq_for_patient):
    icon = {"SQ1": "⚠️", "SQ2": "📈", "SQ3": "🔍", "SQ4": "🔒"}.get(sq["id"], "💬")
    with col:
        if st.button(f"{icon} {sq['label']}", key=sq["id"], use_container_width=True,
                     help=sq["query"]):
            pending_query, pending_mrd = sq["query"], sq["mrd"]

# ── Render history ─────────────────────────────────────────────────────────────
def render_result(res):
    st.markdown(res["answer"])

    if res.get("guardrail"):
        st.markdown('<span class="mrx-badge badge-amber">🛡 Guardrail: blocked before the '
                    'model ever ran — zero tokens spent</span>', unsafe_allow_html=True)

    # Metrics strip
    c1, c2, c3, c4, c5 = st.columns(5)
    lt = res["latency"]
    conf_class = {"High": "badge-green", "Medium": "badge-amber"}.get(res["confidence"], "badge-blue")
    with c1:
        st.markdown(f'<div class="metric-chip"><div class="v">{lt["total_ms"]:,.0f} ms</div>'
                    f'<div class="l">latency</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-chip"><div class="v">{res["tokens"]["total"]:,}</div>'
                    f'<div class="l">tokens (est.)</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-chip"><div class="v">${res["cost_usd"]:.7f}</div>'
                    f'<div class="l">cost / query</div></div>', unsafe_allow_html=True)
    with c4:
        if res["safety"].get("verified_citations"):
            g_txt, g_lbl = "✓ cited", "grounding"
        else:
            g = res["safety"]["grounding"]
            g_txt, g_lbl = (f"{g:.0%}" if g is not None else "—"), "grounding"
        st.markdown(f'<div class="metric-chip"><div class="v">{g_txt}</div>'
                    f'<div class="l">{g_lbl}</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-chip"><div class="v">{res["confidence"]}</div>'
                    f'<div class="l">confidence</div></div>', unsafe_allow_html=True)

    # Safety line
    phi_ok = res["safety"]["phi_scan_pass"]
    st.markdown(
        f'<span class="mrx-badge {"badge-green" if phi_ok else "badge-amber"}">'
        f'{"✓ PHI scan passed" if phi_ok else "⚠ PHI scan flagged"}</span>'
        f'<span class="mrx-badge badge-blue">retrieval {res["latency"]["retrieval_ms"]:.0f} ms · '
        f'generation {res["latency"]["generation_ms"]:.0f} ms</span>'
        + (f'<span class="mrx-badge badge-blue">mode: {res["mode"]}</span>'),
        unsafe_allow_html=True,
    )

    # Sources
    if res["chunks"]:
        with st.expander(f"📚 Why this answer? — {len(res['chunks'])} retrieved sources with ranking scores"):
            st.caption("Bracketed numbers in the answer cite these sources. Guideline documents "
                       "(teal) are shared knowledge; record chunks (blue) are scoped to this patient only.")
            for i, c in enumerate(res["chunks"]):
                cls = "guideline" if c["source_type"] == "clinical_guideline" else ""
                icon = "📖" if cls else "🗂"
                st.markdown(
                    f'<div class="chunk-card {cls}">'
                    f'<span class="score-pill">score {c["score"]:.2f} · rel {c["relevance"]:.0%}</span>'
                    f'<div class="meta">[{i+1}] {icon} <b>{c["title"]}</b> · {c["date"]} · {c["author"]}</div>'
                    f'{c["text"]}</div>',
                    unsafe_allow_html=True,
                )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍⚕️" if msg["role"] == "user" else "🩺"):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_result(msg["result"])

# ── Input ──────────────────────────────────────────────────────────────────────
typed = st.chat_input(f"Ask about {p['patient_name'].title()}'s record…")
if typed:
    pending_query, pending_mrd = typed, mrd

if pending_query:
    q_mrd = pending_mrd or mrd
    qp = corpus["patients"][q_mrd]
    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(f"**[{qp['patient_name'].title()} · MRD {q_mrd}]** {pending_query}")
    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Retrieving records → generating grounded answer → running safety checks…"):
            res = engine.ask(pending_query, q_mrd, corpus, retriever)
        render_result(res)

    st.session_state.messages.append(
        {"role": "user", "content": f"**[{qp['patient_name'].title()} · MRD {q_mrd}]** {pending_query}"})
    st.session_state.messages.append({"role": "assistant", "result": res})
    st.session_state.latencies.append(res["latency"]["total_ms"])
    st.session_state.total_cost += res["cost_usd"]
    st.rerun()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
f1, f2, f3 = st.columns(3)
with f1:
    st.markdown("**🔒 Safety by architecture**")
    st.caption("Scope guardrails run *before* retrieval. PHI scans run on every output. "
               "In production, a flagged answer is regenerated once with a stricter prompt "
               "— and blocked entirely if it's still unsafe.")
with f2:
    st.markdown("**💰 Cost by design**")
    st.caption(f"Local Phi-3 inference costs ≈ ${engine.COST_PER_QUERY_LOCAL_USD * 1000:.4f} "
               f"per 1,000 queries (electricity) vs ≈ ${engine.COST_PER_QUERY_CLOUD_USD * 1000:.0f} "
               f"per 1,000 on a hosted GPT-4o-class API. At classroom scale, that difference "
               f"decides whether a feature ships.")
with f3:
    st.markdown("**📊 Evaluation, not vibes**")
    st.caption("The production system ships with a RAGAS-style benchmark suite (faithfulness, "
               "relevance, groundedness, safety pass-rate) run in CI on every change — "
               "24 automated tests via GitHub Actions.")
