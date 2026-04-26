# =============================================================================
# frontend/ui_components.py — Reusable Streamlit widgets
# =============================================================================

import streamlit as st
from backend.config import DOMAINS


def apply_custom_css():
    st.markdown("""
    <style>
    .agent-card {
        background: #1e2130; border-left: 4px solid #4f8ef7;
        border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;
    }
    .agent-label {
        font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: #4f8ef7; margin-bottom: 4px;
    }
    .confidence-bar-wrap {
        background: #2d3a5a; border-radius: 8px; height: 10px;
        width: 100%; margin: 6px 0 12px;
    }
    .confidence-bar-fill {
        height: 10px; border-radius: 8px;
        background: linear-gradient(90deg, #4f8ef7, #22d3ee);
    }
    .badge-verified   { color: #22c55e; font-weight: 600; }
    .badge-uncertain  { color: #f59e0b; font-weight: 600; }
    .badge-unverified { color: #ef4444; font-weight: 600; }
    .progress-item { padding: 3px 0; color: #94a3b8; font-size: 0.85rem; }
    .progress-item:last-child { color: #e2e8f0; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, version: str):
    c1, c2 = st.columns([6, 1])
    with c1:
        st.markdown(f"# ⚖️ {title}")
        st.markdown(f"*{subtitle}*")
    with c2:
        st.markdown(f"<br><small>v{version}</small>", unsafe_allow_html=True)
    st.divider()


def render_sidebar(domains: dict, current_domain: str, long_mem) -> dict:
    st.markdown("## ⚙️ Configuration")

    domain_labels = {k: v["label"] for k, v in domains.items()}
    domain_keys   = list(domain_labels.keys())
    current_idx   = domain_keys.index(current_domain) if current_domain in domain_keys else 0

    selected_label = st.selectbox(
        "🌐 Domain",
        options=[domain_labels[k] for k in domain_keys],
        index=current_idx,
    )
    selected_domain = domain_keys[
        [domain_labels[k] for k in domain_keys].index(selected_label)
    ]

    st.markdown("---")
    use_rag        = st.toggle("📚 Enable RAG (Document Retrieval)", value=False)
    uploaded_files = None

    if use_rag:
        uploaded_files = st.file_uploader(
            "Upload PDF / TXT",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) ready")

    st.markdown("---")
    st.markdown("### 🧠 Memory")
    st.metric("Stored Debates", long_mem.count())
    st.markdown("---")
    st.markdown(
        "<small>Groq · FAISS · SentenceTransformers · Streamlit</small>",
        unsafe_allow_html=True,
    )

    return {
        "domain":         selected_domain,
        "use_rag":        use_rag,
        "uploaded_files": uploaded_files,
    }


def render_decision_card(verdict: dict):
    if not verdict:
        return

    decision    = verdict.get("final_decision",   "No decision reached.")
    confidence  = verdict.get("confidence_score", 0)
    risk        = verdict.get("risk_level",       "Unknown")
    winner      = verdict.get("winning_agent",    "N/A")
    reasons     = verdict.get("key_reasons",      [])
    dissent     = verdict.get("dissenting_view",  "")
    explanation = verdict.get("explanation",      "")

    conf_label = (
        "High"   if confidence >= 75 else
        "Medium" if confidence >= 50 else "Low"
    )

    st.markdown("### ⚖️ Judge's Final Decision")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence",    f"{confidence}%", conf_label)
    c2.metric("Risk Level",    risk)
    c3.metric("Winning Agent", winner.replace("_", " ").title() if winner else "N/A")
    c4.metric("Status",        "✅ Complete")

    st.markdown(f"**Decision:** {decision}")

    bar = max(2, confidence)
    st.markdown(
        f'<div class="confidence-bar-wrap">'
        f'<div class="confidence-bar-fill" style="width:{bar}%"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    cr, cd = st.columns(2)
    with cr:
        if reasons:
            st.markdown("**Key Reasons:**")
            for i, r in enumerate(reasons, 1):
                st.markdown(f"{i}. {r}")
    with cd:
        if dissent:
            st.markdown("**Strongest Dissent:**")
            st.markdown(f"*{dissent}*")

    if explanation:
        with st.expander("📖 Full Reasoning"):
            st.markdown(explanation)


def render_progress_status(messages: list):
    if not messages:
        return
    st.markdown("**🔄 Progress**")
    for msg in messages[-6:]:
        st.markdown(
            f'<div class="progress-item">→ {msg}</div>',
            unsafe_allow_html=True,
        )


def render_history_tab(long_mem):
    st.markdown("### 📜 Past Debates")
    records = long_mem.get_all(limit=20)

    if not records:
        st.info("No debates stored yet.")
        return

    search = st.text_input("🔍 Search past debates", placeholder="e.g. renewable energy...")

    if search:
        hits = long_mem.search(search, top_k=5)
        if hits:
            for h in hits:
                v = h.get("verdict", {})
                with st.expander(
                    f"[{h.get('similarity',0):.0%}] {h.get('topic','')[:70]}"
                ):
                    st.markdown(f"**Decision:** {v.get('final_decision','N/A')}")
                    st.markdown(f"**Confidence:** {v.get('confidence_score','?')}%")
                    st.caption(h.get("timestamp", "")[:10])
        else:
            st.info("No similar debates found.")
        return

    for rec in records:
        topic  = rec.get("topic", "")[:70]
        ts     = rec.get("timestamp", "")[:10]
        v      = rec.get("verdict", {})
        with st.expander(f"🗓 {ts} · {topic}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Domain",     rec.get("domain","").replace("_"," ").title())
            c2.metric("Confidence", f"{v.get('confidence_score','?')}%")
            c3.metric("Risk",       v.get("risk_level","?"))
            c4.metric("Tokens",     f"{rec.get('total_tokens',0):,}")
            st.markdown(f"**Decision:** {v.get('final_decision','N/A')}")
            st.caption(
                f"Duration: {rec.get('duration_sec',0)}s · "
                f"ID: {rec.get('debate_id','')[:8]}..."
            )