# =============================================================================
# app.py — Main Streamlit entry point
# =============================================================================

import streamlit as st
from backend.config import (
    APP_TITLE, APP_SUBTITLE, APP_VERSION,
    DOMAINS, DEFAULT_DOMAIN, NUM_ROUNDS,
)
from backend.debate_engine   import DebateEngine
from rag.retriever           import RAGRetriever
from memory.long_term        import LongTermMemory
from memory.short_term       import ShortTermMemory
from frontend.debate_view    import render_debate_view
from frontend.analytics_view import render_analytics_view
from frontend.ui_components  import (
    render_header,
    render_sidebar,
    render_decision_card,
    render_history_tab,
    render_progress_status,
    apply_custom_css,
)

# -----------------------------------------------------------------------------
# Page config — MUST be first Streamlit call
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
def init_session():
    defaults = {
        "debate_result":   None,
        "rag_retriever":   None,
        "short_mem":       ShortTermMemory(),
        "long_mem":        LongTermMemory(),
        "domain":          DEFAULT_DOMAIN,
        "last_query":      "",
        "running":         False,
        "progress_msgs":   [],
        "token_breakdown": {},
        "files_indexed":   [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()
apply_custom_css()

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
render_header(APP_TITLE, APP_SUBTITLE, APP_VERSION)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    sidebar_state = render_sidebar(
        domains=DOMAINS,
        current_domain=st.session_state["domain"],
        long_mem=st.session_state["long_mem"],
    )
    if sidebar_state["domain"] != st.session_state["domain"]:
        st.session_state["domain"] = sidebar_state["domain"]

    uploaded_files = sidebar_state["uploaded_files"]
    use_rag        = sidebar_state["use_rag"]

# -----------------------------------------------------------------------------
# Query input
# -----------------------------------------------------------------------------
st.markdown("### 💬 Enter Your Decision Query")
col1, col2 = st.columns([5, 1])

with col1:
    query = st.text_area(
        label="query",
        label_visibility="collapsed",
        placeholder=(
            "e.g. Should our startup pivot from B2C to B2B?\n"
            "e.g. Should we invest in renewable energy this quarter?\n"
            "e.g. Should AI-generated content require mandatory disclosure?"
        ),
        height=100,
        value=st.session_state.get("last_query", ""),
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run_button = st.button(
        "⚖️ Run Debate",
        use_container_width=True,
        type="primary",
        disabled=st.session_state["running"],
    )

domain_label = DOMAINS.get(st.session_state["domain"], {}).get("label", "")
st.caption(
    f"Domain: **{domain_label}** · "
    f"Rounds: **{NUM_ROUNDS}** · "
    f"Agents: **8**"
)

# -----------------------------------------------------------------------------
# RAG helper
# -----------------------------------------------------------------------------
def build_rag_context(files, query_text: str) -> str:
    retriever = RAGRetriever()
    if files:
        with st.spinner("📚 Indexing documents..."):
            for f in files:
                n = retriever.index_uploaded_file(f)
                st.session_state["files_indexed"].append(f"{f.name} ({n} chunks)")
    st.session_state["rag_retriever"] = retriever
    if retriever.is_ready and query_text:
        return retriever.get_context(query_text, top_k=5)
    return ""

# -----------------------------------------------------------------------------
# Run debate
# -----------------------------------------------------------------------------
if run_button:
    if not query or not query.strip():
        st.warning("⚠️ Please enter a query first.")
    else:
        st.session_state.update({
            "running":       True,
            "debate_result": None,
            "progress_msgs": [],
            "last_query":    query,
            "files_indexed": [],
        })

        mem = st.session_state["short_mem"]
        mem.reset()
        mem.start_debate(query, st.session_state["domain"])

        rag_context = ""
        if use_rag:
            rag_context = build_rag_context(
                uploaded_files or [], query
            )

        progress_box = st.empty()

        def on_progress(stage: str, message: str):
            st.session_state["progress_msgs"].append(message)
            mem.log_progress(stage, message)
            with progress_box.container():
                render_progress_status(st.session_state["progress_msgs"])

        engine = DebateEngine(domain=st.session_state["domain"])

        with st.spinner("🧠 Agents debating..."):
            result = engine.run(
                query=query,
                rag_context=rag_context,
                on_progress=on_progress,
            )

        st.session_state["debate_result"]   = result
        st.session_state["token_breakdown"] = engine.get_agent_token_breakdown()
        st.session_state["running"]         = False

        mem.set_scores(result.scores)
        mem.set_verdict(result.verdict)
        mem.add_flags(result.all_flags)
        mem.complete()

        if result.success:
            st.session_state["long_mem"].save_from_result(result)
            st.success("✅ Debate complete!")
        else:
            st.error(f"❌ Error: {result.error}")

        progress_box.empty()
        st.rerun()

# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
result = st.session_state.get("debate_result")

if result:
    st.divider()
    render_decision_card(result.verdict)
    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "🗣️ Debate Transcript",
        "📊 Analytics",
        "📜 History",
    ])
    with tab1:
        render_debate_view(result)
    with tab2:
        render_analytics_view(
            result=result,
            token_breakdown=st.session_state["token_breakdown"],
        )
    with tab3:
        render_history_tab(st.session_state["long_mem"])

else:
    st.divider()
    st.markdown("### 🚀 Example Queries")
    examples = DOMAINS.get(st.session_state["domain"], {}).get("example_topics", [])
    for ex in examples:
        if st.button(f"💡 {ex}", key=ex):
            st.session_state["last_query"] = ex
            st.rerun()
    st.info(
        "Select a domain in the sidebar · enter your query · click **Run Debate**"
    )