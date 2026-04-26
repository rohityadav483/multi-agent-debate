# =============================================================================
# frontend/debate_view.py — Debate Transcript Tab
# =============================================================================

import streamlit as st
from backend.config import NUM_ROUNDS, AGENT_SPEAKING_ORDER

AGENT_COLORS = {
    "optimist":      "#22c55e",
    "skeptic":       "#ef4444",
    "analyst":       "#3b82f6",
    "domain_expert": "#a855f7",
    "critic":        "#f59e0b",
    "fact_checker":  "#06b6d4",
    "judge":         "#e2e8f0",
    "planner":       "#94a3b8",
}

AGENT_ICONS = {
    "optimist":      "🌟",
    "skeptic":       "🔍",
    "analyst":       "📊",
    "domain_expert": "🎓",
    "critic":        "⚡",
    "fact_checker":  "✅",
    "judge":         "⚖️",
    "planner":       "🗂️",
}


def render_debate_view(result):
    """Render the full debate transcript with all rounds."""

    st.markdown("### 🗂️ Debate Agenda")
    st.info(result.agenda if result.agenda else "No agenda generated.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Per-round tabs
    # -------------------------------------------------------------------------
    round_tabs = st.tabs([f"Round {i}" for i in range(1, NUM_ROUNDS + 1)])

    for round_idx, round_tab in enumerate(round_tabs):
        round_num = round_idx + 1

        # Find this round's data
        round_data = next(
            (r for r in result.rounds if r.get("round") == round_num), None
        )

        with round_tab:
            if not round_data:
                st.warning(f"No data for Round {round_num}.")
                continue

            arguments    = round_data.get("arguments",    {})
            critic_data  = round_data.get("critic",       {})
            fc_data      = round_data.get("fact_checker", {})

            # Debate agents
            st.markdown(f"#### 🥊 Round {round_num} — Arguments")
            for agent_name in AGENT_SPEAKING_ORDER:
                text = arguments.get(agent_name, "")
                _render_agent_card(agent_name, text)

            st.markdown("---")

            # Critic + Fact-checker side by side
            col_c, col_f = st.columns(2)

            with col_c:
                st.markdown("#### ⚡ Critic Scores")
                critic_content = critic_data.get("content", "")
                if critic_content:
                    st.code(critic_content, language=None)
                else:
                    _render_critic_scores(critic_data.get("metadata", {}))

            with col_f:
                st.markdown("#### 🔍 Fact-Check Report")
                fc_content = fc_data.get("content", "")
                if fc_content:
                    _render_fact_check_badges(fc_data.get("metadata", {}))
                else:
                    st.info("No fact-check data.")

    # -------------------------------------------------------------------------
    # All flags summary at bottom
    # -------------------------------------------------------------------------
    if result.all_flags:
        st.markdown("---")
        st.markdown("### 🚩 All Fact-Check Flags")
        _render_flags_table(result.all_flags)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _render_agent_card(agent_name: str, text: str):
    color = AGENT_COLORS.get(agent_name, "#94a3b8")
    icon  = AGENT_ICONS.get(agent_name, "🤖")
    label = agent_name.replace("_", " ").title()

    if not text or text.startswith("["):
        text = "*No response recorded.*"

    with st.expander(f"{icon} {label}", expanded=True):
        st.markdown(
            f'<div style="border-left: 3px solid {color}; padding-left: 12px;">'
            f'{text}'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_critic_scores(metadata: dict):
    scores = metadata.get("scores", {})
    if not scores:
        st.info("No scores parsed.")
        return

    for agent, data in scores.items():
        if not isinstance(data, dict):
            continue
        total = (
            data.get("logical_consistency", 0) +
            data.get("evidence_usage",      0) +
            data.get("relevance",           0) +
            data.get("persuasiveness",      0)
        )
        icon  = AGENT_ICONS.get(agent, "🤖")
        label = agent.replace("_", " ").title()
        st.markdown(f"**{icon} {label}** — `{total}/100`")

        cols = st.columns(4)
        cols[0].metric("Logic",    data.get("logical_consistency", 0))
        cols[1].metric("Evidence", data.get("evidence_usage",      0))
        cols[2].metric("Relev.",   data.get("relevance",           0))
        cols[3].metric("Persuade", data.get("persuasiveness",      0))

        just = data.get("justification", "")
        if just:
            st.caption(f"*{just}*")
        st.markdown("---")


def _render_fact_check_badges(metadata: dict):
    fact_checks = metadata.get("fact_checks", {})
    if not fact_checks:
        st.info("No fact-check data.")
        return

    badge_map = {
        "verified":   ("✅", "badge-verified"),
        "uncertain":  ("⚠️",  "badge-uncertain"),
        "unverified": ("❌", "badge-unverified"),
    }

    for agent, claims in fact_checks.items():
        if not isinstance(claims, list):
            continue
        icon  = AGENT_ICONS.get(agent, "🤖")
        label = agent.replace("_", " ").title()
        st.markdown(f"**{icon} {label}**")
        for item in claims:
            status = item.get("status", "uncertain").lower()
            emoji, css = badge_map.get(status, ("❓", ""))
            claim = item.get("claim", "")[:100]
            note  = item.get("note",  "")
            st.markdown(
                f'<span class="{css}">{emoji} {status.upper()}</span> — {claim}',
                unsafe_allow_html=True,
            )
            if note:
                st.caption(f"  └─ {note}")
        st.markdown("")


def _render_flags_table(flags: list):
    if not flags:
        return

    verified   = [f for f in flags if f.get("status") == "verified"]
    uncertain  = [f for f in flags if f.get("status") == "uncertain"]
    unverified = [f for f in flags if f.get("status") == "unverified"]

    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Verified",   len(verified))
    c2.metric("⚠️ Uncertain",  len(uncertain))
    c3.metric("❌ Unverified", len(unverified))

    if unverified:
        st.markdown("**❌ Potential Hallucinations:**")
        for f in unverified:
            st.markdown(
                f"- **{f.get('agent','?').title()}** (Round {f.get('round','?')}): "
                f"{f.get('claim','')[:100]}"
            )
            if f.get("note"):
                st.caption(f"  {f['note']}")