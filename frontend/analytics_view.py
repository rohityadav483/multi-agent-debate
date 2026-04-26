# =============================================================================
# frontend/analytics_view.py — Analytics Tab (scores + charts + tokens)
# =============================================================================

import streamlit as st


def render_analytics_view(result, token_breakdown: dict):
    """Render the analytics tab: leaderboard, score charts, token usage."""

    scores = result.scores or {}

    # -------------------------------------------------------------------------
    # 1. Leaderboard
    # -------------------------------------------------------------------------
    st.markdown("### 🏆 Agent Leaderboard")
    leaderboard = scores.get("leaderboard", [])

    if leaderboard:
        for entry in leaderboard:
            badge = entry.get("badge", "")
            agent = entry.get("agent", "").replace("_", " ").title()
            score = entry.get("score", 0)
            pct   = score   # score is already weighted 0-100

            cols = st.columns([1, 4, 2])
            cols[0].markdown(f"### {badge}")
            cols[1].progress(
                min(int(pct), 100),
                text=f"**{agent}** — {score:.1f} pts",
            )
            cols[2].markdown(f"### `{score:.1f}`")
    else:
        st.info("No leaderboard data.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 2. Per-metric breakdown (bar chart via st.bar_chart)
    # -------------------------------------------------------------------------
    st.markdown("### 📊 Score Breakdown by Metric")
    per_metric = scores.get("per_metric", {})

    if per_metric:
        try:
            import pandas as pd

            metrics = ["logical_consistency", "evidence_usage", "relevance", "persuasiveness"]
            metric_labels = {
                "logical_consistency": "Logic",
                "evidence_usage":      "Evidence",
                "relevance":           "Relevance",
                "persuasiveness":      "Persuasion",
            }

            chart_data = {}
            for agent, agent_metrics in per_metric.items():
                label = agent.replace("_", " ").title()
                chart_data[label] = [
                    agent_metrics.get(m, 0) for m in metrics
                ]

            df = pd.DataFrame(
                chart_data,
                index=[metric_labels[m] for m in metrics],
            )
            st.bar_chart(df)

        except ImportError:
            # Fallback without pandas
            for agent, agent_metrics in per_metric.items():
                label = agent.replace("_", " ").title()
                st.markdown(f"**{label}**")
                cols = st.columns(4)
                cols[0].metric("Logic",    agent_metrics.get("logical_consistency", 0))
                cols[1].metric("Evidence", agent_metrics.get("evidence_usage",      0))
                cols[2].metric("Relev.",   agent_metrics.get("relevance",           0))
                cols[3].metric("Persuade", agent_metrics.get("persuasiveness",      0))
    else:
        st.info("No metric breakdown data.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 3. Round-by-round scores
    # -------------------------------------------------------------------------
    st.markdown("### 📈 Round-by-Round Scores")
    raw_rounds = scores.get("raw_round_scores", {})

    if raw_rounds:
        try:
            import pandas as pd

            rounds_data = {}
            for round_num in sorted(raw_rounds.keys()):
                for agent, score in raw_rounds[round_num].items():
                    label = agent.replace("_", " ").title()
                    if label not in rounds_data:
                        rounds_data[label] = []
                    rounds_data[label].append(score)

            max_rounds = max(len(v) for v in rounds_data.values())
            df = pd.DataFrame(
                rounds_data,
                index=[f"Round {i+1}" for i in range(max_rounds)],
            )
            st.line_chart(df)

        except Exception:
            for round_num, round_scores in sorted(raw_rounds.items()):
                st.markdown(f"**Round {round_num}**")
                cols = st.columns(len(round_scores))
                for col, (agent, score) in zip(cols, round_scores.items()):
                    col.metric(
                        agent.replace("_", " ").title(),
                        f"{score}/100",
                    )
    else:
        st.info("No round score data.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 4. Token usage
    # -------------------------------------------------------------------------
    st.markdown("### ⚡ API Token Usage")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tokens",  f"{result.total_tokens:,}")
    c2.metric("Duration",      f"{result.duration_sec}s")
    c3.metric("Avg per Agent", f"{result.total_tokens // max(len(token_breakdown), 1):,}")

    if token_breakdown:
        try:
            import pandas as pd
            df_tokens = pd.DataFrame(
                {
                    "Agent": [a.replace("_", " ").title() for a in token_breakdown],
                    "Tokens": list(token_breakdown.values()),
                }
            ).set_index("Agent")
            st.bar_chart(df_tokens)
        except Exception:
            for agent, tokens in token_breakdown.items():
                st.markdown(
                    f"**{agent.replace('_',' ').title()}:** {tokens:,} tokens"
                )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 5. Fact-check summary
    # -------------------------------------------------------------------------
    st.markdown("### 🔍 Fact-Check Summary")
    flags = result.all_flags or []

    if flags:
        verified   = sum(1 for f in flags if f.get("status") == "verified")
        uncertain  = sum(1 for f in flags if f.get("status") == "uncertain")
        unverified = sum(1 for f in flags if f.get("status") == "unverified")
        total      = len(flags)

        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Total Claims",  total)
        cc2.metric("✅ Verified",   verified)
        cc3.metric("⚠️ Uncertain",  uncertain)
        cc4.metric("❌ Unverified", unverified)

        if total > 0:
            pct_verified = int((verified / total) * 100)
            st.progress(pct_verified, text=f"Factual accuracy: {pct_verified}%")
    else:
        st.info("No fact-check data available.")

    # -------------------------------------------------------------------------
    # 6. Raw data expander
    # -------------------------------------------------------------------------
    with st.expander("🔧 Raw Debate Data (JSON)"):
        st.json(result.to_dict())