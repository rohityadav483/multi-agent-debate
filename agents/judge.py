# =============================================================================
# agents/judge.py
# Judge Agent — synthesises all debate rounds into a final decision
# =============================================================================

from agents.base_agent import BaseAgent, AgentResult
from backend.config import JUDGE_TEMPERATURE, MAX_TOKENS_JUDGE, RISK_LEVELS, PRIMARY_MODEL

class JudgeAgent(BaseAgent):
    """
    The Judge is the final agent to run — after all 3 rounds are complete.

    It receives:
        - The full debate transcript (all rounds, all agents)
        - Aggregated critic scores per agent
        - Flat list of all fact-check flags

    It outputs a structured JSON verdict containing:
        final_decision    — Clear, actionable recommendation
        confidence_score  — 0-100 integer
        risk_level        — Low / Medium / High / Critical
        winning_agent     — Which agent made the strongest case
        key_reasons       — Top 3 reasons for the decision
        dissenting_view   — Strongest opposing argument acknowledged
        explanation       — 3-4 sentence reasoning summary

    This is the single output that the UI's Decision Card displays.
    """

    name = "judge"

    def build_prompt(self, **kwargs) -> tuple[str, str]:
        return self.prompt_builder.build_judge_prompt(
            topic=kwargs["topic"],
            agenda=kwargs["agenda"],
            all_rounds=kwargs["all_rounds"],
            scores_summary=kwargs["scores_summary"],
            fact_check_summary=kwargs.get("fact_check_summary", []),
        )


    def run(self, **kwargs) -> AgentResult:
        system_prompt, user_message = self.build_prompt(**kwargs)

        # Override: Judge uses PRIMARY_MODEL (70b), not FAST_MODEL (8b)
        # 8b has 6000 TPM limit — full transcript exceeds it
        parsed = self.llm.extract_json(
            system_prompt=system_prompt,
            user_message=user_message,
            model=PRIMARY_MODEL,
        )

        if "error" in parsed:
            self.logger.warning(f"Judge JSON parse failed: {parsed['error']}")
            return self._error_result(parsed.get("error", "JSON failed"), None)

        content = self._format_verdict(parsed)

        return AgentResult(
            agent_name=self.name,
            content=content,
            round_number=None,
            metadata=parsed,
            success=True,
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _json_to_readable(self, parsed: dict) -> str:
        return self._format_verdict(parsed)

    def _format_verdict(self, verdict: dict) -> str:
        """
        Formats the judge's JSON verdict into a clear readable summary
        for the UI decision card and logs.

        Example output:
            ╔══════════════════════════════════════╗
            ║         JUDGE'S FINAL DECISION        ║
            ╚══════════════════════════════════════╝

            DECISION:
            Based on the evidence, the company should proceed with a phased
            international expansion starting with Southeast Asian markets...

            CONFIDENCE:    78/100  (High)
            RISK LEVEL:    Medium
            WINNING AGENT: Analyst

            KEY REASONS:
              1. Revenue diversification outweighs short-term capital risk
              2. Competitor analysis shows an 18-month window of opportunity
              3. Domain expert confirmed regulatory feasibility in target markets

            DISSENTING VIEW:
            The Skeptic raised valid concerns about currency exposure and
            political instability that should be monitored post-launch.

            EXPLANATION:
            The Judge weighed strong quantitative evidence from the Analyst...
        """
        lines = [
            "╔══════════════════════════════════════╗",
            "║       JUDGE'S FINAL DECISION         ║",
            "╚══════════════════════════════════════╝",
            "",
        ]

        # Final decision
        decision = verdict.get("final_decision", "No decision reached.")
        lines += ["DECISION:", decision, ""]

        # Confidence + risk
        confidence = verdict.get("confidence_score", 0)
        risk       = verdict.get("risk_level", "Unknown")
        winner     = verdict.get("winning_agent", "N/A")

        # Validate risk level
        if risk not in RISK_LEVELS:
            risk = "Medium"

        confidence_label = (
            "High"   if confidence >= 75 else
            "Medium" if confidence >= 50 else
            "Low"
        )

        lines += [
            f"CONFIDENCE:    {confidence}/100  ({confidence_label})",
            f"RISK LEVEL:    {risk}",
            f"WINNING AGENT: {winner.upper() if winner else 'N/A'}",
            "",
        ]

        # Key reasons
        reasons = verdict.get("key_reasons", [])
        if reasons:
            lines.append("KEY REASONS:")
            for i, reason in enumerate(reasons, 1):
                lines.append(f"  {i}. {reason}")
            lines.append("")

        # Dissenting view
        dissent = verdict.get("dissenting_view", "")
        if dissent:
            lines += ["DISSENTING VIEW:", dissent, ""]

        # Explanation
        explanation = verdict.get("explanation", "")
        if explanation:
            lines += ["EXPLANATION:", explanation]

        return "\n".join(lines)

    def get_verdict_dict(self, result: AgentResult) -> dict:
        """
        Convenience method — returns the raw verdict dict from an AgentResult.
        Used by the UI to populate the Decision Card widget.

        Returns safe defaults if the result failed or metadata is missing.
        """
        if not result.success or not result.metadata:
            return {
                "final_decision":   "Unable to reach a decision due to a system error.",
                "confidence_score": 0,
                "risk_level":       "Unknown",
                "winning_agent":    "N/A",
                "key_reasons":      [],
                "dissenting_view":  "",
                "explanation":      "The Judge agent encountered an error.",
            }
        return result.metadata