# =============================================================================
# agents/fact_checker.py
# Fact-Checker Agent — verifies agent claims against retrieved context
# =============================================================================

from agents.base_agent import BaseAgent, AgentResult


class FactCheckerAgent(BaseAgent):
    """
    The Fact-Checker is the system's truth guardian.
    It runs after each debate round alongside the Critic.

    For each agent's argument it:
      1. Extracts 1-3 specific factual claims
      2. Cross-checks each claim against the RAG-retrieved context
      3. Labels each as: verified / uncertain / unverified

    Unverified claims are flagged as potential hallucinations in the UI.

    Output is always structured JSON stored in AgentResult.metadata.
    AgentResult.content holds a human-readable flags summary.
    """

    name = "fact_checker"

    def build_prompt(self, **kwargs) -> tuple[str, str]:
        return self.prompt_builder.build_fact_checker_prompt(
            round_number=kwargs["round_number"],
            round_arguments=kwargs["round_arguments"],
            rag_context=kwargs.get("rag_context", ""),
        )

    def run(self, **kwargs) -> AgentResult:
        """
        Args (via kwargs):
            round_number (int):     The round just completed
            round_arguments (dict): {agent_name: argument_text}
            rag_context (str):      The same RAG context used in the round

        Returns:
            AgentResult with:
                .metadata = full fact-check dict
                .content  = human-readable flag summary
        """
        system_prompt, user_message = self.build_prompt(**kwargs)

        result = self._call_json(
            system_prompt=system_prompt,
            user_message=user_message,
            round_number=kwargs["round_number"],
        )

        if result.success and result.metadata:
            result.content = self._format_fact_summary(
                result.metadata,
                kwargs["round_number"],
            )

        return result

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _json_to_readable(self, parsed: dict) -> str:
        return self._format_fact_summary(parsed, parsed.get("round", "?"))

    def _format_fact_summary(self, fact_data: dict, round_number) -> str:
        """
        Converts the JSON fact-check dict into a readable summary.

        Example output:
            === FACT-CHECK REPORT — ROUND 1 ===
            OPTIMIST:
              ✅ VERIFIED   — "Renewable energy investment grew 25% in 2023"
              ⚠️  UNCERTAIN  — "Solar costs dropped 90% in a decade"
            SKEPTIC:
              ❌ UNVERIFIED — "Battery storage costs are prohibitive at scale"
        """
        status_icons = {
            "verified":   "✅ VERIFIED  ",
            "uncertain":  "⚠️  UNCERTAIN ",
            "unverified": "❌ UNVERIFIED",
        }

        lines = [f"=== FACT-CHECK REPORT — ROUND {round_number} ==="]

        fact_checks = fact_data.get("fact_checks", {})
        if not fact_checks:
            return f"[Fact-Checker: no results parsed for round {round_number}]"

        for agent, claims in fact_checks.items():
            lines.append(f"\n{agent.upper()}:")
            if not isinstance(claims, list):
                lines.append("  [No claims extracted]")
                continue
            for item in claims:
                status = item.get("status", "uncertain").lower()
                icon   = status_icons.get(status, "❓ UNKNOWN   ")
                claim  = item.get("claim", "")[:120]
                note   = item.get("note", "")
                lines.append(f"  {icon} — \"{claim}\"")
                if note:
                    lines.append(f"             └─ {note}")

        return "\n".join(lines)

    def get_all_flags(self, result: AgentResult) -> list[dict]:
        """
        Convenience method used by the Judge and memory system.
        Returns a flat list of all fact-check results across all agents.

        Example return:
            [
                {"agent": "optimist", "claim": "...", "status": "verified",   "note": "..."},
                {"agent": "skeptic",  "claim": "...", "status": "unverified", "note": "..."},
                ...
            ]
        """
        flat = []
        fact_checks = result.metadata.get("fact_checks", {})

        for agent, claims in fact_checks.items():
            if not isinstance(claims, list):
                continue
            for item in claims:
                flat.append({
                    "agent":  agent,
                    "claim":  item.get("claim",  ""),
                    "status": item.get("status", "uncertain"),
                    "note":   item.get("note",   ""),
                    "round":  result.round_number,
                })

        return flat