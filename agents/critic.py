# =============================================================================
# agents/critic.py
# Critic Agent — scores each agent's argument on 4 logical quality metrics
# =============================================================================

import json
from agents.base_agent import BaseAgent, AgentResult
from backend.config import CRITIC_TEMPERATURE, SCORING_METRICS


class CriticAgent(BaseAgent):
    """
    The Critic evaluates argument quality — not positions.
    It runs AFTER all debate agents complete each round.

    Scores each agent on:
        logical_consistency  (0-25)
        evidence_usage       (0-25)
        relevance            (0-25)
        persuasiveness       (0-25)
        ─────────────────────────
        Total per agent:     0-100

    Output is always structured JSON stored in AgentResult.metadata.
    AgentResult.content holds a human-readable score table.
    """

    name = "critic"

    def build_prompt(self, **kwargs) -> tuple[str, str]:
        return self.prompt_builder.build_critic_prompt(
            round_number=kwargs["round_number"],
            round_arguments=kwargs["round_arguments"],
        )

    def run(self, **kwargs) -> AgentResult:
        """
        Args (via kwargs):
            round_number (int):       The round just completed
            round_arguments (dict):   {agent_name: argument_text}

        Returns:
            AgentResult with:
                .metadata = full scores dict
                .content  = human-readable score table
        """
        system_prompt, user_message = self.build_prompt(**kwargs)

        result = self._call_json(
            system_prompt=system_prompt,
            user_message=user_message,
            round_number=kwargs["round_number"],
        )

        # Override the default JSON dump with a readable score table
        if result.success and result.metadata:
            result.content = self._format_score_table(
                result.metadata,
                kwargs["round_number"],
            )

        return result

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _json_to_readable(self, parsed: dict) -> str:
        """Override — delegate to the formatted score table."""
        return self._format_score_table(parsed, parsed.get("round", "?"))

    def _format_score_table(self, scores_data: dict, round_number) -> str:
        """
        Converts the JSON scores dict into a readable table string.

        Example output:
            === CRITIC SCORES — ROUND 1 ===
            OPTIMIST       | Logic: 20 | Evidence: 18 | Relevance: 22 | Persuasion: 21 | Total: 81
            SKEPTIC        | Logic: 23 | Evidence: 15 | Relevance: 24 | Persuasion: 19 | Total: 81
            ...
        """
        lines = [f"=== CRITIC SCORES — ROUND {round_number} ==="]

        scores = scores_data.get("scores", {})
        if not scores:
            return f"[Critic: no scores parsed for round {round_number}]"

        for agent, data in scores.items():
            if not isinstance(data, dict):
                continue

            lc   = data.get("logical_consistency", 0)
            ev   = data.get("evidence_usage",      0)
            rel  = data.get("relevance",            0)
            pers = data.get("persuasiveness",       0)
            total = lc + ev + rel + pers
            just  = data.get("justification", "")

            lines.append(
                f"{agent.upper():<16} | "
                f"Logic: {lc:>2} | "
                f"Evidence: {ev:>2} | "
                f"Relevance: {rel:>2} | "
                f"Persuasion: {pers:>2} | "
                f"Total: {total:>3}/100"
            )
            if just:
                lines.append(f"  └─ {just}")

        return "\n".join(lines)

    def extract_round_scores(self, result: AgentResult) -> dict[str, int]:
        """
        Convenience method used by the scoring system.
        Returns {agent_name: round_total_score} from a critic AgentResult.

        Example return:
            {"optimist": 81, "skeptic": 78, "analyst": 85, "domain_expert": 79}
        """
        totals = {}
        scores = result.metadata.get("scores", {})

        for agent, data in scores.items():
            if not isinstance(data, dict):
                totals[agent] = 0
                continue
            totals[agent] = (
                data.get("logical_consistency", 0)
                + data.get("evidence_usage",    0)
                + data.get("relevance",         0)
                + data.get("persuasiveness",    0)
            )

        return totals