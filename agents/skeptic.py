# =============================================================================
# agents/skeptic.py
# Skeptic Agent — challenges assumptions, exposes risks, demands evidence
# =============================================================================

from agents.base_agent import BaseAgent, AgentResult
from backend.config import DEFAULT_TEMPERATURE, MAX_TOKENS_DEBATE


class SkepticAgent(BaseAgent):
    """
    The Skeptic is the system's precision instrument for risk detection.
    It does not oppose for the sake of opposition — it identifies genuine
    weaknesses, blind spots, and overconfident claims in all other arguments.

    In Round 1: Opens with the strongest risk/concern case.
    In Round 2: Dissects the Optimist and Expert's Round 1 arguments.
    In Round 3: Summarises unresolved risks in a closing statement.
    """

    name = "skeptic"

    def build_prompt(self, **kwargs) -> tuple[str, str]:
        return self.prompt_builder.build_debate_prompt(
            agent_name="skeptic",
            topic=kwargs["topic"],
            agenda=kwargs["agenda"],
            round_number=kwargs["round_number"],
            rag_context=kwargs.get("rag_context", ""),
            debate_history=kwargs.get("debate_history", []),
        )

    def run(self, **kwargs) -> AgentResult:
        """
        Args (via kwargs):
            topic (str):           Original user query
            agenda (str):          Planner's structured debate agenda
            round_number (int):    1, 2, or 3
            rag_context (str):     Retrieved context chunks
            debate_history (list): Prior round dicts

        Returns:
            AgentResult with .content = the skeptic's argument for this round
        """
        system_prompt, user_message = self.build_prompt(**kwargs)

        return self._call(
            system_prompt=system_prompt,
            user_message=user_message,
            round_number=kwargs["round_number"],
            temperature=DEFAULT_TEMPERATURE - 0.1,  # Slightly more analytical
            max_tokens=MAX_TOKENS_DEBATE,
        )