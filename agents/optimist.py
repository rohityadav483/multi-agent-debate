# =============================================================================
# agents/optimist.py
# Optimist Agent — argues best-case scenarios and positive outcomes
# =============================================================================

from agents.base_agent import BaseAgent, AgentResult
from backend.config import DEFAULT_TEMPERATURE, MAX_TOKENS_DEBATE


class OptimistAgent(BaseAgent):
    """
    The Optimist champions opportunity, upside, and bold action.
    In Round 1: Opens with the strongest positive case.
    In Round 2: Counters the Skeptic's criticisms with evidence.
    In Round 3: Delivers a refined closing argument.
    """

    name = "optimist"

    def build_prompt(self, **kwargs) -> tuple[str, str]:
        return self.prompt_builder.build_debate_prompt(
            agent_name="optimist",
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
            debate_history (list): Prior round dicts [{round, arguments}]

        Returns:
            AgentResult with .content = the optimist's argument for this round
        """
        system_prompt, user_message = self.build_prompt(**kwargs)

        return self._call(
            system_prompt=system_prompt,
            user_message=user_message,
            round_number=kwargs["round_number"],
            temperature=DEFAULT_TEMPERATURE + 0.1,  # Slightly creative
            max_tokens=MAX_TOKENS_DEBATE,
        )