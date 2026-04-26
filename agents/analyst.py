# =============================================================================
# agents/analyst.py
# Data Analyst Agent — grounds all debate claims in data and evidence
# =============================================================================

from agents.base_agent import BaseAgent, AgentResult
from backend.config import MAX_TOKENS_DEBATE


class AnalystAgent(BaseAgent):
    """
    The Analyst is the data anchor of the debate.
    Every claim must be supported by numbers, statistics, or empirical evidence.
    It also cross-examines other agents' claims for quantitative accuracy.

    In Round 1: Opens with the data-driven view of the topic.
    In Round 2: Fact-checks Round 1 claims for quantitative accuracy.
    In Round 3: Delivers a data-based closing verdict.
    """

    name = "analyst"

    def build_prompt(self, **kwargs) -> tuple[str, str]:
        return self.prompt_builder.build_debate_prompt(
            agent_name="analyst",
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
            rag_context (str):     Retrieved context chunks (most important for analyst)
            debate_history (list): Prior round dicts

        Returns:
            AgentResult with .content = the analyst's evidence-based argument
        """
        system_prompt, user_message = self.build_prompt(**kwargs)

        return self._call(
            system_prompt=system_prompt,
            user_message=user_message,
            round_number=kwargs["round_number"],
            temperature=0.4,        # Lower — data arguments need precision
            max_tokens=MAX_TOKENS_DEBATE,
        )