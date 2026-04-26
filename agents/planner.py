# =============================================================================
# agents/planner.py
# Planner Agent — breaks the user query into a structured debate agenda
# =============================================================================

from agents.base_agent import BaseAgent, AgentResult
from backend.config import JUDGE_TEMPERATURE


class PlannerAgent(BaseAgent):
    """
    The Planner is the first agent to run in every debate.
    It receives the raw user query and produces a structured agenda
    of 3-5 debate propositions that all other agents will argue over.

    It does NOT take a position — it organises the debate space.
    """

    name = "planner"

    def build_prompt(self, **kwargs) -> tuple[str, str]:
        return self.prompt_builder.build_planner_prompt(
            query=kwargs["query"],
            rag_context=kwargs.get("rag_context", ""),
        )

    def run(self, **kwargs) -> AgentResult:
        """
        Args (via kwargs):
            query (str):       The original user question
            rag_context (str): Pre-retrieved document context (optional)

        Returns:
            AgentResult with .content = numbered agenda string
        """
        system_prompt, user_message = self.build_prompt(**kwargs)

        return self._call(
            system_prompt=system_prompt,
            user_message=user_message,
            round_number=None,
            temperature=JUDGE_TEMPERATURE,   # Low temp — agenda must be consistent
            max_tokens=400,
        )