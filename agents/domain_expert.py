# =============================================================================
# agents/domain_expert.py
# Domain Expert Agent — provides deep domain-specific technical insight
# =============================================================================

from agents.base_agent import BaseAgent, AgentResult
from backend.config import DEFAULT_TEMPERATURE, MAX_TOKENS_DEBATE


class DomainExpertAgent(BaseAgent):
    """
    The Domain Expert brings field-specific depth that generalists miss.
    Its perspective shifts based on the active domain:
      - Finance:   Acts as a senior investment analyst / CFA
      - Strategy:  Acts as a McKinsey-level strategy consultant
      - Policy:    Acts as a policy researcher / ethicist
      - Technology: Acts as a principal engineer / CTO

    This specialisation is handled automatically by the PromptBuilder,
    which injects the domain context into the system prompt.

    In Round 1: Frames the problem through the domain lens.
    In Round 2: Corrects domain misconceptions from other agents.
    In Round 3: Delivers the technically grounded closing recommendation.
    """

    name = "domain_expert"

    def build_prompt(self, **kwargs) -> tuple[str, str]:
        return self.prompt_builder.build_debate_prompt(
            agent_name="domain_expert",
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
            AgentResult with .content = expert's domain-specific argument
        """
        system_prompt, user_message = self.build_prompt(**kwargs)

        return self._call(
            system_prompt=system_prompt,
            user_message=user_message,
            round_number=kwargs["round_number"],
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=MAX_TOKENS_DEBATE,
        )