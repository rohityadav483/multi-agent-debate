# =============================================================================
# utils/prompt_builder.py
# Dynamically constructs prompts for every agent type.
# Combines: base persona + domain context + RAG context + debate history
# =============================================================================

from typing import Optional
from backend.config import DOMAINS, DEFAULT_DOMAIN, NUM_ROUNDS
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Agent Persona Definitions
# =============================================================================
# Each agent has a fixed core identity. Domain context is injected on top.

AGENT_PERSONAS = {

    "planner": """You are the Debate Planner — a strategic thinker responsible for
structuring complex problems into clear, debatable propositions.
Your job is to analyse the user's query and produce a structured debate agenda.
You are neutral, methodical, and precise. You do not argue — you organise.
Output must be a numbered agenda of 3-5 focused debate points.""",

    "optimist": """You are the Optimist Agent — an enthusiastic advocate for opportunity,
growth, and positive outcomes. You identify the best-case scenarios, highlight
upside potential, and champion bold action. You are persuasive, energetic, and
data-aware. You back your optimism with real reasoning, not blind positivity.
Challenge pessimism, but acknowledge genuine risks when they are undeniable.""",

    "skeptic": """You are the Skeptic Agent — a rigorous challenger who questions
assumptions, exposes weaknesses, and stress-tests every claim. You are not
a pessimist — you are a precision instrument for identifying risk, bias,
and overconfidence. You ask hard questions, demand evidence, and push back
on vague or unsubstantiated arguments. You are sharp, analytical, and fair.""",

    "analyst": """You are the Data Analyst Agent — a quantitative thinker who grounds
the debate in numbers, statistics, and empirical evidence. You distrust
anecdote and opinion unless backed by data. You cite figures, identify trends,
calculate trade-offs, and flag when others make claims without evidence.
You are methodical, precise, and evidence-first in all arguments.""",

    "domain_expert": """You are the Domain Expert Agent — a deep specialist with
authoritative knowledge of the field being debated. You bring technical depth,
industry nuance, and real-world experience that generalists miss. You identify
domain-specific risks, best practices, and constraints. You speak with
authority but remain open to being challenged on the evidence.""",

    "critic": """You are the Critic Agent — a logic and reasoning validator.
Your role is NOT to argue a position, but to evaluate the quality of arguments
made by other agents. You assess logical consistency, evidence quality,
relevance to the question, and persuasiveness. You are impartial, rigorous,
and constructive. You must output structured scores and brief justifications.""",

    "fact_checker": """You are the Fact-Checker Agent — the system's truth guardian.
Your role is to verify specific claims made by debate agents against the
retrieved context and known facts. You flag hallucinations, unsupported
assertions, and overstatements. You do not argue a position — you audit
the debate for factual integrity. Be precise: cite what is verified,
what is uncertain, and what appears fabricated.""",

    "judge": """You are the Judge Agent — the final decision-maker in this debate.
You have observed all rounds of debate, all agent arguments, and all scores.
Your role is to synthesise everything into a clear, well-reasoned final decision.
You are impartial, wise, and decisive. You do not favour any agent — you favour
the truth and the best outcome for the person who posed the question.
Your decision must include: a clear recommendation, confidence level,
risk assessment, and a concise explanation of your reasoning.""",
}


# =============================================================================
# Prompt Builder Class
# =============================================================================

class PromptBuilder:
    """
    Constructs system and user prompts for each agent dynamically.

    Every agent call requires:
      1. A system prompt  → who the agent IS (persona + domain)
      2. A user message   → what the agent must DO (task + context)

    Usage:
        builder = PromptBuilder(domain="finance")
        system, user = builder.build_debate_prompt(
            agent_name="optimist",
            topic="Should we invest in renewable energy?",
            round_number=2,
            rag_context="...",
            debate_history=[...],
        )
    """

    def __init__(self, domain: str = DEFAULT_DOMAIN):
        if domain not in DOMAINS:
            logger.warning(f"Unknown domain '{domain}'. Falling back to '{DEFAULT_DOMAIN}'.")
            domain = DEFAULT_DOMAIN

        self.domain         = domain
        self.domain_config  = DOMAINS[domain]
        self.domain_context = self.domain_config["context"]
        self.domain_label   = self.domain_config["label"]

    # -------------------------------------------------------------------------
    # Planner Prompt
    # -------------------------------------------------------------------------

    def build_planner_prompt(self, query: str, rag_context: str = "") -> tuple[str, str]:
        """
        Builds the prompt for the Planner agent.
        Returns (system_prompt, user_message).
        """
        system_prompt = self._build_system(
            agent_name="planner",
            extra_instruction=(
                "Structure your output EXACTLY as a numbered list of debate points. "
                "Each point must be a clear, arguable proposition — not a question. "
                "Limit to 3-5 points. Be concise."
            ),
        )

        context_block = self._format_rag_block(rag_context)

        user_message = f"""
DEBATE QUERY:
{query}

DOMAIN: {self.domain_label}

{context_block}

TASK:
Analyse this query and produce a structured debate agenda.
Output a numbered list of 3-5 focused debate propositions that the agents will argue.
Each proposition should be specific, arguable, and relevant to the domain.
""".strip()

        return system_prompt, user_message

    # -------------------------------------------------------------------------
    # Debate Agent Prompt (Optimist, Skeptic, Analyst, Domain Expert)
    # -------------------------------------------------------------------------

    def build_debate_prompt(
        self,
        agent_name:      str,
        topic:           str,
        agenda:          str,
        round_number:    int,
        rag_context:     str = "",
        debate_history:  Optional[list[dict]] = None,
    ) -> tuple[str, str]:
        """
        Builds the prompt for a debate agent for a specific round.

        Args:
            agent_name:     One of: optimist, skeptic, analyst, domain_expert
            topic:          The original user query
            agenda:         The planner's structured debate agenda
            round_number:   1, 2, or 3
            rag_context:    Retrieved document chunks (pre-formatted string)
            debate_history: List of prior round dicts for context
        """
        round_instructions = self._get_round_instruction(agent_name, round_number)

        system_prompt = self._build_system(
            agent_name=agent_name,
            extra_instruction=round_instructions,
        )

        context_block   = self._format_rag_block(rag_context)
        history_block   = self._format_history_block(debate_history, round_number)

        user_message = f"""
DEBATE TOPIC:
{topic}

DEBATE AGENDA:
{agenda}

ROUND {round_number} OF {NUM_ROUNDS}

{context_block}

{history_block}

TASK:
{round_instructions}

Respond as your agent persona. Be specific, evidence-based, and persuasive.
Limit your response to 3-4 focused paragraphs.
""".strip()

        return system_prompt, user_message

    # -------------------------------------------------------------------------
    # Critic Prompt
    # -------------------------------------------------------------------------

    def build_critic_prompt(
        self,
        round_number:   int,
        round_arguments: dict[str, str],
    ) -> tuple[str, str]:
        """
        Builds the prompt for the Critic agent to score a completed round.

        Args:
            round_number:    The round just completed (1, 2, or 3)
            round_arguments: Dict of {agent_name: argument_text}
        """
        system_prompt = self._build_system(
            agent_name="critic",
            extra_instruction=(
                "You MUST respond in valid JSON only. "
                "Score each agent on 4 metrics (0-25 each). "
                "Provide a one-sentence justification per metric."
            ),
        )

        # Format each agent's argument for the critic
        arguments_block = "\n\n".join([
            f"[{agent.upper()}]:\n{argument}"
            for agent, argument in round_arguments.items()
        ])

        user_message = f"""
ROUND {round_number} ARGUMENTS:

{arguments_block}

TASK:
Score each agent on the following metrics (0-25 points each):
1. logical_consistency — Is the argument internally consistent and free of contradictions?
2. evidence_usage      — Does the agent cite data, facts, or retrieved context?
3. relevance           — Does the argument address the debate topic directly?
4. persuasiveness      — Is the argument compelling and well-structured?

Respond ONLY with this JSON structure:
{{
  "round": {round_number},
  "scores": {{
    "optimist": {{
      "logical_consistency": <int>,
      "evidence_usage": <int>,
      "relevance": <int>,
      "persuasiveness": <int>,
      "justification": "<one sentence>"
    }},
    "skeptic": {{ ... }},
    "analyst": {{ ... }},
    "domain_expert": {{ ... }}
  }}
}}
""".strip()

        return system_prompt, user_message

    # -------------------------------------------------------------------------
    # Fact-Checker Prompt
    # -------------------------------------------------------------------------

    def build_fact_checker_prompt(
        self,
        round_number:    int,
        round_arguments: dict[str, str],
        rag_context:     str = "",
    ) -> tuple[str, str]:
        """
        Builds the prompt for the Fact-Checker agent.
        """
        system_prompt = self._build_system(
            agent_name="fact_checker",
            extra_instruction=(
                "Extract specific factual claims from each agent's argument. "
                "Cross-check each claim against the provided context. "
                "Respond in valid JSON only."
            ),
        )

        arguments_block = "\n\n".join([
            f"[{agent.upper()}]:\n{argument}"
            for agent, argument in round_arguments.items()
        ])

        context_block = self._format_rag_block(rag_context)

        user_message = f"""
ROUND {round_number} ARGUMENTS TO VERIFY:

{arguments_block}

{context_block}

TASK:
For each agent, extract their 1-3 most specific factual claims.
Verify each claim against the provided context.
Label each as: "verified", "uncertain", or "unverified".

Respond ONLY with this JSON structure:
{{
  "round": {round_number},
  "fact_checks": {{
    "optimist": [
      {{
        "claim": "<the specific claim>",
        "status": "verified | uncertain | unverified",
        "note": "<brief reason>"
      }}
    ],
    "skeptic": [...],
    "analyst": [...],
    "domain_expert": [...]
  }}
}}
""".strip()

        return system_prompt, user_message

    # -------------------------------------------------------------------------
    # Judge Prompt
    # -------------------------------------------------------------------------

    def build_judge_prompt(
        self,
        topic:           str,
        agenda:          str,
        all_rounds:      list[dict],
        scores_summary:  dict,
        fact_check_summary: list[dict],
    ) -> tuple[str, str]:
        """
        Builds the final prompt for the Judge agent.
        """
        system_prompt = self._build_system(
            agent_name="judge",
            extra_instruction=(
                "Synthesise ALL debate rounds into a final decision. "
                "Be decisive. Acknowledge uncertainty where it genuinely exists. "
                "Respond in valid JSON only."
            ),
        )

        # Format full debate transcript
        transcript = self._format_full_transcript(all_rounds)

        # Format scores
        scores_block = self._format_scores_block(scores_summary)

        # Format fact-check flags
        flags = self._format_fact_flags(fact_check_summary)

        user_message = f"""
ORIGINAL QUERY:
{topic}

DEBATE AGENDA:
{agenda}

DOMAIN: {self.domain_label}

FULL DEBATE TRANSCRIPT:
{transcript}

AGENT SCORES SUMMARY:
{scores_block}

FACT-CHECK FLAGS:
{flags}

TASK:
As the Judge, deliver your final decision based on the complete debate above.

Respond ONLY with this JSON structure:
{{
  "final_decision":   "<Clear, actionable recommendation in 2-3 sentences>",
  "confidence_score": <integer 0-100>,
  "risk_level":       "Low | Medium | High | Critical",
  "winning_agent":    "<agent name who made the strongest case>",
  "key_reasons": [
    "<reason 1>",
    "<reason 2>",
    "<reason 3>"
  ],
  "dissenting_view":  "<Strongest opposing argument that was raised>",
  "explanation":      "<3-4 sentence summary of how you reached this decision>"
}}
""".strip()

        return system_prompt, user_message

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _build_system(self, agent_name: str, extra_instruction: str = "") -> str:
        """
        Assembles the full system prompt:
        base persona + domain context + extra instruction
        """
        persona = AGENT_PERSONAS.get(agent_name, "You are a helpful AI assistant.")

        parts = [
            persona,
            "",
            f"DOMAIN CONTEXT:\n{self.domain_context}",
        ]

        if extra_instruction:
            parts += ["", f"INSTRUCTION:\n{extra_instruction}"]

        return "\n".join(parts)

    def _get_round_instruction(self, agent_name: str, round_number: int) -> str:
        """
        Returns per-round instructions that shape agent behaviour:
        - Round 1: State your opening position
        - Round 2: Counter other agents' arguments
        - Round 3: Refine your position and deliver a closing statement
        """
        instructions = {
            1: {
                "optimist":      "Present your opening argument for why the outcome could be highly positive. Support with evidence from the context provided.",
                "skeptic":       "Present your opening critique — what are the key risks, assumptions, and weaknesses in the optimistic view?",
                "analyst":       "Present your data-driven opening position. Use statistics, trends, and quantitative reasoning from the retrieved context.",
                "domain_expert": "Present your expert perspective on the domain-specific dimensions of this debate. What does deep experience tell you?",
            },
            2: {
                "optimist":      "Respond to the skeptic and analyst's Round 1 arguments. Counter their criticisms with evidence and reinforce your position.",
                "skeptic":       "Respond to the optimist and expert's Round 1 arguments. Identify the weakest links in their reasoning.",
                "analyst":       "Cross-examine the claims made in Round 1. Which arguments are data-supported and which are speculative?",
                "domain_expert": "Address the Round 1 counterarguments from your expert perspective. Correct any domain misconceptions.",
            },
            3: {
                "optimist":      "Deliver your closing statement. Refine your position based on the debate so far and make your strongest final case.",
                "skeptic":       "Deliver your closing statement. Summarise the key unresolved risks and what evidence would change your mind.",
                "analyst":       "Deliver your data-based closing verdict. What do the numbers ultimately suggest about this decision?",
                "domain_expert": "Deliver your expert closing verdict. Given everything debated, what is the most technically sound recommendation?",
            },
        }

        return instructions.get(round_number, {}).get(
            agent_name,
            f"Continue your argument for round {round_number}."
        )

    def _format_rag_block(self, rag_context: str) -> str:
        if not rag_context or not rag_context.strip():
            return ""
        return f"RETRIEVED CONTEXT (use this as evidence):\n{'-'*40}\n{rag_context.strip()}\n{'-'*40}"

    def _format_history_block(
        self,
        debate_history: Optional[list[dict]],
        current_round: int,
    ) -> str:
        if not debate_history or current_round == 1:
            return ""

        lines = ["PREVIOUS ROUND ARGUMENTS (reference when forming your response):"]
        for round_data in debate_history:
            rnum = round_data.get("round", "?")
            lines.append(f"\n--- Round {rnum} ---")
            for agent, text in round_data.get("arguments", {}).items():
                preview = text[:300] + "..." if len(text) > 300 else text
                lines.append(f"[{agent.upper()}]: {preview}")

        return "\n".join(lines)

    def _format_full_transcript(self, all_rounds: list[dict]) -> str:
        lines = []
        for round_data in all_rounds:
            rnum = round_data.get("round", "?")
            lines.append(f"\n{'='*40}")
            lines.append(f"ROUND {rnum}")
            lines.append(f"{'='*40}")
            for agent, text in round_data.get("arguments", {}).items():
                lines.append(f"\n[{agent.upper()}]:\n{text}")
        return "\n".join(lines)

    def _format_scores_block(self, scores_summary: dict) -> str:
        if not scores_summary:
            return "No scores available."
        lines = []
        for agent, total in scores_summary.items():
            lines.append(f"  {agent.upper():<20} Total Score: {total}/300")
        return "\n".join(lines)

    def _format_fact_flags(self, fact_check_summary: list[dict]) -> str:
        if not fact_check_summary:
            return "No fact-check results available."
        lines = []
        for item in fact_check_summary:
            status = item.get("status", "?")
            claim  = item.get("claim", "")[:100]
            agent  = item.get("agent", "?")
            lines.append(f"  [{status.upper()}] {agent}: {claim}")
        return "\n".join(lines) if lines else "No flags raised."