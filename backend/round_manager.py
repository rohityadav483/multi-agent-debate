# =============================================================================
# backend/round_manager.py
# Manages the per-round debate flow — extracted from debate_engine.py
# so the engine stays thin and round logic is independently testable.
# =============================================================================

from typing import Optional, Callable
from agents.optimist      import OptimistAgent
from agents.skeptic       import SkepticAgent
from agents.analyst       import AnalystAgent
from agents.domain_expert import DomainExpertAgent
from agents.critic        import CriticAgent
from agents.fact_checker  import FactCheckerAgent
from backend.config       import AGENT_SPEAKING_ORDER, NUM_ROUNDS
from utils.logger         import get_logger

logger = get_logger(__name__)


# =============================================================================
# RoundResult — output of a single completed round
# =============================================================================

class RoundResult:
    """
    Everything produced by one debate round.

    Attributes:
        round_number:    1, 2, or 3
        arguments:       {agent_name: argument_text}  — the debate outputs
        critic_result:   AgentResult from CriticAgent
        fc_result:       AgentResult from FactCheckerAgent
        round_scores:    {agent_name: total_score}    — extracted from critic
        flags:           flat list of fact-check dicts — extracted from fc
        success:         False if any critical agent failed
    """

    def __init__(self, round_number: int):
        self.round_number:  int  = round_number
        self.arguments:     dict = {}
        self.critic_result       = None
        self.fc_result           = None
        self.round_scores:  dict = {}
        self.flags:         list = []
        self.success:       bool = True

    def to_dict(self) -> dict:
        """Serialise to the format expected by DebateEngine and Memory."""
        return {
            "round":        self.round_number,
            "arguments":    self.arguments,
            "critic":       self.critic_result.to_dict() if self.critic_result else {},
            "fact_checker": self.fc_result.to_dict()     if self.fc_result     else {},
        }


# =============================================================================
# RoundManager
# =============================================================================

class RoundManager:
    """
    Drives one debate round from start to finish.

    Responsibilities:
        1. Call each debate agent in AGENT_SPEAKING_ORDER
        2. Pass debate_history so agents reference prior rounds
        3. Run Critic on completed round arguments
        4. Run Fact-Checker on completed round arguments
        5. Extract scores and flags into RoundResult
        6. Return RoundResult to DebateEngine

    Usage (inside DebateEngine):
        manager = RoundManager(
            agents={
                "optimist":      self.optimist,
                "skeptic":       self.skeptic,
                "analyst":       self.analyst,
                "domain_expert": self.domain_expert,
            },
            critic=self.critic,
            fact_checker=self.fact_checker,
        )

        result = manager.run_round(
            round_number=1,
            topic=query,
            agenda=agenda,
            rag_context=rag_context,
            debate_history=[],
            on_progress=on_progress,
        )
    """

    def __init__(
        self,
        agents:       dict,     # {agent_name: AgentInstance}
        critic:       CriticAgent,
        fact_checker: FactCheckerAgent,
    ):
        self.agents       = agents
        self.critic       = critic
        self.fact_checker = fact_checker

    # -------------------------------------------------------------------------
    # Main method
    # -------------------------------------------------------------------------

    def run_round(
        self,
        round_number:   int,
        topic:          str,
        agenda:         str,
        rag_context:    str = "",
        debate_history: Optional[list] = None,
        on_progress:    Optional[Callable] = None,
    ) -> RoundResult:
        """
        Execute one complete debate round.

        Args:
            round_number:    1, 2, or 3
            topic:           Original user query
            agenda:          Planner's structured agenda string
            rag_context:     Pre-retrieved context (same for all agents)
            debate_history:  List of prior RoundResult.to_dict() outputs
                             Empty list for Round 1.
            on_progress:     Optional callback(stage, message) for UI updates

        Returns:
            RoundResult with all arguments, scores, and flags populated.
        """
        debate_history = debate_history or []
        result         = RoundResult(round_number)

        logger.info(f"RoundManager: starting round {round_number}/{NUM_ROUNDS}")

        # ------------------------------------------------------------------
        # Step 1 — Each debate agent argues
        # ------------------------------------------------------------------
        for agent_name in AGENT_SPEAKING_ORDER:
            agent = self.agents.get(agent_name)
            if agent is None:
                logger.warning(f"Agent '{agent_name}' not found — skipping.")
                result.arguments[agent_name] = f"[{agent_name} not available]"
                continue

            label = agent_name.replace("_", " ").title()
            self._emit(on_progress, f"r{round_number}_{agent_name}",
                       f"💬 {label} is arguing (Round {round_number})...")

            agent_result = agent.run(
                topic=topic,
                agenda=agenda,
                round_number=round_number,
                rag_context=rag_context,
                debate_history=debate_history,
            )

            result.arguments[agent_name] = (
                agent_result.content if agent_result.success
                else f"[{agent_name} failed to respond in round {round_number}]"
            )

            logger.info(
                f"R{round_number} | {agent_name} | "
                f"tokens={agent_result.tokens_used} | "
                f"success={agent_result.success}"
            )

        # ------------------------------------------------------------------
        # Step 2 — Critic scores the round
        # ------------------------------------------------------------------
        self._emit(on_progress, f"critic_{round_number}",
                   f"📊 Critic scoring Round {round_number}...")

        critic_result = self.critic.run(
            round_number=round_number,
            round_arguments=result.arguments,
        )
        result.critic_result = critic_result
        result.round_scores  = self._extract_scores(critic_result)

        logger.info(
            f"R{round_number} | critic done | "
            f"scores={result.round_scores}"
        )

        # ------------------------------------------------------------------
        # Step 3 — Fact-Checker verifies claims
        # ------------------------------------------------------------------
        self._emit(on_progress, f"fc_{round_number}",
                   f"🔍 Fact-Checker verifying Round {round_number} claims...")

        fc_result    = self.fact_checker.run(
            round_number=round_number,
            round_arguments=result.arguments,
            rag_context=rag_context,
        )
        result.fc_result = fc_result
        result.flags     = self.fact_checker.get_all_flags(fc_result)

        logger.info(
            f"R{round_number} | fact-check done | "
            f"flags={len(result.flags)}"
        )

        # Round failed only if ALL debate agents failed
        agents_failed = sum(
            1 for text in result.arguments.values()
            if text.startswith("[") and "failed" in text
        )
        result.success = agents_failed < len(AGENT_SPEAKING_ORDER)

        logger.info(
            f"RoundManager: round {round_number} complete | "
            f"success={result.success}"
        )
        return result

    # -------------------------------------------------------------------------
    # Batch helper — run all NUM_ROUNDS at once
    # -------------------------------------------------------------------------

    def run_all_rounds(
        self,
        topic:       str,
        agenda:      str,
        rag_context: str = "",
        on_progress: Optional[Callable] = None,
    ) -> list[RoundResult]:
        """
        Convenience method — runs all 3 rounds sequentially,
        passing each completed round into the next as history.

        Returns:
            List of RoundResult objects, one per round.
        """
        results        = []
        debate_history = []

        for round_number in range(1, NUM_ROUNDS + 1):
            self._emit(
                on_progress,
                f"round_{round_number}_start",
                f"🥊 Round {round_number} of {NUM_ROUNDS} beginning...",
            )

            round_result = self.run_round(
                round_number=round_number,
                topic=topic,
                agenda=agenda,
                rag_context=rag_context,
                debate_history=debate_history,
                on_progress=on_progress,
            )

            results.append(round_result)

            # Add this round to history for the next round's agents
            debate_history.append({
                "round":     round_number,
                "arguments": round_result.arguments,
            })

            self._emit(
                on_progress,
                f"round_{round_number}_done",
                f"✅ Round {round_number} complete.",
            )

        return results

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _extract_scores(self, critic_result) -> dict:
        """
        Extract {agent_name: total_score} from a CriticAgent result.
        Returns empty dict if parsing failed.
        """
        scores = {}
        raw    = critic_result.metadata.get("scores", {})

        for agent, data in raw.items():
            if not isinstance(data, dict):
                scores[agent] = 0
                continue
            scores[agent] = (
                data.get("logical_consistency", 0)
                + data.get("evidence_usage",    0)
                + data.get("relevance",         0)
                + data.get("persuasiveness",    0)
            )

        return scores

    def _emit(
        self,
        callback: Optional[Callable],
        stage:    str,
        message:  str,
    ):
        if callback:
            try:
                callback(stage, message)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")