# =============================================================================
# backend/debate_engine.py
# Core orchestrator — wires all agents, drives the 3-round debate loop,
# collects scores, runs fact-checks, invokes Judge, returns full result.
# =============================================================================

import time
from typing import Optional, Callable
from utils.prompt_builder import PromptBuilder
from utils.logger import get_logger
from backend.scoring import ScoringEngine
from backend.config import (
    NUM_ROUNDS,
    AGENT_SPEAKING_ORDER,
    DEFAULT_DOMAIN,
)
from agents.planner       import PlannerAgent
from agents.optimist      import OptimistAgent
from agents.skeptic       import SkepticAgent
from agents.analyst       import AnalystAgent
from agents.domain_expert import DomainExpertAgent
from agents.critic        import CriticAgent
from agents.fact_checker  import FactCheckerAgent
from agents.judge         import JudgeAgent

logger = get_logger(__name__)


# =============================================================================
# DebateResult — complete output of one full debate
# =============================================================================

class DebateResult:
    def __init__(self):
        self.topic:        str   = ""
        self.domain:       str   = ""
        self.agenda:       str   = ""
        self.rounds:       list  = []
        self.scores:       dict  = {}
        self.verdict:      dict  = {}
        self.all_flags:    list  = []
        self.total_tokens: int   = 0
        self.duration_sec: float = 0.0
        self.success:      bool  = True
        self.error:        str   = ""

    def to_dict(self) -> dict:
        return {
            "topic":        self.topic,
            "domain":       self.domain,
            "agenda":       self.agenda,
            "rounds":       self.rounds,
            "scores":       self.scores,
            "verdict":      self.verdict,
            "all_flags":    self.all_flags,
            "total_tokens": self.total_tokens,
            "duration_sec": self.duration_sec,
            "success":      self.success,
            "error":        self.error,
        }


# =============================================================================
# DebateEngine
# =============================================================================

class DebateEngine:

    def __init__(self, domain: str = DEFAULT_DOMAIN):
        self.domain         = domain
        self.prompt_builder = PromptBuilder(domain=domain)
        self.scoring        = ScoringEngine()
        self._init_agents()

    def _init_agents(self):
        pb = self.prompt_builder
        self.planner       = PlannerAgent(pb)
        self.optimist      = OptimistAgent(pb)
        self.skeptic       = SkepticAgent(pb)
        self.analyst       = AnalystAgent(pb)
        self.domain_expert = DomainExpertAgent(pb)
        self.critic        = CriticAgent(pb)
        self.fact_checker  = FactCheckerAgent(pb)
        self.judge         = JudgeAgent(pb)

        self._debate_agents = {
            "optimist":      self.optimist,
            "skeptic":       self.skeptic,
            "analyst":       self.analyst,
            "domain_expert": self.domain_expert,
        }

        logger.info(f"Agents initialised | domain={self.domain}")

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def run(
        self,
        query:       str,
        rag_context: str = "",
        on_progress: Optional[Callable[[str, str], None]] = None,
    ) -> DebateResult:

        result        = DebateResult()
        result.topic  = query
        result.domain = self.domain
        self.scoring.reset()

        start_time = time.time()
        logger.info(f"Debate started | topic='{query[:80]}' | domain={self.domain}")

        try:
            # ------------------------------------------------------------------
            # PHASE 1 — Planning
            # ------------------------------------------------------------------
            self._emit(on_progress, "planner",
                       "Planner is structuring the debate agenda...")

            planner_result = self.planner.run(
                query=query,
                rag_context=rag_context,
            )
            result.agenda = planner_result.content
            logger.info(f"Agenda ready | {result.agenda[:100]}...")
            self._emit(on_progress, "planner_done",
                       f"Agenda ready:\n{result.agenda}")

            # ------------------------------------------------------------------
            # PHASE 2 — Debate Rounds
            # ------------------------------------------------------------------
            debate_history = []
            all_flags      = []

            for round_num in range(1, NUM_ROUNDS + 1):
                logger.info(f"Round {round_num}/{NUM_ROUNDS} starting")
                self._emit(on_progress, f"round_{round_num}_start",
                           f"Round {round_num} of {NUM_ROUNDS} beginning...")

                round_arguments = {}

                # --- 2a. Debate agents argue ----------------------------------
                for agent_name in AGENT_SPEAKING_ORDER:
                    agent = self._debate_agents[agent_name]
                    label = agent_name.replace("_", " ").title()

                    self._emit(on_progress, f"round_{round_num}_{agent_name}",
                               f"{label} is arguing...")

                    agent_result = agent.run(
                        topic=query,
                        agenda=result.agenda,
                        round_number=round_num,
                        rag_context=rag_context,
                        debate_history=debate_history,
                    )

                    round_arguments[agent_name] = (
                        agent_result.content if agent_result.success
                        else f"[{agent_name} failed to respond]"
                    )
                    logger.info(
                        f"R{round_num} | {agent_name} | "
                        f"tokens={agent_result.tokens_used}"
                    )

                # --- 2b. Critic scores ----------------------------------------
                self._emit(on_progress, f"critic_{round_num}",
                           f"Critic scoring Round {round_num}...")

                critic_result = self.critic.run(
                    round_number=round_num,
                    round_arguments=round_arguments,
                )
                self.scoring.add_round_from_critic(round_num, critic_result)

                # FIX 1: sleep between critic and fact_checker
                # prevents rate limit on llama-3.1-8b-instant (30 req/min)
                time.sleep(8)

                # --- 2c. Fact-Checker verifies --------------------------------
                self._emit(on_progress, f"fact_check_{round_num}",
                           f"Fact-Checker verifying Round {round_num} claims...")

                fc_result   = self.fact_checker.run(
                    round_number=round_num,
                    round_arguments=round_arguments,
                    rag_context=rag_context,
                )
                round_flags = self.fact_checker.get_all_flags(fc_result)
                all_flags.extend(round_flags)

                # --- Store round ----------------------------------------------
                result.rounds.append({
                    "round":        round_num,
                    "arguments":    round_arguments,
                    "critic":       critic_result.to_dict(),
                    "fact_checker": fc_result.to_dict(),
                })
                debate_history.append({
                    "round":     round_num,
                    "arguments": round_arguments,
                })

                self._emit(on_progress, f"round_{round_num}_done",
                           f"Round {round_num} complete.")

                # FIX 2: sleep between rounds to stay under rate limits
                if round_num < NUM_ROUNDS:
                    time.sleep(5)

            # ------------------------------------------------------------------
            # PHASE 3 — Scoring
            # ------------------------------------------------------------------
            scores_summary   = self.scoring.get_summary()
            result.scores    = scores_summary
            result.all_flags = all_flags
            winner           = scores_summary.get("winner", "?")

            logger.info(f"Scoring done | winner={winner}")
            self._emit(on_progress, "scoring_done",
                       f"Scoring complete. Leading agent: {winner.upper()}")

            # ------------------------------------------------------------------
            # PHASE 4 — Judge
            # ------------------------------------------------------------------
            self._emit(on_progress, "judge",
                       "Judge is deliberating final decision...")

            # FIX 3: sleep before judge (uses fast model — same rate limit)
            time.sleep(5)

            judge_result   = self.judge.run(
                topic=query,
                agenda=result.agenda,
                all_rounds=result.rounds,
                scores_summary=scores_summary.get("weighted_totals", {}),
                fact_check_summary=all_flags,
            )
            result.verdict = self.judge.get_verdict_dict(judge_result)

            logger.info(
                f"Judge done | "
                f"confidence={result.verdict.get('confidence_score')}%"
            )
            self._emit(
                on_progress, "judge_done",
                f"Decision reached with "
                f"{result.verdict.get('confidence_score', 0)}% confidence.",
            )

        except Exception as e:
            logger.error(f"DebateEngine error: {e}", exc_info=True)
            result.success = False
            result.error   = str(e)
            self._emit(on_progress, "error", f"Debate engine error: {e}")

        finally:
            result.total_tokens = self._count_total_tokens()
            result.duration_sec = round(time.time() - start_time, 2)
            logger.info(
                f"Debate finished | success={result.success} | "
                f"tokens={result.total_tokens} | duration={result.duration_sec}s"
            )

        return result

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

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

    def _count_total_tokens(self) -> int:
        agents = [
            self.planner,  self.optimist,      self.skeptic,
            self.analyst,  self.domain_expert,
            self.critic,   self.fact_checker,  self.judge,
        ]
        return sum(a.total_tokens for a in agents)

    def reset(self):
        for agent in [
            self.planner,  self.optimist,      self.skeptic,
            self.analyst,  self.domain_expert,
            self.critic,   self.fact_checker,  self.judge,
        ]:
            agent.reset_tokens()
        self.scoring.reset()
        logger.info("DebateEngine reset.")

    def get_agent_token_breakdown(self) -> dict:
        return {
            "planner":       self.planner.total_tokens,
            "optimist":      self.optimist.total_tokens,
            "skeptic":       self.skeptic.total_tokens,
            "analyst":       self.analyst.total_tokens,
            "domain_expert": self.domain_expert.total_tokens,
            "critic":        self.critic.total_tokens,
            "fact_checker":  self.fact_checker.total_tokens,
            "judge":         self.judge.total_tokens,
        }