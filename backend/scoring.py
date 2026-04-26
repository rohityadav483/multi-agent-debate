# =============================================================================
# backend/scoring.py
# Aggregates Critic scores across rounds → final leaderboard
# =============================================================================

from backend.config import ROUND_WEIGHTS, TOTAL_MAX_SCORE, SCORING_METRICS
from utils.logger import get_logger

logger = get_logger(__name__)


class ScoringEngine:
    """
    Consumes per-round Critic scores and produces:
        - Weighted total per agent across all rounds
        - Per-metric breakdown
        - Ranked leaderboard
        - Debate winner

    Usage:
        engine = ScoringEngine()
        engine.add_round_scores(round=1, scores={"optimist": 81, ...})
        engine.add_round_scores(round=2, scores={"optimist": 75, ...})
        engine.add_round_scores(round=3, scores={"optimist": 88, ...})
        summary = engine.get_summary()
    """

    def __init__(self):
        # {round_number: {agent_name: total_score}}
        self._round_scores: dict[int, dict[str, int]] = {}

        # {round_number: {agent_name: {metric: score}}}
        self._round_details: dict[int, dict[str, dict]] = {}

    # -------------------------------------------------------------------------
    # Ingest
    # -------------------------------------------------------------------------

    def add_round_scores(
        self,
        round_number: int,
        scores: dict[str, int],
        details: dict[str, dict] | None = None,
    ):
        """
        Register scores for a completed round.

        Args:
            round_number: 1, 2, or 3
            scores:  {agent_name: round_total (0-100)}
            details: {agent_name: {metric: score}} — optional per-metric breakdown
        """
        self._round_scores[round_number]  = scores
        self._round_details[round_number] = details or {}
        logger.info(f"Scores added | round={round_number} | {scores}")

    def add_round_from_critic(self, round_number: int, critic_result):
        """
        Convenience — extract scores directly from a CriticAgent AgentResult.
        Avoids boilerplate in the debate engine.
        """
        from agents.critic import CriticAgent

        # Build a throw-away critic instance just to call extract_round_scores
        scores  = {}
        details = {}

        raw = critic_result.metadata.get("scores", {})
        for agent, data in raw.items():
            if not isinstance(data, dict):
                continue
            lc   = data.get("logical_consistency", 0)
            ev   = data.get("evidence_usage",      0)
            rel  = data.get("relevance",            0)
            pers = data.get("persuasiveness",       0)
            scores[agent]  = lc + ev + rel + pers
            details[agent] = {
                "logical_consistency": lc,
                "evidence_usage":      ev,
                "relevance":           rel,
                "persuasiveness":      pers,
                "justification":       data.get("justification", ""),
            }

        self.add_round_scores(round_number, scores, details)

    # -------------------------------------------------------------------------
    # Compute
    # -------------------------------------------------------------------------

    def get_weighted_totals(self) -> dict[str, float]:
        """
        Apply ROUND_WEIGHTS and sum across rounds.

        Formula:
            weighted_total = Σ (round_score × round_weight) for each round
            Max possible   = 100 × (0.25 + 0.35 + 0.40) = 100.0

        Returns:
            {agent_name: weighted_total}  — floats, rounded to 1 dp
        """
        all_agents = self._get_all_agents()
        totals = {agent: 0.0 for agent in all_agents}

        for round_num, scores in self._round_scores.items():
            weight = ROUND_WEIGHTS.get(round_num, 0.33)
            for agent in all_agents:
                totals[agent] += scores.get(agent, 0) * weight

        return {a: round(t, 1) for a, t in totals.items()}

    def get_leaderboard(self) -> list[dict]:
        """
        Returns agents ranked by weighted total, highest first.

        Example:
            [
                {"rank": 1, "agent": "analyst",      "score": 87.5, "badge": "🥇"},
                {"rank": 2, "agent": "domain_expert","score": 82.0, "badge": "🥈"},
                ...
            ]
        """
        totals  = self.get_weighted_totals()
        ranked  = sorted(totals.items(), key=lambda x: x[1], reverse=True)
        badges  = ["🥇", "🥈", "🥉", "4️⃣"]

        return [
            {
                "rank":  i + 1,
                "agent": agent,
                "score": score,
                "badge": badges[i] if i < len(badges) else f"{i+1}.",
            }
            for i, (agent, score) in enumerate(ranked)
        ]

    def get_winner(self) -> str:
        """Returns the name of the highest-scoring agent."""
        lb = self.get_leaderboard()
        return lb[0]["agent"] if lb else "unknown"

    def get_per_metric_summary(self) -> dict[str, dict[str, float]]:
        """
        Averages each metric per agent across all rounds.

        Returns:
            {
                agent_name: {
                    "logical_consistency": avg,
                    "evidence_usage":      avg,
                    "relevance":           avg,
                    "persuasiveness":      avg,
                }
            }
        """
        all_agents = self._get_all_agents()
        metric_sums   = {a: {m: 0.0 for m in SCORING_METRICS} for a in all_agents}
        metric_counts = {a: 0 for a in all_agents}

        for round_num, details in self._round_details.items():
            for agent, data in details.items():
                if not isinstance(data, dict):
                    continue
                for metric in SCORING_METRICS:
                    metric_sums[agent][metric] += data.get(metric, 0)
                metric_counts[agent] += 1

        averages = {}
        for agent in all_agents:
            count = metric_counts[agent] or 1
            averages[agent] = {
                m: round(metric_sums[agent][m] / count, 1)
                for m in SCORING_METRICS
            }

        return averages

    def get_summary(self) -> dict:
        """
        Master summary dict consumed by the Judge and UI.

        Returns:
            {
                "leaderboard":       [...],
                "weighted_totals":   {agent: score},
                "per_metric":        {agent: {metric: avg}},
                "winner":            "analyst",
                "rounds_scored":     3,
                "raw_round_scores":  {1: {...}, 2: {...}, 3: {...}},
            }
        """
        return {
            "leaderboard":      self.get_leaderboard(),
            "weighted_totals":  self.get_weighted_totals(),
            "per_metric":       self.get_per_metric_summary(),
            "winner":           self.get_winner(),
            "rounds_scored":    len(self._round_scores),
            "raw_round_scores": self._round_scores,
        }

    def reset(self):
        """Clear all scores — call between debates."""
        self._round_scores.clear()
        self._round_details.clear()
        logger.info("ScoringEngine reset.")

    # -------------------------------------------------------------------------
    # Private
    # -------------------------------------------------------------------------

    def _get_all_agents(self) -> list[str]:
        """Collect unique agent names seen across all rounds."""
        agents = set()
        for scores in self._round_scores.values():
            agents.update(scores.keys())
        return sorted(agents)