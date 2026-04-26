# =============================================================================
# memory/memory_schema.py
# Data models for short-term and long-term memory
# =============================================================================

import uuid
from datetime import datetime


def new_debate_record(
    topic:      str,
    domain:     str,
    agenda:     str,
    rounds:     list,
    scores:     dict,
    verdict:    dict,
    all_flags:  list,
    duration_sec: float,
    total_tokens: int,
) -> dict:
    """
    Creates a standardised debate record for long-term storage.

    Schema:
        debate_id     — unique UUID
        timestamp     — ISO 8601 string
        topic         — original user query
        domain        — finance / business_strategy / policy_ethics / technology
        agenda        — planner's structured agenda string
        rounds        — full round data [{round, arguments, critic, fact_checker}]
        scores        — ScoringEngine summary dict
        verdict       — Judge verdict dict
        all_flags     — flat list of fact-check results
        duration_sec  — wall-clock time for the debate
        total_tokens  — total API tokens consumed
        summary       — one-line auto-generated summary for memory search
    """
    decision  = verdict.get("final_decision", "")
    winner    = verdict.get("winning_agent",  "unknown")
    confidence= verdict.get("confidence_score", 0)

    summary = (
        f"Topic: {topic[:80]} | "
        f"Decision: {decision[:80]} | "
        f"Winner: {winner} | "
        f"Confidence: {confidence}%"
    )

    return {
        "debate_id":    str(uuid.uuid4()),
        "timestamp":    datetime.utcnow().isoformat(),
        "topic":        topic,
        "domain":       domain,
        "agenda":       agenda,
        "rounds":       rounds,
        "scores":       scores,
        "verdict":      verdict,
        "all_flags":    all_flags,
        "duration_sec": duration_sec,
        "total_tokens": total_tokens,
        "summary":      summary,
    }


def new_session() -> dict:
    """
    Creates a blank short-term session memory dict.
    Resets between debates within the same app session.
    """
    return {
        "session_id":    str(uuid.uuid4()),
        "started_at":    datetime.utcnow().isoformat(),
        "topic":         "",
        "domain":        "",
        "agenda":        "",
        "rag_context":   "",
        "round_outputs": {},    # {round_num: {agent: text}}
        "critic_outputs":{},    # {round_num: critic AgentResult dict}
        "fc_outputs":    {},    # {round_num: fact_checker AgentResult dict}
        "scores":        {},
        "verdict":       {},
        "all_flags":     [],
        "status":        "idle",    # idle | running | complete | error
        "progress_log":  [],        # [(stage, message), ...]
    }