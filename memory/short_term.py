# =============================================================================
# memory/short_term.py
# Short-term session memory — holds live state during a single debate
# Backed by a plain Python dict (lives in Streamlit session_state)
# =============================================================================

from datetime import datetime
from memory.memory_schema import new_session
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# ShortTermMemory
# =============================================================================

class ShortTermMemory:
    """
    Manages the in-progress state of a single debate.

    The debate engine writes to this as each stage completes.
    The Streamlit UI reads from this to render live updates.

    Lifecycle:
        mem = ShortTermMemory()
        mem.start_debate(topic, domain)
        mem.set_agenda(agenda_text)
        mem.add_round_output(1, "optimist", "argument text")
        mem.add_critic_output(1, critic_result_dict)
        mem.add_fc_output(1, fc_result_dict)
        mem.set_scores(scores_dict)
        mem.set_verdict(verdict_dict)
        mem.complete()
        snapshot = mem.snapshot()   # passed to LongTermMemory.save()
        mem.reset()                 # ready for next debate
    """

    def __init__(self):
        self._state = new_session()

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def start_debate(self, topic: str, domain: str, rag_context: str = ""):
        """Call when the user clicks Run Debate."""
        self._state["topic"]       = topic
        self._state["domain"]      = domain
        self._state["rag_context"] = rag_context
        self._state["status"]      = "running"
        self._state["started_at"]  = datetime.utcnow().isoformat()
        logger.info(f"ShortTermMemory: debate started | topic='{topic[:60]}'")

    def complete(self):
        self._state["status"] = "complete"
        logger.info("ShortTermMemory: debate complete.")

    def set_error(self, error_msg: str):
        self._state["status"] = "error"
        self._state["error"]  = error_msg

    def reset(self):
        """Clear all state — call before starting a new debate."""
        self._state = new_session()
        logger.info("ShortTermMemory: reset.")

    # -------------------------------------------------------------------------
    # Writes
    # -------------------------------------------------------------------------

    def set_agenda(self, agenda: str):
        self._state["agenda"] = agenda

    def add_round_output(self, round_num: int, agent_name: str, text: str):
        """Store one agent's argument for a given round."""
        if round_num not in self._state["round_outputs"]:
            self._state["round_outputs"][round_num] = {}
        self._state["round_outputs"][round_num][agent_name] = text

    def add_critic_output(self, round_num: int, critic_dict: dict):
        self._state["critic_outputs"][round_num] = critic_dict

    def add_fc_output(self, round_num: int, fc_dict: dict):
        self._state["fc_outputs"][round_num] = fc_dict

    def set_scores(self, scores: dict):
        self._state["scores"] = scores

    def set_verdict(self, verdict: dict):
        self._state["verdict"] = verdict

    def add_flags(self, flags: list):
        self._state["all_flags"].extend(flags)

    def log_progress(self, stage: str, message: str):
        """Append a progress event — consumed by the UI status widget."""
        self._state["progress_log"].append({
            "stage":     stage,
            "message":   message,
            "timestamp": datetime.utcnow().isoformat(),
        })

    # -------------------------------------------------------------------------
    # Reads
    # -------------------------------------------------------------------------

    @property
    def topic(self)    -> str:  return self._state["topic"]

    @property
    def domain(self)   -> str:  return self._state["domain"]

    @property
    def agenda(self)   -> str:  return self._state["agenda"]

    @property
    def status(self)   -> str:  return self._state["status"]

    @property
    def scores(self)   -> dict: return self._state["scores"]

    @property
    def verdict(self)  -> dict: return self._state["verdict"]

    @property
    def all_flags(self)-> list: return self._state["all_flags"]

    @property
    def is_complete(self) -> bool:
        return self._state["status"] == "complete"

    @property
    def is_running(self) -> bool:
        return self._state["status"] == "running"

    def get_round_outputs(self, round_num: int) -> dict:
        """Returns {agent_name: argument_text} for a given round."""
        return self._state["round_outputs"].get(round_num, {})

    def get_all_round_outputs(self) -> dict:
        """Returns {round_num: {agent_name: text}} for all rounds."""
        return self._state["round_outputs"]

    def get_critic_output(self, round_num: int) -> dict:
        return self._state["critic_outputs"].get(round_num, {})

    def get_fc_output(self, round_num: int) -> dict:
        return self._state["fc_outputs"].get(round_num, {})

    def get_progress_log(self) -> list:
        return self._state["progress_log"]

    def get_latest_progress(self) -> str:
        """Returns the most recent progress message for the UI status bar."""
        log = self._state["progress_log"]
        return log[-1]["message"] if log else ""

    # -------------------------------------------------------------------------
    # Snapshot
    # -------------------------------------------------------------------------

    def snapshot(self) -> dict:
        """
        Return a full copy of the current state.
        Passed to LongTermMemory.save() after a debate completes.
        """
        import copy
        return copy.deepcopy(self._state)

    def get_debate_result_fields(self) -> dict:
        """
        Returns the fields needed to build a long-term memory record.
        Maps short-term state keys to the memory_schema expected format.
        """
        rounds = []
        for round_num in sorted(self._state["round_outputs"].keys()):
            rounds.append({
                "round":        round_num,
                "arguments":    self._state["round_outputs"].get(round_num, {}),
                "critic":       self._state["critic_outputs"].get(round_num, {}),
                "fact_checker": self._state["fc_outputs"].get(round_num, {}),
            })

        return {
            "topic":       self._state["topic"],
            "domain":      self._state["domain"],
            "agenda":      self._state["agenda"],
            "rounds":      rounds,
            "scores":      self._state["scores"],
            "verdict":     self._state["verdict"],
            "all_flags":   self._state["all_flags"],
        }