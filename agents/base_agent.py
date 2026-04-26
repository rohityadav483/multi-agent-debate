# =============================================================================
# agents/base_agent.py
# Abstract base class that every agent inherits from.
# Defines the shared contract: build_prompt() → call_llm() → run()
# =============================================================================

from abc import ABC, abstractmethod
from typing import Optional
from utils.llm_client import get_llm_client, LLMResponse
from utils.logger import get_logger
from backend.config import (
    DEFAULT_TEMPERATURE,
    MAX_TOKENS_DEBATE,
)

logger = get_logger(__name__)


# =============================================================================
# AgentResult — Standardised output from every agent call
# =============================================================================

class AgentResult:
    """
    Uniform output structure returned by every agent's run() method.
    The debate engine and scoring system always receive this object.

    Attributes:
        agent_name:   Which agent produced this result
        content:      The agent's actual response text
        round_number: Which debate round this belongs to (None for non-debate agents)
        metadata:     Extra structured data (scores, fact-checks, decision, etc.)
        tokens_used:  Token count for monitoring API usage
        success:      False if the LLM call failed
    """

    def __init__(
        self,
        agent_name:   str,
        content:      str,
        round_number: Optional[int]  = None,
        metadata:     Optional[dict] = None,
        tokens_used:  int            = 0,
        success:      bool           = True,
    ):
        self.agent_name   = agent_name
        self.content      = content
        self.round_number = round_number
        self.metadata     = metadata or {}
        self.tokens_used  = tokens_used
        self.success      = success

    def __repr__(self):
        preview = self.content[:80].replace("\n", " ")
        return (
            f"AgentResult("
            f"agent={self.agent_name}, "
            f"round={self.round_number}, "
            f"success={self.success}, "
            f"tokens={self.tokens_used}, "
            f"content='{preview}...')"
        )

    def to_dict(self) -> dict:
        """Serialise to dict for memory storage and UI rendering."""
        return {
            "agent_name":   self.agent_name,
            "content":      self.content,
            "round_number": self.round_number,
            "metadata":     self.metadata,
            "tokens_used":  self.tokens_used,
            "success":      self.success,
        }


# =============================================================================
# BaseAgent — Abstract class all agents inherit from
# =============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all debate system agents.

    Every concrete agent must implement:
        build_prompt()  ->  (system_prompt: str, user_message: str)
        run()           ->  AgentResult

    Shared behaviour provided by BaseAgent:
        - LLM client access via self.llm
        - Logging with agent-specific logger
        - _call() wrapper with error handling
        - _call_json() for structured outputs
        - _call_fast() for lightweight tasks
        - Token tracking across calls
    """

    # Each subclass sets its own name — used for logging, scoring, and UI labels
    name: str = "base"

    def __init__(self, prompt_builder):
        """
        Args:
            prompt_builder: An initialised PromptBuilder instance.
                            Passed in from the debate engine so all agents
                            share the same domain context.
        """
        self.prompt_builder = prompt_builder
        self.llm            = get_llm_client()
        self.logger         = get_logger(f"agent.{self.name}")
        self._total_tokens  = 0

    # -------------------------------------------------------------------------
    # Abstract interface — subclasses MUST implement these
    # -------------------------------------------------------------------------

    @abstractmethod
    def build_prompt(self, **kwargs) -> tuple[str, str]:
        """
        Build and return (system_prompt, user_message) for this agent.
        All kwargs come from the debate engine's call to run().
        """
        ...

    @abstractmethod
    def run(self, **kwargs) -> AgentResult:
        """
        Execute the agent's full logic and return an AgentResult.
        Subclasses call self._call() or self._call_json() internally.
        """
        ...

    # -------------------------------------------------------------------------
    # Shared LLM call wrappers — subclasses call these, not self.llm directly
    # -------------------------------------------------------------------------

    def _call(
        self,
        system_prompt:        str,
        user_message:         str,
        round_number:         Optional[int]   = None,
        temperature:          Optional[float] = None,
        max_tokens:           Optional[int]   = None,
        conversation_history: Optional[list]  = None,
    ) -> AgentResult:
        """
        Standard LLM call returning free-form text.
        Used by: Optimist, Skeptic, Analyst, Domain Expert, Planner
        """
        self.logger.info(
            f"Running | round={round_number} | "
            f"temp={temperature or DEFAULT_TEMPERATURE}"
        )

        try:
            response: LLMResponse = self.llm.chat(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=temperature,
                max_tokens=max_tokens or MAX_TOKENS_DEBATE,
                conversation_history=conversation_history,
            )

            self._total_tokens += response.tokens_used
            self.logger.info(
                f"Done | tokens={response.tokens_used} | "
                f"cumulative={self._total_tokens}"
            )

            return AgentResult(
                agent_name=self.name,
                content=response.content,
                round_number=round_number,
                tokens_used=response.tokens_used,
                success=response.success,
            )

        except Exception as e:
            self.logger.error(f"Unexpected error in _call(): {e}")
            return self._error_result(str(e), round_number)

    def _call_json(
        self,
        system_prompt: str,
        user_message:  str,
        round_number:  Optional[int] = None,
    ) -> AgentResult:
        """
        JSON-mode LLM call — forces structured output and parses it.
        Used by: Critic, Fact-Checker, Judge

        Parsed JSON is stored in AgentResult.metadata.
        AgentResult.content holds a human-readable summary.
        """
        self.logger.info(f"Running (JSON mode) | round={round_number}")

        try:
            parsed: dict = self.llm.extract_json(
                system_prompt=system_prompt,
                user_message=user_message,
            )

            if "error" in parsed:
                self.logger.warning(f"JSON parse failed: {parsed['error']}")
                return AgentResult(
                    agent_name=self.name,
                    content=parsed.get("raw", "[JSON parse failed]"),
                    round_number=round_number,
                    metadata={"parse_error": parsed["error"]},
                    success=False,
                )

            content = self._json_to_readable(parsed)
            self.logger.info(f"JSON call done | keys={list(parsed.keys())}")

            return AgentResult(
                agent_name=self.name,
                content=content,
                round_number=round_number,
                metadata=parsed,
                success=True,
            )

        except Exception as e:
            self.logger.error(f"Unexpected error in _call_json(): {e}")
            return self._error_result(str(e), round_number)

    def _call_fast(
        self,
        system_prompt: str,
        user_message:  str,
        round_number:  Optional[int] = None,
        temperature:   float = 0.3,
        max_tokens:    int   = 400,
    ) -> AgentResult:
        """
        Fast-model call using the lighter 8B model.
        Used for: quick sub-tasks, low-complexity reasoning.
        """
        self.logger.info(f"Running (fast model) | round={round_number}")

        try:
            response: LLMResponse = self.llm.fast_chat(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            self._total_tokens += response.tokens_used

            return AgentResult(
                agent_name=self.name,
                content=response.content,
                round_number=round_number,
                tokens_used=response.tokens_used,
                success=response.success,
            )

        except Exception as e:
            self.logger.error(f"Unexpected error in _call_fast(): {e}")
            return self._error_result(str(e), round_number)

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    def _error_result(
        self,
        error_msg:    str,
        round_number: Optional[int],
    ) -> AgentResult:
        """Returns a safe AgentResult when something goes wrong."""
        return AgentResult(
            agent_name=self.name,
            content=(
                f"[{self.name.upper()} encountered an error: {error_msg}]"
            ),
            round_number=round_number,
            metadata={"error": error_msg},
            success=False,
        )

    def _json_to_readable(self, parsed: dict) -> str:
        """
        Converts a parsed JSON dict into a human-readable string for the UI.
        Subclasses can override this for custom formatting.
        """
        import json
        return json.dumps(parsed, indent=2)

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed by this agent across all calls."""
        return self._total_tokens

    def reset_tokens(self):
        """Reset token counter — called between debates."""
        self._total_tokens = 0

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(name={self.name}, tokens={self._total_tokens})"
        )