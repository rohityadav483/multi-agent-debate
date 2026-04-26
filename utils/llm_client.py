# =============================================================================
# utils/llm_client.py
# Groq API wrapper — single point of contact for all LLM calls
# =============================================================================

import time
import json
import requests
from typing import Optional
from backend.config import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    PRIMARY_MODEL,
    FAST_MODEL,
    DEFAULT_TEMPERATURE,
    MAX_TOKENS_DEBATE,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Retry Configuration
# -----------------------------------------------------------------------------

MAX_RETRIES     = 3
RETRY_DELAY_SEC = 2     # Seconds between retries (doubles each attempt)
REQUEST_TIMEOUT = 60    # Seconds before request times out


# =============================================================================
# Core LLM Client Class
# =============================================================================

class LLMClient:
    """
    Wrapper around the Groq Chat Completions API.

    Usage:
        client = LLMClient()
        response = client.chat(
            system_prompt="You are an expert analyst.",
            user_message="Analyse the risks of this investment.",
            temperature=0.5,
        )
        print(response.content)     # The agent's reply text
        print(response.tokens_used) # Token usage for monitoring
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GROQ_API_KEY
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file: GROQ_API_KEY=your_key_here"
            )
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

    # -------------------------------------------------------------------------
    # Primary call method
    # -------------------------------------------------------------------------

    def chat(
        self,
        system_prompt:  str,
        user_message:   str,
        model:          Optional[str]   = None,
        temperature:    Optional[float] = None,
        max_tokens:     Optional[int]   = None,
        conversation_history: Optional[list] = None,
    ) -> "LLMResponse":
        """
        Send a chat completion request to Groq.

        Args:
            system_prompt:        The agent's persona/instructions.
            user_message:         The current input/task for the agent.
            model:                Override model (defaults to PRIMARY_MODEL).
            temperature:          Override temperature (defaults to DEFAULT_TEMPERATURE).
            max_tokens:           Override max tokens (defaults to MAX_TOKENS_DEBATE).
            conversation_history: List of prior {"role": ..., "content": ...} turns.
                                  Used to give agents memory of earlier rounds.

        Returns:
            LLMResponse with .content (str), .tokens_used (int), .model (str)
        """

        model       = model       or PRIMARY_MODEL
        temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        max_tokens  = max_tokens  or MAX_TOKENS_DEBATE

        # Build message list
        messages = [{"role": "system", "content": system_prompt}]

        # Inject prior conversation turns if provided (multi-round memory)
        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_message})

        payload = {
            "model":       model,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }

        return self._call_with_retry(payload, model)

    # -------------------------------------------------------------------------
    # Retry logic
    # -------------------------------------------------------------------------

    def _call_with_retry(self, payload: dict, model: str) -> "LLMResponse":
        """
        Attempt the API call up to MAX_RETRIES times with exponential backoff.
        On final failure, returns a safe fallback LLMResponse.
        """
        delay = RETRY_DELAY_SEC

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(
                    f"LLM call | model={model} | attempt={attempt}/{MAX_RETRIES}"
                )
                response = requests.post(
                    GROQ_BASE_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                return self._parse_response(response, model)

            except requests.exceptions.Timeout:
                logger.warning(f"Request timed out (attempt {attempt})")

            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error (attempt {attempt})")

            except requests.exceptions.RequestException as e:
                logger.error(f"Request exception: {e} (attempt {attempt})")

            if attempt < MAX_RETRIES:
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff

        # All retries exhausted
        logger.error("All retry attempts failed. Returning fallback response.")
        return LLMResponse(
            content="[Agent unavailable — API call failed after all retries.]",
            tokens_used=0,
            model=model,
            success=False,
        )

    # -------------------------------------------------------------------------
    # Response parser
    # -------------------------------------------------------------------------

    def _parse_response(self, response: requests.Response, model: str) -> "LLMResponse":
        """
        Parse the raw HTTP response from Groq into an LLMResponse object.
        Handles API-level errors (rate limits, auth failures, etc.).
        """
        # HTTP-level errors
        if response.status_code == 401:
            raise ValueError("Invalid GROQ_API_KEY. Check your .env file.")

        if response.status_code == 429:
            raise requests.exceptions.RequestException(
                "Rate limit hit. Backing off..."
            )

        if response.status_code != 200:
            raise requests.exceptions.RequestException(
                f"Groq API error {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except json.JSONDecodeError:
            raise requests.exceptions.RequestException(
                f"Could not parse JSON response: {response.text}"
            )

        # Extract content
        content     = data["choices"][0]["message"]["content"].strip()
        tokens_used = data.get("usage", {}).get("total_tokens", 0)

        logger.info(f"LLM response received | tokens={tokens_used}")

        return LLMResponse(
            content=content,
            tokens_used=tokens_used,
            model=model,
            success=True,
        )

    # -------------------------------------------------------------------------
    # Convenience wrappers
    # -------------------------------------------------------------------------

    def fast_chat(
        self,
        system_prompt: str,
        user_message:  str,
        temperature:   float = 0.3,
        max_tokens:    int   = 400,
    ) -> "LLMResponse":
        """
        Use the lighter FAST_MODEL for scoring, fact-checking, and
        other tasks that don't need the full 70B model.
        """
        return self.chat(
            system_prompt=system_prompt,
            user_message=user_message,
            model=FAST_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def extract_json(
        self,
        system_prompt: str,
        user_message:  str,
        model:         Optional[str] = None,
    ) -> dict:
        """
        Request a JSON-only response from the LLM and parse it safely.
        Used by the Critic and Judge agents for structured outputs.

        Returns:
            Parsed dict, or {"error": "..."} if parsing fails.
        """
        json_system = (
            system_prompt
            + "\n\nCRITICAL: Respond ONLY with valid JSON. "
            "No explanation, no markdown, no code fences. "
            "Raw JSON object only."
        )
        response = self.chat(
            system_prompt=json_system,
            user_message=user_message,
            model=model or FAST_MODEL,
            temperature=0.1,    # Near-zero for deterministic JSON
            max_tokens=1200,
        )

        try:
            # Strip any accidental markdown fences
            raw = response.content
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e} | raw={response.content[:200]}")
            return {"error": f"JSON parse failed: {str(e)}", "raw": response.content}


# =============================================================================
# LLMResponse Data Class
# =============================================================================

class LLMResponse:
    """
    Structured response object returned by all LLM calls.

    Attributes:
        content:     The text response from the model.
        tokens_used: Total tokens consumed (prompt + completion).
        model:       Model that generated the response.
        success:     False if all retries failed.
    """

    def __init__(
        self,
        content:     str,
        tokens_used: int,
        model:       str,
        success:     bool = True,
    ):
        self.content     = content
        self.tokens_used = tokens_used
        self.model       = model
        self.success     = success

    def __repr__(self):
        preview = self.content[:80].replace("\n", " ")
        return (
            f"LLMResponse(success={self.success}, "
            f"tokens={self.tokens_used}, "
            f"model={self.model}, "
            f"content='{preview}...')"
        )


# =============================================================================
# Module-level singleton — all agents share one client instance
# =============================================================================

_client_instance: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """
    Returns a shared LLMClient singleton.
    Avoids re-initializing on every agent call.
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = LLMClient()
    return _client_instance