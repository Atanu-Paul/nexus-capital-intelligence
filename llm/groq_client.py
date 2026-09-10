"""Backup LLM wrapper."""

import json
import os
import re

from dotenv import load_dotenv
from groq import Groq
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True if the exception indicates a quota or rate-limit failure."""
    if getattr(exc, "status_code", None) == 429:
        return True
    if getattr(exc, "code", None) == 429:
        return True

    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True

    message = str(exc).lower()
    return "429" in message or "quota" in message or "rate limit" in message


def _should_retry(exc: BaseException) -> bool:
    """Retry only when the failure is not quota or rate-limit related."""
    return not _is_rate_limit_error(exc)


class GroqClient:
    """Client for generating text and JSON responses via the Groq API."""

    def __init__(self) -> None:
        """Initialize the Groq client with API key from the environment."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in the environment")

        self._client = Groq(api_key=api_key)
        self._model_name = MODEL_NAME

    def generate(self, prompt: str) -> str:
        """Generate a text response for the given prompt.

        Retries up to 3 times with exponential backoff on failure.

        Args:
            prompt: The input prompt to send to the model.

        Returns:
            The raw text response from the model.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        try:
            return self._generate_with_retry(prompt)
        except RetryError as exc:
            last_exc = exc.last_attempt.exception()
            raise RuntimeError(
                f"Groq API call failed after 3 attempts: {last_exc}"
            ) from last_exc

    @retry(
        retry=retry_if_exception(_should_retry),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _generate_with_retry(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Groq API returned an empty response")
        return content

    def generate_json(self, prompt: str) -> dict:
        """Generate a JSON object response for the given prompt.

        Args:
            prompt: The input prompt to send to the model.

        Returns:
            The parsed JSON response as a Python dict.

        Raises:
            RuntimeError: If the Groq API call fails after retries.
            ValueError: If the response cannot be parsed as JSON.
        """
        raw_text = self.generate(prompt)
        cleaned = self._strip_markdown_fences(raw_text)

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse Groq response as JSON: {exc}"
            ) from exc

        if not isinstance(result, dict):
            raise ValueError(
                f"Expected JSON object, got {type(result).__name__}"
            )

        return result

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove optional markdown code fences from a string."""
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        return text.strip()
