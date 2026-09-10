"""LLM router with provider/model fallback chain."""

import logging

from llm.gemini_client import GeminiClient
from llm.groq_client import GroqClient

logger = logging.getLogger(__name__)


class LLMRouter:
    """Routes JSON generation requests across Gemini models with Groq fallback."""

    def generate_json(self, prompt: str) -> dict:
        """Generate a JSON response using a fallback chain of LLM providers.

        Tries gemini-3.5-flash, then gemini-3.5-flash-lite, then Groq.

        Args:
            prompt: The input prompt to send to the model.

        Returns:
            The parsed JSON response as a Python dict.

        Raises:
            RuntimeError: If all providers fail.
        """
        errors: list[str] = []

        try:
            return GeminiClient(model_name="gemini-3.5-flash").generate_json(prompt)
        except Exception as exc:
            logger.warning(
                "Gemini flash failed, falling back to flash-lite: %s", exc
            )
            errors.append(f"gemini-3.5-flash: {exc}")

        try:
            return GeminiClient(model_name="gemini-3.5-flash-lite").generate_json(
                prompt
            )
        except Exception as exc:
            logger.warning(
                "Gemini flash-lite failed, falling back to Groq: %s", exc
            )
            errors.append(f"gemini-3.5-flash-lite: {exc}")

        try:
            return GroqClient().generate_json(prompt)
        except Exception as exc:
            errors.append(f"groq llama-3.3-70b-versatile: {exc}")
            raise RuntimeError(
                "All LLM providers failed:\n" + "\n".join(errors)
            ) from exc
