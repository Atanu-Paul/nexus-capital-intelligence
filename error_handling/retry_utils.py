"""Shared retry/backoff decorator used across agents."""

from collections.abc import Callable
from typing import TypeVar

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

F = TypeVar("F", bound=Callable)


def is_rate_limit_error(exc: BaseException) -> bool:
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


def _should_retry_transient(exc: BaseException) -> bool:
    return not is_rate_limit_error(exc)


def retry_transient(func: F) -> F:
    """Retry transient failures up to 3 times; never retry rate-limit errors."""
    return retry(
        retry=retry_if_exception(_should_retry_transient),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )(func)
