import os
import time
from json import JSONDecodeError
from typing import Any, Optional

from langchain_openai import ChatOpenAI

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


class NormalizedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with normalized content output.

    The Responses API returns content as a list of typed blocks
    (reasoning, text, etc.). This normalizes to string for consistent
    downstream handling.
    """

    def __init__(
        self,
        *args,
        provider_name: str = "openai",
        transport_retries: int = 0,
        transport_retry_delay: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.provider_name = provider_name
        self.transport_retries = max(0, int(transport_retries))
        self.transport_retry_delay = float(transport_retry_delay)

    def invoke(self, input, config=None, **kwargs):
        attempt = 0
        delay = self.transport_retry_delay
        while True:
            try:
                return normalize_content(super().invoke(input, config, **kwargs))
            except Exception as exc:
                if (
                    attempt >= self.transport_retries
                    or not _is_transient_compat_error(exc)
                ):
                    raise
                attempt += 1
                print(
                    f"[LLM-RETRY] provider={self.provider_name} "
                    f"attempt={attempt}/{self.transport_retries} "
                    f"error={exc.__class__.__name__}: {exc}",
                    flush=True,
                )
                time.sleep(delay)
                delay *= 2


def _is_transient_compat_error(exc: Exception) -> bool:
    """Best-effort retry filter for flaky OpenAI-compatible backends."""
    transient_names = {
        "JSONDecodeError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "RateLimitError",
    }
    if exc.__class__.__name__ in transient_names:
        return True
    if isinstance(exc, JSONDecodeError):
        return True
    return False

# Kwargs forwarded from user config to ChatOpenAI
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "reasoning_effort",
    "api_key", "callbacks", "http_client", "http_async_client",
)

# Provider base URLs and API key env vars
_PROVIDER_CONFIG = {
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "modelscope": ("https://api-inference.modelscope.cn/v1", "MODELSCOPE_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI, Ollama, OpenRouter, and xAI providers.

    For native OpenAI models, uses the Responses API (/v1/responses) which
    supports reasoning_effort with function tools across all model families
    (GPT-4.1, GPT-5). Third-party compatible providers (xAI, OpenRouter,
    Ollama) use standard Chat Completions.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return configured ChatOpenAI instance."""
        llm_kwargs = {"model": self.model}

        # Provider-specific base URL and auth
        if self.provider in _PROVIDER_CONFIG:
            base_url, api_key_env = _PROVIDER_CONFIG[self.provider]
            llm_kwargs["base_url"] = base_url
            if api_key_env:
                api_key = os.environ.get(api_key_env)
                if api_key:
                    llm_kwargs["api_key"] = api_key
            else:
                llm_kwargs["api_key"] = "ollama"
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        # Forward user-provided kwargs
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Native OpenAI: use Responses API for consistent behavior across
        # all model families. Third-party compatible providers use Chat Completions.
        if self.provider == "openai":
            llm_kwargs["use_responses_api"] = True

        if self.provider != "openai":
            llm_kwargs["provider_name"] = self.provider
            llm_kwargs["transport_retries"] = self.kwargs.get("transport_retries", 2)
            llm_kwargs["transport_retry_delay"] = self.kwargs.get("transport_retry_delay", 2.0)

        return NormalizedChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for the provider."""
        return validate_model(self.provider, self.model)
