"""LLM client interfaces and provider implementations."""

from c3ae.llm.backends import ChatBackend, VeniceBackend
from c3ae.llm.providers import AnthropicBackend, OllamaBackend, OpenAIBackend
from c3ae.llm.venice_chat import ChatResponse, Message, VeniceChat


def create_chat_backend(provider: str = "venice", **kwargs):
    p = (provider or "venice").strip().lower()
    if p in {"venice", "default"}:
        return VeniceBackend(**kwargs)
    if p in {"openai"}:
        return OpenAIBackend(**kwargs)
    if p in {"anthropic"}:
        return AnthropicBackend(**kwargs)
    if p in {"ollama", "local"}:
        return OllamaBackend(**kwargs)
    raise ValueError(f"Unsupported provider: {provider}")


__all__ = [
    "ChatBackend",
    "VeniceBackend",
    "OpenAIBackend",
    "AnthropicBackend",
    "OllamaBackend",
    "VeniceChat",
    "Message",
    "ChatResponse",
    "create_chat_backend",
]
