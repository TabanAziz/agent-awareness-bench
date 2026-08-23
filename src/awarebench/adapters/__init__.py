"""Model adapters: deterministic stub plus thin vendor wrappers."""

from awarebench.adapters.anthropic import AnthropicAdapter
from awarebench.adapters.base import AdapterError, AdapterResponse, ModelAdapter
from awarebench.adapters.openai import OpenAIAdapter
from awarebench.adapters.stub import StubAdapter

__all__ = [
    "AdapterError",
    "AdapterResponse",
    "AnthropicAdapter",
    "ModelAdapter",
    "OpenAIAdapter",
    "StubAdapter",
]
