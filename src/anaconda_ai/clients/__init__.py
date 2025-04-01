from typing import Any

from ..config import AnacondaAIConfig
from .ai_navigator import AINavigatorClient
from .ollama import OllamaClient
from .base import GenericClient


def get_default_client(*args: Any, **kwargs: Any) -> GenericClient:
    config = AnacondaAIConfig()
    if config.default_backend == "ai-navigator":
        return AINavigatorClient(*args, **kwargs)
    elif config.default_backend == "ollama":
        return OllamaClient(*args, **kwargs)
    else:
        raise ValueError(f"{config.default_backend} is not supported")


__all__ = ["AINavigatorClient", "OllamaClient", "get_default_client"]
