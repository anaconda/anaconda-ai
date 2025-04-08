from typing import Any

from ..config import AnacondaAIConfig
from .ai_navigator import AINavigatorClient
from .base import GenericClient


def get_default_client(*args: Any, **kwargs: Any) -> GenericClient:
    config = AnacondaAIConfig()
    if config.backend == "ai-navigator":
        return AINavigatorClient(*args, **kwargs)
    else:
        raise ValueError(f"{config.backend} is not supported")


__all__ = ["AINavigatorClient", "get_default_client"]
