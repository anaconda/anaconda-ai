from typing import Any

from ..config import AnacondaModelsConfig
from .ai_navigator import AINavigatorClient
from .base import GenericClient


def get_default_client(*args: Any, **kwargs: Any) -> GenericClient:
    config = AnacondaModelsConfig()
    if config.default_backend == "ai-navigator":
        return AINavigatorClient(*args, **kwargs)
    else:
        raise ValueError(f"{config.default_backend} is not supported")


__all__ = ["AINavigatorClient", "get_default_client"]
