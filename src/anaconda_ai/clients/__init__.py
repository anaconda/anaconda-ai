from typing import Any, Optional

from ..config import AnacondaAIConfig
from .base import GenericClient
from .ai_catalog import AICatalogClient
from .ollama import OllamaClient

clients = {"ai-catalog": AICatalogClient, "ollama": OllamaClient}


def make_client(
    backend: Optional[str] = None, site: Optional[str] = None, **kwargs: Any
) -> GenericClient:
    if backend is None:
        config = AnacondaAIConfig()
        return clients[config.backend](site=site, **kwargs)
    else:
        return clients[backend](site=site, **kwargs)


__all__ = ["make_client"]
