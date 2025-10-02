from typing import Any, Optional

from ..config import AnacondaAIConfig
from .base import GenericClient
from .ai_catalyst import AICatalystClient
from .ollama import OllamaClient
from .anaconda_desktop import AnacondaDesktopClient

clients = {
    "ai-catalyst": AICatalystClient,
    "ollama": OllamaClient,
    "anaconda-desktop": AnacondaDesktopClient,
}


def make_client(
    backend: Optional[str] = None, site: Optional[str] = None, **kwargs: Any
) -> GenericClient:
    if backend is None:
        config = AnacondaAIConfig()
        return clients[config.backend](site=site, **kwargs)
    else:
        return clients[backend](site=site, **kwargs)


__all__ = ["make_client"]
