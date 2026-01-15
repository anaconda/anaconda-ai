try:
    from ._version import version as __version__
except ImportError:  # pragma: nocover
    __version__ = "unknown"

from .clients import AnacondaAIClient, get_default_client, get_backends

__all__ = ["__version__", "AnacondaAIClient", "get_default_client", "get_backends"]
