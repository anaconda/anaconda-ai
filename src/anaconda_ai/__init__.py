try:
    from ._version import version as __version__
except ImportError:  # pragma: nocover
    __version__ = "unknown"

from .clients import get_default_client

__all__ = ["__version__", "get_default_client"]
