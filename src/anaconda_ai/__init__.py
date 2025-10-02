try:
    from ._version import version as __version__
except ImportError:  # pragma: nocover
    __version__ = "unknown"

from .clients import make_client

__all__ = ["__version__", "make_client"]
