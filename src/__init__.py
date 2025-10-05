"""Top level package for the dierpian multimodal retrieval toolkit."""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("dierpian")
except _metadata.PackageNotFoundError:  # pragma: no cover - during local usage without installation
    __version__ = "0.0.0"

__all__ = ["__version__"]
