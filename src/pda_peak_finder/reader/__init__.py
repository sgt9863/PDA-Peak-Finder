"""File readers. Import this package to register all built-in readers."""

from __future__ import annotations

from .arw import ArwReader
from .base import (
    SpectralDataReader,
    get_reader,
    load,
    load_many,
    register_reader,
    registered_readers,
)

# Register built-in readers. New formats: add the import + register call here.
register_reader(ArwReader)

__all__ = [
    "SpectralDataReader",
    "ArwReader",
    "register_reader",
    "registered_readers",
    "get_reader",
    "load",
    "load_many",
]
