"""Reader interface and format registry.

Design contract
---------------
A reader converts *one file* into a validated :class:`~pda_peak_finder.models.PDAData`.
Everything downstream depends only on ``PDAData``, so adding a new file
format (ARW, generic CSV, vendor exports, ...) means adding one reader
class here and registering it — no other module changes.

To add a format:

1. Subclass :class:`SpectralDataReader`.
2. Set ``format_name`` and ``file_patterns``, implement ``sniff`` and ``read``.
3. Register it with :func:`register_reader` (done at import time in
   ``reader/__init__.py``).

``read`` must return a ``PDAData`` in canonical units (minutes / nm / AU)
with strictly increasing axes — ``PDAData.__post_init__`` enforces this,
so a reader that constructs the model normally cannot hand malformed data
to the pipeline. Readers raise :class:`~pda_peak_finder.errors.ReaderError`
for anything unparsable.
"""

from __future__ import annotations

import fnmatch
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Iterable

from ..errors import ReaderError
from ..models import PDAData

_REGISTRY: list[type["SpectralDataReader"]] = []


class SpectralDataReader(ABC):
    """Base class for all file format readers."""

    #: Human-readable format name, also used for --format on the CLI.
    format_name: ClassVar[str]
    #: Case-insensitive glob patterns matched against the file name.
    file_patterns: ClassVar[tuple[str, ...]]

    @classmethod
    def matches_name(cls, path: Path) -> bool:
        name = path.name.lower()
        return any(fnmatch.fnmatch(name, pat.lower()) for pat in cls.file_patterns)

    @classmethod
    def sniff(cls, path: Path) -> bool:
        """Cheap check whether this reader can likely handle ``path``.

        Default: file name pattern only. Readers may additionally peek at
        the first bytes/lines, but must not fully parse the file.
        """
        return cls.matches_name(path)

    @abstractmethod
    def read(self, path: Path) -> PDAData:
        """Parse ``path`` into a validated PDAData.

        Raises ReaderError when the file cannot be parsed.
        """


def register_reader(reader_cls: type[SpectralDataReader]) -> type[SpectralDataReader]:
    """Add a reader class to the registry (usable as a class decorator)."""
    if reader_cls not in _REGISTRY:
        _REGISTRY.append(reader_cls)
    return reader_cls


def registered_readers() -> tuple[type[SpectralDataReader], ...]:
    return tuple(_REGISTRY)


def get_reader(path: Path | str, format: str | None = None) -> SpectralDataReader:
    """Pick the reader for ``path``, by explicit format name or by sniffing."""
    path = Path(path)
    if format is not None:
        for cls in _REGISTRY:
            if cls.format_name.lower() == format.lower():
                return cls()
        known = ", ".join(cls.format_name for cls in _REGISTRY)
        raise ReaderError(f"unknown format {format!r} (known: {known})")
    for cls in _REGISTRY:
        if cls.sniff(path):
            return cls()
    raise ReaderError(f"no registered reader recognises {path.name!r}")


def load(path: Path | str, format: str | None = None) -> PDAData:
    """Read one file into PDAData. Main entry point for the pipeline/CLI."""
    path = Path(path)
    if not path.is_file():
        raise ReaderError(f"file not found: {path}")
    return get_reader(path, format=format).read(path)


def load_many(paths: Iterable[Path | str], format: str | None = None) -> list[PDAData]:
    """Read several files (one injection each), preserving input order."""
    return [load(p, format=format) for p in paths]
