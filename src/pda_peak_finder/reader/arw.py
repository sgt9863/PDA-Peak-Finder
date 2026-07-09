"""Waters Empower ARW reader — interface stub.

ARW files are ASCII exports produced by Empower's export methods. Their
exact layout (header lines, delimiter, whether wavelengths run across
columns or rows) depends on the export method definition used at the site,
so parsing is deliberately NOT implemented until sample files are
available. The class exists now so that:

* the registry already routes ``*.arw`` files here, and
* the CLI/pipeline call sites need no change when parsing lands.

Implementation checklist once sample ARW files arrive
-----------------------------------------------------
1. Inspect the header: which lines are metadata (sample name, channel,
   acquisition date) vs. the numeric block.
2. Determine the numeric block orientation and delimiter; typical Empower
   3D exports are tab-separated with a wavelength header row and one row
   per scan time (first column = time in minutes).
3. Parse metadata into ``InjectionMetadata`` (``injection_id`` from the
   header if present, else the file stem; keep unrecognised header fields
   in ``metadata.extra``).
4. Build the (T, W) absorbance matrix; ensure both axes are strictly
   increasing (sort/flip if the export writes them descending).
5. Convert units if needed (Empower may export mAU or seconds depending
   on the method) so the returned PDAData is in AU / minutes / nm.
6. Raise ReaderError with the offending line number for malformed files.
7. Add real ARW fixtures under ``tests/fixtures/`` and golden-value tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ..models import PDAData
from .base import SpectralDataReader


class ArwReader(SpectralDataReader):
    """Reader for Waters Empower ARW exports (parsing not yet implemented)."""

    format_name: ClassVar[str] = "arw"
    file_patterns: ClassVar[tuple[str, ...]] = ("*.arw",)

    def read(self, path: Path) -> PDAData:
        raise NotImplementedError(
            "ARW parsing is not implemented yet: the exact export layout "
            "depends on the Empower export method and will be finalised "
            "once sample ARW files are provided. See the module docstring "
            "of pda_peak_finder.reader.arw for the implementation checklist."
        )
