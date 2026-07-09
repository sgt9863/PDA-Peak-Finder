"""Exception hierarchy for pda_peak_finder."""


class PdaPeakFinderError(Exception):
    """Base class for all errors raised by this package."""


class ReaderError(PdaPeakFinderError):
    """A file could not be read or parsed into a valid PDAData."""


class DataValidationError(PdaPeakFinderError):
    """A data model invariant was violated (axis order, shape mismatch, ...)."""
