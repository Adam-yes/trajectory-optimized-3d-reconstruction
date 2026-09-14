"""Actionable, fail-closed errors at data and external-system boundaries."""


class EyeInHandError(Exception):
    """Base error suitable for presentation at the command line."""


class ValidationError(EyeInHandError, ValueError):
    """An input violates an explicitly documented contract."""


class MissingDependencyError(EyeInHandError, ImportError):
    """An optional dependency or checkpoint is missing."""


class ArtifactConflictError(EyeInHandError):
    """An existing artifact belongs to a different experiment or is incomplete."""


class StageError(EyeInHandError):
    """An external stage failed; the run must not be reported as successful."""
