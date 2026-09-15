"""Errors that must cross recoverable per-input processing boundaries."""


class OutputWriteError(RuntimeError):
    """Output may be partially appended; abort its owner instead of continuing."""
