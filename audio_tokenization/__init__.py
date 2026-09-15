"""Production audio tokenization pipeline.

The package provides dataset conversion, GPU audio tokenization, interleave
materialization, validation, and operator tooling for large audio corpora.
"""

__version__ = "0.1.0"

def main():
    """Entry point for CLI."""
    from .__main__ import main as _main
    return _main()
