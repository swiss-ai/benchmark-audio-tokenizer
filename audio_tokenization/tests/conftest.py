import sys
from pathlib import Path

# scripts/ holds CLI entrypoints that aren't a package; make them importable
# by tests without per-file sys.path hacks.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
