"""Process-local memory policy for CPU conversion."""

import ctypes
import logging
import os
import sys


def disable_transparent_huge_pages() -> None:
    """Avoid large THP allocations before loading audio libraries or forking.

    Linux preserves this setting across fork and exec, including forkserver
    workers. AUDIO_PREPARE_DISABLE_THP=0 keeps the inherited policy for profiling.
    """
    if sys.platform != "linux" or os.environ.get("AUDIO_PREPARE_DISABLE_THP") == "0":
        return
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(41, 1, 0, 0, 0) != 0:  # PR_SET_THP_DISABLE
        error = ctypes.get_errno()
        raise OSError(error, f"Cannot disable conversion huge pages: {os.strerror(error)}")
    logging.getLogger(__name__).info("Disabled transparent huge pages for conversion and its workers")
