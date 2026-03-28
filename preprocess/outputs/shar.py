## Difficulty implementing the checkpoint

# import logging 
# import os
# from pathlib import Path
# from typing import Dict, Any, List

# import numpy as np
# import torch

# from .base import BaseOutputWriter

# logger = logging.getLogger(__name__)


# class SharOutputWriter(BaseOutputWriter):
#     """Write preprocessed results as Lhotse SHAR + metrics sidecar, one shard at a time."""

#     def __init__(self, *, shard_size: int = 1000, output_dir: str | None = None):
#         self._shard_size = shard_size

#         self._writer = None
#         self._metrics_fh = None
#         self._parent: str | None = output_dir
#         self._output_dir_tmp: str | None = None
#         self._output_dir_final: str | None = None
#         self._metrics_tmp: str | None = None
#         self._metrics_final: str | None = None
#         self._count: int = 0
    
#     def open(self, cuts_path: str, *, resume_count: int = 0) -> None:
#         from lhotse.shar import SharWriter

#         shard_name = Path(cuts_path).stem.replace(".jsonl", "")
#         parent = str(Path(cuts_path).parent) if self._parent is None else self._output_dir_final
        
#         self._output_dir_final = os.path.join(parent, f"recon_{shard_name}")
#         self._output_dir_tmp = self._output_dir_final + ".tmp"




