import logging

import gzip
import os
from pathlib import Path
from typing import Dict, Any, List
import json
import shutil

import torch

from .base import BaseOutputWriter


logger = logging.getLogger(__name__)

class CutsOutputWriter(BaseOutputWriter):
    """Write benchmark results as Lhotse cut manifests + metrics sidecar.
 
    No audio is written.  Each cut references the original recording via
    custom.original_cut_id.
    """

    def __init__(self, *, shard_size: int = 1000, output_dir: str | None = None):
        self._shard_size = shard_size

        self._fh = None
        self._tmp_path: str | None = None
        self._parent: str | None = output_dir
        self._final_path: str | None = None
        self._count: int = 0
    
    def open(self, cuts_path: str, *, resume_count: int = 0) -> None:
        """Start (or resume) modifying the shard.
 
        Args:
            cuts_path: Absolute path to the original ``cuts_XXXXXX.jsonl.gz``.
            resume_count: Number of lines already written to the ``.tmp``
                file (from a checkpoint).  The file is truncated to this
                many lines before appending resumes after a failure.
        """
        cut_name = Path(cuts_path).stem.replace(".jsonl", "")
    
        if self._parent is not None:
            source_dir_name = Path(cuts_path).parent.name 
            parent = os.path.join(self._parent, source_dir_name)
        else:
            parent = str(Path(cuts_path).parent)
        
        os.makedirs(parent, exist_ok=True)
        
        self._final_path = os.path.join(parent, cut_name)
        self._tmp_path = self._final_path + ".tmp"
        self._count = resume_count

        if resume_count > 0 and os.path.exists(self._tmp_path):
            self._truncate_tmp(resume_count)
            self._count = resume_count
            self._fh = open(self._tmp_path, "a", encoding="utf-8")
        else:
            self._fh = open(self._tmp_path, "w", encoding="utf-8")
    
    def write_cut(self,
                *,
                cut_dict: Dict[str, Any],
                text: str | None = None,
                extra: Dict[str, Any] = None,
                text_tokens: List[int] = None,
                ) -> None:
        """Add new metadata"""
        custom = cut_dict.get("custom", {})
        if text is not None:
            custom["text"] = text
        if extra is not None:
            custom.update(extra)
        if text_tokens is not None:
            custom["text_tokens"] = text_tokens
        cut_dict["custom"] = custom

        self._fh.write(json.dumps(cut_dict, default=self._json_default) + "\n")
        self._count += 1

    @staticmethod
    def _json_default(obj):
        """Handle non-serializable types in cut_dict."""
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        if isinstance(obj, (torch.Tensor,)):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    
    def finalize(self, rank: int) -> None:
        """Close the temp file and atomically replace the original."""
        self._close_handle()
        if self._tmp_path and self._final_path:
            gz_tmp = self._tmp_path + ".gz"
            with open(self._tmp_path, "rb") as f_in, gzip.open(gz_tmp, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            
            final_path_SHAR_format = self._final_path + ".jsonl.gz"
            os.replace(gz_tmp, final_path_SHAR_format)
            os.unlink(self._tmp_path)
            logger.info(
                f"[rank {rank}] shard finalized: {self._final_path} ({self._count} cuts)",
            )
        self._tmp_path = None
        self._final_path = None
        self._count = 0
    
    def close(self) -> None:
        """Close file without renaming"""
        self._close_handle()
    
    @property
    def cuts_written(self) -> int:
        return self._count
    
    def _close_handle(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
    
    def _truncate_tmp(self, keep_lines: int) -> None:
        """Truncate the jsonl .tmp in-place after keep_lines lines."""
        with open(self._tmp_path, "r+", encoding="utf-8") as f:
            for _ in range(keep_lines):
                if not f.readline():
                    break  
            f.truncate()

        logger.info(
            f"truncated {self._tmp_path} to {keep_lines} lines for resume",
        )   
        