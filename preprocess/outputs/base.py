from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseOutputWriter(ABC):
    """
    Order of function call
        1. open(cuts_path, resume_count=0)
        2. write_cut(...)
        3. finalize()
        4. close()
    """

    @abstractmethod
    def open(self, cuts_path: str, resume_count: int = 0) -> None:
        """Start (or resume) writing an enriched shard.
 
        Args:
            cuts_path: Absolute path to the original cuts_XXXXXX.jsonl.gz.
            resume_count: Number of samples already written (from a
                checkpoint).  Implementations should truncate / skip to
                this position before appending new results.
        """
        ...

    @abstractmethod
    def write_cut(self, 
                  *, 
                  cut_dict: Dict[str, Any], 
                  text: str | None = None,
                  extra: Dict[str, Any]=None,
                  text_tokens: List[int] | None = None
                  ) -> None:     
        ...

    @abstractmethod
    def finalize(self, rank: int) -> None: 
        """Atomic saving of the shard"""
        ...

    @abstractmethod
    def close(self) -> None: 
        """Closing the writer"""
        ...

    @property
    @abstractmethod
    def cuts_written(self) -> int:
        """Number of samples written to the current (open) shard."""
        ...
