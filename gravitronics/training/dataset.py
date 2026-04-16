"""
Dataset utilities for training LGT on source files and repositories.

Tokenisation strategy
---------------------
All variants use *byte-level* encoding (token = byte value, 0–255).
This requires no external tokeniser library and works naturally with
the ``vocab_size=256`` preset of the 150k variant; larger variants
(600k, 2m) have wider vocabularies so the first 256 slots are used.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Default file extensions considered as source code / text
_DEFAULT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py", ".txt", ".md", ".rst", ".json", ".yaml", ".yml",
        ".toml", ".cfg", ".ini", ".sh", ".js", ".ts", ".c", ".cpp",
        ".h", ".rs", ".go", ".java", ".rb", ".kt", ".swift",
    }
)

_DEFAULT_MAX_FILE_BYTES: int = 1 * 1024 * 1024  # 1 MB


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_files(
    roots: Sequence[Union[str, Path]],
    extensions: Optional[frozenset[str]] = None,
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
) -> List[Path]:
    """Walk *roots* and return all matching file paths, sorted for reproducibility."""
    if extensions is None:
        extensions = _DEFAULT_EXTENSIONS

    paths: List[Path] = []
    for root in roots:
        root = Path(root)
        if root.is_file():
            if root.suffix in extensions:
                paths.append(root)
            else:
                logger.debug("Skipping file with unsupported extension: %s", root)
        elif root.is_dir():
            for entry in sorted(root.rglob("*")):
                if not entry.is_file():
                    continue
                if entry.suffix not in extensions:
                    continue
                try:
                    size = entry.stat().st_size
                except OSError:
                    continue
                if size > max_file_bytes:
                    logger.debug("Skipping large file (%d bytes): %s", size, entry)
                    continue
                paths.append(entry)
        else:
            logger.warning("Source path not found, skipping: %s", root)

    return paths


def _read_bytes(path: Path) -> Optional[bytes]:
    """Return raw bytes of *path*, or ``None`` on I/O error."""
    try:
        return path.read_bytes()
    except OSError as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CodeDataset(Dataset):
    """PyTorch :class:`~torch.utils.data.Dataset` for next-token-prediction.

    Loads all source files under the given *sources* (directories or
    individual paths), concatenates their raw bytes into a single flat
    token sequence, then exposes sliding-window chunks as
    ``(input_ids, target_ids)`` pairs.

    Parameters
    ----------
    sources:
        One or more directories or file paths to ingest.
    seq_len:
        Number of tokens per training example (must match the model's
        ``max_seq_len``).
    extensions:
        File extensions to include.  Defaults to :data:`_DEFAULT_EXTENSIONS`.
    max_file_bytes:
        Files larger than this are silently skipped.
    stride:
        Sliding-window step size.  Defaults to *seq_len* (no overlap).
        Use a smaller value for more training examples from the same data.
    """

    def __init__(
        self,
        sources: Sequence[Union[str, Path]],
        seq_len: int = 64,
        extensions: Optional[frozenset[str]] = None,
        max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
        stride: Optional[int] = None,
    ) -> None:
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len

        files = _collect_files(sources, extensions=extensions, max_file_bytes=max_file_bytes)
        if not files:
            raise ValueError(
                f"No matching files found under: {list(sources)}.  "
                "Check that the paths exist and contain files with supported extensions."
            )

        logger.info(
            "CodeDataset: loading %d file(s) from %d source root(s).",
            len(files),
            len(sources),
        )

        # Concatenate all file bytes into one flat token array (byte-level).
        raw: List[int] = []
        for path in files:
            data = _read_bytes(path)
            if data:
                raw.extend(data)

        if len(raw) < seq_len + 1:
            raise ValueError(
                f"Corpus is only {len(raw)} byte(s); need at least {seq_len + 1} "
                "to form one training example.  Add more source files or reduce seq_len."
            )

        self._tokens: torch.Tensor = torch.tensor(raw, dtype=torch.long)
        self._files: List[Path] = files

        logger.info(
            "Corpus: %d tokens (~%.2f MB) from %d file(s).",
            len(raw),
            len(raw) / 1024 / 1024,
            len(files),
        )

        # Build sliding-window start indices.
        n_complete = len(self._tokens) - seq_len  # each sample needs seq_len+1 tokens
        self._starts: List[int] = list(range(0, n_complete, self.stride))

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int):
        """Return ``(input_ids, target_ids)`` tensors of shape ``(seq_len,)``."""
        start = self._starts[idx]
        chunk = self._tokens[start : start + self.seq_len + 1]
        return chunk[:-1].clone(), chunk[1:].clone()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def num_tokens(self) -> int:
        """Total number of tokens in the concatenated corpus."""
        return int(self._tokens.shape[0])

    @property
    def source_files(self) -> List[Path]:
        """List of source files that were loaded."""
        return list(self._files)
