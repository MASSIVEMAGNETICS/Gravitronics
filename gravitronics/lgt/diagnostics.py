"""
Diagnostics utilities for Gravitronics LGT.

``MirrorLayer``      — wraps any ``nn.Module``, intercepts its forward pass and
                       captures diagnostic output into a rolling deque.
``DiagnosticsLogger``— writes JSON-serialisable diagnostic dicts to a file in
                       append (JSONL) mode.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


class MirrorLayer(nn.Module):
    """Transparent wrapper that logs diagnostic output from any ``nn.Module``.

    The wrapped module's ``forward`` must return either:

    * A single ``Tensor`` — diagnostics will be an empty dict.
    * A ``(Tensor, dict)`` tuple — the dict is captured as a snapshot.
    * A ``(Tensor, List[dict])`` tuple — each element is stored individually.

    Parameters
    ----------
    module:
        The ``nn.Module`` to wrap.
    max_snapshots:
        Maximum number of diagnostic snapshots retained in memory.
    """

    def __init__(self, module: nn.Module, max_snapshots: int = 100) -> None:
        super().__init__()
        self.module = module
        self._enabled: bool = True
        self._snapshots: Deque[Dict] = deque(maxlen=max_snapshots)

        logger.debug(
            "MirrorLayer wrapping %s (max_snapshots=%d)",
            type(module).__name__,
            max_snapshots,
        )

    # ------------------------------------------------------------------ #
    # Enable / disable                                                     #
    # ------------------------------------------------------------------ #

    def enable(self) -> None:
        """Enable diagnostic capture."""
        self._enabled = True
        logger.debug("MirrorLayer enabled.")

    def disable(self) -> None:
        """Disable diagnostic capture (zero overhead on the hot path)."""
        self._enabled = False
        logger.debug("MirrorLayer disabled.")

    @property
    def diagnostics_enabled(self) -> bool:
        """Whether the underlying module has diagnostics switched on."""
        return getattr(self.module, "diagnostics_enabled", False)

    @diagnostics_enabled.setter
    def diagnostics_enabled(self, value: bool) -> None:
        if hasattr(self.module, "diagnostics_enabled"):
            self.module.diagnostics_enabled = value

    # ------------------------------------------------------------------ #
    # Snapshot access                                                      #
    # ------------------------------------------------------------------ #

    def get_snapshot(self) -> Dict:
        """Return the most recent diagnostic snapshot, or an empty dict."""
        if self._snapshots:
            return self._snapshots[-1]
        return {}

    def get_history(self) -> List[Dict]:
        """Return all stored diagnostic snapshots as a list."""
        return list(self._snapshots)

    def clear(self) -> None:
        """Discard all stored snapshots."""
        self._snapshots.clear()

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(self, x, **kwargs):
        """Call the wrapped module and capture any diagnostic output.

        Parameters
        ----------
        x:
            Primary input passed directly to the wrapped module.
        **kwargs:
            Additional keyword arguments forwarded unchanged.

        Returns
        -------
        The primary output tensor (or the full tuple if the caller expects it).
        """
        result = self.module(x, **kwargs)

        if not self._enabled:
            return result

        # Unpack output and store diagnostics
        if isinstance(result, tuple) and len(result) == 2:
            output, diag = result
            snapshot: Dict = {}
            if isinstance(diag, dict):
                snapshot = diag
            elif isinstance(diag, list):
                # Merge list of dicts into one snapshot
                for item in diag:
                    if isinstance(item, dict):
                        snapshot.update(item)
            snapshot["_timestamp"] = time.time()
            self._snapshots.append(snapshot)
        else:
            # No diagnostics returned — store a minimal timestamp entry
            self._snapshots.append({"_timestamp": time.time()})

        return result


# --------------------------------------------------------------------------- #


class DiagnosticsLogger:
    """Append-mode JSONL logger for diagnostic dictionaries.

    Each call to :meth:`log` serialises *data* to a single JSON line and
    appends it to the target file.  The file is kept open between calls for
    efficiency; call :meth:`flush` / :meth:`close` when appropriate.

    Parameters
    ----------
    filepath:
        Path to the output JSONL file.  Created if it does not exist.
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = filepath
        try:
            self._file = open(filepath, "a", encoding="utf-8")
            logger.info("DiagnosticsLogger: writing to '%s'.", filepath)
        except OSError as exc:
            logger.error("DiagnosticsLogger: cannot open '%s': %s", filepath, exc)
            self._file = None  # type: ignore[assignment]

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def log(self, data: Dict[str, Any]) -> None:
        """Append a JSON line containing *data* to the log file.

        Non-serialisable values are converted to strings automatically.

        Parameters
        ----------
        data:
            JSON-serialisable diagnostic dictionary.
        """
        if self._file is None:
            return
        try:
            line = json.dumps(data, default=str)
            self._file.write(line + "\n")
        except (TypeError, ValueError) as exc:
            logger.warning("DiagnosticsLogger.log serialisation error: %s", exc)

    def flush(self) -> None:
        """Flush the internal buffer to disk."""
        if self._file is not None:
            try:
                self._file.flush()
            except OSError as exc:
                logger.warning("DiagnosticsLogger.flush failed: %s", exc)

    def close(self) -> None:
        """Flush and close the log file."""
        if self._file is not None:
            self.flush()
            try:
                self._file.close()
            except OSError as exc:
                logger.warning("DiagnosticsLogger.close failed: %s", exc)
            finally:
                self._file = None  # type: ignore[assignment]

    def __del__(self) -> None:
        """Ensure the file is closed on garbage collection."""
        self.close()
