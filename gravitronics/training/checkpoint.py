"""
checkpoint.py — Checkpoint saving, loading, and management for the
Gravitronics training subsystem.

Features
--------
* Periodic checkpoint saves (by epoch or step).
* Best-model checkpoint (lowest validation loss).
* Configurable retention: keep the N most-recent periodic checkpoints.
* Integrity check: verifies the file exists and is non-empty before loading.
* Atomic writes: data is written to a ``.tmp`` file then renamed so partial
  writes never corrupt an existing checkpoint.
"""

from __future__ import annotations

import glob as _glob
import json
import logging
import os
import shutil
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_BEST_FILENAME = "best_model.pt"
_BEST_META_FILENAME = "best_model.meta.json"
_PERIODIC_PREFIX = "checkpoint_step"


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _atomic_save(obj: Any, path: str) -> None:
    """Save *obj* to *path* atomically (write to tmp then rename)."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    shutil.move(tmp, path)


def _atomic_json(data: dict, path: str) -> None:
    """Write *data* as JSON to *path* atomically."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    shutil.move(tmp, path)


# --------------------------------------------------------------------------- #
# CheckpointManager                                                            #
# --------------------------------------------------------------------------- #


class CheckpointManager:
    """Manages checkpoint files for a single training run.

    Parameters
    ----------
    checkpoint_dir:
        Root directory for all checkpoints produced by this run.
    keep_last_n:
        How many periodic checkpoints to retain; older ones are deleted.
    save_best:
        Whether to maintain a ``best_model.pt`` based on validation loss.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 3,
        save_best: bool = True,
    ) -> None:
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.keep_last_n = max(1, keep_last_n)
        self._save_best: bool = save_best
        self._best_val_loss: float = float("inf")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info("CheckpointManager initialised at '%s'.", self.checkpoint_dir)

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        epoch: int,
        loss: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a periodic checkpoint and return the file path.

        The payload includes:
        - ``model_state_dict``
        - ``optimizer_state_dict``
        - ``step``, ``epoch``, ``loss``
        - ``timestamp``
        - any *extra* key-value pairs supplied by the caller

        Oldest checkpoints beyond *keep_last_n* are pruned after saving.
        """
        filename = f"{_PERIODIC_PREFIX}_{step:08d}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        payload: Dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if extra:
            payload.update(extra)
        _atomic_save(payload, path)
        logger.info("Checkpoint saved: '%s' (step=%d, loss=%.4f).", path, step, loss)
        self._prune_old_checkpoints()
        return path

    def save_best(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        epoch: int,
        val_loss: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Save a best-model checkpoint if *val_loss* improves.

        Returns the path if saved, ``None`` if the current checkpoint was
        already better.
        """
        if not self.save_best_enabled:
            return None
        if val_loss >= self._best_val_loss:
            return None
        self._best_val_loss = val_loss
        path = os.path.join(self.checkpoint_dir, _BEST_FILENAME)
        payload: Dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "val_loss": val_loss,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if extra:
            payload.update(extra)
        _atomic_save(payload, path)
        meta = {
            "step": step,
            "epoch": epoch,
            "val_loss": val_loss,
            "path": path,
            "timestamp": payload["timestamp"],
        }
        _atomic_json(meta, os.path.join(self.checkpoint_dir, _BEST_META_FILENAME))
        logger.info(
            "Best-model checkpoint saved: val_loss=%.4f → '%s'.", val_loss, path
        )
        return path

    @property
    def save_best_enabled(self) -> bool:
        return self._save_best

    # ------------------------------------------------------------------ #
    # Load / resume                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def verify(path: str) -> bool:
        """Return ``True`` if *path* points to a non-empty, readable file."""
        if not os.path.isfile(path):
            logger.warning("Checkpoint not found: '%s'.", path)
            return False
        if os.path.getsize(path) == 0:
            logger.warning("Checkpoint is empty: '%s'.", path)
            return False
        return True

    @staticmethod
    def load(path: str, device: str = "cpu") -> Dict[str, Any]:
        """Load and return a checkpoint payload dictionary.

        Parameters
        ----------
        path:
            Path to the checkpoint ``.pt`` file.
        device:
            Device to map tensors onto.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If *path* exists but is empty or cannot be parsed.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: '{path}'.")
        if os.path.getsize(path) == 0:
            raise ValueError(f"Checkpoint file is empty: '{path}'.")
        payload = torch.load(path, map_location=device, weights_only=True)
        logger.info(
            "Checkpoint loaded from '%s' (step=%s).",
            path,
            payload.get("step", "?"),
        )
        return payload

    def resume(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        path: str,
        device: str = "cpu",
    ) -> Tuple[int, int, float]:
        """Restore *model* and *optimizer* state from *path*.

        Returns
        -------
        (step, epoch, loss)
            Training state at the time the checkpoint was saved.
        """
        payload = self.load(path, device=device)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        step = int(payload.get("step", 0))
        epoch = int(payload.get("epoch", 0))
        loss = float(payload.get("loss", float("nan")))
        logger.info(
            "Resumed training from '%s': step=%d, epoch=%d, loss=%.4f.",
            path,
            step,
            epoch,
            loss,
        )
        return step, epoch, loss

    # ------------------------------------------------------------------ #
    # Listing / pruning                                                    #
    # ------------------------------------------------------------------ #

    def list_checkpoints(self) -> list:
        """Return a sorted list of periodic checkpoint file paths (oldest first)."""
        pattern = os.path.join(
            self.checkpoint_dir, f"{_PERIODIC_PREFIX}_*.pt"
        )
        paths = sorted(_glob.glob(pattern))
        return paths

    def latest_checkpoint(self) -> Optional[str]:
        """Return the path of the most recent periodic checkpoint, or ``None``."""
        paths = self.list_checkpoints()
        return paths[-1] if paths else None

    def best_checkpoint(self) -> Optional[str]:
        """Return the path of the best-model checkpoint, or ``None``."""
        path = os.path.join(self.checkpoint_dir, _BEST_FILENAME)
        return path if os.path.isfile(path) else None

    def _prune_old_checkpoints(self) -> None:
        """Delete oldest periodic checkpoints beyond *keep_last_n*."""
        paths = self.list_checkpoints()
        excess = len(paths) - self.keep_last_n
        for old in paths[:excess]:
            try:
                os.remove(old)
                logger.debug("Pruned old checkpoint: '%s'.", old)
            except OSError as exc:
                logger.warning("Could not prune '%s': %s.", old, exc)
