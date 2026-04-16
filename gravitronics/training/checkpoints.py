"""
Checkpoint utilities for saving and loading LGT training state.

A checkpoint is a plain :func:`torch.save` dictionary containing:

* ``"model_state"``   — model ``state_dict``
* ``"optimizer_state"`` — optimizer ``state_dict`` (optional)
* ``"epoch"``         — current epoch (int)
* ``"step"``          — global optimiser step (int)
* ``"loss"``          — last recorded training loss (float)
* ``"config"``        — serialised :class:`~gravitronics.lgt.config.LGTConfig`
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    path: Union[str, Path],
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    step: int = 0,
    loss: float = float("inf"),
    config: Optional[Any] = None,
) -> Path:
    """Persist a training checkpoint to *path*.

    Parameters
    ----------
    model:
        The LGT model (or any :class:`torch.nn.Module`) to checkpoint.
    path:
        Output ``.pt`` file path.
    optimizer:
        If provided, its state is included so training can resume exactly.
    epoch:
        Current training epoch index (0-based).
    step:
        Global optimiser step count.
    loss:
        Most recent training loss value.
    config:
        :class:`~gravitronics.lgt.config.LGTConfig` instance (or any object
        with a ``to_dict()`` method).  Stored as a plain ``dict``.

    Returns
    -------
    Path
        The resolved path where the checkpoint was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "loss": loss,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if config is not None:
        payload["config"] = config.to_dict() if hasattr(config, "to_dict") else config

    torch.save(payload, path)
    logger.info(
        "Checkpoint saved → %s  (epoch=%d, step=%d, loss=%.4f)",
        path,
        epoch,
        step,
        loss,
    )
    return path


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Restore a checkpoint saved by :func:`save_checkpoint`.

    Parameters
    ----------
    path:
        Path to a ``.pt`` checkpoint file.
    model:
        Model whose weights will be restored in-place.
    optimizer:
        If provided, its state is also restored from the checkpoint.
    map_location:
        Passed directly to :func:`torch.load` (e.g. ``"cpu"``).

    Returns
    -------
    dict
        The full checkpoint dictionary (includes ``epoch``, ``step``,
        ``loss``, and optionally ``config``).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload: Dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])

    logger.info(
        "Checkpoint loaded ← %s  (epoch=%d, step=%d, loss=%.4f)",
        path,
        payload.get("epoch", 0),
        payload.get("step", 0),
        payload.get("loss", float("inf")),
    )
    return payload


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def list_checkpoints(checkpoint_dir: Union[str, Path]) -> List[Path]:
    """Return all ``.pt`` files in *checkpoint_dir*, sorted by name.

    Parameters
    ----------
    checkpoint_dir:
        Directory to search.

    Returns
    -------
    list[Path]
        Sorted list of checkpoint paths (may be empty).
    """
    d = Path(checkpoint_dir)
    if not d.is_dir():
        return []
    return sorted(d.glob("*.pt"))
