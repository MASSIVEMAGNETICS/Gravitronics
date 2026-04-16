"""Gravitronics training subsystem — live training, checkpointing, and export.

Exported symbols
----------------
TrainingConfig   Full training configuration (JSON-serialisable).
CheckpointManager Atomic checkpoint save/load/prune with best-model tracking.
Trainer          Feature-rich training loop with callbacks and auto-training.
CodeDataset      Byte-level dataset that loads source files from disk.
FileTrainer      Lightweight file-based training loop (no config file needed).
FileTrainerConfig Hyper-parameters for :class:`FileTrainer`.
save_checkpoint  Persist model + optimiser state to disk.
load_checkpoint  Restore model + optimiser state from disk.
list_checkpoints Enumerate saved checkpoints in a directory.
"""

from .config import TrainingConfig
from .checkpoint import CheckpointManager
from .trainer import Trainer
from .dataset import CodeDataset
from .file_trainer import FileTrainer, FileTrainerConfig
from .checkpoints import (
    save_checkpoint,
    load_checkpoint,
    list_checkpoints,
)

__all__ = [
    "TrainingConfig",
    "CheckpointManager",
    "Trainer",
    "CodeDataset",
    "FileTrainer",
    "FileTrainerConfig",
    "save_checkpoint",
    "load_checkpoint",
    "list_checkpoints",
]
