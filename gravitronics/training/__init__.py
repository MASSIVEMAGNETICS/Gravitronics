"""
gravitronics.training — training pipeline for the LGT model.

Exported symbols
----------------
CodeDataset     Dataset that loads source files at the byte level.
TrainerConfig   Dataclass of all training hyper-parameters.
Trainer         Manages the forward/backward loop, scheduling, and checkpointing.
save_checkpoint Persist model + optimiser state to disk.
load_checkpoint Restore model + optimiser state from disk.
list_checkpoints Enumerate saved checkpoints in a directory.
"""

from gravitronics.training.dataset import CodeDataset
from gravitronics.training.trainer import Trainer, TrainerConfig
from gravitronics.training.checkpoints import (
    save_checkpoint,
    load_checkpoint,
    list_checkpoints,
)

__all__ = [
    "CodeDataset",
    "Trainer",
    "TrainerConfig",
    "save_checkpoint",
    "load_checkpoint",
    "list_checkpoints",
]
