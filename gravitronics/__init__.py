"""
Gravitronics LGT — Lightweight Gravitational Transformer framework.

Exported symbols:
    LGT                    - Main model class
    LGTConfig              - Model configuration dataclass
    GravitationalAttention - Physics-inspired attention mechanism
    CurvedPositionEmbedding- Curvature-modulated position embeddings
    MirrorLayer            - Diagnostics wrapping layer
    TrainingConfig         - Training job configuration dataclass
    Trainer                - Live training loop with self-training support
    CheckpointManager      - Checkpoint save / load / prune utilities
"""

from gravitronics.lgt.config import LGTConfig
from gravitronics.lgt.embeddings import CurvedPositionEmbedding
from gravitronics.lgt.attention import GravitationalAttention
from gravitronics.lgt.diagnostics import MirrorLayer
from gravitronics.lgt.model import LGT, create_lgt
from gravitronics.training.config import TrainingConfig
from gravitronics.training.checkpoint import CheckpointManager
from gravitronics.training.trainer import Trainer

__all__ = [
    "LGT",
    "LGTConfig",
    "GravitationalAttention",
    "CurvedPositionEmbedding",
    "MirrorLayer",
    "create_lgt",
    "TrainingConfig",
    "CheckpointManager",
    "Trainer",
]
