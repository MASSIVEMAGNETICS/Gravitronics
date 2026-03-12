"""
Gravitronics LGT — Lightweight Gravitational Transformer framework.

Exported symbols:
    LGT                    - Main model class
    LGTConfig              - Model configuration dataclass
    GravitationalAttention - Physics-inspired attention mechanism
    CurvedPositionEmbedding- Curvature-modulated position embeddings
    MirrorLayer            - Diagnostics wrapping layer
"""

from gravitronics.lgt.config import LGTConfig
from gravitronics.lgt.embeddings import CurvedPositionEmbedding
from gravitronics.lgt.attention import GravitationalAttention
from gravitronics.lgt.diagnostics import MirrorLayer
from gravitronics.lgt.model import LGT, create_lgt

__all__ = [
    "LGT",
    "LGTConfig",
    "GravitationalAttention",
    "CurvedPositionEmbedding",
    "MirrorLayer",
    "create_lgt",
]
