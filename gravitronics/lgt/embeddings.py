"""
CurvedPositionEmbedding — sinusoidal position embeddings modulated by a
learnable curvature tensor.

The embedding generalises standard sinusoidal encodings to a curved space-time
metaphor: each position has a local metric factor (the curvature tensor) that
scales how strongly its positional signal is added to the token representation.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .config import LGTConfig

logger = logging.getLogger(__name__)


class CurvedPositionEmbedding(nn.Module):
    """Sinusoidal position embeddings scaled by a learnable curvature tensor.

    The forward pass computes:

    .. math::

        \\text{output} = x + \\text{curvature} \\cdot \\text{PE}

    where ``PE`` is the standard (fixed) sinusoidal encoding of shape
    ``(1, seq_len, hidden_dim)`` and ``curvature`` is a learnable
    ``nn.Parameter`` of the same shape initialised near ``1.0``.

    Parameters
    ----------
    config:
        Model configuration supplying ``max_seq_len``, ``hidden_dim``, and
        ``curvature_scale``.
    """

    def __init__(self, config: LGTConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.max_seq_len = config.max_seq_len
        self.curvature_scale = config.curvature_scale

        # Learnable curvature metric — initialised near 1.0 with small noise
        self.curvature = nn.Parameter(
            torch.ones(1, config.max_seq_len, config.hidden_dim)
            + torch.randn(1, config.max_seq_len, config.hidden_dim) * 0.02
        )

        # Pre-compute fixed sinusoidal base and register as a non-trainable buffer
        pe = self._make_sinusoidal(config.max_seq_len, config.hidden_dim)
        self.register_buffer("pe", pe)  # (1, max_seq_len, hidden_dim)

        logger.debug(
            "CurvedPositionEmbedding: max_seq_len=%d, hidden_dim=%d",
            config.max_seq_len,
            config.hidden_dim,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_sinusoidal(max_seq_len: int, hidden_dim: int) -> Tensor:
        """Build a standard sinusoidal position encoding table.

        Returns shape ``(1, max_seq_len, hidden_dim)``.
        """
        pe = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        # Frequency scaling identical to *Attention Is All You Need*
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if hidden_dim % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[: hidden_dim // 2])
        return pe.unsqueeze(0)  # (1, max_seq_len, hidden_dim)

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(
        self, x: Tensor, positions: Optional[Tensor] = None
    ) -> Tensor:
        """Add curvature-modulated positional encoding to token embeddings.

        Parameters
        ----------
        x:
            Token embeddings of shape ``(batch, seq_len, hidden_dim)``.
        positions:
            Optional integer tensor of shape ``(batch, seq_len)`` specifying
            which absolute positions to use.  When *None*, positions
            ``0 … seq_len-1`` are used.

        Returns
        -------
        Tensor
            Same shape as *x* — token embeddings with positional signal added.
        """
        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len})."
            )

        if positions is not None:
            # positions: (batch, seq_len) — use advanced indexing
            # pe: (1, max_seq_len, hidden_dim)
            base_pe = self.pe[0][positions]  # (batch, seq_len, hidden_dim)
            curvature = self.curvature[0][positions]  # (batch, seq_len, hidden_dim)
        else:
            base_pe = self.pe[:, :seq_len, :]      # (1, seq_len, hidden_dim)
            curvature = self.curvature[:, :seq_len, :]  # (1, seq_len, hidden_dim)

        curved_pe = self.curvature_scale * curvature * base_pe
        return x + curved_pe
