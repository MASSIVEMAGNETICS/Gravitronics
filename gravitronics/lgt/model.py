"""
LGT — Lightweight Gravitational Transformer.

Full model wiring:

    Token Embedding
    └─ CurvedPositionEmbedding  (optional)
       └─ Dropout
          └─ N × TransformerBlock
             ├─ MirrorLayer → GravitationalAttention  (wrapped when use_mirror_layer)
             ├─ LayerNorm + residual
             ├─ FFN  (hidden → 4×hidden → hidden)
             └─ LayerNorm + residual
    └─ LayerNorm
    └─ Output Linear  (hidden → vocab_size)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .attention import GravitationalAttention
from .config import LGTConfig
from .diagnostics import MirrorLayer
from .embeddings import CurvedPositionEmbedding

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Feed-Forward Network                                                         #
# --------------------------------------------------------------------------- #


class _FFN(nn.Module):
    """Point-wise two-layer feed-forward network with GELU activation.

    Expands the hidden dimension by ``expansion_factor`` then projects back.
    """

    def __init__(self, hidden_dim: int, dropout: float, expansion_factor: int = 4) -> None:
        super().__init__()
        inner = hidden_dim * expansion_factor
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return self.net(x)


# --------------------------------------------------------------------------- #
# Transformer Block                                                            #
# --------------------------------------------------------------------------- #


class TransformerBlock(nn.Module):
    """Single transformer block with gravitational attention + FFN.

    The attention module is optionally wrapped in a :class:`MirrorLayer` for
    real-time diagnostic capture.

    Parameters
    ----------
    config:
        Model configuration.
    """

    def __init__(self, config: LGTConfig) -> None:
        super().__init__()

        attn = GravitationalAttention(config)

        if config.use_mirror_layer:
            self.attn: nn.Module = MirrorLayer(attn)
        else:
            self.attn = attn

        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.ffn = _FFN(config.hidden_dim, config.dropout)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self._use_mirror = config.use_mirror_layer

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict]:
        """Forward pass with residual connections.

        Parameters
        ----------
        x:
            Hidden states ``(batch, seq_len, hidden_dim)``.
        mask:
            Optional attention mask.

        Returns
        -------
        x:
            Updated hidden states.
        diag:
            Diagnostics dict from the attention module (may be empty).
        """
        # ── Attention sub-layer ──────────────────────────────────────── #
        attn_out = self.attn(x, mask=mask)

        # Unpack whether mirror or raw attention
        if isinstance(attn_out, tuple):
            attn_tensor, diag = attn_out
        else:
            attn_tensor, diag = attn_out, {}

        x = self.norm1(x + self.dropout(attn_tensor))

        # ── FFN sub-layer ────────────────────────────────────────────── #
        x = self.norm2(x + self.ffn(x))

        return x, diag


# --------------------------------------------------------------------------- #
# Full Model                                                                   #
# --------------------------------------------------------------------------- #


class LGT(nn.Module):
    """Lightweight Gravitational Transformer.

    Parameters
    ----------
    config:
        :class:`~gravitronics.lgt.config.LGTConfig` instance controlling all
        architectural choices.
    """

    def __init__(self, config: LGTConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Position embedding
        if config.use_curved_positions:
            self.pos_emb: Optional[nn.Module] = CurvedPositionEmbedding(config)
        else:
            self.pos_emb = None

        self.dropout = nn.Dropout(config.dropout)

        # Stacked transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )

        # Output head
        self.out_norm = nn.LayerNorm(config.hidden_dim)
        self.out_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Weight tying between token embedding and output head
        self.out_head.weight = self.token_emb.weight

        self._init_weights()

        logger.info(
            "LGT created: variant=%s, params=%d",
            config.model_variant,
            self.count_parameters(),
        )

    # ------------------------------------------------------------------ #
    # Weight initialisation                                               #
    # ------------------------------------------------------------------ #

    def _init_weights(self) -> None:
        """Initialise weights with small normal / zeros where appropriate."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Dict]]:
        """Run a full forward pass.

        Parameters
        ----------
        input_ids:
            Token indices ``(batch, seq_len)``.
        mask:
            Optional boolean attention mask ``(batch, 1, seq_len, seq_len)``.

        Returns
        -------
        logits:
            Unnormalised vocabulary scores ``(batch, seq_len, vocab_size)``.
        diagnostics_list:
            One diagnostic dict per transformer block.
        """
        x = self.token_emb(input_ids)  # (B, L, D)

        if self.pos_emb is not None:
            x = self.pos_emb(x)

        x = self.dropout(x)

        diagnostics_list: List[Dict] = []
        for block in self.blocks:
            x, diag = block(x, mask=mask)
            diagnostics_list.append(diag)

        x = self.out_norm(x)
        logits = self.out_head(x)  # (B, L, vocab_size)

        return logits, diagnostics_list

    # ------------------------------------------------------------------ #
    # Utility methods                                                      #
    # ------------------------------------------------------------------ #

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> LGTConfig:
        """Return the :class:`LGTConfig` used to build this model."""
        return self.config

    def get_model_size_mb(self) -> float:
        """Return the approximate model size in megabytes (fp32 parameters)."""
        total_bytes = sum(
            p.numel() * p.element_size() for p in self.parameters()
        )
        return total_bytes / (1024 ** 2)

    def enable_diagnostics(self) -> None:
        """Switch on diagnostic capture in all attention / mirror modules."""
        for block in self.blocks:
            attn = block.attn
            if isinstance(attn, MirrorLayer):
                attn.enable()
                attn.diagnostics_enabled = True
            elif isinstance(attn, GravitationalAttention):
                attn.diagnostics_enabled = True
        logger.info("LGT diagnostics enabled.")

    def disable_diagnostics(self) -> None:
        """Switch off diagnostic capture."""
        for block in self.blocks:
            attn = block.attn
            if isinstance(attn, MirrorLayer):
                attn.disable()
                attn.diagnostics_enabled = False
            elif isinstance(attn, GravitationalAttention):
                attn.diagnostics_enabled = False
        logger.info("LGT diagnostics disabled.")


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #


def create_lgt(variant: str = "150k") -> LGT:
    """Instantiate an :class:`LGT` model from a size variant string.

    Parameters
    ----------
    variant:
        One of ``"150k"``, ``"600k"``, or ``"2m"``.

    Returns
    -------
    LGT
        Freshly initialised model ready for training or inference.
    """
    config = LGTConfig.from_variant(variant)
    return LGT(config)
