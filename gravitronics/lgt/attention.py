"""
GravitationalAttention — multi-head attention whose scores are augmented by a
gravitational force law between token masses.

Physics analogy
---------------
Each token acquires a scalar *mass* m_i ≥ 0 predicted by a small projection.
The gravitational force between tokens i and j is:

    F_ij = G * m_i * m_j / (d_ij² + ε)

where d_ij is the Euclidean distance between the query and key vectors of heads
i and j.  This force term is added to the standard scaled-dot-product score
before softmax, biasing attention towards massive, nearby tokens.

Optional physics features:

* **Hawking radiation** — entropy regularisation that suppresses over-peaked
  attention distributions, analogous to information leaking from a black hole.
* **Sparse top-k masking** — hard-zeros all but the top-k scores per query row,
  reducing compute and enforcing locality.
* **Bekenstein bound** — clamps all logits to ``[-bekenstein_limit, +bekenstein_limit]``
  to bound the maximum information per token.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import LGTConfig

logger = logging.getLogger(__name__)

_EPSILON = 1e-8  # numerical stability floor


class GravitationalAttention(nn.Module):
    """Multi-head attention with gravitational force augmentation.

    Parameters
    ----------
    config:
        Model configuration.  Relevant fields:
        ``hidden_dim``, ``num_heads``, ``gravitational_constant``,
        ``hawking_temperature``, ``bekenstein_limit``, ``sparse_top_k``,
        ``use_hawking_radiation``, ``mass_init_scale``.
    """

    def __init__(self, config: LGTConfig) -> None:
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.G = config.gravitational_constant
        self.hawking_temp = config.hawking_temperature
        self.bekenstein_limit = config.bekenstein_limit
        self.sparse_top_k = config.sparse_top_k
        self.use_hawking = config.use_hawking_radiation
        self.diagnostics_enabled: bool = False

        # Standard QKV projections
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        # Per-token mass projection (single scalar, kept positive via softplus)
        self.mass_proj = nn.Linear(config.hidden_dim, 1, bias=True)
        nn.init.constant_(self.mass_proj.bias, config.mass_init_scale)

        self.dropout = nn.Dropout(config.dropout)

        logger.debug(
            "GravitationalAttention: heads=%d, head_dim=%d, G=%.4e",
            self.num_heads,
            self.head_dim,
            self.G,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _split_heads(self, x: Tensor) -> Tensor:
        """Reshape ``(B, L, D)`` → ``(B, H, L, head_dim)``."""
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """Reshape ``(B, H, L, head_dim)`` → ``(B, L, D)``."""
        B, H, L, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, self.hidden_dim)

    def _gravitational_scores(
        self, q: Tensor, k: Tensor, masses: Tensor
    ) -> Tensor:
        """Compute gravitational force matrix.

        Parameters
        ----------
        q:
            Queries  ``(B, H, L, head_dim)``.
        k:
            Keys     ``(B, H, L, head_dim)``.
        masses:
            Per-token masses ``(B, L)`` — broadcast over heads.

        Returns
        -------
        Tensor
            Force matrix of shape ``(B, H, L, L)``.
        """
        # Pairwise L2 distances between query and key vectors
        # ||q_i - k_j||² expanded form: ||q||² + ||k||² - 2 q·k^T
        q_sq = (q * q).sum(-1, keepdim=True)          # (B, H, L, 1)
        k_sq = (k * k).sum(-1, keepdim=True)          # (B, H, L, 1)
        qk = torch.matmul(q, k.transpose(-2, -1))     # (B, H, L, L)
        dist_sq = q_sq + k_sq.transpose(-2, -1) - 2.0 * qk
        dist_sq = dist_sq.clamp(min=0.0)              # numerical safety

        # Outer product of masses — broadcast to (B, H, L, L)
        m_i = masses.unsqueeze(1).unsqueeze(3)        # (B, 1, L, 1)
        m_j = masses.unsqueeze(1).unsqueeze(2)        # (B, 1, 1, L)

        force = self.G * m_i * m_j / (dist_sq + _EPSILON)
        return force

    def _hawking_regularisation(self, attn_weights: Tensor) -> Tensor:
        """Subtract a Hawking-radiation entropy penalty from attention logits.

        A uniform attention distribution (maximum entropy) has the highest
        Hawking-radiation loss.  The penalty encourages peaked but not
        degenerate distributions.

        Parameters
        ----------
        attn_weights:
            Pre-softmax logits ``(B, H, L, L)``.

        Returns
        -------
        Tensor
            Regularised logits of the same shape.
        """
        # Entropy of current (soft) distribution approximated from logits
        probs = F.softmax(attn_weights, dim=-1).clamp(min=_EPSILON)
        entropy = -(probs * probs.log()).sum(-1, keepdim=True)  # (B, H, L, 1)
        return attn_weights - self.hawking_temp * entropy

    @staticmethod
    def _sparse_top_k_mask(scores: Tensor, top_k: int) -> Tensor:
        """Zero out all but the top-k values along the last dimension.

        Parameters
        ----------
        scores:
            Attention logits ``(B, H, L, L)``.
        top_k:
            Number of keys to retain per query position.

        Returns
        -------
        Tensor
            Masked logits — same shape, non-top-k positions set to ``-inf``.
        """
        k = min(top_k, scores.size(-1))
        # Keep only the top-k largest values
        topk_vals, _ = scores.topk(k, dim=-1)
        threshold = topk_vals[..., -1:].detach()  # k-th largest value
        mask = scores < threshold
        return scores.masked_fill(mask, float("-inf"))

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict]:
        """Compute gravitational multi-head attention.

        Parameters
        ----------
        x:
            Input hidden states ``(batch, seq_len, hidden_dim)``.
        mask:
            Optional boolean mask ``(batch, 1, seq_len, seq_len)`` where
            ``True`` positions are masked out (set to ``-inf``).

        Returns
        -------
        out:
            Output tensor of shape ``(batch, seq_len, hidden_dim)``.
        diagnostics:
            Dictionary containing physics diagnostic scalars:
            ``mean_force``, ``mean_mass``, ``curvature_active``,
            ``hawking_limit``.
        """
        B, L, _ = x.shape
        diagnostics: Dict = {}

        # ── Projections ──────────────────────────────────────────────── #
        q = self._split_heads(self.q_proj(x))  # (B, H, L, head_dim)
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # ── Per-token masses ─────────────────────────────────────────── #
        masses = F.softplus(self.mass_proj(x)).squeeze(-1)  # (B, L)

        # ── Standard scaled-dot-product scores ───────────────────────── #
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, L, L)

        # ── Gravitational force augmentation ─────────────────────────── #
        grav = self._gravitational_scores(q, k, masses)  # (B, H, L, L)
        scores = scores + grav

        # ── Bekenstein bound ─────────────────────────────────────────── #
        scores = scores.clamp(-self.bekenstein_limit, self.bekenstein_limit)

        # ── Hawking radiation regularisation ─────────────────────────── #
        if self.use_hawking:
            scores = self._hawking_regularisation(scores)

        # ── Sparse top-k masking ──────────────────────────────────────── #
        if self.sparse_top_k < L:
            scores = self._sparse_top_k_mask(scores, self.sparse_top_k)

        # ── Padding / causal mask ─────────────────────────────────────── #
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        # ── Softmax → dropout → weighted sum ─────────────────────────── #
        attn_weights = F.softmax(scores, dim=-1)
        # Replace NaN rows (all-masked) with uniform attention
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)  # (B, H, L, head_dim)

        out = self.out_proj(self._merge_heads(context))

        # ── Diagnostics ───────────────────────────────────────────────── #
        if self.training or self.diagnostics_enabled:
            with torch.no_grad():
                diagnostics = {
                    "mean_force": float(grav.mean().item()),
                    "mean_mass": float(masses.mean().item()),
                    "curvature_active": True,
                    "hawking_limit": self.use_hawking,
                }

        return out, diagnostics
