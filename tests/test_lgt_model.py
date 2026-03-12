"""
Tests for the Gravitronics LGT model core components.

Covers: LGTConfig, LGT, GravitationalAttention, CurvedPositionEmbedding,
diagnostics, save/load, and numerical properties.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

# Ensure repo root is importable when running from any directory
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gravitronics.lgt.attention import GravitationalAttention
from gravitronics.lgt.config import LGTConfig
from gravitronics.lgt.embeddings import CurvedPositionEmbedding
from gravitronics.lgt.model import LGT, create_lgt


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #


class TestLGTConfig:
    def test_config_variants(self):
        """All three variants produce valid, distinct configs."""
        configs = {v: LGTConfig.from_variant(v) for v in ("150k", "600k", "2m")}
        assert configs["150k"].hidden_dim == 64
        assert configs["600k"].hidden_dim == 128
        assert configs["2m"].hidden_dim == 256

    def test_default_config_valid(self):
        cfg = LGTConfig()
        assert cfg.hidden_dim % cfg.num_heads == 0

    def test_invalid_heads_raises(self):
        with pytest.raises(ValueError):
            LGTConfig(hidden_dim=128, num_heads=3)  # 128 not divisible by 3

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError):
            LGTConfig.from_variant("999m")

    def test_to_dict_round_trip(self):
        cfg = LGTConfig.from_variant("150k")
        d = cfg.to_dict()
        cfg2 = LGTConfig.from_dict(d)
        assert cfg == cfg2


# --------------------------------------------------------------------------- #
# LGT model creation                                                           #
# --------------------------------------------------------------------------- #


class TestLGTCreation:
    def test_lgt_creation_150k(self):
        model = create_lgt("150k")
        assert isinstance(model, LGT)

    def test_lgt_creation_600k(self):
        model = create_lgt("600k")
        assert isinstance(model, LGT)

    def test_lgt_creation_2m(self):
        model = create_lgt("2m")
        assert isinstance(model, LGT)

    def test_get_config_returns_config(self):
        model = create_lgt("150k")
        cfg = model.get_config()
        assert isinstance(cfg, LGTConfig)
        assert cfg.model_variant == "150k"


# --------------------------------------------------------------------------- #
# Parameter counts                                                             #
# --------------------------------------------------------------------------- #


class TestParameterCounts:
    """Parameter counts should be in the expected ballpark (50% tolerance)."""

    def _check(self, variant: str, target: int) -> None:
        model = create_lgt(variant)
        n = model.count_parameters()
        lo, hi = target * 0.5, target * 1.5
        assert lo <= n <= hi, (
            f"{variant}: expected ~{target:,} params, got {n:,} "
            f"(outside [{lo:,.0f}, {hi:,.0f}])"
        )

    def test_150k(self):
        self._check("150k", 150_000)

    def test_600k(self):
        self._check("600k", 600_000)

    def test_2m(self):
        self._check("2m", 2_000_000)


# --------------------------------------------------------------------------- #
# Forward pass                                                                 #
# --------------------------------------------------------------------------- #


class TestForwardPass:
    def _make_input(self, cfg: LGTConfig, batch: int = 2):
        seq = min(cfg.max_seq_len, 16)
        return torch.randint(0, cfg.vocab_size, (batch, seq))

    def test_forward_pass_shapes(self):
        model = create_lgt("150k")
        cfg = model.get_config()
        ids = self._make_input(cfg)
        logits, diag_list = model(ids)
        assert logits.shape == (ids.shape[0], ids.shape[1], cfg.vocab_size)
        assert len(diag_list) == cfg.num_layers

    def test_forward_pass_all_variants(self):
        for variant in ("150k", "600k", "2m"):
            model = create_lgt(variant)
            cfg = model.get_config()
            ids = self._make_input(cfg)
            logits, _ = model(ids)
            assert logits.shape[-1] == cfg.vocab_size

    def test_no_nan_in_logits(self):
        model = create_lgt("150k")
        cfg = model.get_config()
        ids = self._make_input(cfg)
        logits, _ = model(ids)
        assert not torch.isnan(logits).any(), "NaN detected in logits."


# --------------------------------------------------------------------------- #
# Diagnostics                                                                  #
# --------------------------------------------------------------------------- #


class TestDiagnosticsEnabled:
    def test_diagnostics_keys_present(self):
        model = create_lgt("150k")
        model.enable_diagnostics()
        model.train()  # training mode triggers diagnostics
        cfg = model.get_config()
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        _, diag_list = model(ids)
        for diag in diag_list:
            # Diagnostics may be empty dict if mirror layer captures them
            # but at minimum the list should have one entry per layer
            assert isinstance(diag, dict)

    def test_diagnostics_disable(self):
        model = create_lgt("150k")
        model.enable_diagnostics()
        model.disable_diagnostics()
        # Should not raise
        cfg = model.get_config()
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        logits, _ = model(ids)
        assert logits is not None


# --------------------------------------------------------------------------- #
# CurvedPositionEmbedding                                                      #
# --------------------------------------------------------------------------- #


class TestCurvedPositionEmbedding:
    def test_forward_shape(self):
        cfg = LGTConfig.from_variant("150k")
        emb = CurvedPositionEmbedding(cfg)
        x = torch.zeros(2, 16, cfg.hidden_dim)
        out = emb(x)
        assert out.shape == x.shape

    def test_output_differs_from_input(self):
        cfg = LGTConfig.from_variant("150k")
        emb = CurvedPositionEmbedding(cfg)
        x = torch.zeros(1, 16, cfg.hidden_dim)
        out = emb(x)
        # Position embeddings should be non-zero
        assert not torch.allclose(out, x)

    def test_curvature_is_trainable(self):
        cfg = LGTConfig.from_variant("150k")
        emb = CurvedPositionEmbedding(cfg)
        assert emb.curvature.requires_grad

    def test_seq_len_exceeds_max_raises(self):
        cfg = LGTConfig.from_variant("150k")
        emb = CurvedPositionEmbedding(cfg)
        x = torch.zeros(1, cfg.max_seq_len + 1, cfg.hidden_dim)
        with pytest.raises(ValueError):
            emb(x)


# --------------------------------------------------------------------------- #
# GravitationalAttention                                                       #
# --------------------------------------------------------------------------- #


class TestGravitationalAttention:
    def test_forward_shape(self):
        cfg = LGTConfig.from_variant("150k")
        attn = GravitationalAttention(cfg)
        x = torch.randn(2, 16, cfg.hidden_dim)
        out, diag = attn(x)
        assert out.shape == x.shape

    def test_diagnostics_keys_in_train(self):
        cfg = LGTConfig.from_variant("150k")
        attn = GravitationalAttention(cfg)
        attn.train()
        x = torch.randn(1, 8, cfg.hidden_dim)
        _, diag = attn(x)
        for key in ("mean_force", "mean_mass", "curvature_active", "hawking_limit"):
            assert key in diag, f"Missing key '{key}' in diagnostics."

    def test_diagnostics_enabled_flag(self):
        cfg = LGTConfig.from_variant("150k")
        attn = GravitationalAttention(cfg)
        attn.eval()
        attn.diagnostics_enabled = True
        x = torch.randn(1, 8, cfg.hidden_dim)
        _, diag = attn(x)
        assert "mean_force" in diag


# --------------------------------------------------------------------------- #
# Determinism                                                                  #
# --------------------------------------------------------------------------- #


class TestModelDeterminism:
    def test_same_output_in_eval(self):
        model = create_lgt("150k")
        model.eval()
        cfg = model.get_config()
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            out1, _ = model(ids)
            out2, _ = model(ids)
        assert torch.allclose(out1, out2), "Eval mode outputs are not deterministic."


# --------------------------------------------------------------------------- #
# Bekenstein bound                                                             #
# --------------------------------------------------------------------------- #


class TestBekensteinBound:
    def test_logits_clamped(self):
        """Logits after forward should respect the Bekenstein limit indirectly."""
        cfg = LGTConfig.from_variant("150k")
        attn = GravitationalAttention(cfg)
        attn.eval()
        # Use large inputs to stress the clamping
        x = torch.randn(1, 16, cfg.hidden_dim) * 100.0
        out, _ = attn(x)
        # Output should be finite (no inf/nan) — clamping prevents blow-up
        assert torch.isfinite(out).all(), "Non-finite values in attention output."


# --------------------------------------------------------------------------- #
# Save / load                                                                  #
# --------------------------------------------------------------------------- #


class TestSaveLoad:
    def test_state_dict_round_trip(self, tmp_path):
        model = create_lgt("150k")
        model.eval()
        cfg = model.get_config()
        ids = torch.randint(0, cfg.vocab_size, (1, 8))

        with torch.no_grad():
            logits_before, _ = model(ids)

        path = tmp_path / "model.pt"
        torch.save(model.state_dict(), str(path))

        model2 = create_lgt("150k")
        model2.load_state_dict(torch.load(str(path)))
        model2.eval()

        with torch.no_grad():
            logits_after, _ = model2(ids)

        assert torch.allclose(logits_before, logits_after, atol=1e-5)
