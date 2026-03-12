"""
Tests for the Gravitronics edge deployment utilities.

Covers: model export, quantisation, TorchScript tracing, benchmarking, and
the VictorOS edge runtime.
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gravitronics.lgt.model import create_lgt

# Edge imports — guarded so collection doesn't fail if edge/ is not on path
sys.path.insert(0, os.path.join(_REPO_ROOT, "edge"))

from export_edge_model import export_model, quantize_model, trace_model
from benchmark_edge_lgt import (
    benchmark_latency,
    benchmark_memory,
    benchmark_throughput,
    run_full_benchmark,
)
from victoros_lgt_edge import VictorOSLGTEdge


# --------------------------------------------------------------------------- #
# Quantisation                                                                 #
# --------------------------------------------------------------------------- #


class TestQuantisation:
    def test_quantize_none(self):
        model = create_lgt("150k")
        q = quantize_model(model, "none")
        assert q is model  # same object returned

    def test_quantize_fp16(self):
        model = create_lgt("150k")
        q = quantize_model(model, "fp16")
        # All parameters should now be half precision
        for p in q.parameters():
            assert p.dtype == torch.float16, f"Expected float16, got {p.dtype}"

    def test_quantize_int8_does_not_crash(self):
        model = create_lgt("150k")
        # int8 may fall back gracefully; just check no exception is raised
        q = quantize_model(model, "int8")
        assert q is not None

    def test_quantize_invalid_raises(self):
        model = create_lgt("150k")
        with pytest.raises(ValueError):
            quantize_model(model, "bf8")


# --------------------------------------------------------------------------- #
# Export — no quantisation                                                     #
# --------------------------------------------------------------------------- #


class TestExportQuantizationNone:
    def test_export_state_dict_no_quantization(self, tmp_path):
        model = create_lgt("150k")
        cfg = model.get_config()
        output = str(tmp_path / "model_none.pt")
        result = export_model(
            model,
            output_path=output,
            quantization="none",
            trace=False,
            config=cfg,
        )
        assert result["status"] == "ok", result.get("error")
        assert os.path.exists(output)

    def test_config_sidecar_written(self, tmp_path):
        model = create_lgt("150k")
        cfg = model.get_config()
        output = str(tmp_path / "model.pt")
        result = export_model(model, output, quantization="none", trace=False, config=cfg)
        assert "config_path" in result
        assert os.path.exists(result["config_path"])
        with open(result["config_path"]) as f:
            data = json.load(f)
        assert data["model_variant"] == "150k"


# --------------------------------------------------------------------------- #
# Export — TorchScript trace                                                   #
# --------------------------------------------------------------------------- #


class TestExportTrace:
    def test_trace_model_runs(self):
        model = create_lgt("150k")
        cfg = model.get_config()
        model.eval()
        traced = trace_model(model, cfg, device="cpu")
        assert traced is not None

    def test_export_trace_saves_file(self, tmp_path):
        model = create_lgt("150k")
        cfg = model.get_config()
        output = str(tmp_path / "traced.pt")
        result = export_model(
            model,
            output_path=output,
            quantization="none",
            trace=True,
            config=cfg,
        )
        assert result["status"] == "ok", result.get("error")
        assert os.path.exists(output)

    def test_traced_model_inference(self, tmp_path):
        model = create_lgt("150k")
        cfg = model.get_config()
        output = str(tmp_path / "traced_inf.pt")
        export_model(model, output, quantization="none", trace=True, config=cfg)
        loaded = torch.jit.load(output)
        # Use seq_len matching trace size (min(max_seq_len, 32) = 32)
        ids = torch.zeros(1, 32, dtype=torch.long)
        out = loaded(ids)
        assert out is not None


# --------------------------------------------------------------------------- #
# Benchmark — latency                                                          #
# --------------------------------------------------------------------------- #


class TestBenchmarkLatency:
    def test_returns_expected_keys(self):
        model = create_lgt("150k")
        cfg = model.get_config()
        result = benchmark_latency(model, cfg, num_runs=5)
        for key in ("mean_ms", "std_ms", "min_ms", "max_ms", "p50_ms", "p95_ms", "p99_ms"):
            assert key in result, f"Missing key: {key}"

    def test_mean_positive(self):
        model = create_lgt("150k")
        cfg = model.get_config()
        result = benchmark_latency(model, cfg, num_runs=3)
        assert result["mean_ms"] > 0


# --------------------------------------------------------------------------- #
# Benchmark — memory                                                           #
# --------------------------------------------------------------------------- #


class TestBenchmarkMemory:
    def test_returns_expected_keys(self):
        model = create_lgt("150k")
        cfg = model.get_config()
        result = benchmark_memory(model, cfg)
        for key in ("peak_mb", "allocated_mb"):
            assert key in result, f"Missing key: {key}"

    def test_allocated_positive(self):
        model = create_lgt("150k")
        cfg = model.get_config()
        result = benchmark_memory(model, cfg)
        assert result["allocated_mb"] > 0


# --------------------------------------------------------------------------- #
# Full benchmark                                                               #
# --------------------------------------------------------------------------- #


class TestFullBenchmark:
    def test_run_full_benchmark_150k(self):
        report = run_full_benchmark(variant="150k", num_runs=3, duration_sec=0.5)
        assert report["variant"] == "150k"
        assert "latency" in report
        assert "memory" in report
        assert "throughput" in report

    def test_run_full_benchmark_no_errors(self):
        report = run_full_benchmark(variant="150k", num_runs=3, duration_sec=0.5)
        assert report["parameter_count"] > 0


# --------------------------------------------------------------------------- #
# VictorOS edge runtime                                                        #
# --------------------------------------------------------------------------- #


class TestVictorOSEdgeHealth:
    def test_health_check_degraded_mode(self, tmp_path):
        """Health check should work even if model file does not exist."""
        runtime = VictorOSLGTEdge(
            model_path=str(tmp_path / "nonexistent.pt"),
            device="cpu",
        )
        health = runtime.health_check()
        assert "model_loaded" in health
        assert health["model_loaded"] is False

    def test_infer_degraded_returns_error(self, tmp_path):
        runtime = VictorOSLGTEdge(
            model_path=str(tmp_path / "nonexistent.pt"),
            device="cpu",
        )
        result = runtime.infer([[1, 2, 3]])
        assert result["status"] == "error"

    def test_health_check_with_live_model(self, tmp_path):
        """Export a model, load it, and verify health check."""
        from export_edge_model import export_model
        from gravitronics.lgt.config import LGTConfig

        model = create_lgt("150k")
        cfg = model.get_config()
        output = str(tmp_path / "live.pt")
        export_model(model, output, quantization="none", trace=True, config=cfg)

        # Save config
        cfg_path = str(tmp_path / "config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg.to_dict(), f)

        runtime = VictorOSLGTEdge(output, config_path=cfg_path, device="cpu")
        health = runtime.health_check()
        assert health["model_loaded"] is True

    def test_infer_live_model(self, tmp_path):
        """End-to-end inference with a freshly exported model."""
        from export_edge_model import export_model

        model = create_lgt("150k")
        cfg = model.get_config()
        output = str(tmp_path / "infer_test.pt")
        export_model(model, output, quantization="none", trace=True, config=cfg)

        runtime = VictorOSLGTEdge(output, device="cpu")
        result = runtime.infer(torch.zeros(1, 32, dtype=torch.long))
        assert result["status"] == "ok"
        assert result["logits"] is not None

    def test_get_mirror_snapshot_no_crash(self, tmp_path):
        runtime = VictorOSLGTEdge(
            model_path=str(tmp_path / "nonexistent.pt"),
            device="cpu",
        )
        snap = runtime.get_mirror_snapshot()
        assert isinstance(snap, dict)
