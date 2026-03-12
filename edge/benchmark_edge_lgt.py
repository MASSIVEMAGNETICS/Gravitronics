"""
benchmark_edge_lgt.py — Performance benchmark suite for Gravitronics LGT.

Measures latency statistics, peak memory, and sustained throughput for any
LGT model variant.  Results can be printed as a formatted table and/or saved
to a JSON file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import tracemalloc
from typing import Any, Dict

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gravitronics.lgt.config import LGTConfig
from gravitronics.lgt.model import LGT, create_lgt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _make_dummy(config: LGTConfig, batch_size: int, device: str) -> torch.Tensor:
    """Create a dummy input tensor."""
    return torch.zeros(
        batch_size, min(config.max_seq_len, 64), dtype=torch.long, device=device
    )


# --------------------------------------------------------------------------- #
# Latency                                                                      #
# --------------------------------------------------------------------------- #


def benchmark_latency(
    model: LGT,
    config: LGTConfig,
    num_runs: int = 100,
    batch_size: int = 1,
    device: str = "cpu",
) -> Dict[str, float]:
    """Measure forward-pass latency over *num_runs* iterations.

    Parameters
    ----------
    model:
        LGT model to benchmark (must be in eval mode).
    config:
        Model configuration for input construction.
    num_runs:
        Number of timed iterations.
    batch_size:
        Batch size for each run.
    device:
        Device to run on.

    Returns
    -------
    dict
        Keys: ``mean_ms``, ``std_ms``, ``min_ms``, ``max_ms``,
        ``p50_ms``, ``p95_ms``, ``p99_ms``.
    """
    model = model.to(device)
    model.eval()
    dummy = _make_dummy(config, batch_size, device)

    # Warm-up
    warmup = max(5, num_runs // 10)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    times_ms = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            times_ms.append((time.perf_counter() - t0) * 1_000)

    arr = np.array(times_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


# --------------------------------------------------------------------------- #
# Memory                                                                       #
# --------------------------------------------------------------------------- #


def benchmark_memory(
    model: LGT,
    config: LGTConfig,
    batch_size: int = 1,
    device: str = "cpu",
) -> Dict[str, float]:
    """Measure peak memory consumption during a single forward pass.

    Uses :mod:`tracemalloc` for CPU measurements.

    Parameters
    ----------
    model:
        LGT model (eval mode).
    config:
        Model configuration.
    batch_size:
        Batch size.
    device:
        Device to run on.

    Returns
    -------
    dict
        Keys: ``peak_mb``, ``allocated_mb``.
    """
    model = model.to(device)
    model.eval()
    dummy = _make_dummy(config, batch_size, device)

    tracemalloc.start()
    try:
        with torch.no_grad():
            _ = model(dummy)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    peak_mb = peak / (1024 ** 2)
    allocated_mb = model.get_model_size_mb()

    return {
        "peak_mb": round(peak_mb, 3),
        "allocated_mb": round(allocated_mb, 3),
    }


# --------------------------------------------------------------------------- #
# Throughput                                                                   #
# --------------------------------------------------------------------------- #


def benchmark_throughput(
    model: LGT,
    config: LGTConfig,
    duration_sec: float = 5.0,
) -> Dict[str, float]:
    """Measure sustained throughput over a fixed wall-clock duration.

    Parameters
    ----------
    model:
        LGT model (eval mode).
    config:
        Model configuration.
    duration_sec:
        How many seconds to run.

    Returns
    -------
    dict
        Keys: ``tokens_per_second``, ``inferences_per_second``.
    """
    model.eval()
    seq_len = min(config.max_seq_len, 64)
    dummy = torch.zeros(1, seq_len, dtype=torch.long)
    count = 0
    deadline = time.perf_counter() + duration_sec

    with torch.no_grad():
        while time.perf_counter() < deadline:
            _ = model(dummy)
            count += 1

    actual_duration = duration_sec  # conservative
    tokens_per_second = (count * seq_len) / actual_duration

    return {
        "tokens_per_second": round(tokens_per_second, 1),
        "inferences_per_second": round(count / actual_duration, 1),
    }


# --------------------------------------------------------------------------- #
# Full benchmark                                                               #
# --------------------------------------------------------------------------- #


def run_full_benchmark(
    variant: str = "150k",
    device: str = "cpu",
    num_runs: int = 100,
    batch_size: int = 1,
    duration_sec: float = 5.0,
) -> Dict[str, Any]:
    """Create a model and run all benchmarks.

    Parameters
    ----------
    variant:
        LGT size variant (``"150k"``, ``"600k"``, or ``"2m"``).
    device:
        Compute device.
    num_runs:
        Latency iterations.
    batch_size:
        Batch size for latency/memory benchmarks.
    duration_sec:
        Duration for throughput benchmark.

    Returns
    -------
    dict
        Comprehensive report containing ``variant``, ``device``,
        ``parameter_count``, ``model_size_mb``, ``latency``, ``memory``,
        ``throughput``.
    """
    logger.info("Running full benchmark: variant=%s, device=%s", variant, device)
    model = create_lgt(variant)
    config = model.get_config()
    model.eval()

    latency = benchmark_latency(model, config, num_runs=num_runs, batch_size=batch_size, device=device)
    memory = benchmark_memory(model, config, batch_size=batch_size, device=device)
    throughput = benchmark_throughput(model, config, duration_sec=duration_sec)

    return {
        "variant": variant,
        "device": device,
        "parameter_count": model.count_parameters(),
        "model_size_mb": round(model.get_model_size_mb(), 3),
        "latency": latency,
        "memory": memory,
        "throughput": throughput,
    }


# --------------------------------------------------------------------------- #
# Pretty-print                                                                 #
# --------------------------------------------------------------------------- #


def _print_report(report: Dict[str, Any]) -> None:
    """Print a human-readable benchmark table to stdout."""
    sep = "─" * 50
    print(f"\n{'Gravitronics LGT Benchmark Report':^50}")
    print(sep)
    print(f"  Variant          : {report['variant']}")
    print(f"  Device           : {report['device']}")
    print(f"  Parameters       : {report['parameter_count']:,}")
    print(f"  Model size       : {report['model_size_mb']:.2f} MB")
    print(sep)
    lat = report["latency"]
    print("  Latency (ms)")
    print(f"    mean ± std     : {lat['mean_ms']:.2f} ± {lat['std_ms']:.2f}")
    print(f"    min / max      : {lat['min_ms']:.2f} / {lat['max_ms']:.2f}")
    print(f"    p50 / p95 / p99: {lat['p50_ms']:.2f} / {lat['p95_ms']:.2f} / {lat['p99_ms']:.2f}")
    print(sep)
    mem = report["memory"]
    print("  Memory")
    print(f"    peak (trace)   : {mem['peak_mb']:.3f} MB")
    print(f"    allocated (fp32): {mem['allocated_mb']:.3f} MB")
    print(sep)
    thr = report["throughput"]
    print("  Throughput")
    print(f"    tokens/s       : {thr['tokens_per_second']:.1f}")
    print(f"    inferences/s   : {thr['inferences_per_second']:.1f}")
    print(sep + "\n")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark Gravitronics LGT edge performance.")
    p.add_argument("--variant", default="150k", choices=["150k", "600k", "2m"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--num_runs", type=int, default=100)
    p.add_argument("--output", default=None, help="Save JSON report to this path.")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    report = run_full_benchmark(
        variant=args.variant,
        device=args.device,
        num_runs=args.num_runs,
    )
    _print_report(report)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"Report saved to {args.output}")
