"""
export_edge_model.py — Export an LGT model for edge deployment.

Supports three quantisation modes (fp16, int8, none) and optional
TorchScript tracing.  Saves the exported model alongside a JSON
configuration file for easy loading on edge devices.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# Ensure the repo root is importable when run as a script
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
# Quantisation                                                                 #
# --------------------------------------------------------------------------- #


def quantize_model(model: nn.Module, quantization: str = "fp16") -> nn.Module:
    """Apply quantisation to *model* in-place (or via dynamic quant).

    Parameters
    ----------
    model:
        The PyTorch model to quantise.
    quantization:
        One of ``"fp16"``, ``"int8"``, or ``"none"``.

        * ``"fp16"``  — converts all parameters and buffers to half precision.
        * ``"int8"``  — applies ``torch.quantization.quantize_dynamic`` on
          ``nn.Linear`` layers.  Falls back to fp32 if unavailable.
        * ``"none"``  — returns the model unchanged.

    Returns
    -------
    nn.Module
        Quantised model (may be a new object for int8).
    """
    model.eval()

    if quantization == "fp16":
        logger.info("Applying fp16 quantisation.")
        try:
            return model.half()
        except Exception as exc:
            logger.warning("fp16 quantisation failed (%s); returning fp32 model.", exc)
            return model

    if quantization == "int8":
        logger.info("Applying int8 dynamic quantisation.")
        try:
            return torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
        except Exception as exc:
            logger.warning(
                "int8 quantisation unavailable (%s); returning original model.", exc
            )
            return model

    if quantization == "none":
        logger.info("No quantisation applied.")
        return model

    raise ValueError(
        f"Unknown quantization mode '{quantization}'.  "
        "Choose from: 'fp16', 'int8', 'none'."
    )


# --------------------------------------------------------------------------- #
# Tracing                                                                      #
# --------------------------------------------------------------------------- #


class _LogitsOnlyWrapper(nn.Module):
    """Thin wrapper that returns only logits (a plain Tensor) so that
    ``torch.jit.trace`` can handle the output without choking on the
    ``List[Dict]`` diagnostics payload.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # noqa: D102
        out = self.model(input_ids)
        if isinstance(out, tuple):
            return out[0]
        return out


def trace_model(
    model: nn.Module,
    config: LGTConfig,
    device: str = "cpu",
) -> torch.jit.ScriptModule:
    """Trace *model* to TorchScript using a dummy input.

    The model is wrapped in a thin :class:`_LogitsOnlyWrapper` so that the
    traced module returns a plain ``Tensor`` (logits only) — TorchScript
    cannot handle ``List[Dict]`` outputs from a traced graph.

    Parameters
    ----------
    model:
        The (quantised) model to trace.
    config:
        Model configuration — used to derive a suitable dummy input shape.
    device:
        Target device string (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    torch.jit.ScriptModule
        The traced TorchScript module.
    """
    model = model.to(device)
    model.eval()

    wrapper = _LogitsOnlyWrapper(model).to(device)
    wrapper.eval()

    batch_size = 1
    seq_len = min(config.max_seq_len, 32)  # keep tracing fast
    dummy_input = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    logger.info(
        "Tracing model with dummy input shape %s on device '%s'.",
        tuple(dummy_input.shape),
        device,
    )

    try:
        with torch.no_grad():
            traced = torch.jit.trace(
                wrapper,
                example_inputs=(dummy_input,),
                strict=False,
            )
        logger.info("TorchScript trace successful.")
        return traced
    except Exception as exc:
        logger.error("torch.jit.trace failed: %s", exc)
        raise


# --------------------------------------------------------------------------- #
# Export                                                                       #
# --------------------------------------------------------------------------- #


def export_model(
    model: nn.Module,
    output_path: str,
    quantization: str = "fp16",
    trace: bool = True,
    config: Optional[LGTConfig] = None,
) -> Dict[str, Any]:
    """Quantise, optionally trace, and save *model* to *output_path*.

    A JSON sidecar ``<output_path>.config.json`` is written alongside the
    model file containing the :class:`LGTConfig` and export metadata.

    Parameters
    ----------
    model:
        LGT model to export.
    output_path:
        Destination file path (e.g. ``"model.pt"``).
    quantization:
        Quantisation mode passed to :func:`quantize_model`.
    trace:
        Whether to TorchScript-trace the model before saving.
    config:
        Model configuration — required when *trace* is ``True``.

    Returns
    -------
    dict
        Export metadata dictionary with keys ``status``, ``output_path``,
        ``quantization``, ``traced``, ``size_mb``, ``timestamp``.
    """
    start = time.perf_counter()
    metadata: Dict[str, Any] = {
        "status": "ok",
        "output_path": output_path,
        "quantization": quantization,
        "traced": trace,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    try:
        # 1. Quantise
        q_model = quantize_model(model, quantization)

        # 2. Optionally trace
        if trace:
            if config is None:
                raise ValueError("config must be provided when trace=True.")
            exported = trace_model(q_model, config)
            torch.jit.save(exported, output_path)
            logger.info("Saved traced model to '%s'.", output_path)
        else:
            torch.save(q_model.state_dict(), output_path)
            logger.info("Saved state dict to '%s'.", output_path)

        # 3. Record file size
        size_bytes = os.path.getsize(output_path)
        metadata["size_mb"] = round(size_bytes / (1024 ** 2), 3)

        # 4. Save config sidecar
        if config is not None:
            config_path = output_path + ".config.json"
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump(config.to_dict(), fh, indent=2)
            logger.info("Saved config to '%s'.", config_path)
            metadata["config_path"] = config_path

    except Exception as exc:
        logger.error("export_model failed: %s", exc, exc_info=True)
        metadata["status"] = "error"
        metadata["error"] = str(exc)

    elapsed = time.perf_counter() - start
    metadata["elapsed_sec"] = round(elapsed, 3)
    return metadata


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export a Gravitronics LGT model for edge deployment."
    )
    p.add_argument(
        "--variant",
        default="150k",
        choices=["150k", "600k", "2m"],
        help="Model size variant (default: 150k).",
    )
    p.add_argument(
        "--output",
        default="lgt_edge.pt",
        help="Output file path (default: lgt_edge.pt).",
    )
    p.add_argument(
        "--quantization",
        default="fp16",
        choices=["fp16", "int8", "none"],
        help="Quantisation mode (default: fp16).",
    )
    p.add_argument(
        "--trace",
        action="store_true",
        default=True,
        help="TorchScript-trace the model (default: True).",
    )
    p.add_argument(
        "--no-trace",
        dest="trace",
        action="store_false",
        help="Disable TorchScript tracing.",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    lgt = create_lgt(args.variant)
    cfg = lgt.get_config()
    result = export_model(
        lgt,
        output_path=args.output,
        quantization=args.quantization,
        trace=args.trace,
        config=cfg,
    )
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "ok" else 1)
