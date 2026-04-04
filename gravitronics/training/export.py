"""
export.py — Model export utilities for the Gravitronics training subsystem.

Wraps the existing edge/export_edge_model.py functionality with:
* A single :func:`export_trained_model` entry-point that reads a
  :class:`TrainingConfig` and automatically exports to the configured dir.
* Optional ONNX export (requires ``onnx`` and ``onnxruntime`` packages).
* Metadata sidecar: ``export_metadata.json`` written alongside the model.
* A minimal :func:`load_exported_model` helper for inference.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..lgt.config import LGTConfig
from ..lgt.model import LGT
from .config import TrainingConfig

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_EDGE_DIR = os.path.join(_REPO_ROOT, "edge")
if _EDGE_DIR not in sys.path:
    sys.path.insert(0, _EDGE_DIR)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _save_metadata(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Export metadata saved to '%s'.", path)


# --------------------------------------------------------------------------- #
# Export                                                                       #
# --------------------------------------------------------------------------- #


def export_trained_model(
    model: LGT,
    config: TrainingConfig,
    model_cfg: Optional[LGTConfig] = None,
    step: int = 0,
    epoch: int = 0,
    val_loss: float = float("nan"),
) -> Dict[str, Any]:
    """Export *model* according to the settings in *config*.

    Supports:
    * Native PyTorch ``.pt`` (state dict **and** optionally TorchScript-traced)
    * ONNX (when ``export_format`` is ``"onnx"`` or ``"both"``)

    A ``export_metadata.json`` sidecar is always written to *config.export_dir*.

    Parameters
    ----------
    model:
        Trained :class:`~gravitronics.lgt.model.LGT` instance.
    config:
        :class:`TrainingConfig` controlling export format, quantisation, etc.
    model_cfg:
        Model architecture config; derived from *model* if not supplied.
    step, epoch, val_loss:
        Training state to record in the metadata sidecar.

    Returns
    -------
    dict
        Export summary with keys ``status``, ``files``, ``errors``.
    """
    if model_cfg is None:
        model_cfg = model.get_config()

    os.makedirs(config.export_dir, exist_ok=True)
    summary: Dict[str, Any] = {
        "status": "ok",
        "files": [],
        "errors": [],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_variant": config.model_variant,
        "step": step,
        "epoch": epoch,
        "val_loss": val_loss,
        "export_quantization": config.export_quantization,
    }

    fmt = config.export_format  # "pt" | "onnx" | "both"

    # ── PyTorch export ───────────────────────────────────────────────── #
    if fmt in ("pt", "both"):
        try:
            from export_edge_model import export_model as _edge_export

            pt_path = os.path.join(config.export_dir, "model.pt")
            result = _edge_export(
                model,
                output_path=pt_path,
                quantization=config.export_quantization,
                trace=False,  # state-dict export (safer for resume)
                config=model_cfg,
            )
            if result["status"] == "ok":
                summary["files"].append(pt_path)
                logger.info("PyTorch export OK: '%s'.", pt_path)
            else:
                summary["errors"].append(result.get("error", "unknown"))
                summary["status"] = "partial"
        except Exception as exc:
            logger.error("PyTorch export failed: %s", exc, exc_info=True)
            summary["errors"].append(str(exc))
            summary["status"] = "partial"

    # ── ONNX export ──────────────────────────────────────────────────── #
    if fmt in ("onnx", "both"):
        try:
            onnx_path = os.path.join(config.export_dir, "model.onnx")
            _export_onnx(model, model_cfg, onnx_path)
            summary["files"].append(onnx_path)
            logger.info("ONNX export OK: '%s'.", onnx_path)
        except Exception as exc:
            logger.error("ONNX export failed: %s", exc, exc_info=True)
            summary["errors"].append(f"ONNX: {exc}")
            summary["status"] = "partial"

    # ── Config sidecar ───────────────────────────────────────────────── #
    cfg_path = os.path.join(config.export_dir, "model_config.json")
    try:
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(model_cfg.to_dict(), fh, indent=2)
        summary["files"].append(cfg_path)
    except Exception as exc:
        logger.warning("Could not write model config sidecar: %s", exc)

    # ── Metadata sidecar ────────────────────────────────────────────── #
    meta_path = os.path.join(config.export_dir, "export_metadata.json")
    _save_metadata(meta_path, summary)
    summary["metadata_path"] = meta_path

    if not summary["errors"]:
        summary["status"] = "ok"

    return summary


# --------------------------------------------------------------------------- #
# ONNX helper                                                                  #
# --------------------------------------------------------------------------- #


def _export_onnx(model: LGT, config: LGTConfig, path: str) -> None:
    """Export *model* to ONNX format."""
    try:
        import onnx  # noqa: F401 (presence check)
    except ImportError as exc:
        raise ImportError(
            "ONNX export requires the 'onnx' package.  "
            "Install with: pip install onnx onnxruntime"
        ) from exc

    model.eval()
    seq_len = min(config.max_seq_len, 32)
    dummy = torch.zeros(1, seq_len, dtype=torch.long)

    class _LogitsWrapper(nn.Module):
        def __init__(self, m: LGT) -> None:
            super().__init__()
            self.m = m

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.m(x)
            return out[0] if isinstance(out, tuple) else out

    wrapper = _LogitsWrapper(model)
    wrapper.eval()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "logits": {0: "batch", 1: "seq"},
            },
            opset_version=14,
        )
    logger.info("ONNX model written to '%s'.", path)


# --------------------------------------------------------------------------- #
# Load helper                                                                  #
# --------------------------------------------------------------------------- #


def load_exported_model(
    export_dir: str,
    device: str = "cpu",
) -> LGT:
    """Load an exported LGT model from *export_dir* for inference.

    Expects ``model.pt`` (state dict) and ``model_config.json`` to be present.

    Parameters
    ----------
    export_dir:
        Directory produced by :func:`export_trained_model`.
    device:
        Torch device string.

    Returns
    -------
    LGT
        Model in eval mode on *device*.
    """
    from ..lgt.model import create_lgt

    cfg_path = os.path.join(export_dir, "model_config.json")
    pt_path = os.path.join(export_dir, "model.pt")

    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config sidecar not found: '{cfg_path}'.")
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"Model weights not found: '{pt_path}'.")

    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg_dict = json.load(fh)

    model_cfg = LGTConfig.from_dict(cfg_dict)
    model = create_lgt(model_cfg.model_variant).to(device)

    state = torch.load(pt_path, map_location=device, weights_only=True)
    # Checkpoint payload vs. raw state dict
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    logger.info(
        "Exported model loaded from '%s' on device '%s'.", export_dir, device
    )
    return model
