"""
victoros_lgt_edge.py — VictorOS runtime wrapper for LGT edge inference.

Provides a crash-proof, self-correcting inference interface with:
* Automatic model loading (traced TorchScript or state-dict).
* Timing, diagnostics capture, and a structured audit ledger.
* Health-check and mirror-layer snapshot endpoints.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gravitronics.lgt.config import LGTConfig
from gravitronics.lgt.diagnostics import DiagnosticsLogger, MirrorLayer
from gravitronics.lgt.model import LGT, create_lgt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class VictorOSLGTEdge:
    """Crash-proof edge runtime wrapper for a Gravitronics LGT model.

    Handles model loading, inference, diagnostics, and audit logging.

    Parameters
    ----------
    model_path:
        Path to a saved model file — either a TorchScript ``.pt`` file or a
        plain state-dict ``.pt`` file.  If the path does not exist the wrapper
        operates in *degraded* mode and all inference calls return error dicts.
    config_path:
        Optional path to a JSON file containing a serialised
        :class:`LGTConfig`.  When *None* the default ``"150k"`` config is used.
    device:
        Target inference device (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        self.model: Optional[nn.Module] = None
        self.config: Optional[LGTConfig] = None
        self._diag_logger: Optional[DiagnosticsLogger] = None

        self._load_config(config_path)
        self._load_model(model_path)
        self._setup_diagnostics()

        logger.info(
            "VictorOSLGTEdge ready: device=%s, model_loaded=%s",
            device,
            self.model is not None,
        )

    # ------------------------------------------------------------------ #
    # Initialisation helpers                                               #
    # ------------------------------------------------------------------ #

    def _load_config(self, config_path: Optional[str]) -> None:
        """Load LGTConfig from JSON or fall back to default."""
        try:
            if config_path and os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self.config = LGTConfig.from_dict(data)
                logger.info("Loaded config from '%s'.", config_path)
            else:
                self.config = LGTConfig.from_variant("150k")
                logger.info("Using default 150k config.")
        except Exception as exc:
            logger.error("Config load failed: %s", exc)
            self.config = LGTConfig.from_variant("150k")

    def _load_model(self, model_path: str) -> None:
        """Load model from disk.  Sets ``self.model = None`` on any failure."""
        if not os.path.exists(model_path):
            logger.warning("Model file not found: '%s'.  Running in degraded mode.", model_path)
            return

        try:
            # Attempt TorchScript load first
            loaded = torch.jit.load(model_path, map_location=self.device)
            self.model = loaded
            logger.info("Loaded TorchScript model from '%s'.", model_path)
            return
        except Exception:
            pass  # Not a TorchScript file — try state dict

        try:
            assert self.config is not None
            lgt = create_lgt(self.config.model_variant)
            state = torch.load(model_path, map_location=self.device)
            lgt.load_state_dict(state)
            lgt.to(self.device)
            lgt.eval()
            self.model = lgt
            logger.info("Loaded state-dict model from '%s'.", model_path)
        except Exception as exc:
            logger.error("Model load failed: %s", exc)
            self.model = None

    def _setup_diagnostics(self) -> None:
        """Initialise diagnostics logger (writes to lgt_diagnostics.jsonl)."""
        try:
            self._diag_logger = DiagnosticsLogger("lgt_diagnostics.jsonl")
        except Exception as exc:
            logger.warning("DiagnosticsLogger setup failed: %s", exc)
            self._diag_logger = None

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def infer(
        self,
        input_ids,
        mask=None,
    ) -> Dict[str, Any]:
        """Run a forward pass and return structured result.

        Parameters
        ----------
        input_ids:
            Integer tensor or list of shape ``(batch, seq_len)``.
        mask:
            Optional attention mask.

        Returns
        -------
        dict
            Keys: ``logits``, ``diagnostics``, ``status`` (``"ok"`` or
            ``"error"``), ``error`` (only on error), ``elapsed_ms``.
        """
        if self.model is None:
            return {
                "logits": None,
                "diagnostics": [],
                "status": "error",
                "error": "Model not loaded.",
                "elapsed_ms": 0.0,
            }

        def _run() -> Dict[str, Any]:
            t0 = time.perf_counter()
            if not isinstance(input_ids, torch.Tensor):
                ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
            else:
                ids = input_ids.to(self.device)

            with torch.no_grad():
                out = self.model(ids) if mask is None else self.model(ids, mask)

            elapsed_ms = (time.perf_counter() - t0) * 1_000

            if isinstance(out, tuple) and len(out) == 2:
                logits, diag_list = out
            elif isinstance(out, torch.Tensor):
                # Traced model returns only the logits tensor
                logits, diag_list = out, []
            else:
                logits, diag_list = out, []

            return {
                "logits": logits,
                "diagnostics": diag_list,
                "status": "ok",
                "elapsed_ms": round(elapsed_ms, 3),
            }

        # First attempt
        try:
            result = _run()
        except Exception as exc:
            logger.warning("Inference attempt 1 failed: %s.  Retrying…", exc)
            # Self-correcting: retry once
            try:
                result = _run()
            except Exception as exc2:
                logger.error("Inference attempt 2 failed: %s", exc2)
                result = {
                    "logits": None,
                    "diagnostics": [],
                    "status": "error",
                    "error": str(exc2),
                    "elapsed_ms": 0.0,
                }

        # Append audit ledger entry
        if self._diag_logger and result["status"] == "ok":
            try:
                entry = self.get_ledger_entry(input_ids, result["diagnostics"])
                self._diag_logger.log(entry)
            except Exception as exc:
                logger.debug("Ledger entry failed: %s", exc)

        return result

    # ------------------------------------------------------------------ #
    # Audit ledger                                                         #
    # ------------------------------------------------------------------ #

    def get_ledger_entry(
        self,
        input_ids,
        diagnostics: List[Dict],
    ) -> Dict[str, Any]:
        """Build a structured audit log entry.

        Parameters
        ----------
        input_ids:
            Input token IDs.
        diagnostics:
            Diagnostics list from a forward pass.

        Returns
        -------
        dict
            Audit entry with timestamp, input hash, and diagnostics.
        """
        try:
            if isinstance(input_ids, torch.Tensor):
                id_bytes = input_ids.cpu().numpy().tobytes()
            else:
                id_bytes = str(input_ids).encode()
            input_hash = hashlib.sha256(id_bytes).hexdigest()[:16]
        except Exception:
            input_hash = "unknown"

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_hash": input_hash,
            "diagnostics_summary": [
                {k: v for k, v in d.items() if not k.startswith("_")}
                for d in (diagnostics or [])
            ],
            "device": self.device,
            "model_variant": getattr(self.config, "model_variant", "unknown"),
        }

    # ------------------------------------------------------------------ #
    # Health & introspection                                               #
    # ------------------------------------------------------------------ #

    def health_check(self) -> Dict[str, Any]:
        """Return the current health status of the edge runtime.

        Returns
        -------
        dict
            Keys: ``model_loaded``, ``parameter_count``, ``device``,
            ``memory_mb``, ``config_variant``.
        """
        status: Dict[str, Any] = {
            "model_loaded": self.model is not None,
            "parameter_count": 0,
            "device": self.device,
            "memory_mb": 0.0,
            "config_variant": getattr(self.config, "model_variant", "unknown"),
        }

        try:
            import psutil
            proc = psutil.Process(os.getpid())
            status["memory_mb"] = round(proc.memory_info().rss / (1024 ** 2), 2)
        except Exception:
            pass

        if self.model is not None:
            try:
                if isinstance(self.model, LGT):
                    status["parameter_count"] = self.model.count_parameters()
                else:
                    status["parameter_count"] = sum(
                        p.numel()
                        for p in self.model.parameters()
                        if p.requires_grad
                    )
            except Exception:
                pass

        return status

    def get_mirror_snapshot(self) -> Dict[str, Any]:
        """Return the latest diagnostics snapshot from all MirrorLayers.

        Returns
        -------
        dict
            Mapping ``layer_N → snapshot_dict`` for each mirror layer found
            in the model.
        """
        snapshots: Dict[str, Any] = {}

        if not isinstance(self.model, LGT):
            return snapshots

        try:
            for i, block in enumerate(self.model.blocks):
                attn = block.attn
                if isinstance(attn, MirrorLayer):
                    snapshots[f"layer_{i}"] = attn.get_snapshot()
        except Exception as exc:
            logger.warning("get_mirror_snapshot failed: %s", exc)

        return snapshots

    # ------------------------------------------------------------------ #
    # Cleanup                                                              #
    # ------------------------------------------------------------------ #

    def __del__(self) -> None:
        """Release resources on garbage collection."""
        try:
            if self._diag_logger is not None:
                self._diag_logger.close()
        except Exception:
            pass
