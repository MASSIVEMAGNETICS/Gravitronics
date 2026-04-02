"""
TrainingConfig — Hyperparameter + infrastructure configuration for the
Gravitronics training subsystem.

Covers:
- Dataset / data-loader settings
- Optimiser and scheduler hyper-parameters
- Device selection
- Checkpointing policy
- Export options
- Auto-self-training policy
- Safety guards (wall-clock limit, max-steps, early stopping)

The config is JSON-serialisable and can be saved/reloaded from disk so the
wizard and the CLI share a single source of truth.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Validation helpers                                                           #
# --------------------------------------------------------------------------- #

_VALID_DEVICES = {"cpu", "cuda", "mps", "auto"}
_VALID_PRESETS = {"basic", "advanced"}
_VALID_SELF_TRAIN_POLICIES = {"time", "data_threshold", "metric_threshold", "disabled"}
_VALID_EXPORT_FORMATS = {"pt", "onnx", "both"}


# --------------------------------------------------------------------------- #
# TrainingConfig                                                               #
# --------------------------------------------------------------------------- #


@dataclass
class TrainingConfig:
    """All settings required to configure and run a Gravitronics training job.

    Attributes
    ----------
    model_variant:
        LGT size variant — ``"150k"``, ``"600k"``, or ``"2m"``.
    preset:
        High-level training preset — ``"basic"`` (sane defaults) or
        ``"advanced"`` (expose all hyper-parameters in the wizard).
    data_dir:
        Path to the directory containing training data files.
    val_split:
        Fraction of data to reserve for validation (0 < val_split < 1).
    epochs:
        Number of full passes over the training data.
    max_steps:
        Hard cap on total optimiser steps (overrides *epochs* when reached).
    batch_size:
        Mini-batch size.
    learning_rate:
        Initial learning rate for the optimiser.
    weight_decay:
        L2 regularisation coefficient.
    grad_clip:
        Maximum gradient norm for gradient clipping (0 = disabled).
    seed:
        Random seed for reproducibility.  ``-1`` means non-deterministic.
    device:
        Compute device — ``"cpu"``, ``"cuda"``, ``"mps"``, or ``"auto"``
        (selects CUDA > MPS > CPU automatically).
    checkpoint_dir:
        Directory to write checkpoint files into.
    checkpoint_every_steps:
        Save a periodic checkpoint every N optimiser steps.
    checkpoint_every_epochs:
        Save a periodic checkpoint every N epochs (overrides step-based when
        set to a value > 0).
    keep_last_n_checkpoints:
        How many periodic checkpoints to retain on disk (oldest are deleted).
    save_best_checkpoint:
        Whether to keep a separate ``best_model.pt`` checkpoint based on the
        lowest validation loss.
    export_dir:
        Directory to write exported model files into.
    export_format:
        Export format — ``"pt"``, ``"onnx"``, or ``"both"``.
    export_quantization:
        Quantisation mode applied at export time (``"fp16"``, ``"int8"``,
        or ``"none"``).
    auto_self_train:
        Enable the periodic self-training / fine-tuning mode.
    self_train_policy:
        Trigger policy for self-training iterations —
        ``"time"`` | ``"data_threshold"`` | ``"metric_threshold"`` | ``"disabled"``.
    self_train_interval_sec:
        When *policy* is ``"time"``: retrain every N seconds of wall-clock
        time.
    self_train_data_threshold:
        When *policy* is ``"data_threshold"``: retrain once the dataset has
        grown by at least this many new samples.
    self_train_metric_threshold:
        When *policy* is ``"metric_threshold"``: retrain if validation loss
        degrades beyond this threshold above the best seen.
    self_train_max_rounds:
        Maximum number of self-training rounds (0 = unlimited).
    max_wall_clock_sec:
        Safety guard — abort training after this many seconds (0 = no limit).
    early_stop_patience:
        Stop training if validation loss has not improved for this many
        consecutive epoch evaluations (0 = disabled).
    log_dir:
        Directory to write structured log files into.
    log_level:
        Logging level string (``"DEBUG"``, ``"INFO"``, ``"WARNING"``).
    dry_run:
        When ``True``, validate the config and build the model but do not
        run the training loop.
    """

    # ── Model ────────────────────────────────────────────────────────────── #
    model_variant: str = "150k"
    preset: str = "basic"

    # ── Data ────────────────────────────────────────────────────────────── #
    data_dir: str = ""
    val_split: float = 0.1

    # ── Optimisation ────────────────────────────────────────────────────── #
    epochs: int = 10
    max_steps: int = 0  # 0 → use epochs only
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    seed: int = 42

    # ── Device ──────────────────────────────────────────────────────────── #
    device: str = "auto"

    # ── Checkpointing ───────────────────────────────────────────────────── #
    checkpoint_dir: str = "checkpoints"
    checkpoint_every_steps: int = 500
    checkpoint_every_epochs: int = 1
    keep_last_n_checkpoints: int = 3
    save_best_checkpoint: bool = True

    # ── Export ──────────────────────────────────────────────────────────── #
    export_dir: str = "exports"
    export_format: str = "pt"
    export_quantization: str = "fp16"

    # ── Self-training ───────────────────────────────────────────────────── #
    auto_self_train: bool = False
    self_train_policy: str = "time"
    self_train_interval_sec: float = 3600.0
    self_train_data_threshold: int = 1000
    self_train_metric_threshold: float = 0.05
    self_train_max_rounds: int = 0  # 0 = unlimited

    # ── Safety guards ───────────────────────────────────────────────────── #
    max_wall_clock_sec: float = 0.0  # 0 = no limit
    early_stop_patience: int = 0    # 0 = disabled

    # ── Logging ─────────────────────────────────────────────────────────── #
    log_dir: str = "logs"
    log_level: str = "INFO"

    # ── Misc ────────────────────────────────────────────────────────────── #
    dry_run: bool = False

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        """Validate field consistency after construction."""
        self.validate()

    def validate(self) -> None:
        """Raise :class:`ValueError` if any field is out of range or invalid."""
        errs: List[str] = []

        if self.model_variant not in ("150k", "600k", "2m"):
            errs.append(
                f"model_variant must be one of '150k', '600k', '2m'; "
                f"got '{self.model_variant}'."
            )
        if self.preset not in _VALID_PRESETS:
            errs.append(
                f"preset must be one of {sorted(_VALID_PRESETS)}; "
                f"got '{self.preset}'."
            )
        if not (0.0 < self.val_split < 1.0):
            errs.append(f"val_split must be in (0, 1); got {self.val_split}.")
        if self.epochs < 1:
            errs.append(f"epochs must be ≥ 1; got {self.epochs}.")
        if self.max_steps < 0:
            errs.append(f"max_steps must be ≥ 0; got {self.max_steps}.")
        if self.batch_size < 1:
            errs.append(f"batch_size must be ≥ 1; got {self.batch_size}.")
        if self.learning_rate <= 0:
            errs.append(f"learning_rate must be > 0; got {self.learning_rate}.")
        if self.weight_decay < 0:
            errs.append(f"weight_decay must be ≥ 0; got {self.weight_decay}.")
        if self.grad_clip < 0:
            errs.append(f"grad_clip must be ≥ 0; got {self.grad_clip}.")
        if self.device not in _VALID_DEVICES:
            errs.append(
                f"device must be one of {sorted(_VALID_DEVICES)}; "
                f"got '{self.device}'."
            )
        if self.checkpoint_every_steps < 1:
            errs.append(
                f"checkpoint_every_steps must be ≥ 1; "
                f"got {self.checkpoint_every_steps}."
            )
        if self.checkpoint_every_epochs < 0:
            errs.append(
                f"checkpoint_every_epochs must be ≥ 0; "
                f"got {self.checkpoint_every_epochs}."
            )
        if self.keep_last_n_checkpoints < 1:
            errs.append(
                f"keep_last_n_checkpoints must be ≥ 1; "
                f"got {self.keep_last_n_checkpoints}."
            )
        if self.export_format not in _VALID_EXPORT_FORMATS:
            errs.append(
                f"export_format must be one of {sorted(_VALID_EXPORT_FORMATS)}; "
                f"got '{self.export_format}'."
            )
        if self.export_quantization not in ("fp16", "int8", "none"):
            errs.append(
                f"export_quantization must be 'fp16', 'int8', or 'none'; "
                f"got '{self.export_quantization}'."
            )
        if self.self_train_policy not in _VALID_SELF_TRAIN_POLICIES:
            errs.append(
                f"self_train_policy must be one of "
                f"{sorted(_VALID_SELF_TRAIN_POLICIES)}; "
                f"got '{self.self_train_policy}'."
            )
        if self.self_train_interval_sec <= 0:
            errs.append(
                f"self_train_interval_sec must be > 0; "
                f"got {self.self_train_interval_sec}."
            )
        if self.self_train_data_threshold < 1:
            errs.append(
                f"self_train_data_threshold must be ≥ 1; "
                f"got {self.self_train_data_threshold}."
            )
        if self.self_train_metric_threshold <= 0:
            errs.append(
                f"self_train_metric_threshold must be > 0; "
                f"got {self.self_train_metric_threshold}."
            )
        if self.self_train_max_rounds < 0:
            errs.append(
                f"self_train_max_rounds must be ≥ 0; "
                f"got {self.self_train_max_rounds}."
            )
        if self.max_wall_clock_sec < 0:
            errs.append(
                f"max_wall_clock_sec must be ≥ 0; "
                f"got {self.max_wall_clock_sec}."
            )
        if self.early_stop_patience < 0:
            errs.append(
                f"early_stop_patience must be ≥ 0; "
                f"got {self.early_stop_patience}."
            )
        if errs:
            raise ValueError(
                "TrainingConfig validation failed:\n"
                + "\n".join(f"  • {e}" for e in errs)
            )
        logger.debug("TrainingConfig validated OK.")

    # ------------------------------------------------------------------ #
    # Device resolution                                                    #
    # ------------------------------------------------------------------ #

    def resolve_device(self) -> str:
        """Return a concrete torch device string.

        When :attr:`device` is ``"auto"`` the best available device is
        selected automatically: CUDA > MPS > CPU.
        """
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of all config fields."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        """Construct a :class:`TrainingConfig` from a plain dictionary.

        Unknown keys are silently ignored to support forwards compatibility.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def save(self, path: str) -> None:
        """Serialise the config to a JSON file at *path*."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info("TrainingConfig saved to '%s'.", path)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load a :class:`TrainingConfig` from a JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        cfg = cls.from_dict(data)
        logger.info("TrainingConfig loaded from '%s'.", path)
        return cfg
