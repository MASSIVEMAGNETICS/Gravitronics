"""
Interactive setup wizard for configuring an LGT training run.

:class:`WizardSetup` prompts the user for the key settings and returns
a :class:`RunConfig` that can be passed straight to
:class:`~gravitronics.training.trainer.Trainer`.

The wizard can also run non-interactively: pass ``interactive=False`` to
:func:`run_wizard` and supply all values as keyword arguments, which is
useful for scripted or programmatic invocations (and tests).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

_SUPPORTED_VARIANTS = ("150k", "600k", "2m")


# ---------------------------------------------------------------------------
# RunConfig — the output of the wizard
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """Fully resolved configuration for a single training run.

    Attributes
    ----------
    sources:
        Directories and/or individual files to train on.
    model_variant:
        LGT model size — ``"150k"``, ``"600k"``, or ``"2m"``.
    num_epochs:
        Number of passes over the dataset.
    batch_size:
        Sequences per optimiser step.
    learning_rate:
        Peak AdamW learning rate.
    checkpoint_dir:
        Directory where checkpoints are written.
    device:
        PyTorch device string (``"auto"`` lets the system choose).
    """

    sources: List[str] = field(default_factory=list)
    model_variant: str = "150k"
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 3e-4
    checkpoint_dir: str = "checkpoints"
    device: str = "auto"


# ---------------------------------------------------------------------------
# WizardSetup
# ---------------------------------------------------------------------------

class WizardSetup:
    """Interactive (or scripted) assistant that builds a :class:`RunConfig`.

    Parameters
    ----------
    interactive:
        When ``True`` (default), the wizard prints prompts and reads from
        ``stdin``.  When ``False``, it silently uses the provided defaults.
    """

    def __init__(self, interactive: bool = True) -> None:
        self._interactive = interactive

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        sources: Optional[List[str]] = None,
        model_variant: Optional[str] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        checkpoint_dir: Optional[str] = None,
        device: Optional[str] = None,
    ) -> RunConfig:
        """Collect configuration, optionally prompting the user.

        Any value that is ``None`` is either prompted for (interactive mode)
        or replaced with the sensible default.

        Returns
        -------
        RunConfig
            Fully populated configuration ready for training.
        """
        if self._interactive:
            self._banner()

        cfg = RunConfig()

        # --- sources -------------------------------------------------------
        if sources is not None:
            cfg.sources = list(sources)
        elif self._interactive:
            cfg.sources = self._prompt_sources()
        else:
            cfg.sources = []

        # --- model variant -------------------------------------------------
        if model_variant is not None:
            cfg.model_variant = self._validate_variant(model_variant)
        elif self._interactive:
            cfg.model_variant = self._prompt_variant()

        # --- num_epochs ----------------------------------------------------
        if num_epochs is not None:
            cfg.num_epochs = int(num_epochs)
        elif self._interactive:
            cfg.num_epochs = self._prompt_int(
                "Number of epochs", default=cfg.num_epochs
            )

        # --- batch_size ----------------------------------------------------
        if batch_size is not None:
            cfg.batch_size = int(batch_size)
        elif self._interactive:
            cfg.batch_size = self._prompt_int(
                "Batch size", default=cfg.batch_size
            )

        # --- learning_rate -------------------------------------------------
        if learning_rate is not None:
            cfg.learning_rate = float(learning_rate)
        elif self._interactive:
            cfg.learning_rate = self._prompt_float(
                "Learning rate", default=cfg.learning_rate
            )

        # --- checkpoint_dir ------------------------------------------------
        if checkpoint_dir is not None:
            cfg.checkpoint_dir = str(checkpoint_dir)
        elif self._interactive:
            cfg.checkpoint_dir = self._prompt_str(
                "Checkpoint directory", default=cfg.checkpoint_dir
            )

        # --- device --------------------------------------------------------
        if device is not None:
            cfg.device = str(device)
        elif self._interactive:
            cfg.device = self._prompt_str("Device (auto/cpu/cuda/mps)", default=cfg.device)

        if self._interactive:
            self._summary(cfg)

        return cfg

    # ------------------------------------------------------------------
    # Private: prompting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _banner() -> None:
        print("\n" + "=" * 60)
        print("  Gravitronics LGT — Training Setup Wizard")
        print("=" * 60)
        print("  Press Enter to accept the [default] value.\n")

    @staticmethod
    def _summary(cfg: RunConfig) -> None:
        print("\n--- Configuration Summary ---")
        print(f"  Sources       : {cfg.sources or '(none — add files before training)'}")
        print(f"  Model variant : {cfg.model_variant}")
        print(f"  Epochs        : {cfg.num_epochs}")
        print(f"  Batch size    : {cfg.batch_size}")
        print(f"  Learning rate : {cfg.learning_rate}")
        print(f"  Checkpoints   : {cfg.checkpoint_dir}")
        print(f"  Device        : {cfg.device}")
        print("-" * 30 + "\n")

    def _prompt_sources(self) -> List[str]:
        print(
            "Enter the paths to train on (directories or files).\n"
            "  Separate multiple paths with commas.\n"
            "  Leave empty to add sources later via the CLI."
        )
        raw = input("  Sources: ").strip()
        if not raw:
            return []
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        missing = [p for p in parts if not os.path.exists(p)]
        if missing:
            logger.warning("The following paths do not exist: %s", missing)
        return parts

    def _prompt_variant(self) -> str:
        variants_str = " / ".join(_SUPPORTED_VARIANTS)
        while True:
            raw = input(f"  Model variant [{variants_str}] (default: 150k): ").strip()
            if not raw:
                return "150k"
            if raw in _SUPPORTED_VARIANTS:
                return raw
            print(f"  Unknown variant '{raw}'. Choose from: {variants_str}")

    @staticmethod
    def _validate_variant(v: str) -> str:
        if v not in _SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unknown model variant '{v}'. Choose from: {_SUPPORTED_VARIANTS}"
            )
        return v

    @staticmethod
    def _prompt_int(label: str, default: int) -> int:
        raw = input(f"  {label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            logger.warning("Invalid integer '%s', using default %d.", raw, default)
            return default

    @staticmethod
    def _prompt_float(label: str, default: float) -> float:
        raw = input(f"  {label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            logger.warning("Invalid float '%s', using default %g.", raw, default)
            return default

    @staticmethod
    def _prompt_str(label: str, default: str) -> str:
        raw = input(f"  {label} [{default}]: ").strip()
        return raw if raw else default


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def run_wizard(
    interactive: bool = True,
    **kwargs,
) -> RunConfig:
    """Create a :class:`WizardSetup` and immediately invoke :meth:`~WizardSetup.run`.

    Parameters
    ----------
    interactive:
        Whether to prompt the user (default ``True``).
    **kwargs:
        Keyword arguments forwarded to :meth:`WizardSetup.run`.

    Returns
    -------
    RunConfig
    """
    return WizardSetup(interactive=interactive).run(**kwargs)
