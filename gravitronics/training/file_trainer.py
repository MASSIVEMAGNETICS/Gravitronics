"""
file_trainer.py — Lightweight training loop for training LGT on source files.

This module provides a simpler, file-oriented training interface that
complements the full-featured :class:`~gravitronics.training.trainer.Trainer`.
Use :class:`FileTrainer` when you want to point at one or more local
directories / files and train without creating a :class:`~gravitronics.training.config.TrainingConfig` JSON first.

Byte-level next-token prediction
---------------------------------
The dataset uses byte-level tokenisation (token = raw byte, vocab_size=256).
This requires no external tokeniser library and works naturally with every
LGT model variant.

Typical usage::

    from gravitronics.training.file_trainer import FileTrainer, FileTrainerConfig

    cfg = FileTrainerConfig(sources=["./my_repo", "./extra.py"], num_epochs=3)
    trainer = FileTrainer(cfg)
    losses = trainer.train()
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from gravitronics.lgt.config import LGTConfig
from gravitronics.lgt.model import LGT, create_lgt
from gravitronics.training.checkpoints import save_checkpoint, load_checkpoint
from gravitronics.training.dataset import CodeDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FileTrainerConfig
# ---------------------------------------------------------------------------

@dataclass
class FileTrainerConfig:
    """Hyper-parameters for a file-based training run.

    Attributes
    ----------
    sources:
        List of file/directory paths used to build the training corpus.
    model_variant:
        LGT size tag — ``"150k"``, ``"600k"``, or ``"2m"``.
    num_epochs:
        Number of full passes over the dataset.
    batch_size:
        Number of sequences per optimiser step.
    learning_rate:
        Peak learning rate for AdamW.
    weight_decay:
        L2 regularisation coefficient for AdamW.
    grad_clip:
        Maximum gradient norm (0 to disable clipping).
    warmup_steps:
        Number of linear warm-up steps before cosine decay begins.
    stride:
        Sliding-window stride for :class:`~gravitronics.training.dataset.CodeDataset`.
        Defaults to ``seq_len`` (no overlap).
    checkpoint_dir:
        Directory where checkpoints are written.
    checkpoint_every_n_steps:
        Save a checkpoint every *n* optimiser steps (0 = only at epoch end).
    device:
        Torch device string, e.g. ``"cpu"``, ``"cuda"``, ``"mps"``.
        ``"auto"`` picks CUDA > MPS > CPU.
    seed:
        Random seed for reproducibility (None = do not seed).
    log_every_n_steps:
        Print a progress line every *n* steps.
    """

    sources: List[str] = field(default_factory=list)
    model_variant: str = "150k"
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 100
    stride: Optional[int] = None
    checkpoint_dir: str = "checkpoints"
    checkpoint_every_n_steps: int = 500
    device: str = "auto"
    seed: Optional[int] = 42
    log_every_n_steps: int = 50


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _cosine_lr_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Return a scheduler with linear warmup then cosine decay to 10 % of peak."""

    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


# ---------------------------------------------------------------------------
# FileTrainer
# ---------------------------------------------------------------------------

class FileTrainer:
    """Manages a file-based training run for an LGT model.

    Parameters
    ----------
    cfg:
        A fully populated :class:`FileTrainerConfig`.
    model:
        Optional pre-built :class:`~gravitronics.lgt.model.LGT` instance.
        If ``None``, a new model is created from ``cfg.model_variant``.
    dataset:
        Optional pre-built :class:`~gravitronics.training.dataset.CodeDataset`.
        If ``None``, one is constructed from ``cfg.sources``.
    """

    def __init__(
        self,
        cfg: FileTrainerConfig,
        *,
        model: Optional[LGT] = None,
        dataset: Optional[Dataset] = None,
    ) -> None:
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        # ---- Model --------------------------------------------------------
        self.model_cfg: LGTConfig = LGTConfig.from_variant(cfg.model_variant)
        if model is not None:
            self.model = model
        else:
            self.model = create_lgt(cfg.model_variant)
        self.model.to(self.device)

        # ---- Dataset & DataLoader ----------------------------------------
        if dataset is not None:
            self._dataset = dataset
        else:
            if not cfg.sources:
                raise ValueError(
                    "FileTrainerConfig.sources must contain at least one path when "
                    "no dataset is provided explicitly."
                )
            self._dataset = CodeDataset(
                sources=cfg.sources,
                seq_len=self.model_cfg.max_seq_len,
                stride=cfg.stride,
            )

        self._loader = DataLoader(
            self._dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=(self.device.type == "cuda"),
        )

        # ---- Optimiser & scheduler ---------------------------------------
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        total_steps = len(self._loader) * cfg.num_epochs
        self.scheduler = _cosine_lr_with_warmup(
            self.optimizer, cfg.warmup_steps, total_steps
        )

        # ---- Loss --------------------------------------------------------
        self.criterion = nn.CrossEntropyLoss()

        # ---- State -------------------------------------------------------
        self.global_step: int = 0
        self.current_epoch: int = 0
        self.best_loss: float = float("inf")
        self._loss_history: List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> List[float]:
        """Run the full training loop.

        Returns
        -------
        list[float]
            Per-step training loss values recorded throughout training.
        """
        logger.info(
            "FileTrainer: training %s  |  %d epoch(s)  |  %d batches/epoch  |  device=%s",
            self.cfg.model_variant,
            self.cfg.num_epochs,
            len(self._loader),
            self.device,
        )

        for epoch in range(self.cfg.num_epochs):
            self.current_epoch = epoch
            epoch_loss = self._run_epoch(epoch)
            logger.info(
                "Epoch %d/%d complete — avg loss: %.4f",
                epoch + 1,
                self.cfg.num_epochs,
                epoch_loss,
            )
            self._maybe_save_checkpoint(force=True)

        return list(self._loss_history)

    def train_step(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> float:
        """Execute a single forward + backward + optimise step.

        Parameters
        ----------
        input_ids:
            Token indices of shape ``(batch, seq_len)``.
        target_ids:
            Shifted token indices of shape ``(batch, seq_len)``.

        Returns
        -------
        float
            Scalar cross-entropy loss for this batch.
        """
        self.model.train()
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        logits, _ = self.model(input_ids)

        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
        )

        self.optimizer.zero_grad()
        loss.backward()

        if self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        scalar = loss.item()
        self._loss_history.append(scalar)
        if scalar < self.best_loss:
            self.best_loss = scalar

        return scalar

    def resume_from(self, checkpoint_path: Union[str, Path]) -> None:
        """Load model and optimiser weights from a checkpoint.

        Parameters
        ----------
        checkpoint_path:
            Path to a ``.pt`` file produced by
            :func:`~gravitronics.training.checkpoints.save_checkpoint`.
        """
        payload = load_checkpoint(
            checkpoint_path,
            self.model,
            optimizer=self.optimizer,
            map_location=self.device,
        )
        self.current_epoch = payload.get("epoch", 0)
        self.global_step = payload.get("step", 0)
        self.best_loss = payload.get("loss", float("inf"))
        logger.info(
            "Resumed from checkpoint — epoch=%d, step=%d",
            self.current_epoch,
            self.global_step,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_epoch(self, epoch: int) -> float:
        total_loss = 0.0
        t0 = time.time()

        for batch_idx, (input_ids, target_ids) in enumerate(self._loader):
            step_loss = self.train_step(input_ids, target_ids)
            total_loss += step_loss

            if (
                self.cfg.log_every_n_steps > 0
                and self.global_step % self.cfg.log_every_n_steps == 0
            ):
                elapsed = time.time() - t0
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    "epoch %d  step %d  batch %d/%d  loss=%.4f  lr=%.2e  elapsed=%.1fs",
                    epoch + 1,
                    self.global_step,
                    batch_idx + 1,
                    len(self._loader),
                    step_loss,
                    lr,
                    elapsed,
                )

            if (
                self.cfg.checkpoint_every_n_steps > 0
                and self.global_step % self.cfg.checkpoint_every_n_steps == 0
            ):
                self._maybe_save_checkpoint()

        return total_loss / max(1, len(self._loader))

    def _maybe_save_checkpoint(self, force: bool = False) -> None:
        if not force and self.cfg.checkpoint_every_n_steps <= 0:
            return

        ckpt_path = (
            Path(self.cfg.checkpoint_dir)
            / f"{self.cfg.model_variant}_epoch{self.current_epoch:03d}_step{self.global_step:06d}.pt"
        )
        save_checkpoint(
            self.model,
            ckpt_path,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            step=self.global_step,
            loss=self._loss_history[-1] if self._loss_history else float("inf"),
            config=self.model_cfg,
        )
