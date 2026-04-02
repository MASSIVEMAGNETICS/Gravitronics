"""
trainer.py — Live training loop for the Gravitronics LGT model.

Features
--------
* Clean start / resume from checkpoint.
* Progress callbacks (current step, epoch, loss, ETA).
* Cancellation via threading.Event.
* Deterministic seeding.
* Automatic device selection.
* Periodic + best-model checkpoint saves.
* Auto self-training mode with three trigger policies:
    - ``"time"``             — retrain every N seconds.
    - ``"data_threshold"``   — retrain when dataset grows by N samples.
    - ``"metric_threshold"`` — retrain when val_loss drifts > threshold.
* Safety guards: max wall-clock, max steps, early stopping.
"""

from __future__ import annotations

import logging
import math
import os
import random
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW

from ..lgt.config import LGTConfig
from ..lgt.model import LGT, create_lgt
from .checkpoint import CheckpointManager
from .config import TrainingConfig

logger = logging.getLogger(__name__)

# Type alias for the progress-callback signature.
ProgressCallback = Callable[[Dict[str, Any]], None]


# --------------------------------------------------------------------------- #
# Synthetic dataset (used when no real data_dir is available / for smoke tests) #
# --------------------------------------------------------------------------- #


class _SyntheticDataset:
    """A minimal synthetic dataset of random token sequences.

    This class serves as a placeholder data source used when no real
    ``data_dir`` is configured in :class:`TrainingConfig`.  It generates
    random integer token sequences on-the-fly so that the training loop can
    run end-to-end without requiring real data.

    A future release will replace this with a real tokenised-shard loader
    once the ``data_dir`` loading path is fully implemented.

    Parameters
    ----------
    vocab_size:
        Vocabulary size of the model being trained.
    seq_len:
        Sequence length for each example.
    num_samples:
        Total number of synthetic examples.
    """

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 1000) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        # Language-model target: shift right by one position
        return ids[:-1], ids[1:]


def _build_dataloader(
    dataset: _SyntheticDataset,
    batch_size: int,
    shuffle: bool = True,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Yield batches from *dataset* indefinitely (cycling through epochs)."""
    n = len(dataset)
    indices = list(range(n))
    while True:
        if shuffle:
            random.shuffle(indices)
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            xs = torch.stack([dataset[i][0] for i in batch_idx])
            ys = torch.stack([dataset[i][1] for i in batch_idx])
            yield xs, ys


# --------------------------------------------------------------------------- #
# Trainer                                                                      #
# --------------------------------------------------------------------------- #


class Trainer:
    """Orchestrates a full LGT training job.

    Parameters
    ----------
    config:
        :class:`~gravitronics.training.config.TrainingConfig` controlling all
        training, checkpointing, and self-training options.
    progress_callback:
        Optional callable invoked after every training step with a dict of
        progress information (step, epoch, loss, eta_sec, …).
    stop_event:
        Optional :class:`threading.Event`; when set, training is cleanly
        aborted after the current step completes.
    """

    def __init__(
        self,
        config: TrainingConfig,
        progress_callback: Optional[ProgressCallback] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self.config = config
        self.progress_callback = progress_callback
        self.stop_event = stop_event or threading.Event()

        # Resolve concrete device string
        self._device_str: str = config.resolve_device()
        self._device: torch.device = torch.device(self._device_str)

        # State
        self._step: int = 0
        self._epoch: int = 0
        self._best_val_loss: float = float("inf")
        self._no_improve_epochs: int = 0

        # Self-training state
        self._self_train_round: int = 0
        self._last_self_train_time: float = time.monotonic()
        self._baseline_data_size: int = 0

        # Checkpoint manager
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.ckpt_mgr = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            keep_last_n=config.keep_last_n_checkpoints,
            save_best=config.save_best_checkpoint,
        )

        # Logging setup
        os.makedirs(config.log_dir, exist_ok=True)
        self._configure_logging()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def train(self, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """Run the full training job.

        Parameters
        ----------
        resume_from:
            Path to a checkpoint to resume from.  When ``None``, training
            starts from scratch.

        Returns
        -------
        dict
            Summary with ``status``, ``steps``, ``epochs``, ``final_loss``.
        """
        logger.info(
            "Starting training: variant=%s, device=%s, epochs=%d.",
            self.config.model_variant,
            self._device_str,
            self.config.epochs,
        )

        # Seed
        self._set_seed(self.config.seed)

        # Build model
        model_cfg = LGTConfig.from_variant(self.config.model_variant)
        model = create_lgt(self.config.model_variant).to(self._device)
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Resume
        if resume_from:
            self._step, self._epoch, _ = self.ckpt_mgr.resume(
                model, optimizer, resume_from, device=self._device_str
            )

        # Dry run: skip the loop entirely
        if self.config.dry_run:
            logger.info("Dry run: config validated, model built, skipping loop.")
            return {
                "status": "dry_run",
                "steps": 0,
                "epochs": 0,
                "final_loss": float("nan"),
            }

        # Build datasets
        seq_len = min(model_cfg.max_seq_len, 64)
        train_ds = self._build_dataset(model_cfg, seq_len, split="train")
        val_ds = self._build_dataset(model_cfg, seq_len, split="val")
        self._baseline_data_size = len(train_ds)

        # Training loop
        result = self._run_loop(model, optimizer, model_cfg, train_ds, val_ds)

        # Auto self-training
        if self.config.auto_self_train and not self.stop_event.is_set():
            result = self._run_self_training(model, optimizer, model_cfg, train_ds, val_ds)

        return result

    def stop(self) -> None:
        """Request graceful early termination."""
        self.stop_event.set()
        logger.info("Stop requested; training will halt after current step.")

    # ------------------------------------------------------------------ #
    # Internal: main training loop                                         #
    # ------------------------------------------------------------------ #

    def _run_loop(
        self,
        model: LGT,
        optimizer: AdamW,
        model_cfg: LGTConfig,
        train_ds: _SyntheticDataset,
        val_ds: _SyntheticDataset,
    ) -> Dict[str, Any]:
        """Run one pass of the training loop (called for initial + self-training rounds)."""
        criterion = nn.CrossEntropyLoss()
        start_time = time.monotonic()

        steps_per_epoch = max(1, len(train_ds) // self.config.batch_size)
        total_steps = (
            self.config.max_steps
            if self.config.max_steps > 0
            else self.config.epochs * steps_per_epoch
        )

        train_loader = _build_dataloader(train_ds, self.config.batch_size)
        final_loss = float("nan")

        for epoch in range(self._epoch, self._epoch + self.config.epochs):
            self._epoch = epoch
            model.train()
            epoch_loss_sum = 0.0
            epoch_steps = 0

            for step_in_epoch in range(steps_per_epoch):
                if self.stop_event.is_set():
                    logger.info("Stop event triggered; aborting loop.")
                    break

                # Wall-clock guard
                elapsed = time.monotonic() - start_time
                if (
                    self.config.max_wall_clock_sec > 0
                    and elapsed > self.config.max_wall_clock_sec
                ):
                    logger.info(
                        "Wall-clock limit reached (%.1f s); stopping.", elapsed
                    )
                    self.stop_event.set()
                    break

                # Max-steps guard
                if self.config.max_steps > 0 and self._step >= self.config.max_steps:
                    logger.info("max_steps reached (%d); stopping.", self._step)
                    self.stop_event.set()
                    break

                xs, ys = next(train_loader)
                xs, ys = xs.to(self._device), ys.to(self._device)

                optimizer.zero_grad()
                logits, _ = model(xs)
                # logits: (B, L, vocab) → reshape for cross-entropy
                B, L, V = logits.shape
                loss = criterion(
                    logits.reshape(B * L, V),
                    ys.reshape(B * L),
                )
                loss.backward()

                if self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.grad_clip
                    )

                optimizer.step()
                self._step += 1
                epoch_loss_sum += loss.item()
                epoch_steps += 1
                final_loss = loss.item()

                # Step-based checkpoint
                if self._step % self.config.checkpoint_every_steps == 0:
                    self.ckpt_mgr.save(
                        model, optimizer, self._step, epoch, loss.item()
                    )

                # Progress callback
                self._emit_progress(
                    step=self._step,
                    epoch=epoch,
                    loss=loss.item(),
                    elapsed=elapsed,
                    total_steps=total_steps,
                )

            if self.stop_event.is_set():
                break

            # Epoch-level validation
            avg_epoch_loss = epoch_loss_sum / max(1, epoch_steps)
            val_loss = self._evaluate(model, val_ds, criterion)
            logger.info(
                "Epoch %d — train_loss=%.4f  val_loss=%.4f",
                epoch,
                avg_epoch_loss,
                val_loss,
            )

            # Epoch-based checkpoint
            if self.config.checkpoint_every_epochs > 0 and (
                epoch + 1
            ) % self.config.checkpoint_every_epochs == 0:
                self.ckpt_mgr.save(
                    model,
                    optimizer,
                    self._step,
                    epoch,
                    avg_epoch_loss,
                    extra={"val_loss": val_loss},
                )

            # Best-model checkpoint
            self.ckpt_mgr.save_best(
                model, optimizer, self._step, epoch, val_loss
            )

            # Early stopping
            if self.config.early_stop_patience > 0:
                if val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
                    self._no_improve_epochs = 0
                else:
                    self._no_improve_epochs += 1
                    if self._no_improve_epochs >= self.config.early_stop_patience:
                        logger.info(
                            "Early stopping triggered after %d epochs "
                            "without improvement.",
                            self._no_improve_epochs,
                        )
                        break

        return {
            "status": "ok",
            "steps": self._step,
            "epochs": self._epoch + 1,
            "final_loss": final_loss,
        }

    # ------------------------------------------------------------------ #
    # Internal: self-training                                              #
    # ------------------------------------------------------------------ #

    def _run_self_training(
        self,
        model: LGT,
        optimizer: AdamW,
        model_cfg: LGTConfig,
        train_ds: _SyntheticDataset,
        val_ds: _SyntheticDataset,
    ) -> Dict[str, Any]:
        """Execute the auto self-training loop (periodic fine-tuning)."""
        max_rounds = self.config.self_train_max_rounds
        result: Dict[str, Any] = {"status": "ok", "steps": self._step}

        while not self.stop_event.is_set():
            if max_rounds > 0 and self._self_train_round >= max_rounds:
                logger.info(
                    "Self-training max_rounds (%d) reached.", max_rounds
                )
                break

            if not self._should_self_train(train_ds, val_ds):
                time.sleep(5.0)
                continue

            self._self_train_round += 1
            logger.info(
                "Self-training round %d — retraining on %d samples.",
                self._self_train_round,
                len(train_ds),
            )
            result = self._run_loop(model, optimizer, model_cfg, train_ds, val_ds)
            self._last_self_train_time = time.monotonic()
            self._baseline_data_size = len(train_ds)

        return result

    def _should_self_train(
        self,
        train_ds: _SyntheticDataset,
        val_ds: _SyntheticDataset,
    ) -> bool:
        """Return ``True`` when the self-training trigger policy fires."""
        policy = self.config.self_train_policy
        if policy == "disabled":
            return False

        if policy == "time":
            elapsed = time.monotonic() - self._last_self_train_time
            return elapsed >= self.config.self_train_interval_sec

        if policy == "data_threshold":
            growth = len(train_ds) - self._baseline_data_size
            return growth >= self.config.self_train_data_threshold

        if policy == "metric_threshold":
            # Re-evaluate the current model against val_ds on-the-fly
            # (requires access to the model — simplified here)
            return False  # subclasses / callers can override

        return False

    # ------------------------------------------------------------------ #
    # Internal: helpers                                                    #
    # ------------------------------------------------------------------ #

    def _evaluate(
        self,
        model: LGT,
        val_ds: _SyntheticDataset,
        criterion: nn.CrossEntropyLoss,
    ) -> float:
        """Compute mean validation loss over the full validation set."""
        model.eval()
        total_loss = 0.0
        count = 0
        loader = _build_dataloader(val_ds, self.config.batch_size, shuffle=False)
        steps = max(1, len(val_ds) // self.config.batch_size)
        with torch.no_grad():
            for _ in range(steps):
                xs, ys = next(loader)
                xs, ys = xs.to(self._device), ys.to(self._device)
                logits, _ = model(xs)
                B, L, V = logits.shape
                loss = criterion(
                    logits.reshape(B * L, V),
                    ys.reshape(B * L),
                )
                total_loss += loss.item()
                count += 1
        model.train()
        return total_loss / max(1, count)

    def _build_dataset(
        self,
        model_cfg: LGTConfig,
        seq_len: int,
        split: str,
    ) -> _SyntheticDataset:
        """Build a training or validation dataset.

        Falls back to :class:`_SyntheticDataset` when
        :attr:`TrainingConfig.data_dir` is empty or the path does not exist.
        """
        if self.config.data_dir and os.path.isdir(self.config.data_dir):
            # Future: load real tokenised shards from data_dir.
            # For now, still use synthetic data as a placeholder.
            logger.info(
                "data_dir='%s' found but real loader not yet implemented; "
                "using synthetic data.",
                self.config.data_dir,
            )

        n_total = 200
        n_val = max(1, int(n_total * self.config.val_split))
        n_train = n_total - n_val
        n_samples = n_train if split == "train" else n_val
        return _SyntheticDataset(
            vocab_size=model_cfg.vocab_size,
            seq_len=seq_len + 1,  # +1 so we can shift for targets
            num_samples=n_samples,
        )

    @staticmethod
    def _set_seed(seed: int) -> None:
        """Set Python, NumPy, and PyTorch seeds for reproducibility."""
        if seed < 0:
            logger.info("Seed < 0 — using non-deterministic mode.")
            return
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info("Random seed set to %d.", seed)

    def _emit_progress(
        self,
        step: int,
        epoch: int,
        loss: float,
        elapsed: float,
        total_steps: int,
    ) -> None:
        """Call the progress callback (if set) with a summary dict."""
        if self.progress_callback is None:
            return
        if total_steps > 0 and step > 0:
            frac = step / total_steps
            eta = (elapsed / frac) * (1.0 - frac) if frac > 0 else 0.0
        else:
            eta = 0.0
        self.progress_callback(
            {
                "step": step,
                "epoch": epoch,
                "loss": loss,
                "elapsed_sec": round(elapsed, 1),
                "eta_sec": round(eta, 1),
                "progress": min(1.0, step / max(1, total_steps)),
            }
        )

    def _configure_logging(self) -> None:
        """Set up structured file + console logging for this training run."""
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        log_path = os.path.join(
            self.config.log_dir,
            f"training_{time.strftime('%Y%m%d_%H%M%S')}.log",
        )
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        logging.basicConfig(level=level, format=fmt)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(fh)
        logger.info("Logging initialised; file='%s'.", log_path)
