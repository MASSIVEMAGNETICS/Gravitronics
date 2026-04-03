"""
Gravitronics CLI — command-line interface for the LGT training pipeline.

Usage
-----
::

    # Run the interactive setup wizard and then train
    python -m gravitronics.cli wizard

    # Train directly with explicit options
    python -m gravitronics.cli train \\
        --sources ./my_repo ./extra_file.py \\
        --variant 150k \\
        --epochs 5 \\
        --batch-size 32 \\
        --lr 3e-4 \\
        --checkpoint-dir ./checkpoints

    # Resume from a checkpoint
    python -m gravitronics.cli train \\
        --sources ./my_repo \\
        --resume checkpoints/150k_epoch002_step001000.pt

Sub-commands
------------
``wizard``  Launch the interactive :class:`~gravitronics.wizard.WizardSetup`
            and start training with the collected configuration.

``train``   Non-interactive training run.  All settings are specified via
            command-line flags.
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------

def cmd_wizard(args: argparse.Namespace) -> None:
    """Interactive wizard that collects config and launches training."""
    from gravitronics.wizard import RunConfig, run_wizard
    from gravitronics.training import Trainer, TrainerConfig

    cfg: RunConfig = run_wizard(interactive=True)

    if not cfg.sources:
        print(
            "\n[wizard] No source paths provided — please add at least one "
            "directory or file and re-run the wizard."
        )
        sys.exit(1)

    trainer_cfg = TrainerConfig(
        sources=cfg.sources,
        model_variant=cfg.model_variant,
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        checkpoint_dir=cfg.checkpoint_dir,
        device=cfg.device,
    )
    trainer = Trainer(trainer_cfg)
    trainer.train()
    print("\n[wizard] Training complete.")


def cmd_train(args: argparse.Namespace) -> None:
    """Non-interactive training sub-command."""
    from gravitronics.training import Trainer, TrainerConfig

    trainer_cfg = TrainerConfig(
        sources=args.sources,
        model_variant=args.variant,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every_n_steps=args.checkpoint_steps,
        device=args.device,
        seed=args.seed,
        log_every_n_steps=args.log_steps,
    )
    trainer = Trainer(trainer_cfg)

    if args.resume:
        trainer.resume_from(args.resume)

    losses = trainer.train()
    if losses:
        print(f"\n[train] Done. Final loss: {losses[-1]:.4f}")
    else:
        print("\n[train] Done (no steps were executed).")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gravitronics.cli",
        description="Gravitronics LGT — train a Gravitational Transformer on your code.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ---- wizard -----------------------------------------------------------
    sub.add_parser("wizard", help="Launch the interactive setup wizard.")

    # ---- train ------------------------------------------------------------
    train_p = sub.add_parser("train", help="Train LGT on files / repositories.")
    train_p.add_argument(
        "--sources",
        nargs="+",
        metavar="PATH",
        required=True,
        help="One or more directories / files to train on.",
    )
    train_p.add_argument(
        "--variant",
        choices=("150k", "600k", "2m"),
        default="150k",
        help="Model size variant (default: 150k).",
    )
    train_p.add_argument(
        "--epochs",
        type=int,
        default=3,
        metavar="N",
        help="Number of training epochs (default: 3).",
    )
    train_p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="Batch size (default: 16).",
    )
    train_p.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        metavar="LR",
        help="Peak learning rate for AdamW (default: 3e-4).",
    )
    train_p.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        metavar="DIR",
        help="Directory to write checkpoints (default: checkpoints).",
    )
    train_p.add_argument(
        "--checkpoint-steps",
        type=int,
        default=500,
        metavar="N",
        help="Save checkpoint every N optimiser steps (default: 500).",
    )
    train_p.add_argument(
        "--device",
        default="auto",
        help="Torch device string: auto / cpu / cuda / mps (default: auto).",
    )
    train_p.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="Random seed for reproducibility (default: 42).",
    )
    train_p.add_argument(
        "--log-steps",
        type=int,
        default=50,
        metavar="N",
        help="Log progress every N steps (default: 50).",
    )
    train_p.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT",
        help="Path to a .pt checkpoint to resume training from.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if args.command == "wizard":
        cmd_wizard(args)
    elif args.command == "train":
        cmd_train(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
